"""ANATOMICALLY-CONSTRAINED offset frames: a trunk component only where a trunk component can exist.

The user's point, and it is the right one: an ELBOW or WRIST marker is stuck to the arm segment, so
its displacement from our keypoint must rotate WITH that segment -- a trunk-fixed wrist offset would
mean the wrist marker sits (say) 20mm anterior in TRUNK space no matter which way the forearm points,
which no marker does. The SHOULDER is the exception: the acromion marker sits on the scapula/trunk,
and our COCO shoulder keypoint is a learned surface point, so that displacement is plausibly
trunk-fixed (and the earlier trunk-frame fit put 100-125mm there on exactly the 4 bad arms).

So the defensible models put the trunk block ONLY on the shoulder:
    anat9    shoulder TRUNK ,  elbow SEG   , wrist SEG      (9 params)
    anat12   shoulder TRUNK+SEG, elbow SEG , wrist SEG      (12)
compared against the two single-frame extremes already run (seg9, trunk9) and the unconstrained
hybrid18. If anat9 keeps trunk9's interjoint gain AND seg9's peak-angle gain, the free trunk blocks
on elbow/wrist in hybrid18 were buying nothing but overfit.

    nohup python anat_frame.py > out/scoring/anat_frame.log 2>&1 &
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # anat12 modules live beside this file
import prep_cache
from vel_v2 import pose_of, _basis
from joint_fit import speed_series
from score_vs_automq import _planar_body_angles, _elbow_series
from pipeline.score import _count_movement_units, DEFAULT_MU_AMPLITUDE_MMPS
FPS = 60.0

# per-landmark frame spec: "T" trunk-fixed, "S" segment-fixed, "TS" both
MODELS = {"anat9":  ("T",  "S", "S"),
          "anat12": ("TS", "S", "S")}
import os
# CFG env overrides the config list, e.g. CFG="anat12:0,anat12:1,anat12:4"; TAG suffixes the outputs.
CONFIGS = [(s.split(":")[0], float(s.split(":")[1])) for s in os.environ["CFG"].split(",")] \
    if os.environ.get("CFG") else [("anat9", 2.0), ("anat12", 2.0), ("anat9", 0.0)]
TAG = os.environ.get("TAG", "")
NPAR = {m: sum(6 if s == "TS" else 3 for s in spec) for m, spec in MODELS.items()}


def frames(r):
    P = pose_of(r); side = r["side"]; B = r["B"]
    sh, el, wr = P[f"{side}_shoulder"], P[f"{side}_elbow"], P[f"{side}_wrist"]
    down = np.broadcast_to(B[0], sh.shape)
    u = np.stack(_basis(el - sh, down), 0)            # upper-arm triad
    f = np.stack(_basis(wr - el, el - sh), 0)         # forearm triad
    T = np.broadcast_to(B[:, None, :], (3,) + sh.shape)
    return P, T, (u, u, f)


def apply(r, th, model):
    P, T, seg = frames(r)
    P = {j: v.copy() for j, v in P.items()}
    side = r["side"]; th = np.asarray(th, float); spec = MODELS[model]
    names = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
    i = 0
    for j, nm in enumerate(names):
        d = 0.0
        if "T" in spec[j]:
            a = th[i:i+3]; i += 3
            d = d + sum(a[k] * T[k] for k in range(3))
        if "S" in spec[j]:
            c = th[i:i+3]; i += 3
            d = d + sum(c[k] * seg[j][k] for k in range(3))
        P[nm] = P[nm] + d
    return P


def blocks(r, th, model):
    P = apply(r, th, model)
    fl, ab = _planar_body_angles(P, r["side"], r["other"]); el = _elbow_series(P, r["side"])
    k = min(len(fl), len(r["flex_mmc"]))
    ang = []
    for x, y in ((fl[:k], r["flex_mmc"][:k]), (ab[:k], r["abd_mmc"][:k]), (el[:k], r["elb_mmc"][:k])):
        v = x - y; ang.append(v[np.isfinite(v)])
    so, sm = speed_series(P[f"{r['side']}_wrist"]), r.get("_sm_cached")
    vel = np.zeros(0)
    if so is not None and sm is not None:
        kk = min(len(so), len(sm)); dv = so[:kk] - sm[:kk]; vel = dv[np.isfinite(dv)]
    return np.concatenate(ang) if ang else np.zeros(0), vel


def make_resid(recs, model, w, sA, sV):
    def fn(th):
        A, V = [], []
        for r in recs:
            a, v = blocks(r, th, model); A.append(a); V.append(v)
        A = np.concatenate(A) if A else np.zeros(1)
        V = np.concatenate(V) if V else np.zeros(1)
        return np.concatenate([A/sA, w*V/sV]) if w > 0 else A/sA
    return fn


def _corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 10 or np.std(a[k]) < 1e-9 or np.std(b[k]) < 1e-9: return np.nan
    return float(np.corrcoef(a[k], b[k])[0, 1])


def _meas(f, a, e, wrist, nr):
    sp = speed_series(wrist)
    return dict(flex=np.nanmax(f), abd=np.nanmax(a), elb=np.nanmax(e), ij=_corr(f[:nr], e[:nr]),
                pav=float(np.nanmax(np.abs(np.diff(e[:nr])))*FPS),
                pv=float(np.nanmax(sp)) if sp is not None else np.nan,
                ttp=float(np.nanargmax(sp)/FPS) if sp is not None else np.nan,
                mu=float(_count_movement_units(sp, DEFAULT_MU_AMPLITUDE_MMPS, 3)) if sp is not None else np.nan)


def measures_omc(r, th, model):
    P = apply(r, th, model); side = r["side"]
    f, a = _planar_body_angles(P, side, r["other"]); e = _elbow_series(P, side)
    return _meas(f, a, e, P[f"{side}_wrist"], int(min(r["n_reach"], len(f))))


def measures_mmc(r):
    f, a, e = r["flex_mmc"], r["abd_mmc"], r["elb_mmc"]
    return _meas(f, a, e, r["mmc_wrist"], int(min(r["n_reach"], len(f))))


def perturb(r, th, model):
    """Bias (mean shift) vs SHAPE (within-trial SD) of the angle change -- interjoint sees SHAPE only."""
    if "_raw" not in r:
        P0 = apply(r, np.zeros(NPAR[model]), model)
        r["_raw"] = (_planar_body_angles(P0, r["side"], r["other"])[0], _elbow_series(P0, r["side"]))
    f0, e0 = r["_raw"]
    P = apply(r, th, model)
    f1, _ = _planar_body_angles(P, r["side"], r["other"]); e1 = _elbow_series(P, r["side"])
    nr = int(min(r["n_reach"], len(f0), len(f1)))
    df, de = (f1 - f0)[:nr], (e1 - e0)[:nr]
    return dict(dflex_bias=float(np.nanmean(df)), dflex_shape=float(np.nanstd(df)),
                delb_bias=float(np.nanmean(de)), delb_shape=float(np.nanstd(de)))


KEYS = ["flex", "abd", "elb", "ij", "pav", "pv", "ttp", "mu"]
NAMES = dict(flex="max flexion", abd="max abduction", elb="max elbow ext", ij="interjoint",
             pav="peak elbow ANG vel", pv="peak wrist vel", ttp="time-to-peak", mu="movement units")

if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    print(f"{len(recs)} cached trials, {len(groups)} groups; params {NPAR}", flush=True)
    t0 = time.time(); out = {}; mags = []
    for model, w in CONFIGS:
        npar = NPAR[model]; rows = []; ins, oos = [], []
        for (p, a), rs in sorted(groups.items()):
            for parity in (0, 1):
                tr = [r for i, r in enumerate(rs) if i % 2 != parity]
                te = [r for i, r in enumerate(rs) if i % 2 == parity]
                if len(tr) < 4 or not te: continue
                A0 = np.concatenate([blocks(r, np.zeros(npar), model)[0] for r in tr])
                V0 = np.concatenate([blocks(r, np.zeros(npar), model)[1] for r in tr])
                sA = np.sqrt(np.mean(A0**2)) or 1.0; sV = np.sqrt(np.mean(V0**2)) or 1.0
                th = least_squares(make_resid(tr, model, w, sA, sV), x0=np.zeros(npar),
                                   method="lm", max_nfev=900).x
                ins += [np.sqrt(np.mean(blocks(r, th, model)[0]**2)) for r in tr]
                for r in te:
                    oos.append(np.sqrt(np.mean(blocks(r, th, model)[0]**2)))
                    o, m = measures_omc(r, th, model), measures_mmc(r)
                    rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS},
                                 **perturb(r, th, model), "part": p, "arm": a})
                mags.append(dict(model=model, w=w, part=p, arm=a, parity=parity,
                                 shoulder_trunk_mm=float(np.linalg.norm(th[0:3])),
                                 shoulder_seg_mm=float(np.linalg.norm(th[3:6])) if model == "anat12" else np.nan,
                                 elbow_seg_mm=float(np.linalg.norm(th[-6:-3])),
                                 wrist_seg_mm=float(np.linalg.norm(th[-3:])),
                                 # PERSIST theta itself -- every earlier script threw the fit away and
                                 # any new diagnostic then needed a full refit.
                                 **{f"th{i}": float(v) for i, v in enumerate(th)}))
            print(f"  [{time.time()-t0:5.0f}s] {model} w={w}  {p} {a}  ({len(rows)} test trials)", flush=True)
        out[(model, w)] = pd.DataFrame(rows)
        print(f"[{time.time()-t0:5.0f}s] {model} w={w} DONE  angle RMSE in-sample {np.median(ins):5.2f} "
              f"out-of-sample {np.median(oos):5.2f} deg  ({len(rows)} trials)", flush=True)

    rows = []
    for (p, a), rs in sorted(groups.items()):
        for r in rs:
            o, m = measures_omc(r, np.zeros(NPAR["anat9"]), "anat9"), measures_mmc(r)
            rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS}})
    out[("raw", 0.0)] = pd.DataFrame(rows)

    cols = [("raw", 0.0)] + CONFIGS
    print(f"\n{'measure':22}" + "".join(f"{(m if m=='raw' else m+f'/w{w:g}'):>13}" for m, w in cols) + "   <- r_s")
    for k in KEYS:
        cells = []
        for c in cols:
            D = out[c].dropna(subset=[f"o_{k}", f"m_{k}"])
            cells.append(spearmanr(D[f"o_{k}"], D[f"m_{k}"]).correlation if len(D) > 3 else np.nan)
        print(f"{NAMES[k]:22}" + "".join(f"{c:>13.3f}" for c in cells), flush=True)

    print("\nBIAS vs SHAPE of the angle change over reaching (deg):")
    print(f"{'model':14}{'flex bias':>11}{'flex SHAPE':>12}{'elbow bias':>12}{'elbow SHAPE':>13}")
    for m, w in CONFIGS:
        D = out[(m, w)]
        print(f"{m+'/w'+f'{w:g}':14}{D.dflex_bias.abs().median():>11.2f}{D.dflex_shape.median():>12.2f}"
              f"{D.delb_bias.abs().median():>12.2f}{D.delb_shape.median():>13.2f}", flush=True)

    M = pd.DataFrame(mags)
    print("\nFitted offset magnitudes (mm), median over folds:")
    print(M.groupby(["model", "w"])[["shoulder_trunk_mm", "shoulder_seg_mm", "elbow_seg_mm",
                                     "wrist_seg_mm"]].median().round(1))
    M.to_csv(ROOT/f"out/scoring/anat_frame_mags{TAG}.csv", index=False)
    pd.concat({f"{m}_w{w:g}": v for (m, w), v in out.items()}).to_csv(ROOT/f"out/scoring/anat_frame{TAG}.csv")
    print("\nDONE", flush=True)
