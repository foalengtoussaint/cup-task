"""HYBRID landmark-offset frame: trunk-fixed component + segment-fixed (rotating) component.

Previous two models each assumed ONE frame for the MMC-vs-OMC landmark displacement:
  trunk9  d_j = sum_k th_k * B_k          (B = frozen per-trial trunk basis: down/fwd/lat)
  seg9    d_j = sum_k th_k * u_k(t)       (u = rotating upper-arm / forearm basis)
A real marker-vs-keypoint displacement need not be either: e.g. an acromion marker sits on the
SCAPULA (trunk-ish) while a soft-tissue elbow/wrist offset rides the SEGMENT. So fit BOTH parts:

  hybrid18   d_j(t) = A_j . B  +  C_j . u_j(t)      (6 params per landmark x 3 landmarks)

Identifiability: the two blocks separate only where the arm ROTATES relative to the trunk within a
trial -- which it does here (reach->drink->return). Split-half OOS is the judge: if hybrid is only
absorbing noise it will LOSE out of sample despite fitting better in sample. In-sample angle RMSE is
printed alongside so the gap is visible.

Same cache, same measures, same split-half protocol as all_measures_w.py.
    nohup python hybrid_frame.py > out/scoring/hybrid_frame.log 2>&1 &
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


def frames(r):
    """Per-landmark basis triads: trunk (frozen) and segment (rotating)."""
    P = pose_of(r); side = r["side"]; B = r["B"]
    sh, el, wr = P[f"{side}_shoulder"], P[f"{side}_elbow"], P[f"{side}_wrist"]
    down = np.broadcast_to(B[0], sh.shape)
    u = np.stack(_basis(el - sh, down), 0)          # upper-arm triad (3,n,3)
    f = np.stack(_basis(wr - el, el - sh), 0)       # forearm triad
    T = np.broadcast_to(B[:, None, :], (3,) + sh.shape)   # trunk triad, constant over the trial
    return P, T, u, f


def apply(r, th, model):
    """th: 9 params (trunk9 | seg9) or 18 (hybrid: [trunk9, seg9])."""
    P, T, u, f = frames(r)
    P = {j: v.copy() for j, v in P.items()}
    side = r["side"]; th = np.asarray(th, float)
    seg = (u, u, f)                                  # shoulder & elbow on upper arm, wrist on forearm
    names = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
    for i, nm in enumerate(names):
        d = 0.0
        if model in ("trunk9", "hybrid18"):
            a = th[3*i:3*i+3]
            d = d + sum(a[k] * T[k] for k in range(3))
        if model in ("seg9", "hybrid18"):
            off = 9 if model == "hybrid18" else 0
            c = th[off + 3*i: off + 3*i + 3]
            d = d + sum(c[k] * seg[i][k] for k in range(3))
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


def measures_omc(r, th, model):
    P = apply(r, th, model); side = r["side"]
    f, a = _planar_body_angles(P, side, r["other"]); e = _elbow_series(P, side)
    nr = int(min(r["n_reach"], len(f)))
    sp = speed_series(P[f"{side}_wrist"])
    return dict(flex=np.nanmax(f), abd=np.nanmax(a), elb=np.nanmax(e), ij=_corr(f[:nr], e[:nr]),
                pav=float(np.nanmax(np.abs(np.diff(e[:nr])))*FPS),
                pv=float(np.nanmax(sp)) if sp is not None else np.nan,
                ttp=float(np.nanargmax(sp)/FPS) if sp is not None else np.nan,
                mu=float(_count_movement_units(sp, DEFAULT_MU_AMPLITUDE_MMPS, 3)) if sp is not None else np.nan)


def measures_mmc_local(r):
    side = r["side"]; f, a, e = r["flex_mmc"], r["abd_mmc"], r["elb_mmc"]
    nr = int(min(r["n_reach"], len(f))); sp = speed_series(r["mmc_wrist"])
    return dict(flex=np.nanmax(f), abd=np.nanmax(a), elb=np.nanmax(e), ij=_corr(f[:nr], e[:nr]),
                pav=float(np.nanmax(np.abs(np.diff(e[:nr])))*FPS),
                pv=float(np.nanmax(sp)) if sp is not None else np.nan,
                ttp=float(np.nanargmax(sp)/FPS) if sp is not None else np.nan,
                mu=float(_count_movement_units(sp, DEFAULT_MU_AMPLITUDE_MMPS, 3)) if sp is not None else np.nan)


def perturb(r, th, model):
    """How the correction DEFORMS the angle time course: bias (mean shift) vs SHAPE (within-trial SD).

    Interjoint is corr(flexion, elbow) over reaching -- blind to any constant shift, sensitive only to
    shape. A segment-fixed offset rotates WITH the arm, so to first order it is a near-CONSTANT
    angular rotation (all bias, no shape); a trunk-fixed offset is posture-dependent
    (err ~ -(d.p_hat(phi))/L) so it varies through the reach. This measures that directly.
    """
    if "_raw_ang" not in r:
        P0 = apply(r, np.zeros(18), "hybrid18")
        f0, _ = _planar_body_angles(P0, r["side"], r["other"])
        r["_raw_ang"] = (f0, _elbow_series(P0, r["side"]))
    f0, e0 = r["_raw_ang"]
    P = apply(r, th, model)
    f1, _ = _planar_body_angles(P, r["side"], r["other"]); e1 = _elbow_series(P, r["side"])
    nr = int(min(r["n_reach"], len(f0), len(f1)))
    df, de = (f1 - f0)[:nr], (e1 - e0)[:nr]
    g = lambda v: (float(np.nanmean(v)), float(np.nanstd(v)))
    (fb, fs), (eb, es) = g(df), g(de)
    return dict(dflex_bias=fb, dflex_shape=fs, delb_bias=eb, delb_shape=es)


KEYS = ["flex", "abd", "elb", "ij", "pav", "pv", "ttp", "mu"]
NAMES = dict(flex="max flexion", abd="max abduction", elb="max elbow ext", ij="interjoint",
             pav="peak elbow ANG vel", pv="peak wrist vel", ttp="time-to-peak", mu="movement units")
CONFIGS = [("seg9", 2.0), ("trunk9", 2.0), ("hybrid18", 2.0), ("hybrid18", 0.0)]

if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    print(f"{len(recs)} cached trials, {len(groups)} participant x arm groups", flush=True)
    t0 = time.time(); out = {}; mags = []
    for model, w in CONFIGS:
        npar = 18 if model == "hybrid18" else 9
        rows = []; ins, oos = [], []
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
                    o, m = measures_omc(r, th, model), measures_mmc_local(r)
                    rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS},
                                 **perturb(r, th, model)})
                # fitted magnitudes (mm) per landmark
                for i, nm in enumerate(("shoulder", "elbow", "wrist")):
                    tp = np.linalg.norm(th[3*i:3*i+3]) if model in ("trunk9", "hybrid18") else 0.0
                    off = 9 if model == "hybrid18" else 0
                    sp_ = np.linalg.norm(th[off+3*i:off+3*i+3]) if model in ("seg9", "hybrid18") else 0.0
                    mags.append(dict(model=model, w=w, part=p, arm=a, parity=parity, landmark=nm,
                                     trunk_mm=tp, seg_mm=sp_))
            print(f"  [{time.time()-t0:5.0f}s] {model} w={w}  {p} {a}  ({len(rows)} test trials so far)",
                  flush=True)
        out[(model, w)] = pd.DataFrame(rows)
        print(f"[{time.time()-t0:5.0f}s] {model} w={w} DONE  angle RMSE in-sample {np.median(ins):5.2f}  "
              f"out-of-sample {np.median(oos):5.2f} deg  ({len(rows)} trials)", flush=True)

    rows = []
    for (p, a), rs in sorted(groups.items()):
        for r in rs:
            o, m = measures_omc(r, np.zeros(9), "seg9"), measures_mmc_local(r)
            rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS}})
    out[("raw", 0.0)] = pd.DataFrame(rows)

    cols = [("raw", 0.0)] + CONFIGS
    print(f"\n{'measure':22}" + "".join(f"{(m if m=='raw' else m+f'/w{w:g}'):>14}" for m, w in cols) + "   <- r_s")
    for k in KEYS:
        cells = []
        for c in cols:
            D = out[c].dropna(subset=[f"o_{k}", f"m_{k}"])
            cells.append(spearmanr(D[f"o_{k}"], D[f"m_{k}"]).correlation if len(D) > 3 else np.nan)
        print(f"{NAMES[k]:22}" + "".join(f"{c:>14.3f}" for c in cells), flush=True)

    print("\nHOW THE CORRECTION DEFORMS THE ANGLES over reaching (median |mean shift| = BIAS vs "
          "within-trial SD = SHAPE change, deg). Interjoint only responds to SHAPE:")
    print(f"{'model':16}{'flex bias':>11}{'flex SHAPE':>12}{'elbow bias':>12}{'elbow SHAPE':>13}")
    for m, w in CONFIGS:
        D = out[(m, w)]
        print(f"{m+'/w'+f'{w:g}':16}{D.dflex_bias.abs().median():>11.2f}{D.dflex_shape.median():>12.2f}"
              f"{D.delb_bias.abs().median():>12.2f}{D.delb_shape.median():>13.2f}", flush=True)

    M = pd.DataFrame(mags)
    print("\nFitted offset magnitude (mm), median over folds -- how the hybrid SPLITS the displacement:")
    print(M[M.model == "hybrid18"].groupby(["w", "landmark"])[["trunk_mm", "seg_mm"]].median().round(1))
    M.to_csv(ROOT/"out/scoring/hybrid_frame_mags.csv", index=False)
    pd.concat({f"{m}_w{w:g}": v for (m, w), v in out.items()}).to_csv(ROOT/"out/scoring/hybrid_frame.csv")
    print("\nDONE", flush=True)
