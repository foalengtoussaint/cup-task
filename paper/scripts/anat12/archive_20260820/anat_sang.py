"""anat12 with a FOURTH residual block: SHOULDER angular speed (flexion rate + abduction rate).

Symmetric with the elbow block in anat_evel.py: |d(angle)/dt| * FPS on both sides, from the same
low-passed series the measures use. Flexion and abduction rates go in ONE block sharing one RMS
normaliser, so the shoulder gets the same total weight as the elbow rather than double.

    residual = [ angles/sA , 2*wrist_speed/sV , 2*elbow_ang_speed/sW , 2*shoulder_ang_speed/sS ]

Two configs, both anat12, both split-half OOS:
    +sang       angles + wrist speed + shoulder angular speed        (elbow rate OFF)
    +eang+sang  all four blocks
so the shoulder block's effect is visible both alone and on top of the elbow one.

    nohup python anat_sang.py > out/scoring/anat_sang.log 2>&1 &
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
from anat_frame import apply, measures_omc, measures_mmc, KEYS, NAMES, NPAR
from joint_fit import speed_series
from score_vs_automq import _planar_body_angles, _elbow_series
FPS = 60.0
MODEL = "anat12"; W_VEL = 2.0; W_EANG = 2.0; W_SANG = 2.0
CONFIGS = [("+eang+sang", True, True)]   # all three velocity blocks at once


def rate(x):
    return np.abs(np.diff(x)) * FPS


def four_blocks(r, th):
    P = apply(r, th, MODEL)
    fl, ab = _planar_body_angles(P, r["side"], r["other"]); el = _elbow_series(P, r["side"])
    k = min(len(fl), len(r["flex_mmc"]))
    fin = lambda v: v[np.isfinite(v)]
    ang = [fin(x[:k] - y[:k]) for x, y in ((fl, r["flex_mmc"]), (ab, r["abd_mmc"]), (el, r["elb_mmc"]))]
    eang = fin(rate(el[:k]) - rate(r["elb_mmc"][:k]))
    sang = np.concatenate([fin(rate(fl[:k]) - rate(r["flex_mmc"][:k])),
                           fin(rate(ab[:k]) - rate(r["abd_mmc"][:k]))])
    so, sm = speed_series(P[f"{r['side']}_wrist"]), r.get("_sm_cached")
    vel = np.zeros(0)
    if so is not None and sm is not None:
        kk = min(len(so), len(sm)); vel = fin(so[:kk] - sm[:kk])
    return np.concatenate(ang), vel, eang, sang


def make_resid(recs, use_e, use_s, sA, sV, sW, sS):
    def fn(th):
        B = [four_blocks(r, th) for r in recs]
        out = [np.concatenate([b[0] for b in B])/sA, W_VEL*np.concatenate([b[1] for b in B])/sV]
        if use_e: out.append(W_EANG*np.concatenate([b[2] for b in B])/sW)
        if use_s: out.append(W_SANG*np.concatenate([b[3] for b in B])/sS)
        return np.concatenate(out)
    return fn


if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    npar = NPAR[MODEL]
    print(f"{len(recs)} trials, {len(groups)} groups, {MODEL}, configs {[c[0] for c in CONFIGS]}",
          flush=True)
    t0 = time.time(); out = {}; mags = []
    for name, use_e, use_s in CONFIGS:
        rows = []; ins, oos = [], []
        for (p, a), rs in sorted(groups.items()):
            for parity in (0, 1):
                tr = [r for i, r in enumerate(rs) if i % 2 != parity]
                te = [r for i, r in enumerate(rs) if i % 2 == parity]
                if len(tr) < 4 or not te: continue
                b0 = [four_blocks(r, np.zeros(npar)) for r in tr]
                nrm = [np.sqrt(np.mean(np.concatenate([x[j] for x in b0])**2)) or 1.0 for j in range(4)]
                th = least_squares(make_resid(tr, use_e, use_s, *nrm), x0=np.zeros(npar),
                                   method="lm", max_nfev=900).x
                ins += [np.sqrt(np.mean(four_blocks(r, th)[0]**2)) for r in tr]
                for r in te:
                    oos.append(np.sqrt(np.mean(four_blocks(r, th)[0]**2)))
                    o, m = measures_omc(r, th, MODEL), measures_mmc(r)
                    rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS}})
                mags.append(dict(config=name, part=p, arm=a, parity=parity,
                                 shoulder_trunk_mm=float(np.linalg.norm(th[0:3])),
                                 shoulder_seg_mm=float(np.linalg.norm(th[3:6])),
                                 elbow_seg_mm=float(np.linalg.norm(th[6:9])),
                                 wrist_seg_mm=float(np.linalg.norm(th[9:12])),
                                 **{f"th{i}": float(v) for i, v in enumerate(th)}))
            print(f"  [{time.time()-t0:5.0f}s] {name}  {p} {a}  ({len(rows)} test trials)", flush=True)
        out[name] = pd.DataFrame(rows)
        print(f"[{time.time()-t0:5.0f}s] {name} DONE  angle RMSE in-sample {np.median(ins):.2f} "
              f"out-of-sample {np.median(oos):.2f} deg  ({len(rows)} trials)", flush=True)

    names = [c[0] for c in CONFIGS]
    print(f"\nPROCESSING CHECK: " + ", ".join(
        f"{n} {len(out[n])} trials, non-finite {int(out[n][[f'o_{k}' for k in KEYS]].isna().sum().sum())}"
        for n in names))
    print(f"\n{'measure':22}" + "".join(f"{n:>14}" for n in names) + "   <- r_s")
    for k in KEYS:
        cells = []
        for n in names:
            g = out[n].dropna(subset=[f"o_{k}", f"m_{k}"])
            cells.append(spearmanr(g[f"o_{k}"], g[f"m_{k}"]).correlation)
        print(f"{NAMES[k]:22}" + "".join(f"{c:>14.3f}" for c in cells), flush=True)
    M = pd.DataFrame(mags)
    print("\nFitted magnitudes (mm, median over folds):")
    print(M.groupby("config")[["shoulder_trunk_mm", "shoulder_seg_mm", "elbow_seg_mm",
                               "wrist_seg_mm"]].median().round(1))
    M.to_csv(ROOT/"out/scoring/anat_sang_mags.csv", index=False)
    pd.concat(out, names=["config"]).to_csv(ROOT/"out/scoring/anat_sang.csv")
    print("\nDONE", flush=True)
