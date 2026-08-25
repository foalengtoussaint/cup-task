"""anat12 with a THIRD residual block: elbow ANGULAR speed.

Until now the objective saw four series -- flexion, abduction, elbow angle, wrist linear speed. Elbow
angular velocity was evaluation-only (it read 0.865 held-out at w=2). Wrist speed observes a POINT;
elbow angular rate depends on both segment DIRECTIONS, so it constrains the fit differently.

    residual = [ angles/sA , w * wrist_speed/sV , wE * elbow_angular_speed/sW ]
each block normalised by its own raw RMS. Config: anat12, w=2, wE=2 (matching the operating point).
Angular speed = |d(elbow angle)/dt| * FPS on BOTH sides, from the same low-passed angle series the
measure itself uses -- no separate smoothing, or the two sides would not be comparable.

    nohup python anat_evel.py > out/scoring/anat_evel.log 2>&1 &
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # anat12 modules live beside this file
import prep_cache
from anat_frame import apply, measures_omc, measures_mmc, KEYS, NAMES, NPAR
from joint_fit import speed_series
from score_vs_automq import _planar_body_angles, _elbow_series
FPS = 60.0
MODEL = "anat12"; W_VEL = 2.0; W_EANG = 2.0


def three_blocks(r, th):
    P = apply(r, th, MODEL)
    fl, ab = _planar_body_angles(P, r["side"], r["other"]); el = _elbow_series(P, r["side"])
    k = min(len(fl), len(r["flex_mmc"]))
    ang = []
    for x, y in ((fl[:k], r["flex_mmc"][:k]), (ab[:k], r["abd_mmc"][:k]), (el[:k], r["elb_mmc"][:k])):
        v = x - y; ang.append(v[np.isfinite(v)])
    # elbow angular speed, same operator both sides
    ao = np.abs(np.diff(el[:k])) * FPS; am = np.abs(np.diff(r["elb_mmc"][:k])) * FPS
    de = ao - am; eang = de[np.isfinite(de)]
    so, sm = speed_series(P[f"{r['side']}_wrist"]), r.get("_sm_cached")
    vel = np.zeros(0)
    if so is not None and sm is not None:
        kk = min(len(so), len(sm)); dv = so[:kk] - sm[:kk]; vel = dv[np.isfinite(dv)]
    return (np.concatenate(ang) if ang else np.zeros(0)), vel, eang


def make_resid(recs, sA, sV, sW):
    def fn(th):
        A, V, W = [], [], []
        for r in recs:
            a, v, e = three_blocks(r, th); A.append(a); V.append(v); W.append(e)
        return np.concatenate([np.concatenate(A)/sA, W_VEL*np.concatenate(V)/sV,
                               W_EANG*np.concatenate(W)/sW])
    return fn


if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    npar = NPAR[MODEL]
    print(f"{len(recs)} trials, {len(groups)} groups, {MODEL} w={W_VEL} wE={W_EANG}", flush=True)
    t0 = time.time(); rows = []; mags = []; ins, oos = [], []
    for (p, a), rs in sorted(groups.items()):
        for parity in (0, 1):
            tr = [r for i, r in enumerate(rs) if i % 2 != parity]
            te = [r for i, r in enumerate(rs) if i % 2 == parity]
            if len(tr) < 4 or not te: continue
            b0 = [three_blocks(r, np.zeros(npar)) for r in tr]
            sA = np.sqrt(np.mean(np.concatenate([x[0] for x in b0])**2)) or 1.0
            sV = np.sqrt(np.mean(np.concatenate([x[1] for x in b0])**2)) or 1.0
            sW = np.sqrt(np.mean(np.concatenate([x[2] for x in b0])**2)) or 1.0
            th = least_squares(make_resid(tr, sA, sV, sW), x0=np.zeros(npar), method="lm",
                               max_nfev=900).x
            ins += [np.sqrt(np.mean(three_blocks(r, th)[0]**2)) for r in tr]
            for r in te:
                oos.append(np.sqrt(np.mean(three_blocks(r, th)[0]**2)))
                o, m = measures_omc(r, th, MODEL), measures_mmc(r)
                rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS}})
            mags.append(dict(part=p, arm=a, parity=parity,
                             shoulder_trunk_mm=float(np.linalg.norm(th[0:3])),
                             shoulder_seg_mm=float(np.linalg.norm(th[3:6])),
                             elbow_seg_mm=float(np.linalg.norm(th[6:9])),
                             wrist_seg_mm=float(np.linalg.norm(th[9:12])),
                             **{f"th{i}": float(v) for i, v in enumerate(th)}))
        print(f"  [{time.time()-t0:5.0f}s] {p} {a}  ({len(rows)} test trials)", flush=True)
    D = pd.DataFrame(rows)
    print(f"\nangle RMSE in-sample {np.median(ins):.2f}  out-of-sample {np.median(oos):.2f} deg "
          f"({len(D)} trials)")
    print(f"PROCESSING CHECK: trials {len(D)}, non-finite per measure: "
          + ", ".join(f"{k} {int(D[f'o_{k}'].isna().sum())}" for k in KEYS))
    print(f"\n{'measure':22}{'anat12/w2+eang':>16}   <- r_s")
    for k in KEYS:
        g = D.dropna(subset=[f"o_{k}", f"m_{k}"])
        print(f"{NAMES[k]:22}{spearmanr(g[f'o_{k}'], g[f'm_{k}']).correlation:>16.3f}", flush=True)
    M = pd.DataFrame(mags)
    print("\nFitted magnitudes (mm, median over folds):")
    print(M[["shoulder_trunk_mm", "shoulder_seg_mm", "elbow_seg_mm", "wrist_seg_mm"]].median().round(1))
    M.to_csv(ROOT/"out/scoring/anat_evel_mags.csv", index=False)
    D.to_csv(ROOT/"out/scoring/anat_evel.csv", index=False)
    print("\nDONE", flush=True)
