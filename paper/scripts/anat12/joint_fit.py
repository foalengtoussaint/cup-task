"""Fit the segment-frame landmark offsets to ANGLES **and** VELOCITY at once -- is there a d that
serves both, or do the objectives genuinely conflict?

Objective per participant x arm:
    residual = [ angle residuals (deg, 3 series) ,  w * velocity residuals (mm/s, per-frame speed) ]
with each block normalised by its OWN raw RMS so w is dimensionless and w=1 means "weigh the two
equally". Sweeping w traces the PARETO FRONT: w=0 is the angle-only fit already run (which HURT
velocity: peak-speed r_s 0.798 -> 0.266); large w is velocity-first.

All out-of-sample (split-half), all from the cached prep. Velocity uses the scorer's own smoothing.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # anat12 modules live beside this file
import prep_cache
from vel_v2 import apply_seg, pose_of, pv_scorer, JN
from score_vs_automq import _planar_body_angles, _elbow_series
from pipeline.score import _smoothed_xyz, _hand_speed_mmps, DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER
FPS = 60.0

def speed_series(track):
    ok = np.isfinite(track).all(1)
    if ok.sum() < 20: return None
    return _hand_speed_mmps(_smoothed_xyz(track[ok], FPS, DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER), FPS)

def blocks(r, th):
    P = apply_seg(r, th)
    f, a = _planar_body_angles(P, r["side"], r["other"]); e = _elbow_series(P, r["side"])
    k = min(len(f), len(r["flex_mmc"]))
    ang = []
    for x,y in ((f[:k],r["flex_mmc"][:k]),(a[:k],r["abd_mmc"][:k]),(e[:k],r["elb_mmc"][:k])):
        v = x-y; ang.append(v[np.isfinite(v)])
    so = speed_series(P[f"{r['side']}_wrist"]); sm = r.get("_sm_cached")
    vel = np.zeros(0)
    if so is not None and sm is not None:
        kk = min(len(so), len(sm)); d = so[:kk]-sm[:kk]; vel = d[np.isfinite(d)]
    return np.concatenate(ang) if ang else np.zeros(0), vel

def make_resid(recs, w, sA, sV):
    def f(th):
        A, V = [], []
        for r in recs:
            a, v = blocks(r, th); A.append(a); V.append(v)
        A = np.concatenate(A) if A else np.zeros(1)
        V = np.concatenate(V) if V else np.zeros(1)
        return np.concatenate([A/sA, w*V/sV]) if w > 0 else A/sA
    return f

if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs:                       # precompute the fixed MMC speed series ONCE
        r["_sm_cached"] = speed_series(r["mmc_wrist"])
    import time as _t; _t0 = _t.time()
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    WEIGHTS = [10.0, 25.0]   # far end of the frontier
    rows = []
    for w in WEIGHTS:
        ang_err, pv_o, pv_m, flex_o, flex_m = [], [], [], [], []
        for (p,a), rs in sorted(groups.items()):
            for parity in (0,1):
                tr = [r for i,r in enumerate(rs) if i%2!=parity]
                te = [r for i,r in enumerate(rs) if i%2==parity]
                if len(tr)<4 or not te: continue
                A0 = np.concatenate([blocks(r, np.zeros(9))[0] for r in tr])
                V0 = np.concatenate([blocks(r, np.zeros(9))[1] for r in tr])
                sA = np.sqrt(np.mean(A0**2)) or 1.0; sV = np.sqrt(np.mean(V0**2)) or 1.0
                th = least_squares(make_resid(tr, w, sA, sV), x0=np.zeros(9),
                                   method="lm", max_nfev=600).x
                for r in te:
                    A, _ = blocks(r, th)
                    ang_err.append(np.sqrt(np.mean(A**2)))
                    P = apply_seg(r, th)
                    pv_o.append(pv_scorer(P[f"{r['side']}_wrist"])); pv_m.append(pv_scorer(r["mmc_wrist"]))
                    f_, _a = _planar_body_angles(P, r["side"], r["other"])
                    kk = min(len(f_), len(r["flex_mmc"]))
                    flex_o.append(float(np.nanmax(f_[:kk]))); flex_m.append(float(np.nanmax(r["flex_mmc"][:kk])))
        D = pd.DataFrame(dict(pv_o=pv_o, pv_m=pv_m, fo=flex_o, fm=flex_m)).dropna()
        rows.append(dict(w=w, ang_rmse=float(np.nanmedian(ang_err)),
                         flex_rs=spearmanr(D.fo, D.fm).correlation,
                         pv_rs=spearmanr(D.pv_o, D.pv_m).correlation,
                         pv_abserr=float((D.pv_o-D.pv_m).abs().median()), n=len(D)))
        print(f"[{_t.time()-_t0:5.0f}s] w={w:<5} angleRMSE {rows[-1]['ang_rmse']:6.2f}  flexion r_s {rows[-1]['flex_rs']:.3f}"
              f"   peakvel r_s {rows[-1]['pv_rs']:.3f}  |err| {rows[-1]['pv_abserr']:6.1f}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT/"out/scoring/joint_fit_pareto.csv", index=False)
    print("\nw=0 is the angle-only fit. RAW (no correction) reference: flexion r_s 0.449, "
          "peak-vel r_s 0.798, |err| 68.7")
