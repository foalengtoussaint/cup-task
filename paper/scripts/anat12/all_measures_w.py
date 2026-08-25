"""EVERY measure the rotating landmark correction can touch, at w = 0 / 1 / 2 / 4 (split-half OOS).

Under segment-frame (rotating) offsets the WRIST TRACK changes, so wrist-derived measures move too --
not just the angles. Computed here:
  angles      max flexion, max abduction, max elbow extension, interjoint (corr over reaching)
  angular vel peak elbow angular velocity  <- the measure that resisted every previous correction
  wrist       peak velocity, time-to-peak, movement units (scorer's own counter), trunk displacement
NOT computed: total movement time -- it is fixed by AutoMQ's phase boundaries in this pose-isolated
setup, hence identical on both sides by construction (Table III reads 1.00).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # anat12 modules live beside this file
import prep_cache
from joint_fit import blocks, make_resid, speed_series
from vel_v2 import apply_seg, pose_of
from score_vs_automq import _planar_body_angles, _elbow_series
from pipeline.score import _count_movement_units, DEFAULT_MU_AMPLITUDE_MMPS
FPS = 60.0

def _corr(a,b):
    k = np.isfinite(a)&np.isfinite(b)
    if k.sum()<10 or np.std(a[k])<1e-9 or np.std(b[k])<1e-9: return np.nan
    return float(np.corrcoef(a[k],b[k])[0,1])

def measures_omc(r, th):
    P = apply_seg(r, th)
    side = r["side"]
    f, a = _planar_body_angles(P, side, r["other"]); e = _elbow_series(P, side)
    nr = int(min(r["n_reach"], len(f)))
    sp = speed_series(P[f"{side}_wrist"])
    trunk = (P[f"{side}_shoulder"] + P[f"{r['other']}_shoulder"])/2.0
    td = np.linalg.norm(trunk - trunk[0], axis=1)
    return dict(flex=np.nanmax(f), abd=np.nanmax(a), elb=np.nanmax(e),
                ij=_corr(f[:nr], e[:nr]),
                pav=float(np.nanmax(np.abs(np.diff(e[:nr])))*FPS),
                pv=float(np.nanmax(sp)) if sp is not None else np.nan,
                ttp=float(np.nanargmax(sp)/FPS) if sp is not None else np.nan,
                mu=float(_count_movement_units(sp, DEFAULT_MU_AMPLITUDE_MMPS, 3)) if sp is not None else np.nan,
                trunk=float(np.nanmax(td)))

def measures_mmc(r):
    side = r["side"]
    f, a, e = r["flex_mmc"], r["abd_mmc"], r["elb_mmc"]
    nr = int(min(r["n_reach"], len(f)))
    sp = speed_series(r["mmc_wrist"])
    return dict(flex=np.nanmax(f), abd=np.nanmax(a), elb=np.nanmax(e),
                ij=_corr(f[:nr], e[:nr]),
                pav=float(np.nanmax(np.abs(np.diff(e[:nr])))*FPS),
                pv=float(np.nanmax(sp)) if sp is not None else np.nan,
                ttp=float(np.nanargmax(sp)/FPS) if sp is not None else np.nan,
                mu=float(_count_movement_units(sp, DEFAULT_MU_AMPLITUDE_MMPS, 3)) if sp is not None else np.nan,
                trunk=np.nan)   # MMC trunk needs both shoulders; not in cache -> skipped

KEYS = ["flex","abd","elb","ij","pav","pv","ttp","mu"]
if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    import time as T; t0=T.time()
    out = {}
    for w in (0.0, 1.0, 2.0, 4.0):
        rows=[]
        for (p,a), rs in sorted(groups.items()):
            for parity in (0,1):
                tr = [r for i,r in enumerate(rs) if i%2!=parity]
                te = [r for i,r in enumerate(rs) if i%2==parity]
                if len(tr)<4 or not te: continue
                A0 = np.concatenate([blocks(r, np.zeros(9))[0] for r in tr])
                V0 = np.concatenate([blocks(r, np.zeros(9))[1] for r in tr])
                sA = np.sqrt(np.mean(A0**2)) or 1.0; sV = np.sqrt(np.mean(V0**2)) or 1.0
                th = least_squares(make_resid(tr, w, sA, sV), x0=np.zeros(9), method="lm",
                                   max_nfev=600).x
                for r in te:
                    o, m = measures_omc(r, th), measures_mmc(r)
                    rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS}})
        out[w] = pd.DataFrame(rows)
        print(f"[{T.time()-t0:5.0f}s] w={w} done ({len(rows)} trials)", flush=True)
    # raw reference
    rows=[]
    for (p,a), rs in sorted(groups.items()):
        for r in rs:
            o, m = measures_omc(r, np.zeros(9)), measures_mmc(r)
            rows.append({**{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS}})
    out["raw"] = pd.DataFrame(rows)
    NAMES = dict(flex="max flexion", abd="max abduction", elb="max elbow ext",
                 ij="interjoint", pav="peak elbow ANG vel", pv="peak wrist vel",
                 ttp="time-to-peak", mu="movement units")
    print(f"\n{'measure':22}" + "".join(f"{str(k):>10}" for k in ("raw",0.0,1.0,2.0,4.0)) + "   <- r_s")
    for k in KEYS:
        cells=[]
        for w in ("raw",0.0,1.0,2.0,4.0):
            D = out[w].dropna(subset=[f"o_{k}", f"m_{k}"])
            cells.append(spearmanr(D[f"o_{k}"], D[f"m_{k}"]).correlation if len(D)>3 else np.nan)
        print(f"{NAMES[k]:22}" + "".join(f"{c:>10.3f}" for c in cells), flush=True)
    pd.concat({str(k): v for k,v in out.items()}).to_csv(ROOT/"out/scoring/all_measures_w.csv")
    print("\nTotal movement time omitted: fixed by AutoMQ phases here, identical both sides.")
