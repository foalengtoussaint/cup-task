"""Does the SEGMENT-FRAME landmark correction improve the VELOCITY measures?

Fixed from the previous attempt: peak speed is now computed the way pipeline.score does it --
_smoothed_xyz (Butterworth 4Hz, order 2) then _hand_speed_mmps -- on BOTH sides. The last run
differentiated the RAW MMC track against smoothed mocap, inflating our peak ~2x (1540 vs 660 mm/s)
and making the comparison meaningless.

Reads the cached prep (cache/omc_prep/*.npz), so no C3D parsing and no re-derivation.
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
from score_vs_automq import _planar_body_angles, _elbow_series
from pipeline.score import _smoothed_xyz, _hand_speed_mmps, DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER
FPS = 60.0
JN = ["right_shoulder","left_shoulder","right_elbow","left_elbow","right_wrist","left_wrist",
      "right_hip","left_hip","nose"]

def pose_of(r):  return {j: r[f"omc_{j}"] for j in JN}

def _basis(v, ref):
    nn = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
    e1 = nn(v); c = np.cross(e1, ref)
    bad = np.linalg.norm(c, axis=-1) < 1e-6
    if bad.any():
        alt = np.zeros_like(ref); alt[...,0] = 1.0
        c = np.where(bad[...,None], np.cross(e1, alt), c)
    e2 = nn(c); return e1, e2, nn(np.cross(e1, e2))

def apply_seg(r, th):
    P = {j: v.copy() for j, v in pose_of(r).items()}
    side = r["side"]; B = r["B"]; th = np.asarray(th, float)
    sh, el, wr = P[f"{side}_shoulder"], P[f"{side}_elbow"], P[f"{side}_wrist"]
    down = np.broadcast_to(B[0], sh.shape)
    u1,u2,u3 = _basis(el-sh, down); f1,f2,f3 = _basis(wr-el, el-sh)
    P[f"{side}_shoulder"] = sh + th[0]*u1 + th[1]*u2 + th[2]*u3
    P[f"{side}_elbow"]    = el + th[3]*u1 + th[4]*u2 + th[5]*u3
    P[f"{side}_wrist"]    = wr + th[6]*f1 + th[7]*f2 + th[8]*f3
    return P

def resid(th, recs):
    out=[]
    for r in recs:
        P = apply_seg(r, th)
        f, a = _planar_body_angles(P, r["side"], r["other"]); e = _elbow_series(P, r["side"])
        k = min(len(f), len(r["flex_mmc"]))
        for x,y in ((f[:k],r["flex_mmc"][:k]),(a[:k],r["abd_mmc"][:k]),(e[:k],r["elb_mmc"][:k])):
            v = x-y; out.append(v[np.isfinite(v)])
    return np.concatenate(out) if out else np.zeros(1)

def pv_scorer(track):
    """EXACTLY the scorer's path: smooth (Butterworth 4Hz) then differentiate."""
    ok = np.isfinite(track).all(1)
    if ok.sum() < 20: return np.nan
    sm = _smoothed_xyz(track[ok], FPS, DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER)
    return float(np.nanmax(_hand_speed_mmps(sm, FPS)))

if __name__ == "__main__":
    recs = prep_cache.load_all()
    print(f"loaded {len(recs)} cached trials (no C3D parsing)", flush=True)
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    raw, corr = [], []
    for (p,a), rs in sorted(groups.items()):
        for r in rs:
            raw.append(dict(part=p, arm=a, pv_omc=pv_scorer(pose_of(r)[f"{r['side']}_wrist"]),
                            pv_mmc=pv_scorer(r["mmc_wrist"])))
        for parity in (0,1):
            tr = [r for i,r in enumerate(rs) if i%2!=parity]
            te = [r for i,r in enumerate(rs) if i%2==parity]
            if len(tr)<4 or not te: continue
            th = least_squares(resid, x0=np.zeros(9), args=(tr,), method="lm", max_nfev=900).x
            for r in te:
                corr.append(dict(part=p, arm=a, pv_omc=pv_scorer(apply_seg(r, th)[f"{r['side']}_wrist"]),
                                 pv_mmc=pv_scorer(r["mmc_wrist"])))
        print(f"  {p} {a}", flush=True)
    R, C = pd.DataFrame(raw).dropna(), pd.DataFrame(corr).dropna()
    print(f"\nPEAK WRIST SPEED, scorer-matched smoothing on BOTH sides (Butterworth "
          f"{DEFAULT_LOWPASS_HZ}Hz order {DEFAULT_BUTTER_ORDER}):")
    print(f"{'set':16}{'r_s':>8}{'med OMC':>10}{'med MMC':>10}{'med bias':>10}{'med |err|':>11}   n")
    for lab, D in (("raw", R), ("seg-frame OOS", C)):
        b = D.pv_omc - D.pv_mmc
        print(f"{lab:16}{spearmanr(D.pv_omc,D.pv_mmc).correlation:>8.3f}{D.pv_omc.median():>10.1f}"
              f"{D.pv_mmc.median():>10.1f}{b.median():>10.1f}{b.abs().median():>11.1f}   {len(D)}")
    R.to_csv(ROOT/"out/scoring/vel_v2_raw.csv", index=False); C.to_csv(ROOT/"out/scoring/vel_v2_corr.csv", index=False)
    print("\nMMC side is IDENTICAL in both rows (our track is never corrected) -- sanity: "
          f"med MMC {R.pv_mmc.median():.1f} vs {C.pv_mmc.median():.1f}")
