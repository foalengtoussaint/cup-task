"""CoTracker point-track vs flow vs SmoothNet vs blend on the MURPHY metrics (peak-speed, off-peak,
peak-timing). Reads CoTracker's cached tracks (cache/pointtrack/) — no re-tracking. P07+P08.

Answers: does tracking-through-time (CoTracker) fix optical-flow's +61mm/s peak over-shoot (blur)?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import flow_velocity_probe as F
from pipeline import pose_smooth
from pipeline.kalman_3d import triangulate_dlt
FPS = 60.0


def flow_sp(part, trial, joint, cams, n, method="pyrlk"):
    px = F.load_wrist_px(part, trial, joint); flow = {}
    for c in px:
        f = ROOT / "cache" / "flow_vel" / f"delta_{part}_{trial}.{c.split('_')[1]}__{method}.npy"
        if f.exists() and c in cams: flow[c] = np.load(f)
    sp = np.full(n, np.nan)
    for fr in range(n):
        op, opv = {}, {}
        for c in flow:
            if fr < len(px[c]) and np.isfinite(px[c][fr]).all() and fr < len(flow[c]) and np.isfinite(flow[c][fr]).all():
                op[c] = px[c][fr]; opv[c] = px[c][fr] + flow[c][fr]
        if len(op) < 2: continue
        Xp = triangulate_dlt([cams[c] for c in op], [np.array(op[c]) for c in op])
        Xv = triangulate_dlt([cams[c] for c in opv], [np.array(opv[c]) for c in opv])
        sp[fr] = np.linalg.norm(np.array(Xv) - np.array(Xp)) * FPS
    return sp


def cotrack_sp(part, trial, joint, cams, n):
    px = F.load_wrist_px(part, trial, joint); trk = {}
    for c in px:
        f = ROOT / "cache" / "pointtrack" / f"delta_{part}_{trial}.{c.split('_')[1]}__cotracker3.npy"
        if f.exists() and c in cams: trk[c] = np.load(f)
    X = np.full((n, 3), np.nan)
    for fr in range(n):
        obs = {c: trk[c][fr] for c in trk if fr < len(trk[c]) and np.isfinite(trk[c][fr]).all()}
        if len(obs) < 2: continue
        X[fr] = triangulate_dlt([cams[c] for c in obs], [np.array(obs[c]) for c in obs])
    sp = np.r_[np.nan, np.linalg.norm(np.diff(X, axis=0), axis=1) * FPS]
    both = np.isfinite(X[:-1]).all(1) & np.isfinite(X[1:]).all(1)
    out = np.full(n, np.nan); out[1:][both] = sp[1:][both]
    return out


def align(sig, oS):
    fl = H._lp(sig); best, bo = 1e9, H._lp(oS)
    for dl in range(-5, 6):
        os = H._lp(np.roll(oS, dl)); m = np.isfinite(os) & np.isfinite(fl)
        if m.sum() > 20 and np.median(np.abs(fl[m] - os[m])) < best:
            best = np.median(np.abs(fl[m] - os[m])); bo = os
    return fl, bo


def main():
    H.use_good_cams()
    TR = {"P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
          "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right")}
    R = {m: {"pf": [], "pk": [], "off": [], "tt": []} for m in
         ["smoothnet", "flow", "cotrack", "blend"]}
    for part, (trials, side) in TR.items():
        for trial in trials:
            joint = f"{side}_wrist"
            mmc, n = H._load_mmc(part, trial); omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[joint], omc[joint]); oS = H._speed(F._shift(omc[joint], lag))
            cams = H._load_calib_mm(part)
            if part in H.GOOD_CAMS: cams = {c: v for c, v in cams.items() if c in H.GOOD_CAMS[part]}
            snt = pose_smooth.smooth_track([{"frame": k, "X": (None if not np.isfinite(p).all() else list(p))}
                                            for k, p in enumerate(mmc[joint])])
            sp_sn = H._speed(np.array([t["X"] if t["X"] else [np.nan] * 3 for t in snt]))
            sp_fl = flow_sp(part, trial, joint, cams, n)
            sp_ct = cotrack_sp(part, trial, joint, cams, n)
            flL, snL = H._lp(sp_fl), H._lp(sp_sn)
            wb = 1 / (1 + np.exp(-(flL - 350) / 120))
            blend = (1 - wb) * np.where(np.isfinite(flL), flL, snL) + wb * snL
            for m, s in [("smoothnet", sp_sn), ("flow", sp_fl), ("cotrack", sp_ct), ("blend", blend)]:
                fl, o = align(s, oS)
                pf = np.isfinite(fl) & np.isfinite(o); R[m]["pf"].append(np.median(np.abs(fl[pf] - o[pf])))
                off = pf & (o < 300); R[m]["off"].append(np.median(np.abs(fl[off] - o[off])))
                for p in find_peaks(o, height=300, distance=30, prominence=150)[0]:
                    spk, _ = find_peaks(fl, height=200, distance=25, prominence=100)
                    if len(spk):
                        j = spk[np.argmin(np.abs(spk - p))]
                        if abs(j - p) <= 20:
                            R[m]["pk"].append(abs(fl[j] - o[p])); R[m]["tt"].append(abs(j - p) / FPS * 1000)
    print(f"{'method':10} {'per-frame':>10} {'PEAK':>7} {'off-peak':>9} {'peak-time mean':>15} {'time max':>9}")
    print("-" * 66)
    for m in ["smoothnet", "flow", "cotrack", "blend"]:
        a = R[m]
        print(f"{m:10} {np.median(a['pf']):8.1f}mm {np.median(a['pk']):5.1f}mm {np.median(a['off']):7.1f}mm "
              f"{np.mean(a['tt']):12.0f}ms {np.max(a['tt']) if a['tt'] else 0:6.0f}ms")
    print("\n(P07+P08. Does CoTracker fix flow's +61 peak over-shoot? cotrack = differentiate tracked pos.)")


if __name__ == "__main__":
    main()
