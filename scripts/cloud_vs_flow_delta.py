"""Is the cloud pipeline better than the shipping flow-speed path? Measured on real DELTA data.

Both produce the SAME quantity -- 3D speed of the acting wrist, mm/s -- from the same cameras and
the same 2D keypoints, so they are directly comparable against OMC on the same frames.

    cloud   cup_task.cloud_velocity: Shi-Tomasi sub-features in a ROI, epipolar NCC matching,
            triangulate, Kabsch/RANSAC between consecutive clouds. Re-detects every frame.
    flow    cup_task.flow_speed (SHIPPING): PyrLK at the single keypoint, lift to 3D via the
            Jacobian with an l1 fuser. Tracks the point rather than re-detecting it.

Scored on MOVING frames (OMC > 50mm/s) and at the peaks, the same way as everywhere else in this
repo, plus coverage -- a method that only answers on easy frames is not comparable on error alone,
which is the selection-bias trap that made "v1 better for cup speed" look true once before.

    python scripts/cloud_vs_flow_delta.py --trials 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

MOVING = 50.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--roi", type=int, default=35)
    a = ap.parse_args(argv)

    import cv2
    from scipy.signal import find_peaks
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import flow_velocity_probe as F
    from cup_task import flow_speed as FS
    from cup_task.cloud_velocity import CloudVelocityTracker

    H.use_good_cams()
    FPS = 60.0
    acc = {k: {"mv": [], "peak": [], "cov": []} for k in ("cloud", "flow")}
    npts = []
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            ow = R._shift(omc[f"{side}_wrist"], lag)
            so = H._lp(H._speed(ow))

            px = F.load_wrist_px(part, trial, f"{side}_wrist")
            fl = {}
            for c in px:
                p = ROOT / "cache" / "flow_vel" / f"delta_{part}_{trial}.{c.split('_')[1]}__pyrlk.npy"
                if p.exists() and c in calib:
                    fl[c] = np.load(p)
            cams = [c for c in fl if c in px and c in calib]
            if len(cams) < 2:
                continue

            # --- SHIPPING flow path (cached, no video needed) ---
            s_flow = H._lp(FS.speed_from_cached_flow(px, fl, calib, n))

            # --- CLOUD path: needs the actual pixels ---
            # cloud_velocity takes plain dicts; adapt CamCalib without duplicating the geometry.
            cams_d = {c: {"K": np.asarray(calib[c].K), "R": np.asarray(calib[c].R),
                          "t": np.asarray(calib[c].t).ravel(), "P": None} for c in cams}
            caps = {}
            for c in cams:
                v = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{c.split('_')[1]}.mp4"
                if v.exists():
                    caps[c] = cv2.VideoCapture(str(v))
            if not caps:
                continue
            trk = CloudVelocityTracker({c: cams_d[c] for c in caps}, units_per_metre=1.0,
                                       roi_px=a.roi, min_features=4)
            s_cloud = np.full(n, np.nan)
            per_n = []
            for f in range(n):
                gray = {}
                for c, cap in caps.items():
                    ok, im = cap.read()
                    if ok:
                        gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                kp = {c: px[c][f] for c in caps
                      if f < len(px[c]) and np.isfinite(px[c][f]).all()}
                if len(kp) < 2 or len(gray) < 2:
                    continue
                # units_per_metre=1.0 => "m/s" comes back in mm/s, matching OMC directly
                r = trk.update(gray, kp, dt=1.0 / FPS)
                if r is not None and r.linear_speed is not None:
                    s_cloud[f] = r.linear_speed
                    per_n.append(r.active_3d_points_count)
            for c in caps.values():
                c.release()
            if per_n:
                npts.append(float(np.median(per_n)))
            s_cloud = H._lp(s_cloud)

            mv = np.isfinite(so) & (so > MOVING)
            opk = find_peaks(so, height=300, distance=30, prominence=150)[0]
            for k, s in (("cloud", s_cloud), ("flow", s_flow)):
                m = np.isfinite(s) & np.isfinite(so) & mv
                if m.sum() > 20:
                    acc[k]["mv"].append(float(np.median(np.abs(s[m] - so[m]))))
                acc[k]["cov"].append(float(np.isfinite(s[mv]).mean() * 100) if mv.sum() else np.nan)
                pk, _ = find_peaks(s, height=200, distance=25, prominence=100)
                for p_ in opk:
                    if len(pk):
                        j = pk[np.argmin(np.abs(pk - p_))]
                        if abs(j - p_) <= 20:
                            acc[k]["peak"].append(float(abs(s[j] - so[p_])))
            print(f"  {part}_{trial.split('_')[1]:>10}  "
                  f"cloud {acc['cloud']['mv'][-1] if acc['cloud']['mv'] else float('nan'):7.1f}  "
                  f"flow {acc['flow']['mv'][-1] if acc['flow']['mv'] else float('nan'):6.1f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    f_ = lambda v: np.median(v) if len(v) else float("nan")
    print(f"\n=== WRIST SPEED vs OMC (n={len(acc['flow']['mv'])} trials) ===")
    print(f"{'method':8} {'MOVING err':>11} {'PEAK err':>9} {'coverage':>9}")
    print("-" * 42)
    for k in ("cloud", "flow"):
        print(f"{k:8} {f_(acc[k]['mv']):9.1f}mm/s {f_(acc[k]['peak']):8.1f} "
              f"{f_(acc[k]['cov']):8.1f}%")
    if npts:
        print(f"\n  cloud inlier points per frame (median): {np.median(npts):.1f}")
    print("\n  Coverage matters as much as error: a method that answers only on easy frames wins")
    print("  on error for the wrong reason. Compare both columns together.")


if __name__ == "__main__":
    main()
