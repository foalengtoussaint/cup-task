"""Tracked surface cloud (cloud_track) vs the shipping flow path, on real DELTA cup data.

Scores the SAME quantity -- 3D cup speed, mm/s -- against OMC on the same frames, plus the thing
only a cloud can give: angular velocity. There is no OMC rotation truth for the cup here, so
rotation is reported as a plausibility check (magnitude, and whether it is quiet while the cup is
stationary) rather than an accuracy claim.

    python scripts/cloud_track_delta.py --trials 2
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
    ap.add_argument("--nseed", type=int, default=24)
    a = ap.parse_args(argv)

    import cv2
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    from cup_task import flow_speed as FS
    from cup_task.cloud_track import CloudTracker

    H.use_good_cams()
    FPS = 60.0
    acc = {k: {"mv": [], "cov": []} for k in ("cloud", "flow")}
    rot_moving, rot_rest, npts = [], [], []
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            oc = R._shift(R._omc_cup(part, trial, n), lag)
            so = H._lp(H._speed(oc))

            px = C.cup_px(part, trial, n)
            cup3 = R._smooth_joint(R._cup_v3(part, trial, calib, n))     # 3D cup, for seeding
            fl = {}
            for c in px:
                p = ROOT / "cache" / "cup_flow_vel" / f"delta_{part}_{trial}.{c.split('_')[1]}__pyrlk.npy"
                if p.exists() and c in calib:
                    fl[c] = np.load(p)
            cams = [c for c in px if c in calib]
            if len(cams) < 2:
                continue
            s_flow = H._lp(FS.speed_from_cached_flow(px, fl, calib, n)) if fl else np.full(n, np.nan)

            caps = {}
            for c in cams:
                v = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{c.split('_')[1]}.mp4"
                if v.exists():
                    caps[c] = cv2.VideoCapture(str(v))
            if not caps:
                continue
            trk = CloudTracker({c: calib[c] for c in caps}, n_seed=a.nseed, units_per_metre=1.0)
            s_cloud = np.full(n, np.nan)
            w_cloud = np.full(n, np.nan)
            per_n = []
            for f in range(n):
                gray = {}
                for c, cap in caps.items():
                    ok, im = cap.read()
                    if ok:
                        gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if len(gray) < 2 or f >= len(cup3) or not np.isfinite(cup3[f]).all():
                    continue
                r = trk.update(gray, cup3[f], dt=1.0 / FPS)
                if r is not None and r.linear_speed is not None:
                    s_cloud[f] = r.linear_speed
                    w_cloud[f] = r.angular_speed
                    per_n.append(r.active_3d_points_count)
            for c in caps.values():
                c.release()
            if per_n:
                npts.append(float(np.median(per_n)))
            # ⚠ coverage MUST be measured on the RAW signal: H._lp interpolates across NaNs, so a
            # low-passed track always reads 100% covered and the gate's refusals become invisible.
            raw_cloud, raw_flow = s_cloud.copy(), s_flow.copy()
            s_cloud = H._lp(s_cloud)

            mv = np.isfinite(so) & (so > MOVING)
            rest = np.isfinite(so) & (so < 10.0)
            for k, s, raw in (("cloud", s_cloud, raw_cloud), ("flow", s_flow, raw_flow)):
                m = np.isfinite(s) & np.isfinite(so) & np.isfinite(raw) & mv
                if m.sum() > 20:
                    acc[k]["mv"].append(float(np.median(np.abs(s[m] - so[m]))))
                if mv.sum():
                    acc[k]["cov"].append(float(np.isfinite(raw[mv]).mean() * 100))
            q = np.isfinite(w_cloud) & mv
            if q.sum() > 10:
                rot_moving.append(float(np.median(w_cloud[q])))
            q = np.isfinite(w_cloud) & rest
            if q.sum() > 10:
                rot_rest.append(float(np.median(w_cloud[q])))
            cm = acc['cloud']['mv'][-1] if acc['cloud']['mv'] else float('nan')
            fm = acc['flow']['mv'][-1] if acc['flow']['mv'] else float('nan')
            print(f"  {part}_{trial.split('_')[1]:>10}  cloud {cm:8.1f}  flow {fm:6.1f}  "
                  f"pts {per_n[len(per_n)//2] if per_n else 0:3d}  ({time.time()-t0:.0f}s)",
                  flush=True)

    f_ = lambda v: np.median(v) if len(v) else float("nan")
    print(f"\n=== CUP SPEED vs OMC (n={len(acc['flow']['mv'])} trials) ===")
    print(f"{'method':8} {'MOVING err':>12} {'coverage':>10}")
    print("-" * 34)
    for k in ("cloud", "flow"):
        print(f"{k:8} {f_(acc[k]['mv']):10.1f}mm/s {f_(acc[k]['cov']):9.1f}%")
    if npts:
        print(f"\n  cloud inlier points per frame (median): {np.median(npts):.1f}")
    if rot_moving or rot_rest:
        print(f"\n  ANGULAR (no OMC truth -- plausibility only):")
        print(f"    while the cup MOVES : {f_(rot_moving):6.2f} rad/s")
        print(f"    while the cup RESTS : {f_(rot_rest):6.2f} rad/s   <- should be near 0")


if __name__ == "__main__":
    main()
