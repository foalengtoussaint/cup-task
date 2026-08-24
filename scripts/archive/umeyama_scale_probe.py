"""Does the Umeyama SCALE predict per-frame speed error -- without any ground truth?

A rigid cup cannot change size, so a fitted scale s != 1 means the tracked cloud DEFORMED between
frames (tracks sliding on the surface). That deformation is the mechanism already measured behind
the cloud's under-read. The question here is whether s is USABLE at runtime: if |s-1| correlates
with the error, it is a free per-frame confidence signal that needs no OMC, and the pipeline can
down-weight or refuse bad frames the way `min_inliers` already does with a cruder proxy.

Also asks the follow-on: does DIVIDING by s (undoing the fitted shrink) reduce the error? That
would be a mechanistic de-bias rather than a fitted gain -- if the cloud contracted by 3% this
frame, its reported translation is short by roughly the same factor.

⚠ The scale is estimated on the SAME points whose motion is being measured, so dividing by it can
also amplify noise. Both are reported; neither is assumed.

    python scripts/umeyama_scale_probe.py --trials 2
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
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args(argv)

    import cv2
    from scipy.stats import spearmanr
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    from pipeline.cloud_track import CloudTracker
    from pipeline.cloud_velocity import kabsch_ransac, umeyama

    H.use_good_cams()
    FPS = 60.0
    rows = []                      # (omc, cloud_speed, scale, n_inliers)
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            so = H._lp(H._speed(R._shift(R._omc_cup(part, trial, n), lag)))
            px = C.cup_px(part, trial, n)
            cup3 = R._smooth_joint(R._cup_v3(part, trial, calib, n))
            cams = [c for c in px if c in calib]
            caps = {}
            for c in cams:
                v = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{c.split('_')[1]}.mp4"
                if v.exists():
                    caps[c] = cv2.VideoCapture(str(v))
            if not caps:
                continue

            trk = CloudTracker({c: calib[c] for c in caps}, n_seed=48, units_per_metre=1.0,
                               min_inliers=8, anchor_px=30.0)
            prev_cloud, prev_ids = None, None
            spd = np.full(n, np.nan); sca = np.full(n, np.nan); nin = np.full(n, np.nan)
            for f in range(n):
                gray = {}
                for c, cap in caps.items():
                    ok, im = cap.read()
                    if ok:
                        gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if len(gray) < 2 or f >= len(cup3) or not np.isfinite(cup3[f]).all():
                    continue
                kp = {c: px[c][f] for c in cams
                      if f < len(px[c]) and np.isfinite(px[c][f]).all()}
                r = trk.update(gray, cup3[f], 1.0 / FPS, kp)
                if r is None or r.cloud is None or trk._prev_ids is None:
                    continue
                ids = np.array(trk._prev_ids, int)
                if prev_cloud is not None and len(r.cloud) >= 4:
                    common, ia, ib = np.intersect1d(prev_ids, ids, return_indices=True)
                    if len(common) >= 4:
                        A, B = prev_cloud[ia], r.cloud[ib]
                        # inliers from the RIGID fit, then measure scale on those same inliers, so
                        # the scale describes the points actually used for the motion estimate
                        _, _, mask = kabsch_ransac(A, B, thresh=5.0)
                        if mask is not None and mask.sum() >= 4:
                            _, _, s = umeyama(A[mask], B[mask])
                            sca[f] = s
                            nin[f] = mask.sum()
                if r.linear_speed is not None:
                    spd[f] = r.linear_speed
                prev_cloud, prev_ids = r.cloud, ids
            for c in caps.values():
                c.release()

            sl = H._lp(spd)
            for f in range(n):
                if np.isfinite(sl[f]) and np.isfinite(so[f]) and np.isfinite(sca[f]):
                    rows.append((so[f], sl[f], sca[f], nin[f]))
            print(f"  {part}_{trial.split('_')[1]:>10}  ({time.time()-t0:.0f}s)", flush=True)

    A = np.array(rows)
    np.save(ROOT / "out" / "figures" / "umeyama_scale.npy", A)
    so, sp, sc, ni = A.T
    m = so > MOVING
    err = np.abs(sp - so)

    print(f"\n=== SCALE distribution (n={len(A)} frames, {m.sum()} moving) ===")
    q = np.percentile(sc, [5, 25, 50, 75, 95])
    print(f"  p5 {q[0]:.4f}  p25 {q[1]:.4f}  MED {q[2]:.4f}  p75 {q[3]:.4f}  p95 {q[4]:.4f}")
    print(f"  median |s-1| = {np.median(np.abs(sc-1))*100:.2f}%   "
          f"(a rigid object should give 0)")

    print(f"\n=== Does |s-1| PREDICT the error? (moving frames) ===")
    dev = np.abs(sc[m] - 1)
    print(f"  Pearson  r(|s-1|, err) = {np.corrcoef(dev, err[m])[0,1]:+.3f}")
    print(f"  Spearman rho          = {spearmanr(dev, err[m]).statistic:+.3f}")
    print(f"  (compare) rho(n_inliers, err) = {spearmanr(ni[m], err[m]).statistic:+.3f}")
    print(f"\n  {'|s-1| quartile':>16} {'n':>6} {'median err':>11}")
    qs = np.percentile(dev, [25, 50, 75])
    for lo, hi, lab in [(0, qs[0], "Q1 most rigid"), (qs[0], qs[1], "Q2"),
                        (qs[1], qs[2], "Q3"), (qs[2], 1e9, "Q4 most deformed")]:
        s_ = (dev >= lo) & (dev < hi)
        if s_.sum() > 20:
            print(f"  {lab:>16} {s_.sum():6d} {np.median(err[m][s_]):10.1f}")

    print(f"\n=== Does DIVIDING by s reduce the error? ===")
    for nm, est in (("raw", sp[m]), ("/ s", sp[m] / np.maximum(sc[m], 1e-6))):
        e = np.abs(est - so[m])
        print(f"  {nm:5} median {np.median(e):6.2f}  p90 {np.percentile(e,90):7.2f}  "
              f"ratio {np.median(est/so[m]):.3f}")
    print("\n  A gate on |s-1| is only worth adding if the quartile spread is LARGE -- a weak")
    print("  correlation means it duplicates what min_inliers already rejects.")


if __name__ == "__main__":
    main()
