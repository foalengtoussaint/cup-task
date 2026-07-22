"""Score the cloud against a RIGID-BODY truth built from the 4 cup markers, not a single centroid.

⚠ THE MARKERS ARE TRUTH ONLY. Nothing here feeds mocap into the tracker -- that would be circular
and would produce a pipeline that cannot run without a mocap lab. They are used exclusively to
build a better reference to score against.

WHY THE CURRENT METRIC IS PARTLY WRONG. Everything so far compares the speed of the CLOUD'S
CENTROID against the speed of the OMC MARKER CENTROID. Those are two DIFFERENT PHYSICAL POINTS on
the cup -- measured 34.6mm apart, because the cloud only ever sees the near hemisphere and its
centroid sits toward the camera-facing surface. For a purely translating body that does not matter
(every point moves identically), but under ROTATION two body points move at genuinely different
speeds: v_P = v_C + omega x (P - C). The cup rotates a lot during drinking. So part of the residual
"error" is not tracker error at all -- it is a comparison between two different points.

WHAT THIS SCRIPT DOES. With >=3 markers the cup's full 6-DoF pose is known per frame, so the OMC
truth can be evaluated AT THE CLOUD'S OWN CENTROID:
  1. establish, once per trial, where the cloud's centroid sits in the cup's own marker frame
     (median over frames of the centroid expressed in that frame -- a body-fixed offset);
  2. per frame, map that body point back to world with the marker pose and difference IT.
This is the apples-to-apples comparison. Reported next to the old centroid-vs-centroid number, so
the size of my own metric error is visible.

    python scripts/cloud_truth_rigid.py --trials 6
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


def marker_poses(mk: np.ndarray):
    """(n,M,3) markers -> per-frame (R, c) mapping the cup's REFERENCE marker frame to world.

    The reference frame is the first fully-visible frame's marker set, centred. Kabsch per frame
    against that reference gives the cup's orientation and centroid without any model of the cup.
    """
    from cup_task.cloud_velocity import kabsch
    n = len(mk)
    ok = np.isfinite(mk).all(2).all(1)
    if not ok.any():
        return None, None, ok
    ref = mk[np.flatnonzero(ok)[0]]
    ref_c = ref.mean(0)
    Rs = np.full((n, 3, 3), np.nan)
    Cs = np.full((n, 3), np.nan)
    for f in np.flatnonzero(ok):
        R_, t_ = kabsch(ref - ref_c, mk[f])
        Rs[f] = R_
        Cs[f] = mk[f].mean(0)
    return Rs, Cs, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args(argv)

    import cv2
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    import cloud_rotation_truth as CR
    from cup_task.cloud_track import CloudTracker

    H.use_good_cams()
    FPS = 60.0
    acc = {k: [] for k in ("centroid-truth", "bodypoint-truth")}
    offs = []
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            mk, names = CR.omc_cup_markers(part, trial, n, FPS)
            if mk is None:
                continue
            mk = np.stack([R._shift(mk[:, j], lag) for j in range(mk.shape[1])], 1)
            Rs, Cs, ok = marker_poses(mk)
            if Rs is None:
                continue

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
            spd = np.full(n, np.nan)
            cen = np.full((n, 3), np.nan)
            for f in range(n):
                gray = {}
                for c, cap in caps.items():
                    okr, im = cap.read()
                    if okr:
                        gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if len(gray) < 2 or f >= len(cup3) or not np.isfinite(cup3[f]).all():
                    continue
                kp = {c: px[c][f] for c in cams
                      if f < len(px[c]) and np.isfinite(px[c][f]).all()}
                r = trk.update(gray, cup3[f], 1.0 / FPS, kp)
                if r is None:
                    continue
                if r.linear_speed is not None:
                    spd[f] = r.linear_speed
                if r.cloud is not None and len(r.cloud):
                    cen[f] = r.cloud.mean(0)
            for c in caps.values():
                c.release()

            # (1) The cloud lives in the MMC world and the markers in the MOCAP world -- two
            # different coordinate frames, so `Rs[f].T @ (cen[f] - Cs[f])` is meaningless
            # (it produced a 1842mm "offset", i.e. room scale, not a point on a cup).
            # Work with the LEVER ARM instead, which is frame-free: the cloud's centroid sits at
            # some body-fixed offset from the cup centroid, and its DISTANCE is observable in the
            # MMC world alone. Estimate the offset DIRECTION in the cup frame by asking which
            # body-fixed lever best explains the cloud's speed -- but the honest, assumption-free
            # version is simply to test a RANGE of lever magnitudes and report the sensitivity.
            good = ok & np.isfinite(cen).all(1)
            if good.sum() < 30:
                continue
            # distance from the cloud centroid to the cup centroid, in the MMC world, using the
            # detected cup 3D as the MMC-side stand-in for the cup centroid
            lever = float(np.median(np.linalg.norm(cen[good] - cup3[good], axis=1)))
            offs.append(lever)

            # (2) Truth at a body point that lever-arm distance away. Direction is unknown, so
            # take the WORST and BEST case over a set of directions on that sphere: if even the
            # best case does not beat the centroid truth, the metric error is not the story.
            rng = np.random.default_rng(0)
            dirs = rng.normal(size=(24, 3))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            best = None
            for u in dirs:
                p = np.full((n, 3), np.nan)
                for f in np.flatnonzero(ok):
                    p[f] = Cs[f] + Rs[f] @ (lever * u)
                s_ = H._lp(H._speed(p))
                mm = np.isfinite(H._lp(spd)) & np.isfinite(spd) & np.isfinite(s_) & (s_ > MOVING)
                if mm.sum() > 20:
                    e = float(np.median(np.abs(H._lp(spd)[mm] - s_[mm])))
                    if best is None or e < best[0]:
                        best = (e, p)
            p_body = best[1] if best is not None else np.full((n, 3), np.nan)
            so_cent = H._lp(H._speed(Cs))
            so_body = H._lp(H._speed(p_body))
            sl = H._lp(spd)
            for k, so in (("centroid-truth", so_cent), ("bodypoint-truth", so_body)):
                m = np.isfinite(sl) & np.isfinite(spd) & np.isfinite(so) & (so > MOVING)
                if m.sum() > 20:
                    e = np.abs(sl[m] - so[m])
                    acc[k].append((float(np.median(e)), float(np.percentile(e, 90)),
                                   float(np.median(sl[m] / so[m]))))
            c_ = acc['centroid-truth'][-1] if acc['centroid-truth'] else (float('nan'),)*3
            b_ = acc['bodypoint-truth'][-1] if acc['bodypoint-truth'] else (float('nan'),)*3
            print(f"  {part}_{trial.split('_')[1]:>10}  centroid {c_[0]:6.2f}  body {b_[0]:6.2f}  "
                  f"lever {lever:5.1f}mm  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== CLOUD vs two different TRUTHS (n={len(acc['centroid-truth'])} trials) ===")
    print(f"{'truth':18} {'median':>8} {'mean':>7} {'p90':>8} {'ratio':>7}")
    for k in ("centroid-truth", "bodypoint-truth"):
        A = np.array(acc[k])
        if len(A):
            print(f"{k:18} {np.median(A[:,0]):7.2f} {A[:,0].mean():6.2f} "
                  f"{np.median(A[:,1]):7.2f} {np.median(A[:,2]):6.3f}")
    if offs:
        print(f"\n  cloud centroid sits {np.median(offs):.1f}mm from the marker centroid "
              f"(body-fixed)")
    print("\n  ⚠ bodypoint-truth is an ORACLE: the lever DIRECTION is chosen per trial as the best")
    print("  of 24 candidates, scored against the very error it is minimising. It is a CEILING on")
    print("  how much of the residual could be metric error, not an adoptable number. If even this")
    print("  oracle cannot beat the plain centroid truth, the two-different-points effect is not")
    print("  what is limiting the cloud, and the residual is genuine tracker error.")


if __name__ == "__main__":
    main()
