"""Fit each frame to ONE rigid reference shape instead of chaining frame-to-frame Kabsch.

THE PROBLEM THIS ATTACKS (measured, P07 trial_10): the tracked cloud is NOT rigid. Points that
should sit at a fixed radius on a ~40mm cup wander with 29mm std -- comparable to the cup itself --
because PyrLK tracks slide across a smooth specular surface. Frame-to-frame Kabsch has no way to
notice: every consecutive pair is fitted independently, so a slow deformation is absorbed into the
motion estimate. Measured consequence: pairwise distances swell 1.6% per frame at speed (spread
0.948-1.047), and the reported speed under-reads by 15-30%.

THE IDEA, still pure Kabsch/Umeyama -- only the REFERENCE changes:

  chained    X(t) -> X(t+1)              each pair independent, deformation invisible
  model      M    -> X(t)   for every t  one shared rigid shape M, so a sliding point has a
                                         LARGE RESIDUAL against M and RANSAC can reject it

M is built by averaging each track's position in the cloud's own centroid frame over the frames
where it exists -- a translation-invariant shape estimate, no ground truth involved. Velocity then
comes from differencing consecutive POSES of that one model (t_model(t+1) - t_model(t)), which is
the Umeyama transform per frame rather than a chain of pairwise ones. Errors no longer accumulate
along the chain, and a point that slides is an outlier rather than a deformation.

Reports both, on the same tracks and the same frames, so the comparison is paired.

    python scripts/rigid_model_probe.py --trials 2
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


def build_model(tracks: dict, min_frames: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """{id: [(frame, xyz), ...]} -> (model (K,3), ids (K,)) in the cloud's own centroid frame.

    Each track's model position is the MEDIAN of its centroid-relative offsets. Median rather than
    mean because sliding is one-sided and heavy-tailed -- a mean would be dragged by the frames
    where the point has already slipped, which is exactly the corruption being removed.
    """
    ids, pts = [], []
    for k, v in tracks.items():
        if len(v) < min_frames:
            continue
        off = np.array([o for _, o in v])
        pts.append(np.median(off, 0)); ids.append(k)
    if not pts:
        return np.empty((0, 3)), np.empty(0, int)
    return np.array(pts), np.array(ids, int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=2)
    a = ap.parse_args(argv)

    import cv2
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    from pipeline.cloud_track import CloudTracker
    from pipeline.cloud_velocity import kabsch_ransac

    H.use_good_cams()
    FPS = 60.0
    acc = {k: [] for k in ("chained", "model")}
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
            cup3 = R._smooth_joint(R._cup_v3(part, trial, calib, n))
            cams = [c for c in px if c in calib]
            caps = {}
            for c in cams:
                v = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{c.split('_')[1]}.mp4"
                if v.exists():
                    caps[c] = cv2.VideoCapture(str(v))
            if not caps:
                continue

            # ---- PASS 1: collect every track's centroid-relative offset, plus the chained result
            trk = CloudTracker({c: calib[c] for c in caps}, n_seed=48, units_per_metre=1.0,
                               min_inliers=8)
            tracks: dict[int, list] = {}
            frames: dict[int, tuple] = {}
            s_chain = np.full(n, np.nan)
            for f in range(n):
                gray = {}
                for c, cap in caps.items():
                    ok, im = cap.read()
                    if ok:
                        gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if len(gray) < 2 or f >= len(cup3) or not np.isfinite(cup3[f]).all():
                    continue
                r = trk.update(gray, cup3[f], 1.0 / FPS)
                if r is not None and r.linear_speed is not None:
                    s_chain[f] = r.linear_speed
                if r is None or r.cloud is None or trk._prev_ids is None or len(r.cloud) < 4:
                    continue
                cen = r.cloud.mean(0)
                frames[f] = (np.array(trk._prev_ids, int), r.cloud.copy(), cen)
                for i, k in enumerate(trk._prev_ids):
                    tracks.setdefault(int(k), []).append((f, r.cloud[i] - cen))
            for c in caps.values():
                c.release()

            # ---- PASS 2: one rigid model, fit every frame to it (Umeyama per frame, no chain)
            M, mids = build_model(tracks)
            s_model = np.full(n, np.nan)
            if len(M) >= 4:
                pos = {}
                idx = {k: i for i, k in enumerate(mids)}
                for f, (ids_f, cloud_f, cen_f) in frames.items():
                    sel = [(idx[int(k)], j) for j, k in enumerate(ids_f) if int(k) in idx]
                    if len(sel) < 4:
                        continue
                    A = M[[i for i, _ in sel]]
                    B = cloud_f[[j for _, j in sel]]
                    Rm, tm, mask = kabsch_ransac(A, B, thresh=5.0)
                    if Rm is None or mask.sum() < 4:
                        continue
                    # the model's POSE this frame: where the model's own centroid lands
                    pos[f] = (Rm @ A[mask].mean(0)) + tm
                ks = sorted(pos)
                for i in range(1, len(ks)):
                    if ks[i] - ks[i - 1] == 1:
                        s_model[ks[i]] = float(np.linalg.norm(pos[ks[i]] - pos[ks[i - 1]])) * FPS

            for k, s in (("chained", s_chain), ("model", s_model)):
                raw = np.isfinite(s)
                sl = H._lp(s)
                m = raw & np.isfinite(sl) & np.isfinite(so) & (so > MOVING)
                if m.sum() > 20:
                    acc[k].append((float(np.median(np.abs(sl[m] - so[m]))),
                                   float(raw.mean() * 100)))
            c_ = acc['chained'][-1] if acc['chained'] else (float('nan'),) * 2
            m_ = acc['model'][-1] if acc['model'] else (float('nan'),) * 2
            print(f"  {part}_{trial.split('_')[1]:>10}  chained {c_[0]:7.1f}  model {m_[0]:7.1f}  "
                  f"(model cov {m_[1]:4.0f}%, {len(M)} model pts)  ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\n=== CUP SPEED vs OMC ===")
    print(f"{'method':10} {'moving err':>11} {'coverage':>10} {'n':>4}")
    for k in ("chained", "model"):
        A = np.array(acc[k]) if acc[k] else np.zeros((0, 2))
        if len(A):
            print(f"{k:10} {np.median(A[:,0]):9.1f}mm/s {np.median(A[:,1]):9.1f}% {len(A):4d}")


if __name__ == "__main__":
    main()
