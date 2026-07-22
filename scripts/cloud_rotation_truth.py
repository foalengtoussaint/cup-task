"""Does the tracked cloud's ANGULAR velocity match the OMC cup's actual rotation?

The cup has SEVERAL `cluster_cup*` markers in the C3D, and results_v3_delta._omc_cup averages them
into a single point -- which is right for position but throws away exactly the information needed
here. Keeping them separate gives a rigid marker set per frame, and Kabsch between consecutive
frames gives the cup's TRUE angular velocity. That turns the cloud's rotation output from a
plausibility check into a measured claim.

⚠ The comparison is on angular SPEED (magnitude), not the axis. MMC world and mocap world are
different frames, so an axis comparison needs the session rotation; the magnitude is
frame-invariant and needs nothing. Correlation over time is the honest metric -- a constant offset
would show as a scale error, and that is visible in the ratio.

    python scripts/cloud_rotation_truth.py --trials 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def omc_cup_markers(part: str, trial: str, n: int, fps: float = 60.0):
    """(n, M, 3) -- each cup marker kept SEPARATE, resampled onto the video grid."""
    import ezc3d
    import compare_pose_omc_delta as H
    c = ezc3d.c3d(str(H.DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]
    mk = [nm for nm in L if "cluster_cup" in nm.lower() or nm.lower().startswith("cup")]
    if len(mk) < 3:
        return None, mk
    out = []
    for m in mk:
        raw = P[:3, L.index(m), :].T
        g = H._resample(raw, H.C3D_RATE, fps)
        if len(g) < n:
            g = np.vstack([g, np.full((n - len(g), 3), np.nan)])
        out.append(g[:n])
    return np.stack(out, 1), mk


def omc_angular_speed(mk: np.ndarray, fps: float = 60.0) -> np.ndarray:
    """(n,M,3) marker cloud -> (n,) angular speed in rad/s, by Kabsch between consecutive frames."""
    from scipy.spatial.transform import Rotation
    from cup_task.cloud_velocity import kabsch
    n = len(mk)
    w = np.full(n, np.nan)
    for f in range(n - 1):
        A, B = mk[f], mk[f + 1]
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if ok.sum() < 3:
            continue
        R, _ = kabsch(A[ok], B[ok])
        w[f] = np.linalg.norm(Rotation.from_matrix(R).as_rotvec()) * fps
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--nseed", type=int, default=48)
    a = ap.parse_args(argv)

    import cv2
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    from cup_task.cloud_track import CloudTracker

    H.use_good_cams()
    FPS = 60.0
    corrs, ratios, rest_o, rest_m = [], [], [], []
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            mk, names = omc_cup_markers(part, trial, n, FPS)
            if mk is None:
                print(f"  {part}_{trial}: only {names} -- need >=3 markers, skipping", flush=True)
                continue
            w_omc = R._shift(omc_angular_speed(mk, FPS).reshape(-1, 1), lag).ravel()
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
            trk = CloudTracker({c: calib[c] for c in caps}, n_seed=a.nseed, units_per_metre=1.0)
            w_mmc = np.full(n, np.nan)
            for f in range(n):
                gray = {}
                for c, cap in caps.items():
                    ok, im = cap.read()
                    if ok:
                        gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if len(gray) < 2 or f >= len(cup3) or not np.isfinite(cup3[f]).all():
                    continue
                r = trk.update(gray, cup3[f], dt=1.0 / FPS)
                if r is not None and r.angular_speed is not None:
                    w_mmc[f] = r.angular_speed
            for c in caps.values():
                c.release()

            wo, wm = H._lp(w_omc), H._lp(w_mmc)
            m = np.isfinite(wo) & np.isfinite(wm) & np.isfinite(w_mmc) & np.isfinite(w_omc)
            if m.sum() > 30:
                cc = float(np.corrcoef(wo[m], wm[m])[0, 1])
                rt = float(np.median(wm[m]) / max(np.median(wo[m]), 1e-6))
                corrs.append(cc); ratios.append(rt)
                print(f"  {part}_{trial.split('_')[1]:>10}  corr {cc:+.3f}  ratio {rt:5.2f}  "
                      f"n={m.sum():4d}  ({time.time()-t0:.0f}s)", flush=True)
            q = np.isfinite(so) & (so < 10) & m
            if q.sum() > 10:
                rest_o.append(float(np.median(wo[q]))); rest_m.append(float(np.median(wm[q])))

    f_ = lambda v: np.median(v) if len(v) else float("nan")
    print(f"\n=== CUP ANGULAR SPEED, cloud vs OMC-marker truth (n={len(corrs)} trials) ===")
    print(f"  correlation over time : {f_(corrs):+.3f}")
    print(f"  magnitude ratio       : {f_(ratios):5.2f}   (1.0 = right scale)")
    if rest_o:
        print(f"  at rest, OMC          : {f_(rest_o):5.3f} rad/s")
        print(f"  at rest, cloud        : {f_(rest_m):5.3f} rad/s")
    print("\n  Correlation is the claim; the ratio says whether it is also on the right SCALE.")
    print("  A high corr with ratio 2 means the shape is right but the magnitude is not.")


if __name__ == "__main__":
    main()
