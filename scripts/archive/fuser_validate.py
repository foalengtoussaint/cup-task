"""Does the `l1` fuser's peak-error win survive the FULL cohort? (n=12, both targets)

The probe that motivated this (flow_3d_survival.py) compared five fusers on n=6 trials while ALSO
recomputing RAFT, which costs minutes of GPU per trial. The fuser question does not need RAFT at
all -- it is about how the per-camera flow vectors are COMBINED, not where they came from. So this
runs on the cached PyrLK flow only: all 12 trials, both targets, seconds instead of minutes.

FUSERS (all solve the same model u_dot = J(X) v, differing only in the loss):
    plain      least squares. Breakdown point 0 -- one bad camera moves the answer.
    loo        SHIPPING. Leave-one-out: drop the camera whose removal most changes the fused
               velocity, while that change exceeds a hand-set 20mm/s. Hard in/out, one knob.
    huber      IRLS, Huber loss: quadratic near zero, linear in the tail. Down-weights smoothly.
    l1         IRLS toward an L1 objective (w = 1/|r| => minimising sum|r|, a geometric-median-like
               solution). A camera's INFLUENCE is capped by its own error. No threshold at all.
    trimmed    drop the single worst-residual camera. No threshold, no iteration.

WHY THE PEAK IS THE METRIC THAT DECIDES. Peak frames are where one camera goes badly wrong (motion
blur smears the patch; an occluder crosses the ray), so they are exactly where a mean-like estimator
is dragged and where LOO's fixed threshold is most likely to be on the wrong side. Median moving-
frame error is dominated by the easy 90% of frames and barely separates the fusers -- but
peak_velocity is a reported Murphy measure, and one bad frame sets it for the whole trial.

Reports the peak-error DISTRIBUTION, not just its median: the claim being tested is about the tail.

    python scripts/fuser_validate.py                  # all 12 trials, wrist + cup
    python scripts/fuser_validate.py --trials 2       # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from flow_3d_survival import jacobian, solve_velocity   # ONE implementation of the solver

MOVING = 50.0        # mm/s: "the target is actually moving"
FUSERS = ["plain", "loo", "huber", "l1", "trimmed"]


def _tracks(px, fl, calib, n, fps=60.0):
    """Per-frame 3D speed under every fuser, from ONE pass over the frames.

    All fusers share the same triangulation and the same Jacobian rows, so they differ only in the
    solve. Computing them together means any difference in the output is attributable to the fuser
    and nothing else.
    """
    from pipeline.kalman_3d import triangulate_dlt

    out = {k: np.full(n, np.nan) for k in FUSERS}
    cams = [c for c in fl if c in px and c in calib]
    for f in range(n):
        obs = {c: px[c][f] for c in cams
               if f < len(px[c]) and f < len(fl[c])
               and np.isfinite(px[c][f]).all() and np.isfinite(fl[c][f]).all()}
        if len(obs) < 2:
            continue
        X = triangulate_dlt([calib[c] for c in obs], [np.asarray(obs[c]) for c in obs])
        if X is None:
            continue
        rows, use = [], []
        for c in obs:
            J = jacobian(calib[c], X)
            if J is not None:
                rows.append((J, np.asarray(fl[c][f]))); use.append(c)
        if len(rows) < 2:
            continue
        for k in FUSERS:
            if k == "loo":
                P = {c: obs[c] for c in use}
                PV = {c: obs[c] + fl[c][f] for c in use}
                keep = flow_consensus(P, PV, calib) if len(P) >= 3 else list(P)
                sub = [rows[i] for i, c in enumerate(use) if c in keep]
                v = solve_velocity(sub if len(sub) >= 2 else rows, "plain", fps)
            else:
                v = solve_velocity(rows, k, fps)
            if v is not None:
                out[k][f] = float(np.linalg.norm(v))
    return out


def flow_consensus(P, PV, calib):
    from pipeline import flow_speed
    return flow_speed.flow_consensus_cams(P, PV, calib)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=6, help="trials per participant (6 = full)")
    a = ap.parse_args(argv)

    from scipy.signal import find_peaks
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    import flow_velocity_probe as F

    H.use_good_cams()
    acc = {t: {k: {"mv": [], "peak": []} for k in FUSERS} for t in ("wrist", "cup")}
    ntr = {"wrist": 0, "cup": 0}
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            ow = R._shift(omc[f"{side}_wrist"], lag)
            oc = R._shift(R._omc_cup(part, trial, n), lag)

            targets = {
                "wrist": (F.load_wrist_px(part, trial, f"{side}_wrist"), "flow_vel", ow),
                "cup":   (C.cup_px(part, trial, n), "cup_flow_vel", oc),
            }
            for tname, (px, cdir, truth) in targets.items():
                fl = {}
                for cam in px:
                    p = (ROOT / "cache" / cdir /
                         f"delta_{part}_{trial}.{cam.split('_')[1]}__pyrlk.npy")
                    if p.exists() and cam in calib:
                        fl[cam] = np.load(p)
                if not fl:
                    continue
                so = H._lp(H._speed(truth))
                tr = _tracks(px, fl, calib, n)
                ntr[tname] += 1
                # OMC's own peaks are the reference events; each fuser is asked for ITS speed at the
                # matching peak. Same peak set for every fuser, so the comparison is paired.
                opk = find_peaks(so, height=300, distance=30, prominence=150)[0]
                for k in FUSERS:
                    s = H._lp(tr[k])
                    m = np.isfinite(s) & np.isfinite(so) & (so > MOVING)
                    if m.sum() > 20:
                        acc[tname][k]["mv"].append(float(np.median(np.abs(s[m] - so[m]))))
                    pk, _ = find_peaks(s, height=200, distance=25, prominence=100)
                    for p_ in opk:
                        if len(pk):
                            j = pk[np.argmin(np.abs(pk - p_))]
                            if abs(j - p_) <= 20:
                                acc[tname][k]["peak"].append(float(abs(s[j] - so[p_])))
            print(f"  {part}_{trial.split('_')[1]:>10}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez(ROOT / "out" / "figures" / "fuser_validate.npz",
             **{f"{t}_{k}_{fld}": np.array(acc[t][k][fld])
                for t in acc for k in FUSERS for fld in ("mv", "peak")})

    for tname in ("wrist", "cup"):
        pk_n = len(acc[tname]["loo"]["peak"])
        print(f"\n=== {tname.upper()}  (n={ntr[tname]} trials, {pk_n} matched peaks) ===")
        print(f"{'fuser':10} {'MOVING err':>11} | {'PEAK err  med':>14} {'p75':>7} {'p90':>7} "
              f"{'max':>8} {'mean':>7}")
        print("-" * 70)
        for k in FUSERS:
            mv = np.array(acc[tname][k]["mv"]); pk = np.array(acc[tname][k]["peak"])
            if not len(pk):
                continue
            q = np.percentile(pk, [50, 75, 90])
            print(f"{k:10} {np.median(mv):9.2f}   | {q[0]:13.2f} {q[1]:7.2f} {q[2]:7.2f} "
                  f"{pk.max():8.2f} {pk.mean():7.2f}")

        # PAIRED comparison: the same peak events under both fusers. A median-of-medians can move
        # for reasons that have nothing to do with the fuser (different peaks matched); this cannot.
        base = np.array(acc[tname]["loo"]["peak"])
        print(f"\n  paired vs shipping `loo` (same {len(base)} peak events):")
        for k in FUSERS:
            if k == "loo":
                continue
            o = np.array(acc[tname][k]["peak"])
            if len(o) != len(base) or not len(base):
                continue
            d = o - base
            print(f"    {k:9} better on {(d < 0).mean()*100:5.1f}% of peaks   "
                  f"median delta {np.median(d):+8.2f} mm/s   mean {d.mean():+8.2f}")

    print("\n  MOVING err = median |speed - OMC| over frames where OMC speed > 50mm/s.")
    print("  PEAK err   = |speed at the matched peak - OMC peak height|, one per OMC peak event.")
    print("  The peak is what feeds the Murphy peak_velocity measure, and one bad frame sets it")
    print("  for the whole trial -- so read the tail (p90/max), not only the median.")


if __name__ == "__main__":
    main()
