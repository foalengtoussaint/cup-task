"""Score a cached MVGFormer wrist trajectory against OMC and the YOLO+robust-triangulation incumbent.

MVGFormer targets OCCLUSION (volumetric multi-view fusion), so the PRIMARY metric is position vs OMC
+ coverage, not the fast-frame speed thread. We report both. All alignment/sync/speed math is reused
from compare_pose_omc_delta (H) so there is no reimplementation. MVGFormer lives in the rig/calib WORLD
frame (mm); OMC lives in the mocap frame — so we rigid-Procrustes-align MVGFormer->OMC (same treatment
the incumbent gets in project_yolo_sigma_inference; a constant rigid offset does not affect speed).

Usage:  python scripts/mvgformer_score.py --part P07 --trial trial_13_L_unaffected
"""
import sys, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import compare_pose_omc_delta as H

CACHE = Path("/home/imove/Documents/cup-task/cache/mvgformer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P07")
    ap.add_argument("--trial", default="trial_13_L_unaffected")
    ap.add_argument("--tag", default="", help="MVGFormer npz suffix (e.g. drop13 for a dropout run)")
    ap.add_argument("--incumbent-cams", default="", help="restrict the INCUMBENT triangulation to "
                    "these cam numbers (e.g. 1,3) so a dropout run is apples-to-apples vs MVGFormer")
    args = ap.parse_args()
    H.use_good_cams()
    if args.incumbent_cams:      # same-cameras comparison for the dropout test
        want = {f"cam_{x.strip()}" for x in args.incumbent_cams.split(",")}
        H.GOOD_CAMS = {args.part: want}
    side = "left" if "_L_" in args.trial else "right"
    joint = f"{side}_wrist"

    npz = np.load(CACHE / f"{args.part}_{args.trial}{('_' + args.tag) if args.tag else ''}.npz")
    mvg = npz["wrist"].astype(float)        # (n,3) mm, rig/calib world frame
    score = npz["score"].astype(float)
    mvg[score < 0.1] = np.nan               # drop unconfident/empty queries
    # MVGFormer picks argmax query per frame with NO temporal continuity -> occasional single-frame
    # query swaps = isolated >100mm teleport spikes (see out/mvgformer plot). Despike them the SAME
    # way the incumbent triangulation already does (H._despike, isolated fast-jump removal) so the
    # comparison is like-for-like: both are raw-but-despiked per-frame estimators.
    mvg = H._despike(mvg)

    mmc, n = H._load_mmc(args.part, args.trial)     # incumbent YOLO+robust-triangulation, mm world
    inc = mmc[joint]
    omc = H._load_omc(args.part, args.trial, n)[joint]
    m = min(len(mvg), len(inc), len(omc))
    mvg, inc, omc = mvg[:m], inc[:m], omc[:m]

    # coverage
    cov = lambda a: np.isfinite(a).all(1).mean() * 100
    print(f"\n{args.part} {args.trial}  ({side} wrist, {m} frames)")
    print(f"coverage %%: MVGFormer {cov(mvg):.0f}   incumbent {cov(inc):.0f}   OMC {cov(omc):.0f}")

    # rigid align each estimator -> OMC (kills the constant rig<->mocap offset; speed unaffected)
    def align(a):
        if not (np.isfinite(a).all(1) & np.isfinite(omc).all(1)).any():
            return a            # nothing to align (e.g. incumbent collapsed on a dropout run)
        R, t, _ = H._kabsch(a, omc)
        return a @ R.T + t
    mvg_a, inc_a = align(mvg), align(inc)

    def pos_err(a):
        d = np.linalg.norm(a - omc, axis=1)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return None
        return np.median(d), np.percentile(d, 90)
    for name, a in (("MVGFormer", mvg_a), ("incumbent", inc_a)):
        pe = pos_err(a)
        if pe is None:
            print(f"position vs OMC  {name:10}: NO VALID FRAMES (estimator produced nothing)")
        else:
            print(f"position vs OMC  {name:10}: median {pe[0]:5.1f} mm   p90 {pe[1]:5.1f} mm")

    # RAW JITTER — the fair like-for-like metric between two RAW per-frame estimators (no smoother):
    # median magnitude of the 3D 2nd difference (accel proxy), same metric as the SmoothNet /
    # point-tracking memos. Frame-invariant, so no alignment needed. Both are raw here.
    def jitter(a):
        j = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1)
        j = j[np.isfinite(j)]
        return (np.median(j), np.percentile(j, 90)) if j.size else None
    for name, a in (("MVGFormer", mvg), ("incumbent", inc)):
        jr = jitter(a)
        print(f"raw jitter (|d2X|) {name:10}: " +
              (f"median {jr[0]:5.2f} mm   p90 {jr[1]:5.2f} mm" if jr else "NO VALID FRAMES"))

    # THE occlusion question: frames the incumbent LOST (nan) but OMC has a real wrist. MVGFormer's
    # volumetric fusion is meant to survive exactly these. Report how many it recovers + how well.
    inc_gap = ~np.isfinite(inc).all(1) & np.isfinite(omc).all(1)
    if inc_gap.sum():
        mvg_here = np.isfinite(mvg_a[inc_gap]).all(1)
        d = np.linalg.norm(mvg_a[inc_gap] - omc[inc_gap], axis=1)
        d = d[np.isfinite(d)]
        print(f"\nINCUMBENT-GAP frames: {inc_gap.sum()}  ->  MVGFormer fills {mvg_here.sum()} "
              f"({100*mvg_here.mean():.0f}%)"
              + (f", median err {np.median(d):.0f} mm / p90 {np.percentile(d,90):.0f} mm" if d.size else ""))
    else:
        print("\nINCUMBENT-GAP frames: 0 (incumbent has full coverage on this trial — no occlusion to test)")

    # speed lens (frame-invariant; sync via incumbent lag). low-pass + median |dspeed|.
    def _shift(v, l):
        out = np.full_like(v, np.nan)
        if l >= 0: out[l:] = v[:len(v)-l] if l else v
        else: out[:l] = v[-l:]
        return out
    ref = inc if np.isfinite(inc).all(1).any() else mvg   # sync ref (incumbent may be empty)
    lag, _ = H._find_lag(ref, omc)
    o = H._lp(H._speed(_shift(omc, lag)))

    def pos_lp(a):        # low-pass POSITION per axis, THEN differentiate (vs speed-domain low-pass)
        out = a.copy()
        for k in range(3):
            out[:, k] = H._lp(a[:, k])
        return out

    # (A) speed-domain low-pass (what the doc/incumbent uses); (B) POSITION low-pass then differentiate
    for name, a in (("MVGFormer", mvg), ("incumbent", inc)):
        for variant, sig in ((" [speed-lp]", H._lp(H._speed(a))),
                             (" [pos-lp]  ", H._speed(pos_lp(a)))):
            mm = np.isfinite(sig) & np.isfinite(o)
            if mm.sum() > 20:
                dmm = np.median(np.abs(sig[mm] - o[mm]))
                pk = abs((np.nanmax(sig) - np.nanmax(o)) / np.nanmax(o) * 100)
                print(f"speed vs OMC {name:10}{variant}: |dspeed| {dmm:6.1f} mm/s   peak err {pk:4.0f}%")
    print(f"(MVGFormer score: median {np.nanmedian(score):.2f}, min {np.nanmin(score):.2f})")


if __name__ == "__main__":
    main()
