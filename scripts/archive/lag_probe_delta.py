"""Decide desync-vs-miscalib GEOMETRICALLY: does a time-shift make the camera reproject?

The reaudit classifier split desync/miscalib partly on a WRIST-SPEED lag correlation, which is
viewpoint-confounded (the exact signal we retracted). This probe drops correlation entirely:

  1. Build the consensus 3D wrist per frame by RANSAC over ALL OTHER cams (suspect LEFT OUT, so a
     desynced camera cannot drag its own reference).
  2. For the suspect cam, sweep integer lag L: reproject consensus[f] against detection[f+L],
     take the median error over STILL frames and over MOVE frames separately.
  3. Read the curve:
       min at L==0, low                         -> FINE
       min at L!=0, low, and << error at L=0    -> DESYNC by L frames  (re-cut fixes it)
       high at EVERY lag                        -> MISCALIB            (geometry wrong, recalibrate)

A time shift can only fix a timing error; it can never fix wrong camera geometry. That is the whole
discriminator, and it needs no correlation.

    python scripts/lag_probe_delta.py --part P13 --cams 2 --maxlag 200
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from pipeline.kalman_3d import project  # noqa: E402
from reaudit_cam_quality import ransac_point, _side, _trials, STILL_MM  # noqa: E402

LOW = 15.0  # px; reproj at or below this = geometry consistent


def _reproj_at_lag(dets_c, cams, cam, X3, spd, L, n):
    """Median reproj of suspect cam's detection at f+L vs consensus X3[f], split still/move."""
    still, move = [], []
    for f in range(1, n):
        g = f + L
        if X3[f] is None or spd[f] is None or g < 0 or g >= n:
            continue
        p = FN(dets_c[g])
        if p is None:
            continue
        e = float(np.hypot(*(project(cams[cam], X3[f])[0] - p)))
        (still if spd[f] < STILL_MM else move).append(e)
    return (np.median(still) if still else float("nan"),
            np.median(move) if move else float("nan"),
            len(still), len(move))


def probe(part, cam_ids, maxlag, n_trials):
    global SIDE, FN
    SIDE = _side(part)
    FN = C._kp_point(f"{SIDE}_wrist")
    cams = C._load_calib_mm(part)
    cams = {c: cams[c] for c in cams if int(c.split("_")[1]) <= 5}  # only trusted 1-5
    trials = _trials(part, n_trials)

    for cid in cam_ids:
        suspect = f"cam_{cid}"
        agg = {}  # lag -> [still_meds, move_meds]
        for t in trials:
            per = {}
            for pj in sorted(glob.glob(str(C.DELTA / part / "dets" / f"delta_{part}_{t}.*.pose.json"))):
                c = "cam_" + Path(pj).name.split(".")[1]
                if c in cams:
                    per[c] = json.loads(Path(pj).read_text())["frames"]
            if suspect not in per:
                continue
            n = min(len(v) for v in per.values())
            # consensus from OTHER cams only
            others = {c: per[c] for c in per if c != suspect}
            X3 = [None] * n
            for f in range(n):
                pts = {c: FN(others[c][f]) for c in others if FN(others[c][f]) is not None}
                if len(pts) >= 2:
                    X, _ = ransac_point({c: cams[c] for c in pts}, pts)
                    X3[f] = X
            spd = [None] * n
            for f in range(1, n):
                if X3[f] is not None and X3[f - 1] is not None:
                    spd[f] = float(np.linalg.norm(X3[f] - X3[f - 1]))
            for L in range(-maxlag, maxlag + 1):
                s, m, ns, nm = _reproj_at_lag(per[suspect], cams, suspect, X3, spd, L, n)
                a = agg.setdefault(L, [[], []])
                if np.isfinite(s):
                    a[0].append(s)
                if np.isfinite(m):
                    a[1].append(m)

        # collapse trials -> median curve
        lags = sorted(agg)
        still = {L: (np.median(agg[L][0]) if agg[L][0] else float("nan")) for L in lags}
        move = {L: (np.median(agg[L][1]) if agg[L][1] else float("nan")) for L in lags}
        combo = {L: np.nanmean([still[L], move[L]]) for L in lags}
        finite = {L: v for L, v in combo.items() if np.isfinite(v)}
        if not finite:
            print(f"\n{part} {suspect}: NO reference (other cams can't form consensus) -> UNUSABLE")
            continue
        bestL = min(finite, key=finite.get)
        e0 = combo.get(0, float("nan"))
        eB = finite[bestL]
        print(f"\n{part} {suspect}  (leave-one-out consensus from "
              f"{sorted(set(cams)-{suspect}, key=lambda z:int(z.split('_')[1]))})")
        print(f"  reproj @lag0 : still {still.get(0,float('nan')):.0f}px  move {move.get(0,float('nan')):.0f}px  combo {e0:.0f}px")
        print(f"  best lag L={bestL:+d} : still {still[bestL]:.0f}px  move {move[bestL]:.0f}px  combo {eB:.0f}px")
        # show the curve at a few offsets around 0 and around best
        show = sorted(set([0, bestL] + [bestL + d for d in (-2, -1, 1, 2)] + [-5, 5, -50, 50]))
        show = [L for L in show if L in combo]
        print("   lag  combo(px):  " + "  ".join(f"{L:+d}:{combo[L]:.0f}" for L in show))
        # verdict
        if eB <= LOW and abs(bestL) <= 1:
            v = "FINE (already aligned)"
        elif eB <= LOW and e0 > 2 * eB and abs(bestL) >= 2:
            v = f"DESYNC -> re-cut by {bestL:+d} frames (reproj {e0:.0f}->{eB:.0f}px)"
        elif eB > LOW * 2:
            v = f"MISCALIB (no lag helps; best still {eB:.0f}px at L={bestL:+d})"
        else:
            v = f"AMBIGUOUS (best {eB:.0f}px at L={bestL:+d}, lag0 {e0:.0f}px)"
        print(f"  => {v}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cams", nargs="+", type=int, required=True)
    ap.add_argument("--maxlag", type=int, default=200)
    ap.add_argument("--n-trials", type=int, default=6)
    a = ap.parse_args(argv)
    probe(a.part, a.cams, a.maxlag, a.n_trials)


if __name__ == "__main__":
    main()
