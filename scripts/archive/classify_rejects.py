"""Classify each GATE-rejected cam-trial: DESYNC (recoverable) vs MISCALIB (not).

For a rejected (trial, cam): build the >=3-cam RANSAC consensus wrist from the KEPT cams
(work/dets), then sweep an integer frame lag L on the rejected cam's detections (work/rejected)
and take the median reproj at each L.
  best reproj <= LOW at L != 0  -> DESYNC by L frames (a bad re-cut/timing -> re-cut fixes it)
  best reproj <= LOW at L == 0  -> borderline calib (was just over the gate)
  best reproj still high at every L -> MISCALIB (geometry; recalibration, not re-cut)

    python scripts/classify_rejects.py --parts P13 P12 P07 P08 P15
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
from cup_task.kalman_3d import project  # noqa: E402
from reaudit_cam_quality import ransac_point  # noqa: E402

LOW = 15.0
MAXLAG = 90


def classify(part):
    cams = C._load_calib_mm(part)
    work = C.DELTA / part / "work"
    dets, rej = work / "dets", work / "rejected"
    rejected = sorted(glob.glob(str(rej / "*.pose.json")))
    out = []
    for rp in rejected:
        name = Path(rp).name                      # delta_<P>_<trial>.<cam>.pose.json
        cam = "cam_" + name.split(".")[1]
        trial = name.split(".")[0].replace(f"delta_{part}_", "")
        side = "right" if "_R_" in trial else "left"
        fn = C._kp_point(f"{side}_wrist")
        # kept cams for this trial
        per = {}
        for pj in glob.glob(str(dets / f"delta_{part}_{trial}.*.pose.json")):
            c = "cam_" + Path(pj).name.split(".")[1]
            if c in cams:
                per[c] = json.loads(Path(pj).read_text())["frames"]
        if len(per) < 2 or cam not in cams:
            out.append((trial, cam, None, None, "no-ref"))
            continue
        sus = json.loads(Path(rp).read_text())["frames"]
        n = min([len(v) for v in per.values()] + [len(sus)])
        X = [None] * n
        for f in range(n):
            pts = {c: fn(per[c][f]) for c in per if fn(per[c][f]) is not None}
            if len(pts) >= 2:
                X[f], _ = ransac_point({c: cams[c] for c in pts if c in cams}, pts)
        best = (0, 1e9)
        for L in range(-MAXLAG, MAXLAG + 1):
            errs = []
            for f in range(n):
                g = f + L
                if X[f] is None or g < 0 or g >= n:
                    continue
                p = fn(sus[g])
                if p is not None:
                    errs.append(float(np.hypot(*(project(cams[cam], X[f])[0] - p))))
            if len(errs) >= 10:
                m = float(np.median(errs))
                if m < best[1]:
                    best = (L, m)
        L, e = best
        if e <= LOW and abs(L) >= 1:
            v = f"DESYNC {L:+d}fr ({e:.0f}px)"
        elif e <= LOW:
            v = f"borderline-ok ({e:.0f}px @0)"
        else:
            v = f"MISCALIB (best {e:.0f}px @{L:+d})"
        out.append((trial, cam, L, round(e), v))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P12", "P13", "P15"])
    a = ap.parse_args(argv)
    for part in a.parts:
        res = classify(part)
        desync = sum(1 for r in res if r[4].startswith("DESYNC"))
        misc = sum(1 for r in res if r[4].startswith("MISCALIB"))
        bord = sum(1 for r in res if r[4].startswith("borderline"))
        print(f"\n### {part}: {len(res)} rejected -> {desync} DESYNC(recoverable), "
              f"{misc} MISCALIB, {bord} borderline", flush=True)
        for trial, cam, L, e, v in sorted(res):
            print(f"  {trial:26} {cam:6} {v}", flush=True)


if __name__ == "__main__":
    main()
