"""Multi-joint reprojection check: is the "miscalibration" global (real) or wrist-specific (fake)?

A camera with wrong GEOMETRY displaces every joint across the whole frame. A camera whose wrist
detection is merely quirky (motion blur, viewpoint bias) reprojects fine on the stable joints.
Only meaningful for cameras whose cuts are correctly PLACED (cut_placement_audit CUTS-OK) --
for mis-cut cams reprojection is meaningless (different repetition).

Per joint: RANSAC consensus from the reference (FINE) cams only, then median reproj of the
suspect cam. Stable joints (nose/shoulders/hips) barely move -> their error IS the geometry error.

    python scripts/multijoint_reproj.py --part P17 --cam 5 --ref 1 2 3 4
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
from reaudit_cam_quality import ransac_point, _trials  # noqa: E402

JOINTS = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip"]


def run(part, cam, refs, n_trials=6):
    cams = C._load_calib_mm(part)
    sus = f"cam_{cam}"
    refks = [f"cam_{r}" for r in refs]
    errs = {j: [] for j in JOINTS}
    for t in _trials(part, n_trials):
        per = {}
        for pj in sorted(glob.glob(str(C.DELTA / part / "dets" / f"delta_{part}_{t}.*.pose.json"))):
            ck = "cam_" + Path(pj).name.split(".")[1]
            if ck in refks + [sus]:
                per[ck] = json.loads(Path(pj).read_text())["frames"]
        if sus not in per or sum(k in per for k in refks) < 2:
            continue
        n = min(len(v) for v in per.values())
        for j in JOINTS:
            fn = C._kp_point(j)
            for f in range(0, n, 3):          # every 3rd frame is plenty
                pts = {k: fn(per[k][f]) for k in refks if k in per and fn(per[k][f]) is not None}
                if len(pts) < 2:
                    continue
                X, _ = ransac_point({k: cams[k] for k in pts if k in cams}, pts)
                if X is None:
                    continue
                p = fn(per[sus][f])
                if p is None:
                    continue
                errs[j].append(float(np.hypot(*(project(cams[sus], X)[0] - p))))
    print(f"\n{part} cam{cam}  (consensus from {refs}, median reproj px per joint)")
    glob_high = 0; joints_seen = 0
    for j in JOINTS:
        if errs[j]:
            m = float(np.median(errs[j])); joints_seen += 1
            if m > 25:
                glob_high += 1
            print(f"  {j:15} {m:7.1f}px  (n={len(errs[j])})")
    if joints_seen:
        frac = glob_high / joints_seen
        v = ("GLOBAL geometry error -> REAL MISCALIBRATION" if frac >= 0.7 else
             "mixed -- partial geometry error, inspect" if frac >= 0.3 else
             "stable joints reproject FINE -> wrist-specific detection issue, NOT miscalibration")
        print(f"  => {glob_high}/{joints_seen} joints >25px: {v}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--ref", nargs="+", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=6)
    a = ap.parse_args(argv)
    run(a.part, a.cam, a.ref, a.n_trials)


if __name__ == "__main__":
    main()
