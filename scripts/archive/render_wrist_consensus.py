"""5-cam WRIST consensus grid on the work clips (reliable detector, unlike the cup) to judge a
camera: GREEN = detected wrist (from work/dets pose json), RED = >=3-cam RANSAC consensus wrist
reprojected into that view. Reads cached pose -- no model, fast.

  red constantly offset from green EVEN WHEN STILL  -> MISCALIBRATION (geometry)
  red==green when still, diverges in MOTION         -> DESYNC / bad cut (timing)
Tile shows per-cam gap px + a STILL/MOVE tag (from consensus speed)."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from pipeline.kalman_3d import project  # noqa: E402
from reaudit_cam_quality import ransac_point  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
               "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/wrist_consensus")
STILL = 3.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--side", default=None, help="left/right; default from trial name")
    a = ap.parse_args(argv)
    side = a.side or ("right" if "_R_" in a.trial else "left")
    cams = C._load_calib_mm(a.part)
    fn = C._kp_point(f"{side}_wrist")
    dets = C.DELTA / a.part / "work" / "dets"
    clipd = C.DELTA / a.part / "work" / "clips"
    per = {}
    for pj in sorted(glob.glob(str(dets / f"delta_{a.part}_{a.trial}.*.pose.json"))):
        c = "cam_" + Path(pj).name.split(".")[1]
        if c in cams:
            per[c] = json.loads(Path(pj).read_text())["frames"]
    caps = {c: cv2.VideoCapture(str(clipd / f"delta_{a.part}_{a.trial}.{c.split('_')[1]}.mp4"))
            for c in per}
    order = sorted(per, key=lambda z: int(z.split("_")[1]))
    n = min(len(per[c]) for c in per)
    # consensus + speed
    X = [None] * n
    for f in range(n):
        pts = {c: fn(per[c][f]) for c in per if fn(per[c][f]) is not None}
        if len(pts) >= 2:
            X[f], _ = ransac_point({c: cams[c] for c in pts if c in cams}, pts)
    spd = [None] * n
    for f in range(1, n):
        if X[f] is not None and X[f - 1] is not None:
            spd[f] = float(np.linalg.norm(X[f] - X[f - 1]))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / f"{a.part}_{a.trial}_wrist.mp4"
    W, H = 480, 270; cols = len(order)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W * cols, H))
    for f in range(n):
        ims = {}
        for c in order:
            ok, im = caps[c].read()
            ims[c] = cv2.resize(im, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
        moving = spd[f] is not None and spd[f] >= STILL
        tiles = []
        for c in order:
            im = ims[c]; sx, sy = W / 1920.0, H / 1080.0
            d = fn(per[c][f]); gp = (int(d[0] * sx), int(d[1] * sy)) if d is not None else None
            rp = None
            if X[f] is not None:
                pr = project(cams[c], X[f])[0]; rp = (int(pr[0] * sx), int(pr[1] * sy))
            gap = int(np.hypot(gp[0] - rp[0], gp[1] - rp[1]) / sx) if (gp and rp) else -1
            if gp and rp:
                cv2.line(im, gp, rp, (0, 255, 255), 1)
            if rp:
                cv2.circle(im, rp, 7, (0, 0, 255), 2)
            if gp:
                cv2.circle(im, gp, 5, (0, 220, 0), -1)
            cv2.rectangle(im, (0, 0), (W - 1, H - 1), (200, 200, 200), 2)
            cv2.rectangle(im, (0, 0), (W, 18), (0, 0, 0), -1)
            cv2.putText(im, f"{c} gap{gap}px", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            tiles.append(im)
        grid = np.hstack(tiles)
        cv2.putText(grid, f"{a.part} {a.trial} f{f} {'MOVING' if moving else 'still'}  "
                    "GREEN=detected RED=consensus-reproj", (8, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if moving else (255, 255, 255), 1)
        vw.write(grid)
    vw.release()
    for cap in caps.values():
        cap.release()
    print(f"{a.part} {a.trial} -> {out}", flush=True)


if __name__ == "__main__":
    main()
