"""5-cam grid render of the CUP as the 3D metric sees it: GREEN = raw per-cam detection,
RED = the >=3-cam <=30px 3D consensus reprojected into that view (drawn in EVERY camera, even
ones that missed -- that is the 'fill'). Uses the SAME consensus() as eval_cup_3d_delta, so the
picture matches the numbers (the shared-code rule). A yellow line links green<->red when both exist.
Tile border: green=in-consensus (inlier), red=detection rejected, grey=no detection."""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from pipeline.kalman_3d import project  # noqa: E402
from eval_cup_3d_delta import consensus, cup_center  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
               "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/cup_consensus")


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cams", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args(argv)
    calib = C._load_calib_mm(a.part)
    clipdir = C.DELTA / a.part / "work" / "clips"
    caps = {}
    for c in a.cams:
        f = clipdir / f"delta_{a.part}_{a.trial}.{c}.mp4"
        if f.exists() and f"cam_{c}" in calib:
            caps[f"cam_{c}"] = cv2.VideoCapture(str(f))
    if len(caps) < 3:
        raise SystemExit("need >=3 cams")
    model = YOLO(a.model)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / f"{a.part}_{a.trial}_consensus.mp4"
    cols = len(caps); W, H = 480, 270
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W * cols, H))
    order = sorted(caps, key=lambda z: int(z.split("_")[1]))
    ngood = n = 0
    while True:
        imgs = {}
        for c in order:
            ok, im = caps[c].read()
            if ok:
                imgs[c] = im
        if len(imgs) < len(caps):
            break
        n += 1
        det = {c: cup_center(model, imgs[c]) for c in order}
        obs = {c: det[c] for c in order if det[c] is not None}
        X, kept, _ = consensus(obs, calib) if len(obs) >= 2 else (None, set(), None)
        if X is not None:
            ngood += 1
        tiles = []
        for c in order:
            im = cv2.resize(imgs[c], (W, H)); sx, sy = W / 1920.0, H / 1080.0
            gp = (int(det[c][0] * sx), int(det[c][1] * sy)) if det[c] is not None else None
            rp = None
            if X is not None:
                pr = project(calib[c], X)[0]
                rp = (int(pr[0] * sx), int(pr[1] * sy))
            if gp and rp:
                cv2.line(im, gp, rp, (0, 255, 255), 1)
            if rp:
                cv2.circle(im, rp, 7, (0, 0, 255), 2)          # consensus reprojection = RED
            if gp:
                cv2.circle(im, gp, 5, (0, 220, 0), -1)         # raw detection = GREEN
            border = ((0, 200, 0) if c in kept else (0, 0, 255) if det[c] is not None
                      else (120, 120, 120))
            cv2.rectangle(im, (0, 0), (W - 1, H - 1), border, 3)
            cv2.putText(im, c, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 2)
            tiles.append(im)
        grid = np.hstack(tiles)
        cv2.putText(grid, f"{a.part} {a.trial} f{n}  GREEN=det RED=consensus-reproj  "
                    f"consensus={'YES' if X is not None else 'no'}",
                    (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(grid)
    vw.release()
    for cap in caps.values():
        cap.release()
    print(f"{a.part} {a.trial}: consensus in {ngood}/{n} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
