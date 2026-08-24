"""What is cam_4 confidently looking at? Its OWN 2D keypoint vs the 10-camera consensus.

THE ANOMALY THIS EXISTS TO EXPLAIN. On the left (task) wrist, cam_4 is the worst camera by
far -- 20.9mm reprojection error vs cam_8's 7.5mm -- and that survives correcting for camera
distance, so it is not an optical artifact. All four pose models (n/s/m/x) show it
identically, so it is not the network's capacity.

The clue is the WITHIN-CAMERA correlation between confidence and error, which holds geometry
fixed and varies only confidence:

    cam_8:  r = -0.402   (sane: less sure -> less accurate)
    cam_2:  r = -0.364   (sane)
    cam_4:  r = +0.669   (INVERTED: the MORE confident it is, the MORE wrong it is)

A merely *fuzzy landmark* -- "different viewpoints see slightly different points on a wrist"
-- would be UNCORRELATED with confidence. A strong POSITIVE correlation means cam_4 is
confidently locking onto something, and that something is not where the other nine cameras
say the wrist is. Confidently-wrong, not vaguely-wrong. (Compare the drink_study note that
cam_4's wrist marker sits so close to the cup that it dominated the agreement metric.)

So: draw what cam_4 SAID (its raw 2D keypoint, with its confidence) next to where the
consensus of all 10 cameras puts the wrist (triangulated 3D, reprojected into cam_4), and
look at the frames where they diverge most.

    python scripts/render_cam_disagree.py --clip CLIP.mp4 --calib CALIB --stem STEM -o out.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.kalman_3d import load_calibration, project

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "pose_models"
FPS = 60.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--model", default="s")
    ap.add_argument("--joint", default="left_wrist")
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(argv)

    n_cam = int(re.search(r"\.(\d+)\.mp4$", a.clip.name).group(1))
    ck = f"cam_{n_cam}"
    cam = load_calibration(a.calib, target_size=(1920, 1080))[ck]

    d = CACHE / a.stem
    per = json.loads((d / f"yolo26{a.model}.2d.json").read_text())[ck]
    tr = json.loads((d / f"yolo26{a.model}.3d.json").read_text())["targets"][a.joint]

    cap = cv2.VideoCapture(str(a.clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    print(f"{ck}  joint={a.joint}  model=yolo26{a.model} -> {a.out}", flush=True)

    worst = []
    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break

        own = per[fr]["kps"].get(a.joint) if fr < len(per) else None
        X = tr[fr]["X"] if fr < len(tr) else None

        pc = None
        if X:
            (u, v), infront = project(cam, np.array(X, float))
            if infront:
                pc = (int(u), int(v))
                cv2.drawMarker(img, pc, (60, 240, 60), cv2.MARKER_TILTED_CROSS, 30, 2)
                cv2.putText(img, "10-cam consensus", (pc[0] + 18, pc[1] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 240, 60), 1, cv2.LINE_AA)

        if own:
            po = (int(own[0]), int(own[1]))
            cv2.circle(img, po, 10, (60, 60, 245), 2, cv2.LINE_AA)
            cv2.putText(img, f"{ck} says  conf={own[2]:.3f}", (po[0] + 14, po[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 245), 1, cv2.LINE_AA)
            if pc:
                cv2.line(img, po, pc, (0, 165, 255), 2, cv2.LINE_AA)
                dpx = float(np.hypot(po[0] - pc[0], po[1] - pc[1]))
                worst.append((dpx, fr, own[2]))
                cv2.putText(img, f"disagree {dpx:.0f}px",
                            ((po[0] + pc[0]) // 2, (po[1] + pc[1]) // 2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(img, f"{ck}  {a.joint}   t={fr/FPS:5.2f}s   "
                         f"RED = what this camera says   GREEN = all 10 cameras agree",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        vw.write(img)
        fr += 1

    cap.release(); vw.release()
    worst.sort(reverse=True)
    print(f"  wrote {a.out} ({fr} frames)", flush=True)
    if worst:
        w = np.array([x[0] for x in worst])
        print(f"\n  disagreement {ck} vs consensus: median {np.median(w):.1f}px  "
              f"p90 {np.percentile(w,90):.1f}px  max {w.max():.1f}px", flush=True)
        print(f"  WORST frames (look at these):", flush=True)
        for dpx, f, cf in worst[:8]:
            print(f"    t={f/FPS:5.2f}s  {dpx:5.1f}px apart, and it was {cf:.3f} confident",
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
