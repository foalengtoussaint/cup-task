"""Render the cached pose models' 3D wrists reprojected into one camera, to judge the
REFERENCE by eye.

The point is not to admire the tracking. It is to answer ONE question you cannot answer
from a table: **is yolo26x actually right?** Everything numeric we have ranks n/s/m by
distance to x, which is only meaningful if x is correct -- and x is the same architecture on
the same training data, so it shares the family's biases.

REPROJECTION RESIDUAL CANNOT SETTLE THIS, and it took a correction to see why. It looks like
a model-quality metric but it is contaminated by CALIBRATION ERROR and CAMERA GEOMETRY: a
joint that only badly-spread cameras can see triangulates sloppily no matter how good the 2D
detection is. Hold the rig fixed and vary only the model, and the floor is visible:

    joint          n       s       m       x     spread
    left_wrist   10.39    6.65    6.25    7.12   4.14px
    right_wrist   7.13    7.13    6.58    3.61   3.52px
    mouth         5.16    3.49    3.45    3.25   1.91px

s, m and x all sit at ~6-7px on the left wrist -- that is the GEOMETRY FLOOR, not a ranking.
(I briefly read x's 7.12 vs m's 6.25 as "x is broken on the task hand". It is not: that gap
is inside the floor. Do not read a signal out of a difference smaller than the noise.)
Only **n** is genuinely above the floor (10.39px), and that is a real deficiency.

This draws each model's TRIANGULATED 3D wrist reprojected back into the chosen camera, so
what you see is what the 3D fusion actually produced -- not the raw 2D detections (dumb
player: no re-derivation). Watch the drink: if x's marker drifts off her hand while m's
stays on it, x is not ground truth and the ranking above is circular.

    python scripts/render_pose_ref.py --clip CLIP.mp4 --calib CALIB.toml --stem STEM -o out.mp4
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
COLOR = {"n": (80, 80, 245), "s": (80, 235, 245), "m": (120, 245, 120),
         "x": (255, 255, 255)}
FPS = 60.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--joint", default="left_wrist")
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(argv)

    n_cam = int(re.search(r"\.(\d+)\.mp4$", a.clip.name).group(1))
    ck = f"cam_{n_cam}"
    cam = load_calibration(a.calib, target_size=(1920, 1080))[ck]

    d = CACHE / a.stem
    tracks = {}
    for v in "nsmx":
        t = json.loads((d / f"yolo26{v}.3d.json").read_text())["targets"][a.joint]
        tracks[v] = np.array([x["X"] if x["X"] else [np.nan] * 3 for x in t], float)

    cap = cv2.VideoCapture(str(a.clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    print(f"{a.clip.name} [{ck}] joint={a.joint} -> {a.out}", flush=True)

    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for v in "nsmx":                        # x last -> drawn on top
            X = tracks[v][fr] if fr < len(tracks[v]) else None
            if X is None or not np.isfinite(X).all():
                continue
            (u, vv), infront = project(cam, X)
            if not infront:
                continue
            u, vv = int(round(u)), int(round(vv))
            if v == "x":                         # the reference: big hollow cross
                cv2.drawMarker(img, (u, vv), COLOR[v], cv2.MARKER_TILTED_CROSS, 26, 2)
            else:
                cv2.circle(img, (u, vv), 7, COLOR[v], 2, cv2.LINE_AA)

        cv2.rectangle(img, (0, 0), (W, 42), (0, 0, 0), -1)
        for i, (v, lbl) in enumerate([("n", "n 3.7M"), ("s", "s 11.8M"),
                                      ("m", "m 24.2M"), ("x", "x 69M = REFERENCE")]):
            cv2.circle(img, (20 + i * 200, 21), 7, COLOR[v], -1)
            cv2.putText(img, lbl, (34 + i * 200, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR[v], 1, cv2.LINE_AA)
        cv2.putText(img, f"{a.joint}   t={fr/FPS:5.2f}s", (W - 330, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "is the WHITE cross on her wrist? if not, x is not truth",
                    (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)
        vw.write(img)
        fr += 1

    cap.release(); vw.release()
    print(f"  wrote {a.out} ({fr} frames)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
