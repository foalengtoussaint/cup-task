"""Overlay the triangulated 3D points back onto one camera's video.

Each frame's 3D cup/mouth/wrist point is projected into the chosen camera with
that camera's calibration and drawn on the real footage. This is the honest
test of the 3D: a correct triangulation reprojects exactly onto the object in
every view — including views that never detected it themselves (the 3D point
comes from the OTHER cameras). That's the whole value of fusion made visible.

    python scripts/reproject_video.py CLIP.mp4 out/rep_P07_3d.json \
        --calib data/P07_calibration.toml -o out/reproj_cam1.mp4
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

COLORS = {"cup": (0, 0, 255), "mouth": (255, 128, 0),
          "left_wrist": (0, 200, 0), "right_wrist": (0, 165, 255)}


def _cam_key(clip_name: str) -> str:
    m = re.search(r"\.(\d+)\.mp4$", clip_name)
    if not m:
        raise ValueError(f"no .N.mp4 camera suffix in {clip_name}")
    return f"cam_{int(m.group(1))}"


def _dense(track, n):
    a = [None] * n
    for t in track:
        if t["X"] is not None and t["frame"] < n:
            a[t["frame"]] = np.array(t["X"], dtype=float)
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("tri_json", type=Path)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cams = load_calibration(args.calib, target_size=(1920, 1080))
    ckey = _cam_key(args.clip.name)
    if ckey not in cams:
        raise SystemExit(f"{ckey} not in calib {sorted(cams)}")
    cam = cams[ckey]
    print(f"reprojecting into {ckey}", flush=True)

    data = json.loads(args.tri_json.read_text())
    cap = cv2.VideoCapture(str(args.clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w, h = int(cap.get(3)), int(cap.get(4))
    tracks = {name: _dense(tr, n) for name, tr in data["targets"].items()}

    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        y0 = 40
        for name, seq in tracks.items():
            col = COLORS.get(name, (255, 255, 255))
            X = seq[i] if i < len(seq) else None
            if X is not None:
                uv, infront = project(cam, X)
                if infront and -50 <= uv[0] <= w + 50 and -50 <= uv[1] <= h + 50:
                    u, v = int(uv[0]), int(uv[1])
                    cv2.circle(frame, (u, v), 9, col, 2, cv2.LINE_AA)
                    cv2.circle(frame, (u, v), 2, col, -1, cv2.LINE_AA)
                    cv2.putText(frame, name, (u + 12, v),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            # status legend, top-left
            tag = "--" if X is None else "3D"
            cv2.putText(frame, f"{name}: {tag}", (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            y0 += 28
        cv2.putText(frame, f"{ckey}  frame {i}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
        i += 1
        if i % 120 == 0:
            print(f"  {i}/{n} frames", flush=True)
    cap.release()
    vw.release()
    print(f"done: {i} frames -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
