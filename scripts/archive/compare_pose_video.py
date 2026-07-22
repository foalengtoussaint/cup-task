"""Overlay TWO pose models on one clip to eyeball where they agree/diverge.

Separate yolo11n-pose (CYAN) vs the merged person+cup model (MAGENTA), person
keypoints only. Where the two colors sit on top of each other the models agree
(expected on the head); where they separate you can see the offset the numbers
flagged (elbows/hips). A short line links each joint's two positions so the gap
is obvious.

    python scripts/compare_pose_video.py CLIP.mp4 --merged runs/.../best.pt -o out.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (0, 1), (0, 2), (1, 3), (2, 4),
]
SEP_COLOR = (255, 255, 0)     # cyan  = separate pose model
MRG_COLOR = (255, 0, 255)     # magenta = merged model
GAP_COLOR = (0, 0, 255)       # red link line between the two
MIN_KP = 0.30


def _person_kpts(model, frame, cls_person=0):
    r = model.predict(frame, imgsz=1280, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    cls = r.boxes.cls.cpu().numpy(); conf = r.boxes.conf.cpu().numpy()
    idx = [i for i in range(len(cls)) if int(cls[i]) == cls_person]
    if not idx:
        return None
    b = max(idx, key=lambda i: conf[i])
    return r.keypoints.xy[b].cpu().numpy(), r.keypoints.conf[b].cpu().numpy()


def _draw(frame, kp, color):
    if kp is None:
        return
    xy, cf = kp
    for a, b in COCO_EDGES:
        if cf[a] >= MIN_KP and cf[b] >= MIN_KP:
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)),
                     color, 1, cv2.LINE_AA)
    for j in range(len(xy)):
        if cf[j] >= MIN_KP:
            cv2.circle(frame, tuple(xy[j].astype(int)), 4, color, -1, cv2.LINE_AA)


def main(argv=None) -> int:
    from ultralytics import YOLO
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--sep", default="yolo11n-pose.pt")
    ap.add_argument("--merged", required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args(argv)

    sep = YOLO(args.sep); mrg = YOLO(args.merged)
    cap = cv2.VideoCapture(str(args.clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"compare -> {args.out.name}", flush=True)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ks = _person_kpts(sep, frame, 0)
        km = _person_kpts(mrg, frame, 0)
        # red gap lines first (underneath the dots)
        if ks is not None and km is not None:
            sxy, scf = ks; mxy, mcf = km
            for j in range(len(sxy)):
                if scf[j] >= MIN_KP and mcf[j] >= MIN_KP:
                    cv2.line(frame, tuple(sxy[j].astype(int)), tuple(mxy[j].astype(int)),
                             GAP_COLOR, 1, cv2.LINE_AA)
        _draw(frame, ks, SEP_COLOR)
        _draw(frame, km, MRG_COLOR)
        cv2.putText(frame, "CYAN=separate  MAGENTA=merged  RED=gap", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
        i += 1
        if i % 120 == 0:
            print(f"  {i} frames", flush=True)
    cap.release(); vw.release()
    print(f"done: {i} frames -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
