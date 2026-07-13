"""Overlay the MERGED person+cup model on a clip — one model, one pass, both drawn.

Draws the person skeleton + mouth-proxy (from the pose head) and the cup box
(class 1) from a single forward pass of the merged yolo-pose model.

    python scripts/visualize_merged.py CLIP.mp4 --model runs/.../best.pt -o out.mp4
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
KP_COLOR = (0, 255, 0)
EDGE_COLOR = (255, 200, 0)
MOUTH_COLOR = (0, 0, 255)
CUP_COLOR = (0, 255, 255)
MIN_KP = 0.30


def _mouth(kxy, kcf):
    if kcf[0] >= MIN_KP:
        return kxy[0]
    pts = [kxy[j] for j in (1, 2, 3, 4) if kcf[j] >= MIN_KP]
    return np.mean(pts, axis=0) if pts else None


def main(argv=None) -> int:
    from ultralytics import YOLO
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args(argv)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"merged overlay -> {args.out.name}", flush=True)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=args.imgsz, device=0, verbose=False)[0]
        if r.boxes is not None:
            for bi in range(len(r.boxes)):
                cls = int(r.boxes.cls[bi]); conf = float(r.boxes.conf[bi])
                x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[bi].cpu().numpy())
                if cls == 1:  # cup
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CUP_COLOR, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"cup {conf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CUP_COLOR, 2, cv2.LINE_AA)
                else:  # person + keypoints
                    kxy = r.keypoints.xy[bi].cpu().numpy()
                    kcf = r.keypoints.conf[bi].cpu().numpy()
                    for a, b in COCO_EDGES:
                        if kcf[a] >= MIN_KP and kcf[b] >= MIN_KP:
                            cv2.line(frame, tuple(kxy[a].astype(int)),
                                     tuple(kxy[b].astype(int)), EDGE_COLOR, 2, cv2.LINE_AA)
                    for j in range(len(kxy)):
                        if kcf[j] >= MIN_KP:
                            cv2.circle(frame, tuple(kxy[j].astype(int)), 4,
                                       KP_COLOR, -1, cv2.LINE_AA)
                    mp = _mouth(kxy, kcf)
                    if mp is not None:
                        cv2.circle(frame, tuple(mp.astype(int)), 8, MOUTH_COLOR, 2, cv2.LINE_AA)
                        cv2.putText(frame, "mouth", tuple((mp + [10, 0]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MOUTH_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "MERGED person+cup (1 model)", (20, 40),
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
