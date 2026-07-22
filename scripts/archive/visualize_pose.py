"""Render a cup + keypoint overlay video so you can eyeball the detection.

Draws, per frame: the trained cup box (yellow), the upper-body skeleton, and the
mouth-proxy point (the QTM-head replacement). Runs both models if no cached JSON
is given.

Usage:
    python scripts/visualize_pose.py CLIP.mp4 -o overlay.mp4
    python scripts/visualize_pose.py CLIP.mp4 --pose out.pose.json --cup out.cup.json -o overlay.mp4
    python scripts/visualize_pose.py CLIP.mp4 --no-cup -o overlay.mp4   # pose only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task.pose_keypoints import (  # noqa: E402
    FramePose, extract_pose, mouth_proxy,
)
from cup_task.cup_detect import CupDet, detect_cup  # noqa: E402

# Upper-body skeleton edges (by joint name) — knees/ankles intentionally absent.
EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]
KP_COLOR = (0, 255, 0)       # green joints
EDGE_COLOR = (255, 200, 0)   # cyan-ish bones
MOUTH_COLOR = (0, 0, 255)    # red mouth-proxy (the point scoring depends on)
CUP_COLOR = (0, 255, 255)    # yellow cup box


def _draw_cup(frame, cd: CupDet):
    if cd is None or cd.box is None:
        return frame
    x1, y1, x2, y2 = (int(v) for v in cd.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), CUP_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, f"cup {cd.conf:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CUP_COLOR, 2, cv2.LINE_AA)
    if cd.center is not None:
        cv2.circle(frame, (int(cd.center[0]), int(cd.center[1])), 4,
                   CUP_COLOR, -1, cv2.LINE_AA)
    return frame


def _draw(frame, fp: FramePose):
    for a, b in EDGES:
        if a in fp.kps and b in fp.kps:
            pa, pb = fp.kps[a], fp.kps[b]
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     EDGE_COLOR, 2, cv2.LINE_AA)
    for name, p in fp.kps.items():
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, KP_COLOR, -1, cv2.LINE_AA)
    mp = mouth_proxy(fp)
    if mp is not None:
        cv2.circle(frame, (int(mp[0]), int(mp[1])), 8, MOUTH_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "mouth", (int(mp[0]) + 10, int(mp[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MOUTH_COLOR, 2, cv2.LINE_AA)
    return frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--pose", type=Path, help="cached .pose.json (else extract now)")
    ap.add_argument("--cup", type=Path, help="cached .cup.json (else detect now)")
    ap.add_argument("--no-cup", action="store_true", help="skip cup detection")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--cup-model", default="cup_clean3d_refill.pt")
    ap.add_argument("--device", default=0)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args(argv)

    if args.pose and args.pose.exists():
        payload = json.loads(args.pose.read_text())
        frames = [FramePose(**f) for f in payload["frames"]]
        print(f"loaded {len(frames)} pose frames from {args.pose.name}", flush=True)
    else:
        frames = extract_pose(args.clip, model_path=args.model,
                              device=args.device, imgsz=args.imgsz)

    cups: list[CupDet] = []
    if not args.no_cup:
        if args.cup and args.cup.exists():
            cpay = json.loads(args.cup.read_text())
            cups = [CupDet(**d) for d in cpay["frames"]]
            print(f"loaded {len(cups)} cup frames from {args.cup.name}", flush=True)
        else:
            cups = detect_cup(args.clip, model_path=args.cup_model,
                              device=args.device, imgsz=args.imgsz)

    cap = cv2.VideoCapture(str(args.clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))
    print(f"writing {args.out.name}  ({w}x{h} @ {fps:.0f}fps)", flush=True)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < len(cups):
            _draw_cup(frame, cups[i])
        if i < len(frames):
            _draw(frame, frames[i])
        vw.write(frame)
        i += 1
        if i % 120 == 0:
            print(f"  {i} frames rendered", flush=True)
    cap.release()
    vw.release()
    print(f"done: {i} frames -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
