"""Real-time proof: run the merged person+cup+keypoints model per-frame on a
recorded clip and prove it keeps up with the source framerate.

This is the "real-time demo" without needing a live camera plugged in: we read a
real rig clip (1920x1080 @ 60fps), run ONE model forward per frame, draw the cup
box + person keypoints, and burn a live FPS / per-frame-latency counter into the
video. If the measured model FPS >= the clip's source FPS, the pipeline runs in
real time on this hardware — a webcam feed would behave identically.

We report inference-only FPS (the model) separately from end-to-end FPS
(decode + model + draw + encode), because the supervisor question is "is the
MODEL fast enough to run live" — decode/encode is demo overhead, not the system.

    python scripts/realtime_demo.py CLIP.mp4 --model runs/.../best.pt -o out/realtime.mp4

Prints a summary table at the end and (with --json PATH) writes the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (0, 1), (0, 2), (1, 3), (2, 4),
]
KP_COLOR = (0, 255, 0)      # green skeleton
CUP_COLOR = (0, 165, 255)   # orange cup box
MIN_KP = 0.30
CUP_CLS = 1                 # merged model: 0=person, 1=cup
PERSON_CLS = 0


def _draw(frame, r):
    """Draw the single best person's keypoints + all cup boxes on `frame`."""
    if r.boxes is None or len(r.boxes) == 0:
        return
    cls = r.boxes.cls.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()

    # cups
    for i in range(len(cls)):
        if int(cls[i]) == CUP_CLS:
            x1, y1, x2, y2 = xyxy[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), CUP_COLOR, 2)
            cv2.putText(frame, f"cup {conf[i]:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CUP_COLOR, 2, cv2.LINE_AA)

    # best person's skeleton
    pidx = [i for i in range(len(cls)) if int(cls[i]) == PERSON_CLS]
    if pidx and r.keypoints is not None:
        b = max(pidx, key=lambda i: conf[i])
        xy = r.keypoints.xy[b].cpu().numpy()
        cf = r.keypoints.conf[b].cpu().numpy()
        for a, c in COCO_EDGES:
            if cf[a] >= MIN_KP and cf[c] >= MIN_KP:
                cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[c].astype(int)),
                         KP_COLOR, 2, cv2.LINE_AA)
        for j in range(len(xy)):
            if cf[j] >= MIN_KP:
                cv2.circle(frame, tuple(xy[j].astype(int)), 4, KP_COLOR, -1, cv2.LINE_AA)


def _hud(frame, src_fps, model_fps, e2e_fps, infer_ms):
    """Top-left heads-up display with the live framerate numbers."""
    ok = model_fps >= src_fps
    lines = [
        f"source: {src_fps:.0f} fps  (real-time bar)",
        f"model:  {model_fps:5.1f} fps  ({infer_ms:4.1f} ms/frame)",
        f"end2end:{e2e_fps:5.1f} fps  (+decode/draw/encode)",
        "REAL-TIME: YES" if ok else "REAL-TIME: NO",
    ]
    y = 34
    for i, t in enumerate(lines):
        color = (255, 255, 255)
        if i == 3:
            color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(frame, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)          # outline
        cv2.putText(frame, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2, cv2.LINE_AA)
        y += 34


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--model", default="runs/pose/runs/cup_head_frozen_p3/weights/best.pt")
    ap.add_argument("-o", "--out", type=Path, default=Path("out/realtime.mp4"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=0)
    ap.add_argument("--warmup", type=int, default=5,
                    help="frames to run before timing (GPU warmup, first-call JIT)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        print(f"cannot open {args.clip}", file=sys.stderr)
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w, h = int(cap.get(3)), int(cap.get(4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    print(f"clip {args.clip.name}  {w}x{h} @ {src_fps:.0f}fps  model={Path(args.model).name}",
          flush=True)
    print(f"real-time bar = {1000.0/src_fps:.1f} ms/frame\n", flush=True)

    infer_times: list[float] = []      # model-only, post-warmup
    e2e_start = None
    n = 0
    # rolling display numbers so the HUD is smooth
    disp_model_fps = disp_e2e_fps = disp_ms = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_infer0 = time.perf_counter()
        r = model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        t_infer = time.perf_counter() - t_infer0

        if n == args.warmup:                 # start end-to-end clock after warmup
            e2e_start = time.perf_counter()
        if n >= args.warmup:
            infer_times.append(t_infer)
            recent = infer_times[-30:]
            disp_ms = 1000.0 * float(np.mean(recent))
            disp_model_fps = 1000.0 / disp_ms if disp_ms else 0.0
            elapsed = time.perf_counter() - e2e_start
            done = n - args.warmup + 1
            disp_e2e_fps = done / elapsed if elapsed > 0 else 0.0

        _draw(frame, r)
        _hud(frame, src_fps, disp_model_fps, disp_e2e_fps, disp_ms)
        vw.write(frame)

        n += 1
        if n % 60 == 0:
            print(f"  frame {n:4d}  model {disp_model_fps:5.1f}fps "
                  f"({disp_ms:4.1f}ms)  e2e {disp_e2e_fps:5.1f}fps", flush=True)

    cap.release(); vw.release()

    arr = np.array(infer_times)
    model_fps = 1.0 / arr.mean() if len(arr) else 0.0
    p50_ms = 1000.0 * float(np.median(arr)) if len(arr) else 0.0
    p95_ms = 1000.0 * float(np.percentile(arr, 95)) if len(arr) else 0.0
    e2e_fps = (len(arr)) / (time.perf_counter() - e2e_start) if e2e_start else 0.0
    bar_ms = 1000.0 / src_fps

    print("\n" + "=" * 54)
    print(f"  RESULT  ({len(arr)} timed frames, {args.warmup} warmup)")
    print("=" * 54)
    print(f"  source framerate (bar) : {src_fps:6.1f} fps  ({bar_ms:.1f} ms)")
    print(f"  model inference        : {model_fps:6.1f} fps  "
          f"(p50 {p50_ms:.1f} / p95 {p95_ms:.1f} ms)")
    print(f"  end-to-end (w/ io+draw): {e2e_fps:6.1f} fps")
    verdict = "REAL-TIME  (model faster than source)" if model_fps >= src_fps \
        else "NOT real-time on this hardware"
    print(f"  verdict                : {verdict}")
    print(f"  overlay video          : {args.out}")
    print("=" * 54, flush=True)

    if args.json:
        args.json.write_text(json.dumps({
            "clip": str(args.clip), "model": str(args.model),
            "resolution": [w, h], "source_fps": src_fps,
            "model_fps": model_fps, "p50_ms": p50_ms, "p95_ms": p95_ms,
            "e2e_fps": e2e_fps, "timed_frames": int(len(arr)),
            "realtime": bool(model_fps >= src_fps),
        }, indent=2))
        print(f"json -> {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
