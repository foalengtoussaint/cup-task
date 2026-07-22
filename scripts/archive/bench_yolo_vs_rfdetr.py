"""Speed: our YOLO-seg cup detector vs the finetuned RF-DETR, batched across cameras.

Same protocol as bench_streams.py (which produced docs/realtime.md):
  * frames PRELOADED -> decode is EXCLUDED from the budget (a live camera hands you the frame
    already decoded; you pay that in a capture thread, in parallel)
  * WARMUP iterations before timing (cuDNN autotunes)
  * torch.cuda.synchronize() before stopping the clock, else we time LAUNCHES not completions
  * median latency over N iterations, swept over camera counts (batch = n cameras)

⚠ RESOLUTION CAVEAT: RF-DETR Nano runs at 384 (its config; must be divisible by 56, so 640 is
not available), while our YOLO runs at 640. 384 is ~2.8x fewer pixels, so RF-DETR is FLATTERED
on speed here. This benchmark measures the models AS CONFIGURED, not at matched resolution.

    python scripts/bench_yolo_vs_rfdetr.py --clip CLIP.mp4 \
        --yolo runs/.../best.pt --rfdetr runs/rfdetr/P07/checkpoint_best_regular.pth \
        --cams 1 2 4 6 8 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WARMUP = 12


def load_frames(clip: Path, n: int):
    cap = cv2.VideoCapture(str(clip))
    fr = []
    while len(fr) < n:
        ok, img = cap.read()
        if not ok:
            break
        fr.append(img)
    cap.release()
    return fr


def _batch(frames, i, b):
    return [frames[(i + k) % len(frames)] for k in range(b)]


def time_yolo(model, frames, imgsz, b):
    for i in range(WARMUP):
        model.predict(_batch(frames, i, b), imgsz=imgsz, device=0, verbose=False)
    torch.cuda.synchronize()
    lat = []
    for i in range(len(frames)):
        x = _batch(frames, i, b)
        t0 = time.perf_counter()
        model.predict(x, imgsz=imgsz, device=0, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def time_rfdetr(model, frames, b, thr=0.25):
    # RF-DETR takes RGB; predict() accepts a list -> batched.
    # ascontiguousarray: the ::-1 flip yields NEGATIVE strides, which torch refuses.
    # Done ONCE here (outside the timed loop) so it costs nothing in the measurement.
    rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
    for i in range(WARMUP):
        model.predict(_batch(rgb, i, b), threshold=thr)
    torch.cuda.synchronize()
    lat = []
    for i in range(len(rgb)):
        x = _batch(rgb, i, b)
        t0 = time.perf_counter()
        model.predict(x, threshold=thr)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--yolo", type=Path, required=True)
    ap.add_argument("--rfdetr", type=Path, required=True)
    ap.add_argument("--variant", default="RFDETRNano")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)

    frames = load_frames(a.clip, a.n)
    print(f"{len(frames)} frames preloaded (decode EXCLUDED) from {a.clip.name}", flush=True)

    from ultralytics import YOLO
    import rfdetr
    y = YOLO(str(a.yolo))
    r = getattr(rfdetr, a.variant)(pretrain_weights=str(a.rfdetr))
    res = getattr(r.model_config, "resolution", "?") if hasattr(r, "model_config") else "?"
    print(f"YOLO @ imgsz {a.imgsz}   |   {a.variant} @ resolution {res}  "
          f"(⚠ not matched -- RF-DETR flattered on speed)\n", flush=True)

    print(f"{'cams':>5} | {'YOLO ms':>8} {'YOLO fps':>9} | {'RFDETR ms':>10} {'RFDETR fps':>11}",
          flush=True)
    for b in a.cams:
        my = time_yolo(y, frames, a.imgsz, b)
        try:
            mr = time_rfdetr(r, frames, b)
            rs = f"{mr:10.1f} {1000.0/mr:11.1f}"
        except Exception as e:                     # OOM at high batch is a real result
            rs = f"{'OOM/err':>10} {str(e)[:18]:>11}"
            torch.cuda.empty_cache()
        print(f"{b:5d} | {my:8.1f} {1000.0/my:9.1f} | {rs}", flush=True)


if __name__ == "__main__":
    main()
