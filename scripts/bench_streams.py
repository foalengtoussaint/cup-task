"""Run the cup and pose nets on SEPARATE CUDA STREAMS, not just separate Python threads.

WHY THE THREADED VERSION WAS LEAVING SOMETHING ON THE TABLE. Two Python threads calling
model.predict() both submit into the SAME (default) CUDA stream. CUDA serializes work
within a stream, so the two nets' kernels still run one after the other on the device --
all the threading bought was overlapping one net's CPU work (letterbox, HWC->CHW, host->
device copy) with the other's GPU work. That is real, and it measured 1.30x at 640. But
the GPU itself never ran the two nets concurrently.

A separate torch.cuda.Stream per net lets the scheduler INTERLEAVE their kernels on the
device. This matters here specifically because we are LAUNCH-BOUND at batch=1: inference is
flat at ~2.8-3.2ms from imgsz 128 to 640, i.e. the GPU spends most of its time waiting
between small kernels rather than computing. Those gaps are exactly what a second stream
can fill.

It does NOT reduce the number of launches -- so the win should SHRINK as the batch grows and
the GPU becomes genuinely compute-bound (at batch=10 there are no idle gaps left to fill).
That prediction is the point of sweeping batch here rather than testing one size.

CORRECTNESS. Each stream needs its own warmup (cuDNN autotunes per stream), and we must
synchronize BOTH streams before stopping the clock or we time launches, not completions.

    python scripts/bench_streams.py --clip CLIP.mp4 --imgsz 640 --cams 1 2 4 10
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CUP = ROOT / "models" / "cup_clean3d_refill.pt"
POSE = ROOT / "models" / "yolo26n-pose.pt"
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
    return [frames[(i + k) % len(frames)] for k in range(b)] if b > 1 \
        else frames[i % len(frames)]


def time_serial(cup, pose, frames, imgsz, b):
    """Both nets back-to-back, one thread, default stream. The baseline."""
    for i in range(WARMUP):
        x = _batch(frames, i, b)
        cup.predict(x, imgsz=imgsz, device=0, verbose=False)
        pose.predict(x, imgsz=imgsz, device=0, verbose=False)
    torch.cuda.synchronize()
    lat = []
    for i in range(len(frames)):
        x = _batch(frames, i, b)
        t0 = time.perf_counter()
        cup.predict(x, imgsz=imgsz, device=0, verbose=False)
        pose.predict(x, imgsz=imgsz, device=0, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def time_threads(cup, pose, frames, imgsz, b):
    """Two Python threads, SAME default stream: overlaps CPU work only."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        def run(x):
            fa = ex.submit(cup.predict, x, imgsz=imgsz, device=0, verbose=False)
            fb = ex.submit(pose.predict, x, imgsz=imgsz, device=0, verbose=False)
            fa.result(); fb.result()
        for i in range(WARMUP):
            run(_batch(frames, i, b))
        torch.cuda.synchronize()
        lat = []
        for i in range(len(frames)):
            x = _batch(frames, i, b)
            t0 = time.perf_counter()
            run(x)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def time_streams(cup, pose, frames, imgsz, b):
    """Two threads, each on its OWN CUDA STREAM: lets the kernels interleave on-device."""
    s_cup = torch.cuda.Stream()
    s_pose = torch.cuda.Stream()

    with ThreadPoolExecutor(max_workers=2) as ex:
        def on(stream, model, x):
            # the stream must be current INSIDE the worker thread -- stream context is
            # thread-local, so setting it on the main thread would do nothing here
            with torch.cuda.stream(stream):
                return model.predict(x, imgsz=imgsz, device=0, verbose=False)

        def run(x):
            fa = ex.submit(on, s_cup, cup, x)
            fb = ex.submit(on, s_pose, pose, x)
            fa.result(); fb.result()

        for i in range(WARMUP):          # each stream autotunes cuDNN separately
            run(_batch(frames, i, b))
        torch.cuda.synchronize()

        lat = []
        for i in range(len(frames)):
            x = _batch(frames, i, b)
            t0 = time.perf_counter()
            run(x)
            torch.cuda.synchronize()      # BOTH streams, else we time launches not results
            lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    print(f"GPU: {torch.cuda.get_device_name(0)}   imgsz={a.imgsz}", flush=True)
    frames = load_frames(a.clip, a.n)
    print(f"{len(frames)} frames\n", flush=True)

    cup, pose = YOLO(str(CUP)), YOLO(str(POSE))

    print(f"{'cams':>5} {'serial':>8} {'threads':>9} {'streams':>9} "
          f"{'thr gain':>9} {'str gain':>9} {'fps':>7}")
    print("-" * 64)
    for b in a.cams:
        s = time_serial(cup, pose, frames, a.imgsz, b)
        t = time_threads(cup, pose, frames, a.imgsz, b)
        st = time_streams(cup, pose, frames, a.imgsz, b)
        best = min(t, st)
        print(f"{b:>5} {s:>8.1f} {t:>9.1f} {st:>9.1f} "
              f"{s/t:>8.2f}x {s/st:>8.2f}x {1000/best:>7.1f}", flush=True)

    print("\nPREDICTION being tested: streams should help at SMALL batch (GPU is idle "
          "between\nlaunches -- gaps to fill) and stop helping at LARGE batch (compute-bound, "
          "no gaps).")


if __name__ == "__main__":
    sys.exit(main())
