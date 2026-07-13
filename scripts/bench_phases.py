"""WHERE does the per-frame time actually go, and can the two nets overlap?

Two questions the first benchmark could not answer, because it only timed the whole
predict() call:

1. PHASE SPLIT. Ultralytics reports preprocess / inference / postprocess separately
   (r.speed). Only `inference` is GPU model work. `preprocess` (letterbox, HWC->CHW,
   host->device copy) and `postprocess` (NMS, keypoint decode) are largely CPU and do
   NOT shrink when you pick a smaller model. Published "yolo26n = 2ms @ 640" figures are
   INFERENCE ONLY, on a bigger GPU. If our fixed CPU cost is ~4ms, then swapping s->n
   saves almost nothing at 640 and the model choice is a red herring -- you have to see
   the split to know.

2. OVERLAP. cup and pose are independent (neither reads the other's output) and both fit
   in 8GB. Run back-to-back in one thread, the GPU sits idle during each net's CPU phases.
   Two threads let net B's preprocessing overlap net A's inference.

   This wins only if we are CPU/launch-bound. If we are GPU-compute-bound the two nets
   just contend for the same SMs and the total is unchanged (or worse, from contention).
   Which regime we are in is exactly what question 1 answers -- so measure, don't assume.

    python scripts/bench_phases.py --clip CLIP.mp4 --imgsz 640 960 1280
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

MODELS = Path(__file__).resolve().parents[1] / "models"
CUP_MODEL = MODELS / "cup_clean3d_refill.pt"
POSE_MODELS = {"yolo26n-pose": MODELS / "yolo26n-pose.pt",
               "yolo26s-pose": MODELS / "yolo26s-pose.pt"}
WARMUP = 15


def load_frames(clip: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while len(frames) < n:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames


def phase_split(model, frames, imgsz, device="0", half=False):
    """Median preprocess / inference / postprocess (ms), straight from Ultralytics."""
    for img in frames[:WARMUP]:
        model.predict(img, imgsz=imgsz, device=device, half=half, verbose=False)
    torch.cuda.synchronize()

    pre, inf, post, wall = [], [], [], []
    for img in frames:
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, device=device, half=half, verbose=False)[0]
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    med = lambda x: float(np.median(x))
    return med(pre), med(inf), med(post), med(wall)


def timed_serial(cup, pose, frames, imgsz, device="0", half=False):
    """Both nets, back to back, in one thread -- what the pipeline does today."""
    for img in frames[:WARMUP]:
        cup.predict(img, imgsz=imgsz, device=device, half=half, verbose=False)
        pose.predict(img, imgsz=imgsz, device=device, half=half, verbose=False)
    torch.cuda.synchronize()
    lat = []
    for img in frames:
        t0 = time.perf_counter()
        cup.predict(img, imgsz=imgsz, device=device, half=half, verbose=False)
        pose.predict(img, imgsz=imgsz, device=device, half=half, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def timed_threaded(cup, pose, frames, imgsz, device="0", half=False):
    """Both nets at once, one thread each. Python releases the GIL inside the CUDA call,
    so net B's CPU preprocessing can proceed while net A's kernels run."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        def both(img):
            fa = ex.submit(cup.predict, img, imgsz=imgsz, device=device, half=half,
                           verbose=False)
            fb = ex.submit(pose.predict, img, imgsz=imgsz, device=device, half=half,
                           verbose=False)
            return fa.result(), fb.result()

        for img in frames[:WARMUP]:
            both(img)
        torch.cuda.synchronize()
        lat = []
        for img in frames:
            t0 = time.perf_counter()
            both(img)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    frames = load_frames(a.clip, a.n + WARMUP)
    print(f"{len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}\n", flush=True)

    cup = YOLO(str(CUP_MODEL))
    poses = {n_: YOLO(str(p)) for n_, p in POSE_MODELS.items()}

    for half in (False, True):
        tag = "FP16" if half else "FP32"
        print(f"\n{'='*82}\n{tag}  -- per-frame phase split, 1 camera\n{'='*82}", flush=True)
        print(f"{'imgsz':>5} {'model':>14} {'pre':>6} {'infer':>7} {'post':>6} {'wall':>7}",
              flush=True)
        for imgsz in a.imgsz:
            pr, inf, po, w = phase_split(cup, frames, imgsz, a.device, half)
            print(f"{imgsz:>5} {'cup(seg)':>14} {pr:>6.1f} {inf:>7.1f} {po:>6.1f} {w:>7.1f}",
                  flush=True)
            for nm, m in poses.items():
                pr, inf, po, w = phase_split(m, frames, imgsz, a.device, half)
                print(f"{imgsz:>5} {nm:>14} {pr:>6.1f} {inf:>7.1f} {po:>6.1f} {w:>7.1f}",
                      flush=True)
            print(flush=True)

        print(f"{tag}  -- cup + pose together: SERIAL vs 2 THREADS", flush=True)
        print(f"{'imgsz':>5} {'pose net':>14} {'serial':>8} {'threaded':>9} {'gain':>7}",
              flush=True)
        for imgsz in a.imgsz:
            for nm, m in poses.items():
                s = timed_serial(cup, m, frames, imgsz, a.device, half)
                t = timed_threaded(cup, m, frames, imgsz, a.device, half)
                print(f"{imgsz:>5} {nm:>14} {s:>8.1f} {t:>9.1f} {s/t:>6.2f}x", flush=True)


if __name__ == "__main__":
    sys.exit(main())
