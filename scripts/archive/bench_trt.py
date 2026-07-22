"""What fps do we ACTUALLY get, per camera count -- PyTorch vs TensorRT?

Answers the only question that matters for the rig: point at a camera count and a
resolution, get an fps.

TENSORRT (TRT) is NVIDIA's inference compiler. It takes the trained net and emits a
GPU-specific engine: layers fused (conv+bn+act -> one kernel), kernels chosen by timing
candidates on THIS card, and weights actually executed in fp16 -- not merely cast around
a float32 graph, which is why ultralytics `half=True` on a .pt changed nothing (measured:
4.0ms vs 4.1ms). Same weights, no retraining. The cost: an engine is compiled for one GPU,
one imgsz, one MAX batch, so every (imgsz, cams) pair needs its own build (~1-3 min each).

We also combine the two speedups instead of assuming they multiply:
  * BATCH   -- N cameras are N independent images of the same instant, so one forward pass.
  * THREADS -- cup and pose are independent nets; 2 threads overlap one's CPU phase with
               the other's GPU phase. Measured 1.30x at 640, decaying to 1.08x at 1280
               (small images underfeed the GPU; large ones already saturate it).
Whether batching and threading COMPOSE is an empirical question -- a saturated GPU has no
idle gaps left for threading to fill, so the gain should shrink as the batch grows. Hence
we measure the combination rather than multiplying the two numbers together.

    python scripts/bench_trt.py --clip CLIP.mp4 --imgsz 640 --cams 1 2 4 6 8 10
    python scripts/bench_trt.py --clip CLIP.mp4 --imgsz 640 960 --cams 1 4 10 --no-trt
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
MODELS = ROOT / "models"
CUP = MODELS / "cup_clean3d_refill.pt"
POSE = MODELS / "yolo26n-pose.pt"      # the fast one; s costs ~2x infer at high res
WARMUP = 10
TARGETS = (60.0, 30.0)


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


def engine_path(pt: Path, imgsz: int, batch: int) -> Path:
    return pt.with_name(f"{pt.stem}_{imgsz}_b{batch}.engine")


def build_engine(pt: Path, imgsz: int, batch: int):
    """Export .pt -> TensorRT fp16 engine, pinned to this imgsz and max batch. Cached."""
    from ultralytics import YOLO
    out = engine_path(pt, imgsz, batch)
    if out.exists():
        print(f"    engine cached: {out.name}", flush=True)
        return out
    print(f"    building {out.name} (a few minutes)...", flush=True)
    t0 = time.time()
    m = YOLO(str(pt))
    exported = m.export(format="engine", imgsz=imgsz, batch=batch, half=True,
                        device=0, verbose=False)
    Path(exported).rename(out)
    print(f"    built in {time.time()-t0:.0f}s", flush=True)
    return out


def time_pair(cup_m, pose_m, frames, imgsz, batch, threaded=True):
    """Median ms for ONE frame-instant: both nets over a batch of `batch` camera views."""
    def _b(i):
        return [frames[(i + k) % len(frames)] for k in range(batch)]

    if threaded:
        ex = ThreadPoolExecutor(max_workers=2)

        def run(imgs):
            fa = ex.submit(cup_m.predict, imgs, imgsz=imgsz, device=0, verbose=False)
            fb = ex.submit(pose_m.predict, imgs, imgsz=imgsz, device=0, verbose=False)
            fa.result(); fb.result()
    else:
        def run(imgs):
            cup_m.predict(imgs, imgsz=imgsz, device=0, verbose=False)
            pose_m.predict(imgs, imgsz=imgsz, device=0, verbose=False)

    for i in range(WARMUP):
        run(_b(i))
    torch.cuda.synchronize()

    lat = []
    for i in range(len(frames)):
        t0 = time.perf_counter()
        run(_b(i))
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    if threaded:
        ex.shutdown()
    return float(np.median(lat))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--no-trt", action="store_true", help="skip the TensorRT build")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    frames = load_frames(a.clip, a.n + WARMUP)
    print(f"{len(frames)} frames  |  cup={CUP.name}  pose={POSE.name}\n", flush=True)

    cup_pt, pose_pt = YOLO(str(CUP)), YOLO(str(POSE))
    rows = []

    for imgsz in a.imgsz:
        for ncam in a.cams:
            print(f"imgsz {imgsz}  cams {ncam}", flush=True)
            ms_pt = time_pair(cup_pt, pose_pt, frames, imgsz, ncam, threaded=True)
            print(f"    pytorch  {ms_pt:6.1f} ms  -> {1000/ms_pt:5.1f} fps", flush=True)
            ms_trt = None
            if not a.no_trt:
                try:
                    ce = build_engine(CUP, imgsz, ncam)
                    pe = build_engine(POSE, imgsz, ncam)
                    ms_trt = time_pair(YOLO(str(ce)), YOLO(str(pe)), frames, imgsz, ncam,
                                       threaded=True)
                    print(f"    tensorrt {ms_trt:6.1f} ms  -> {1000/ms_trt:5.1f} fps"
                          f"   ({ms_pt/ms_trt:.2f}x)", flush=True)
                except Exception as e:
                    print(f"    tensorrt FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append((imgsz, ncam, ms_pt, ms_trt))

    print("\n" + "=" * 74)
    print(f"{'imgsz':>5} {'cams':>5} | {'pytorch ms':>10} {'fps':>6} | "
          f"{'trt ms':>7} {'fps':>6} | {'60fps?':>7} {'30fps?':>7}")
    print("=" * 74)
    for imgsz, ncam, pt, trt in rows:
        best = trt if trt else pt
        ok60 = "OK" if best <= 1000 / 60 else "--"
        ok30 = "OK" if best <= 1000 / 30 else "--"
        t_s = f"{trt:7.1f} {1000/trt:6.1f}" if trt else f"{'--':>7} {'--':>6}"
        print(f"{imgsz:>5} {ncam:>5} | {pt:>10.1f} {1000/pt:>6.1f} | {t_s} | "
              f"{ok60:>7} {ok30:>7}")
    print("=" * 74)
    print("fps = sustainable rate for ALL cameras (one batched pass per net, threaded).")


if __name__ == "__main__":
    sys.exit(main())
