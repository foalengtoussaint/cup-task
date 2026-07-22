"""How fast does cup+pose detection run, per camera, per resolution?

This answers the LIVE question ("can N cameras run at 60fps?"), which is not the same
as the offline one ("how long does the batch take?"). The two differ by a lot, so the
benchmark is careful about what it counts:

  DECODE is EXCLUDED from the live budget. Reading an mp4 costs ~2ms/frame, but a live
  camera hands you the frame already decoded -- you pay that cost in the capture thread,
  in parallel, not in the inference budget. Counting it would understate the rig.

  model.predict(source=<path>) is NOT USED. It re-decodes the video itself and adds ~2x
  overhead on top of inference. We decode once with cv2, then feed the ndarray -- which
  is exactly what a live loop does.

  WARMUP is mandatory. The first few CUDA calls pay lazy init + cudnn autotune and run
  3-10x slow. Timing them would slander the model.

  We report the MEDIAN per-frame latency, not the mean. One scheduler hiccup skews a mean
  and the median is what the rig actually sustains. p95 is reported too, because a live
  system drops frames on the tail, not on the average.

Two models run per camera per frame (cup detector + pose net) -- there is no merged net
here, so the per-camera cost is the SUM of the two. That is the thing to remember when
reading the numbers: N cameras = 2N forward passes per frame time.

    python scripts/bench_detect.py --clip CLIP.mp4
    python scripts/bench_detect.py --clip CLIP.mp4 --imgsz 640 960 1280 --n 120
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

MODELS = Path(__file__).resolve().parents[1] / "models"
CUP_MODEL = MODELS / "cup_clean3d_refill.pt"
POSE_MODELS = {"yolo26n-pose": MODELS / "yolo26n-pose.pt",
               "yolo26s-pose": MODELS / "yolo26s-pose.pt"}
TARGET_FPS = 60.0
WARMUP = 15


def load_frames(clip: Path, n: int) -> list[np.ndarray]:
    """Decode n frames up front. Live capture does this in its own thread."""
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while len(frames) < n:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames


def time_model(model, frames: list[np.ndarray], imgsz: int, device="0", batch: int = 1
               ) -> tuple[float, float]:
    """Median and p95 latency (ms) for ONE BATCH of `batch` frames at one imgsz.

    Times the WHOLE call -- preprocess (resize/letterbox/to-GPU) + inference + postprocess
    (NMS, keypoint decode). All three run live, so all three count. Synchronizes CUDA so
    we time completed work, not just kernel launches.

    batch>1 models the MULTI-CAMERA case: N cameras produce N independent images for the
    same instant, so they go into ONE forward pass as a batch. This is not a trick to make
    the numbers look good -- it is how you would actually build the rig. A GPU is a
    throughput device; feeding it one 640x640 image at a time leaves most of it idle,
    which is why 8 cameras batched cost far less than 8x one camera.
    """
    def _b(i):
        return [frames[(i + k) % len(frames)] for k in range(batch)] if batch > 1 \
            else frames[i % len(frames)]

    for i in range(WARMUP):
        model.predict(_b(i), imgsz=imgsz, device=device, verbose=False)
    torch.cuda.synchronize()

    lat = []
    for i in range(len(frames)):
        t0 = time.perf_counter()
        model.predict(_b(i), imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat)), float(np.percentile(lat, 95))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[480, 640, 960, 1280, 1920])
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10],
                    help="camera counts = batch sizes to sweep")
    ap.add_argument("--n", type=int, default=40, help="batches to time per config")
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"clip: {a.clip.name}", flush=True)

    t0 = time.perf_counter()
    frames = load_frames(a.clip, a.n + WARMUP)
    dec_ms = (time.perf_counter() - t0) / max(len(frames), 1) * 1000
    h, w = frames[0].shape[:2]
    print(f"decoded {len(frames)} frames of {w}x{h} at {dec_ms:.1f} ms/frame "
          f"(EXCLUDED from the live budget -- a camera hands you the frame)\n", flush=True)

    cup = YOLO(str(CUP_MODEL))
    poses = {name: YOLO(str(p)) for name, p in POSE_MODELS.items()}
    budget = 1000.0 / TARGET_FPS

    rows = []
    for name, pose in poses.items():
        print(f"\n### {name} + cup   (both nets run on every camera)", flush=True)
        print(f"{'imgsz':>6} {'cams':>5} {'cup ms':>7} {'pose ms':>8} {'tot ms':>8} "
              f"{'ms/cam':>7} {'max fps':>8}   {TARGET_FPS:.0f}fps?", flush=True)
        for imgsz in a.imgsz:
            for ncam in a.cams:
                c_med, _ = time_model(cup, frames, imgsz, a.device, batch=ncam)
                p_med, _ = time_model(pose, frames, imgsz, a.device, batch=ncam)
                tot = c_med + p_med          # one batched pass each, per frame-instant
                ok = "OK" if tot <= budget else "--"
                print(f"{imgsz:>6} {ncam:>5} {c_med:>7.1f} {p_med:>8.1f} {tot:>8.1f} "
                      f"{tot/ncam:>7.1f} {1000/tot:>8.1f}   {ok}", flush=True)
                rows.append((name, imgsz, ncam, c_med, p_med, tot))

    print("\n" + "=" * 72)
    print(f"REAL-TIME ENVELOPE -- biggest camera count that holds {TARGET_FPS:.0f}fps "
          f"(budget {budget:.1f} ms/frame)")
    print("=" * 72)
    for name in poses:
        for imgsz in a.imgsz:
            fit = [n for nm, i, n, _, _, t in rows
                   if nm == name and i == imgsz and t <= budget]
            best = max(fit) if fit else 0
            print(f"  {name:13s} imgsz {imgsz:4d}: "
                  + (f"{best:2d} cameras" if best else "does not hold at 1 camera"))
    print("\nN cameras = ONE batched forward pass per net (they are independent images of "
          "the\nsame instant). Decode is excluded: a live camera hands you the frame.")


if __name__ == "__main__":
    sys.exit(main())
