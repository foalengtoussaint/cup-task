"""Crop-tracking ("hybrid") inference: is it faster, and does it still see the cup?

THE IDEA (standard top-down tracking): detect the subject on a full frame once, then for
every subsequent frame crop to the previous box + margin, run the net on the small crop,
and add the crop offset back to the keypoints. A 1080p frame becomes a ~256-512px crop,
so inference cost falls with pixel count.

TWO THINGS THIS SCRIPT CHECKS, because both can kill the idea:

1. IS THE SPEEDUP REAL? Cropping shrinks INFERENCE, but preprocess/postprocess and the
   Python/CUDA launch overhead are ~1ms of FIXED cost that does not shrink, and we ADD a
   crop+resize. Measured phase split at 640: infer 3.1ms of a 4.1ms call. So a 16x pixel
   cut cannot buy 16x. Amdahl, not pixels.

2. DOES THE CUP SURVIVE? The crop tracks the PERSON. The cup starts on the table and may
   sit OUTSIDE the person's box exactly during rest_pre -- which is the phase the grasp
   detector needs it in. If the cup leaves the person box, a person-crop pipeline goes
   blind to it and the segmentation breaks. This script measures the containment rate
   directly: what fraction of frames is the cup box inside the padded person box?

DRIFT GUARD (not optional). Deriving the next crop from the current prediction is a
FEEDBACK LOOP: one bad frame moves the crop, which makes the next frame worse. A tracker
that only ever looks at its own crop can lock onto nothing and never recover. So we
(a) force a full-frame re-detect every REDETECT_EVERY frames, and (b) re-detect whenever
confidence drops or a keypoint lands on the crop border. Both are cheap; without them the
failure is silent, which is the worst kind.

    python scripts/bench_crop.py --clip CLIP.mp4 --imgsz 640
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

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CUP = MODELS / "cup_clean3d_refill.pt"
POSE = MODELS / "yolo26n-pose.pt"

MARGIN = 0.25            # pad the person box by this fraction of its size
REDETECT_EVERY = 30      # forced full-frame re-detect (drift guard)
MIN_BOX_CONF = 0.50      # below this, stop trusting the crop and re-detect
WARMUP = 10


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


def pad_box(box, W, H, margin=MARGIN):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 -= w * margin; x2 += w * margin
    y1 -= h * margin; y2 += h * margin
    return [max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))]


def person_box(model, img, imgsz):
    r = model.predict(img, imgsz=imgsz, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, 0.0
    i = int(r.boxes.conf.cpu().numpy().argmax())
    return r.boxes.xyxy[i].cpu().numpy().tolist(), float(r.boxes.conf[i])


def cup_box(model, img, imgsz):
    r = model.predict(img, imgsz=imgsz, device=0, conf=0.25, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    i = int(r.boxes.conf.cpu().numpy().argmax())
    return r.boxes.xyxy[i].cpu().numpy().tolist()


def contains(outer, inner) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640, help="imgsz for the FULL-frame path")
    ap.add_argument("--crop-imgsz", type=int, default=320, help="imgsz fed for a crop")
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    frames = load_frames(a.clip, a.n + WARMUP)
    H, W = frames[0].shape[:2]
    print(f"{len(frames)} frames of {W}x{H}\n", flush=True)

    pose = YOLO(str(POSE))
    cup = YOLO(str(CUP))

    # ---------- Q2 FIRST: does the cup stay inside the person box? ----------
    # If it does not, the speed question is moot -- a person-crop pipeline cannot see the
    # cup, and the cup is the whole task.
    print("=== containment: is the cup inside the padded person box? ===", flush=True)
    inside = out = nocup = 0
    areas = []
    for img in frames[WARMUP:]:
        pb, _ = person_box(pose, img, a.imgsz)
        cb = cup_box(cup, img, a.imgsz)
        if pb is None:
            continue
        pbp = pad_box(pb, W, H)
        areas.append((pbp[2] - pbp[0]) * (pbp[3] - pbp[1]) / (W * H))
        if cb is None:
            nocup += 1
        elif contains(pbp, cb):
            inside += 1
        else:
            out += 1
    tot = inside + out
    print(f"  cup detected in {tot}/{tot+nocup} frames", flush=True)
    if tot:
        print(f"  INSIDE padded person box : {inside:4d}  ({inside/tot:.0%})", flush=True)
        print(f"  OUTSIDE (crop would MISS): {out:4d}  ({out/tot:.0%})", flush=True)
    print(f"  padded person box covers {np.mean(areas):.0%} of the frame "
          f"(a crop only helps if this is SMALL)\n", flush=True)

    # ---------- Q1: is the crop path actually faster? ----------
    print("=== speed: full-frame vs crop-tracked (pose only, 1 camera) ===", flush=True)

    for img in frames[:WARMUP]:
        pose.predict(img, imgsz=a.imgsz, device=0, verbose=False)
    torch.cuda.synchronize()

    lat = []
    for img in frames[WARMUP:]:
        t0 = time.perf_counter()
        pose.predict(img, imgsz=a.imgsz, device=0, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    full_ms = float(np.median(lat))
    print(f"  full frame  @{a.imgsz:4d}: {full_ms:5.2f} ms  ({1000/full_ms:.0f} fps)",
          flush=True)

    # crop-tracked, with the drift guard
    box = None
    lat, redetects, crop_px = [], 0, []
    for i, img in enumerate(frames[WARMUP:]):
        t0 = time.perf_counter()
        need_full = box is None or i % REDETECT_EVERY == 0
        if need_full:
            b, conf = person_box(pose, img, a.imgsz)
            redetects += 1
            box = pad_box(b, W, H) if b is not None else None
        else:
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]
            crop_px.append(crop.shape[0] * crop.shape[1])
            r = pose.predict(crop, imgsz=a.crop_imgsz, device=0, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0 or \
                    float(r.boxes.conf.max()) < MIN_BOX_CONF:
                box = None                    # drift guard: force a full re-detect
            else:
                j = int(r.boxes.conf.cpu().numpy().argmax())
                bb = r.boxes.xyxy[j].cpu().numpy()
                # remap crop coords -> original frame coords
                bb = [bb[0] + x1, bb[1] + y1, bb[2] + x1, bb[3] + y1]
                box = pad_box(bb, W, H)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)

    crop_ms = float(np.median(lat))
    mean_crop = np.mean(crop_px) if crop_px else 0
    print(f"  crop-track  @{a.crop_imgsz:4d}: {crop_ms:5.2f} ms  ({1000/crop_ms:.0f} fps)"
          f"   {full_ms/crop_ms:.2f}x", flush=True)
    print(f"  {redetects} full-frame re-detects of {len(lat)} frames; "
          f"mean crop {np.sqrt(mean_crop):.0f}x{np.sqrt(mean_crop):.0f} px", flush=True)
    print(f"\n  NOTE: this is the POSE net only. The CUP net cannot use a person-crop "
          f"unless the\n  containment number above is ~100%.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
