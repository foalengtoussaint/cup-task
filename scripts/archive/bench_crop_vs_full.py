"""Does a person-crop at imgsz=640 match a full frame at imgsz=1280 -- at half the cost?

THE HYPOTHESIS. The pipeline runs imgsz=1280 on the full 1920x1080 frame, presumably
because at 640 the cup is too few pixels to detect. But the subject fills ~50% of the
frame, so cropping to the (padded) person box magnifies everything ~2x. The cup then
occupies the SAME number of pixels in a 640 crop as it did in a 1280 full frame -- so
detection should be unchanged, while inference drops from the compute-bound region
(1280: 6.6ms) into the flat overhead-bound floor (640: 3.2ms).

Measured cost curve (batch=1, either net):
    imgsz   128   512   640  1024  1280  1920
    ms      2.8   2.9   3.2   4.3   6.6  13.3
    ^--------- FLAT (launch-bound) ---^  ^-- compute-bound --^
A crop only wins if it moves you from ABOVE the ~640-1024 knee to BELOW it. Cropping
640->320 buys NOTHING (both on the floor) -- an earlier version of this test made exactly
that mistake and measured 1.08x, which said nothing about the idea.

WHAT WE COMPARE. Same frames, same model, two paths:
    A) full frame            @ 1280   (what the pipeline does today)
    B) padded person crop    @  640   (the proposal, coords remapped back to frame space)
and we ask: same detections? The cup CENTRE is what gets triangulated, so centre
agreement in ORIGINAL-frame pixels is the number that matters -- a few px of box-edge
disagreement is irrelevant, a moved centre is not.

The crop must come from the POSE box, not the cup box: at rest the cup is stationary and
a cup-tracking crop has nothing to re-acquire from if it ever loses it, whereas the person
is always present. (Containment measured separately: the cup is inside the padded person
box in 100% of frames -- it is either at rest on the table or approaching the subject.)

    python scripts/bench_crop_vs_full.py --clip CLIP.mp4
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

MARGIN = 0.25
MIN_CONF = 0.25
WARMUP = 8


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


def pad_box(b, W, H, margin=MARGIN):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    return [max(0, int(x1 - w * margin)), max(0, int(y1 - h * margin)),
            min(W, int(x2 + w * margin)), min(H, int(y2 + h * margin))]


def best_box(res, conf_thr=0.0):
    if res.boxes is None or len(res.boxes) == 0:
        return None, 0.0
    c = res.boxes.conf.cpu().numpy()
    i = int(c.argmax())
    if c[i] < conf_thr:
        return None, 0.0
    return res.boxes.xyxy[i].cpu().numpy().astype(float), float(c[i])


def centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--full-imgsz", type=int, default=1280)
    ap.add_argument("--crop-imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    frames = load_frames(a.clip, a.n)
    H, W = frames[0].shape[:2]
    print(f"{len(frames)} frames of {W}x{H}", flush=True)
    print(f"A: full frame @ {a.full_imgsz}   vs   B: person crop @ {a.crop_imgsz}\n",
          flush=True)

    cup = YOLO(str(CUP))
    pose = YOLO(str(POSE))

    for f in frames[:WARMUP]:
        cup.predict(f, imgsz=a.full_imgsz, device=0, verbose=False)
        pose.predict(f, imgsz=a.crop_imgsz, device=0, verbose=False)
    torch.cuda.synchronize()

    rows = []
    t_full, t_crop = [], []
    cup_px_full, cup_px_crop = [], []

    for img in frames:
        # ---- A: full frame @ full_imgsz ----
        t0 = time.perf_counter()
        rf = cup.predict(img, imgsz=a.full_imgsz, device=0, conf=MIN_CONF,
                         verbose=False)[0]
        torch.cuda.synchronize()
        t_full.append((time.perf_counter() - t0) * 1000)
        bf, cf = best_box(rf)

        # ---- B: person crop @ crop_imgsz (pose to find the box, then cup on the crop) ----
        t0 = time.perf_counter()
        rp = pose.predict(img, imgsz=a.crop_imgsz, device=0, verbose=False)[0]
        pb, _ = best_box(rp)
        bc, cc = None, 0.0
        if pb is not None:
            x1, y1, x2, y2 = pad_box(pb, W, H)
            crop = img[y1:y2, x1:x2]
            rc = cup.predict(crop, imgsz=a.crop_imgsz, device=0, conf=MIN_CONF,
                             verbose=False)[0]
            b, cc = best_box(rc)
            if b is not None:
                bc = np.array([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])  # remap
        torch.cuda.synchronize()
        t_crop.append((time.perf_counter() - t0) * 1000)

        if bf is not None:
            cup_px_full.append(max(bf[2] - bf[0], bf[3] - bf[1]))
        if bc is not None and pb is not None:
            # cup size AS THE NET SAW IT in the crop (that is what detectability tracks)
            scale = a.crop_imgsz / max(x2 - x1, y2 - y1)
            cup_px_crop.append(max(bc[2] - bc[0], bc[3] - bc[1]) * scale)

        rows.append((bf, cf, bc, cc))

    # ---------------- agreement ----------------
    both = [(f, c) for f, _, c, _ in rows if f is not None and c is not None]
    only_full = sum(1 for f, _, c, _ in rows if f is not None and c is None)
    only_crop = sum(1 for f, _, c, _ in rows if f is None and c is not None)
    neither = sum(1 for f, _, c, _ in rows if f is None and c is None)
    n = len(rows)

    print("=" * 66)
    print("DETECTION AGREEMENT")
    print("=" * 66)
    print(f"  both found the cup     : {len(both):4d} / {n}  ({len(both)/n:.0%})")
    print(f"  only FULL found it     : {only_full:4d}   <- crop path LOST these")
    print(f"  only CROP found it     : {only_crop:4d}   <- crop path GAINED these")
    print(f"  neither                : {neither:4d}")

    if both:
        d = np.array([np.linalg.norm(centre(f) - centre(c)) for f, c in both])
        print(f"\n  cup-centre disagreement (original-frame px, the number that "
              f"reaches 3D):")
        print(f"    median {np.median(d):5.1f}  p90 {np.percentile(d,90):5.1f}  "
              f"max {d.max():5.1f}")
        print(f"    within  5px: {(d<5).mean():.0%}     within 10px: {(d<10).mean():.0%}")

    if cup_px_full and cup_px_crop:
        print(f"\n  cup size AS THE NET SEES IT (px on the model's input canvas):")
        print(f"    full @{a.full_imgsz}: {np.median(cup_px_full)*a.full_imgsz/W:5.1f} px")
        print(f"    crop @{a.crop_imgsz}: {np.median(cup_px_crop):5.1f} px   "
              f"<- the crop's whole point: keep this the same")

    print("\n" + "=" * 66)
    print("COST (cup net; the crop path also pays for the pose box it needs anyway)")
    print("=" * 66)
    tf, tc = np.median(t_full), np.median(t_crop)
    print(f"  A full  @{a.full_imgsz:4d}: {tf:6.2f} ms")
    print(f"  B crop  @{a.crop_imgsz:4d}: {tc:6.2f} ms   ({tf/tc:.2f}x)")
    print(f"\n  NOTE: B includes the pose forward pass. In the real pipeline pose runs")
    print(f"  ANYWAY, so its cost is shared -- the marginal cup cost is lower than shown.")


if __name__ == "__main__":
    sys.exit(main())
