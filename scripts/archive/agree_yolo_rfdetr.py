"""Do YOLO and RF-DETR reach consensus on the SAME frames, or different ones?

good_frame% is an aggregate: 80% vs 88% could mean RF-DETR's good frames are a strict SUPERSET of
YOLO's (it simply recalls more), or that each fails on frames the other handles (complementary --
in which case an ensemble/either-of gets you more than either alone). Only a per-frame comparison
separates those.

Reports the 2x2 contingency over frames, plus -- where BOTH reach consensus -- the 3D distance
between their cup points (are they even tracking the same object?).

    python scripts/agree_yolo_rfdetr.py --part P15 --yolo <best.pt> --rfdetr <ckpt.pth>
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_cup_3d_delta as E  # noqa: E402
import compare_pose_omc_delta as C  # noqa: E402


def yolo_center(model, img):
    r = model.predict(img, imgsz=640, conf=E.CONF, verbose=False, device=0)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return None
    i = int(np.argmax(b.conf.cpu().numpy()))
    x1, y1, x2, y2 = b.xyxy.cpu().numpy()[i]
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def rfdetr_center(model, img):
    d = model.predict(np.ascontiguousarray(img[:, :, ::-1]), threshold=E.CONF)
    if d is None or len(d) == 0:
        return None
    i = int(np.argmax(d.confidence))
    x1, y1, x2, y2 = d.xyxy[i]
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def main(argv=None):
    from ultralytics import YOLO
    import rfdetr
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--rfdetr", required=True)
    ap.add_argument("--variant", default="RFDETRNano")
    ap.add_argument("--max-trials", type=int, default=4)
    ap.add_argument("--fstride", type=int, default=4)
    a = ap.parse_args(argv)

    y = YOLO(a.yolo)
    r = getattr(rfdetr, a.variant)(pretrain_weights=a.rfdetr)
    calib = C._load_calib_mm(a.part)
    work = C.DELTA / a.part / "work" / "clips"
    stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{a.part}_", "")
                    for p in glob.glob(str(work / "*.mp4"))})[:a.max_trials]

    both = yonly = ronly = neither = 0
    d3 = []
    for stem in stems:
        caps = {}
        for c in calib:
            v = work / f"delta_{a.part}_{stem}.{int(c.split('_')[1])}.mp4"
            if v.exists():
                caps[c] = cv2.VideoCapture(str(v))
        if len(caps) < E.MINC:
            continue
        fi = -1
        while True:
            fi += 1
            got = {}
            for c, cap in caps.items():
                ok, img = cap.read()
                if ok:
                    got[c] = img
            if len(got) < len(caps):
                break
            if fi % a.fstride:
                continue
            yo = {c: p for c in got if (p := yolo_center(y, got[c])) is not None}
            ro = {c: p for c in got if (p := rfdetr_center(r, got[c])) is not None}
            Xy = E.consensus(yo, calib)[0] if len(yo) >= 2 else None
            Xr = E.consensus(ro, calib)[0] if len(ro) >= 2 else None
            if Xy is not None and Xr is not None:
                both += 1
                d3.append(float(np.linalg.norm(np.asarray(Xy) - np.asarray(Xr))))
            elif Xy is not None:
                yonly += 1
            elif Xr is not None:
                ronly += 1
            else:
                neither += 1
        for cap in caps.values():
            cap.release()

    tot = both + yonly + ronly + neither
    print(f"\n=== {a.part}: per-frame consensus agreement (n={tot}) ===", flush=True)
    print(f"  BOTH good      : {both:5d}  ({both/tot*100:5.1f}%)")
    print(f"  YOLO only      : {yonly:5d}  ({yonly/tot*100:5.1f}%)   <- RF-DETR misses these")
    print(f"  RF-DETR only   : {ronly:5d}  ({ronly/tot*100:5.1f}%)   <- YOLO misses these")
    print(f"  NEITHER        : {neither:5d}  ({neither/tot*100:5.1f}%)  <- genuinely unrecoverable")
    print(f"  union (either) : {(both+yonly+ronly)/tot*100:5.1f}%   "
          f"vs YOLO {(both+yonly)/tot*100:.1f}%  RF-DETR {(both+ronly)/tot*100:.1f}%")
    if d3:
        d3 = np.array(d3)
        print(f"\n  where BOTH good, 3D cup distance: median {np.median(d3):.1f}mm  "
              f"p90 {np.percentile(d3,90):.1f}mm  max {d3.max():.1f}mm")
        print(f"  frames >50mm apart (tracking different things?): "
              f"{(d3>50).sum()} ({(d3>50).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
