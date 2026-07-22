"""Show frames where RF-DETR reached >=3-cam consensus but YOLO did NOT.

The question: RF-DETR scores 100% good_frame on P15 while YOLO gets 82.9%, yet we measured the
losses are PHYSICAL occlusion (hand wraps the cup at the lips, mid-trial). Either RF-DETR really
sees through it, or it is locking onto the hand/cup-complex and >=3 cameras agree on the WRONG
thing (consensus cannot catch an error all cameras share).

So: find the disagreement frames and LOOK at them. Draws each camera view with
  GREEN = RF-DETR box      RED = YOLO box (absent if YOLO found nothing)
and writes a contact sheet.

    python scripts/viz_rfdetr_vs_yolo_disagree.py --part P15 \
        --yolo runs/.../best.pt --rfdetr runs/rfdetr/P15/checkpoint_best_regular.pth \
        --out /tmp/disagree_P15.png
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


def yolo_box(model, img):
    r = model.predict(img, imgsz=640, conf=E.CONF, verbose=False, device=0)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return None
    i = int(np.argmax(b.conf.cpu().numpy()))
    return b.xyxy.cpu().numpy()[i]


def rfdetr_box(model, img):
    d = model.predict(np.ascontiguousarray(img[:, :, ::-1]), threshold=E.CONF)
    if d is None or len(d) == 0:
        return None
    i = int(np.argmax(d.confidence))
    return d.xyxy[i]


def ctr(b):
    return None if b is None else (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2))


def main(argv=None):
    from ultralytics import YOLO
    import rfdetr
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--rfdetr", required=True)
    ap.add_argument("--variant", default="RFDETRNano")
    ap.add_argument("--trial", default=None, help="default: first trial")
    ap.add_argument("--fstride", type=int, default=4)
    ap.add_argument("--max-show", type=int, default=4, help="disagreement frames to render")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    y = YOLO(a.yolo)
    r = getattr(rfdetr, a.variant)(pretrain_weights=a.rfdetr)
    calib = C._load_calib_mm(a.part)
    work = C.DELTA / a.part / "work" / "clips"
    stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{a.part}_", "")
                    for p in glob.glob(str(work / "*.mp4"))})
    stem = a.trial or stems[0]
    caps = {}
    for c in calib:
        v = work / f"delta_{a.part}_{stem}.{int(c.split('_')[1])}.mp4"
        if v.exists():
            caps[c] = cv2.VideoCapture(str(v))
    print(f"{a.part} {stem}: {len(caps)} cams", flush=True)

    rows, fi = [], -1
    while len(rows) < a.max_show:
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
        yb = {c: yolo_box(y, im) for c, im in got.items()}
        rb = {c: rfdetr_box(r, im) for c, im in got.items()}
        yo = {c: ctr(v) for c, v in yb.items() if ctr(v) is not None}
        ro = {c: ctr(v) for c, v in rb.items() if ctr(v) is not None}
        Xy = E.consensus(yo, calib)[0] if len(yo) >= 2 else None
        Xr = E.consensus(ro, calib)[0] if len(ro) >= 2 else None
        if Xr is not None and Xy is None:          # RF-DETR consensus, YOLO none
            tiles = []
            for c in sorted(got, key=lambda z: int(z.split('_')[1])):
                im = got[c].copy()
                if rb[c] is not None:
                    x1, y1, x2, y2 = [int(v) for v in rb[c]]
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if yb[c] is not None:
                    x1, y1, x2, y2 = [int(v) for v in yb[c]]
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cx, cy = (rb[c][:2] + rb[c][2:]) / 2 if rb[c] is not None else (960, 540)
                x0, y0 = int(max(0, cx - 220)), int(max(0, cy - 220))
                crop = im[y0:y0 + 440, x0:x0 + 440]
                crop = cv2.resize(crop, (260, 260))
                cv2.putText(crop, f"{c} f{fi}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
                tiles.append(crop)
            rows.append(np.hstack(tiles))
            print(f"  disagreement at frame {fi} (RF-DETR consensus, YOLO none)", flush=True)
    for cap in caps.values():
        cap.release()
    if not rows:
        print("no disagreement frames found", flush=True)
        return
    w = max(r_.shape[1] for r_ in rows)
    rows = [np.pad(r_, ((0, 0), (0, w - r_.shape[1]), (0, 0))) for r_ in rows]
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"wrote {a.out}  ({len(rows)} disagreement frames; GREEN=RF-DETR RED=YOLO)", flush=True)


if __name__ == "__main__":
    main()
