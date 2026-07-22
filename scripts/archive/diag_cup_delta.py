"""Why does the cup detector miss on DELTA? Draw its RAW output (conf>=0.01, all classes)
on a montage of frames, per camera, at the model's actual 640 input.

The recall table says cam4=0%, cam1=49%. That is the SHIPPING threshold (0.25). This asks the
prior question: is the detector seeing the cup at all (a box near it, just low-conf -> a
threshold/finetune problem), or does it fire nowhere / on the wrong thing (a real domain gap)?
Boxes are labelled with conf; the shipping 0.25 line is noted so sub-threshold hits are visible.

    python scripts/diag_cup_delta.py STAGE_DIR REP_PREFIX --cams 4 1 -o out/diag_cup_P14.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "models/cup_clean3d_refill.pt"
IMGSZ = 640
SHIP_CONF = 0.25
RAW_CONF = 0.01     # draw everything the net proposes, however weak


def _grab(clip, idxs):
    cap = cv2.VideoCapture(str(clip))
    out = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if ok:
            out[i] = img
    cap.release()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage_dir", type=Path)
    ap.add_argument("rep_prefix")
    ap.add_argument("--cams", type=int, nargs="+", required=True)
    ap.add_argument("--nframes", type=int, default=6)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--device", default=0)
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(MODEL)
    names = model.names
    print(f"model classes: {names}", flush=True)

    # sample frames spread across the trial (skip the very start/end)
    cap = cv2.VideoCapture(str(next(a.stage_dir.glob(f"{a.rep_prefix}.{a.cams[0]}.mp4"))))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(total * 0.15, total * 0.85, a.nframes).astype(int).tolist()
    print(f"total {total} fr, sampling {idxs}", flush=True)

    rows = []
    for cam in a.cams:
        clip = a.stage_dir / f"{a.rep_prefix}.{cam}.mp4"
        frames = _grab(clip, idxs)
        cells = []
        for i in idxs:
            img = frames.get(i)
            if img is None:
                continue
            # run at the model's real input size, raw threshold
            r = model.predict(img, imgsz=IMGSZ, conf=RAW_CONF, device=a.device,
                              verbose=False)[0]
            disp = cv2.resize(img, (640, 360))
            sx, sy = 640 / img.shape[1], 360 / img.shape[0]
            nb = 0 if r.boxes is None else len(r.boxes)
            best = 0.0
            if nb:
                for b in range(nb):
                    x1, y1, x2, y2 = (r.boxes.xyxy[b].cpu().numpy() * [sx, sy, sx, sy])
                    conf = float(r.boxes.conf[b]); cls = int(r.boxes.cls[b])
                    best = max(best, conf)
                    col = (0, 255, 0) if conf >= SHIP_CONF else (0, 165, 255)  # green ship / orange sub
                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                    cv2.putText(disp, f"{names[cls]}:{conf:.2f}", (int(x1), int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            tag = f"cam{cam} f{i}  {nb}box best{best:.2f}"
            cv2.rectangle(disp, (0, 0), (640, 18), (0, 0, 0), -1)
            cv2.putText(disp, tag, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cells.append(disp)
        if cells:
            rows.append(np.hstack(cells))
    montage = np.vstack(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), montage)
    print(f"wrote {a.out}  ({montage.shape[1]}x{montage.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
