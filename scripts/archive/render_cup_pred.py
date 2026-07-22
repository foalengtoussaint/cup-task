"""Overlay the trained cup detector's predictions on a work-set clip -> mp4, to eyeball why
cross-participant generalization fails (scene/cup/viewpoint gap vs a real bug). Draws GREEN pred
box+conf; if a pool LABEL exists for that frame (train frames), draws it YELLOW so label quality
is visible too."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

SCRATCH = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
               "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/cup_pred")
POOL = Path("/home/imove/Documents/cup-task/data/delta_cup_final")


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--cam", type=int, default=4)
    ap.add_argument("--model", default="runs/segment/runs/cup_seg/cup_seg_3k_p07p08/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args(argv)
    clip = Path(f"cache/delta/{a.part}/work/clips/delta_{a.part}_{a.trial}.{a.cam}.mp4")
    if not clip.exists():
        raise SystemExit(f"no clip {clip}")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / f"{a.part}_{a.trial}_cam{a.cam}_pred.mp4"
    m = YOLO(a.model)
    cap = cv2.VideoCapture(str(clip))
    W, H = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    fi = -1; nhit = 0; n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fi += 1; n += 1
        r = m.predict(img, imgsz=640, conf=a.conf, verbose=False, device=0)[0]
        b = r.boxes
        if b is not None and len(b):
            nhit += 1
            i = int(np.argmax(b.conf.cpu().numpy()))
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[i].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 3)
            cv2.putText(img, f"cup {float(b.conf[i]):.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        cv2.putText(img, f"{a.part} {a.trial} cam{a.cam} f{fi}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(img)
    vw.release(); cap.release()
    print(f"{a.part} {a.trial} cam{a.cam}: cup in {nhit}/{n} frames  -> {out}", flush=True)


if __name__ == "__main__":
    main()
