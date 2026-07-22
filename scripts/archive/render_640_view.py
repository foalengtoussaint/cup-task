"""Render the clip AT THE RESOLUTION THE MODEL ACTUALLY SEES (640), with its detections.

The normal overlay draws on the full 1920x1080 frame, which flatters the detector: you
look at a crisp 1080p cup and think "of course it found that". The model never sees that
image. It sees a 640-wide letterboxed version, in which the cup is ~14x15 px.

This renders THAT -- the actual model input -- with the actual per-frame detections drawn
on it. It is the honest picture of how much signal the network has to work with, and it is
the frame to look at when deciding whether a miss was resolution or occlusion.

    python scripts/render_640_view.py --clip CLIP.mp4 -o out.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CUP = ROOT / "models" / "cup_clean3d_refill.pt"
POSE = ROOT / "models" / "yolo26n-pose.pt"
IMGSZ = 640
FPS = 60.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    cup, pose = YOLO(str(CUP)), YOLO(str(POSE))

    cap = cv2.VideoCapture(str(a.clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = a.imgsz / max(W, H)                      # the letterbox scale the model applies
    ow, oh = int(W * s), int(H * s)
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ow, oh))
    print(f"{a.clip.name}: {W}x{H} -> model sees {ow}x{oh}", flush=True)

    fr = found = 0
    sizes = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(img, (ow, oh))

        rc = cup.predict(img, imgsz=a.imgsz, device=0, conf=0.25, verbose=False)[0]
        rp = pose.predict(img, imgsz=a.imgsz, device=0, verbose=False)[0]

        # pose skeleton (thin -- it is context, the cup is the subject here)
        if rp.keypoints is not None and len(rp.keypoints.xy):
            k = rp.keypoints.xy[0].cpu().numpy() * s
            kc = rp.keypoints.conf[0].cpu().numpy()
            for j in (0, 9, 10, 5, 6):          # nose, wrists, shoulders
                if kc[j] > 0.3:
                    cv2.circle(small, (int(k[j][0]), int(k[j][1])), 3, (90, 235, 90), -1)

        if rc.boxes is not None and len(rc.boxes):
            i = int(rc.boxes.conf.cpu().numpy().argmax())
            b = rc.boxes.xyxy[i].cpu().numpy() * s
            conf = float(rc.boxes.conf[i])
            w_px, h_px = int(b[2] - b[0]), int(b[3] - b[1])
            sizes.append(max(w_px, h_px))
            found += 1
            cv2.rectangle(small, (int(b[0]) - 2, int(b[1]) - 2),
                          (int(b[2]) + 2, int(b[3]) + 2), (60, 220, 255), 1)
            cv2.putText(small, f"cup {conf:.2f}  {w_px}x{h_px}px",
                        (int(b[0]) - 20, int(b[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (60, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(small, "NO CUP", (int(ow * 0.42), 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (60, 60, 245), 2, cv2.LINE_AA)

        cv2.putText(small, f"MODEL INPUT {ow}x{oh}   t={fr/FPS:5.2f}s",
                    (8, oh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                    cv2.LINE_AA)
        vw.write(small)
        fr += 1
        if fr % 100 == 0:
            print(f"  {fr} frames, cup in {found} ({found/fr:.0%})", flush=True)

    cap.release(); vw.release()
    print(f"\n  {fr} frames, cup found in {found} ({found/fr:.0%})", flush=True)
    if sizes:
        print(f"  cup size on the {a.imgsz} canvas: median {np.median(sizes):.0f}px  "
              f"min {min(sizes)}px  max {max(sizes)}px", flush=True)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
