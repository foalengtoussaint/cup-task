"""n vs s vs m -pose, drawn on the same frames so the difference is visible, not just tabulated.

The numbers said s beats n on wrist jitter (4.37 -> 3.55mm 3D jerk) and reprojection
(7.28 -> 6.27px), and that m adds almost nothing over s. Those are small numbers. This
renders the three nets' keypoints on the SAME video so you can see whether that gap is real
to the eye or a difference only a table can love.

Three markers per joint, one per model, plus a per-frame 2D spread readout: how far apart do
the three nets place the SAME joint? When they disagree by 20px on a wrist, one of them is
wrong, and the 3D triangulation is being fed that error.

    python scripts/render_pose_compare.py --clip CLIP.mp4 -o out.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
COLOR = {"n": (80, 80, 245), "s": (80, 235, 245), "m": (120, 245, 120)}  # BGR
JOINTS = {"left_wrist": 9, "right_wrist": 10, "nose": 0}
FPS = 60.0
MIN_CONF = 0.30


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--zoom", action="store_true",
                    help="crop to the hand/face region so the pixel differences are visible")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    nets = {k: YOLO(str(ROOT / "models" / f"yolo26{k}-pose.pt")) for k in ("n", "s", "m")}

    cap = cv2.VideoCapture(str(a.clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    print(f"{a.clip.name} -> {a.out}", flush=True)

    spreads = []
    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break

        kp = {}
        for k, m in nets.items():
            r = m.predict(img, imgsz=a.imgsz, device=0, verbose=False)[0]
            if r.keypoints is not None and len(r.keypoints.xy):
                i = int(r.boxes.conf.cpu().numpy().argmax())
                kp[k] = (r.keypoints.xy[i].cpu().numpy(),
                         r.keypoints.conf[i].cpu().numpy())

        # draw each model's joints
        for jname, j in JOINTS.items():
            pts = []
            for k, (xy, cf) in kp.items():
                if cf[j] < MIN_CONF:
                    continue
                p = xy[j]
                pts.append(p)
                cv2.circle(img, (int(p[0]), int(p[1])), 6, COLOR[k], 2, cv2.LINE_AA)
            if len(pts) == 3:                       # how far apart are the three nets?
                P = np.array(pts)
                spread = float(np.max(np.linalg.norm(P - P.mean(0), axis=1)))
                if "wrist" in jname:
                    spreads.append(spread)
                if spread > 15:                      # flag a real disagreement
                    c = P.mean(0).astype(int)
                    cv2.circle(img, tuple(c), int(spread) + 8, (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, f"{spread:.0f}px", (c[0] + 12, c[1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, 0), (W, 40), (0, 0, 0), -1)
        for i, (k, lbl) in enumerate([("n", "yolo26n 3.7M"), ("s", "yolo26s 11.8M"),
                                      ("m", "yolo26m 24.2M")]):
            cv2.circle(img, (20 + i * 210, 20), 7, COLOR[k], -1)
            cv2.putText(img, lbl, (34 + i * 210, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR[k], 1, cv2.LINE_AA)
        cv2.putText(img, f"t={fr/FPS:5.2f}s", (W - 150, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (230, 230, 230), 1, cv2.LINE_AA)
        vw.write(img)
        fr += 1
        if fr % 120 == 0:
            print(f"  {fr} frames", flush=True)

    cap.release(); vw.release()
    if spreads:
        s = np.array(spreads)
        print(f"\n  wrist disagreement between the 3 nets (px):", flush=True)
        print(f"    median {np.median(s):5.1f}   p90 {np.percentile(s,90):5.1f}   "
              f"max {s.max():5.1f}", flush=True)
        print(f"    frames where they disagree >15px: {(s>15).mean():.0%}", flush=True)
    print(f"  wrote {a.out} ({fr} frames)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
