"""5-camera grid montage for one DELTA trial, with pose overlay, to SEE why a camera detects nothing.

Draws each cam's staged clip tiled 2x3, overlays the detected pose keypoints (wrist highlighted) from
that cam's dets/*.pose.json. A camera with an EMPTY dets file (detector found no person) is labelled
NO DETECTION so you can eyeball the raw footage (out of frame? wrong recut? blank?).

    python scripts/montage_5cam.py --part P12 --trial trial_1_L_unaffected
Output: out/viz/<part>_<trial>_5cam.mp4
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

EDGES = [("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
         ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),("left_shoulder","left_hip"),
         ("right_shoulder","right_hip"),("left_hip","right_hip"),("nose","left_shoulder"),
         ("nose","right_shoulder")]


def draw_pose(img, kps, side_wrist):
    for a, b in EDGES:
        if a in kps and b in kps:
            pa, pb = kps[a], kps[b]
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 2)
    for j, p in kps.items():
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 200, 255), -1)
    if side_wrist in kps:
        p = kps[side_wrist]
        cv2.circle(img, (int(p[0]), int(p[1])), 12, (0, 0, 255), 3)
        cv2.putText(img, "WRIST", (int(p[0]) + 14, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--tile-w", type=int, default=480)
    a = ap.parse_args(argv)
    side_wrist = "left_wrist" if "_L_" in a.trial else "right_wrist"
    base = ROOT / "cache" / "delta" / a.part
    tw, th = a.tile_w, int(a.tile_w * 9 / 16)

    caps, dets = {}, {}
    for c in "12345":
        clip = base / "staged" / f"delta_{a.part}_{a.trial}.{c}.mp4"
        det = base / "dets" / f"delta_{a.part}_{a.trial}.{c}.pose.json"
        caps[c] = cv2.VideoCapture(str(clip)) if clip.exists() else None
        frames = json.loads(det.read_text())["frames"] if det.exists() else []
        dets[c] = frames
        n = len(frames)
        print(f"cam{c}: clip={'OK' if caps[c] else 'MISSING'}  dets={n} frames "
              f"({'EMPTY - detector found no person' if n == 0 else 'ok'})", flush=True)

    nframes = max((int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in "12345" if caps[c]), default=0)
    out = ROOT / "out" / "viz" / f"{a.part}_{a.trial}_5cam.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid_w, grid_h = tw * 3, th * 2
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (grid_w, grid_h))
    layout = {"1": (0, 0), "2": (0, 1), "3": (0, 2), "4": (1, 0), "5": (1, 1)}

    for f in range(nframes):
        canvas = np.zeros((grid_h, grid_w, 3), np.uint8)
        for c in "12345":
            cap = caps[c]
            if cap is None:
                continue
            ok, img = cap.read()
            if not ok:
                continue
            frames = dets[c]
            wrist_pct = (sum(1 for fr in frames if side_wrist in fr.get("kps", {})) /
                         max(len(frames), 1)) if frames else 0.0
            if frames and f < len(frames):
                kps = frames[f].get("kps", {})
                if kps:
                    draw_pose(img, kps, side_wrist)
            img = cv2.resize(img, (tw, th))
            label = f"cam{c}  wrist {wrist_pct:.0%}" if frames else f"cam{c}  NO DETECTION"
            col = (0, 255, 0) if frames else (0, 0, 255)
            cv2.rectangle(img, (0, 0), (tw - 1, th - 1), col, 3)
            cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            r, cc = layout[c]
            canvas[r * th:(r + 1) * th, cc * tw:(cc + 1) * tw] = img
        cv2.putText(canvas, f"{a.part} {a.trial}  frame {f}/{nframes}", (tw * 2 + 8, grid_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(canvas)
        if f % 100 == 0:
            print(f"  frame {f}/{nframes}", flush=True)
    vw.release()
    for c in "12345":
        if caps[c]:
            caps[c].release()
    # re-encode to h264 for broad playability
    import subprocess
    h264 = out.with_name(out.stem + "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264",
                    "-crf", "23", "-pix_fmt", "yuv420p", str(h264)], check=False)
    print(f"\nDONE -> {out}\n       -> {h264} (h264)", flush=True)


if __name__ == "__main__":
    main()
