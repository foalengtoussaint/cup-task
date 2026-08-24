"""Render a BAD-cup trial: the cached MMC cup 3D and the wrist, reprojected onto the staged clips.

Dumb player -- it draws exactly the arrays the segmenter reads (cache/seg_inputs), so what you see is
what the numbers are. RED = MMC cup 3D (the one that breaks), GREEN = OMC cup (mocap truth, lag-
shifted onto the video timebase), BLUE = MMC wrist. A red X marks frames where the MMC cup is NaN.
Header prints the frame, the two distances and the sequential segmenter's current phase.

    python render_cup_vs_wrist.py --part P13 --trial trial_34_L_unaffected
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cache_seg_inputs as CSI
import compare_pose_omc_delta as C
C.use_good_cams()          # SAME camera set the 3D pipeline uses (P13 -> cam_1, cam_3, cam_4)
import results_v3_delta as R
from pipeline.kalman_3d import project
from pipeline.segment import _butter_lp, _interp_nan_xyz, FPS
from seg_sequential import segment_sequential

OUT = Path(__file__).resolve().parents[3] / "out" / "renders"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--max-cam", type=int, default=5)
    a = ap.parse_args()
    rec = {(r["part"], r["trial"]): r for r in CSI.load_all()}[(a.part, a.trial)]
    cams = R._calib(a.part)                       # good-cams filtered, as the pipeline does
    trk2d = json.loads((R.TRACKS / f"{a.part}__{a.trial}__uetrack__fs1.json").read_text())
    staged = C.DELTA / a.part / "staged"
    vids = {c: staged / f"delta_{a.part}_{a.trial}.{c}.mp4" for c in range(1, a.max_cam + 1)}
    vids = {c: v for c, v in vids.items() if v.exists() and f"cam_{c}" in cams}
    if not vids:
        raise SystemExit("no staged clips")

    cup_m, cup_o, wr_m = rec["cup_mmc"], rec["cup_omc"], rec["wrist_mmc"]
    nan_m = ~np.isfinite(cup_m).all(1)
    f_ = lambda x: _butter_lp(_interp_nan_xyz(np.asarray(x, float)), FPS)
    d_cm = np.linalg.norm(f_(cup_m) - f_(rec["nose_mmc"]), axis=1)
    d_wm = np.linalg.norm(f_(wr_m) - f_(rec["nose_mmc"]), axis=1)
    seg = segment_sequential(cup_m, wr_m, rec["nose_mmc"])
    phase_of = np.full(len(cup_m), "-", dtype=object)
    for nm, s, e in seg:
        phase_of[s:min(e, len(phase_of))] = nm

    caps = {c: cv2.VideoCapture(str(v)) for c, v in sorted(vids.items())}
    n = min(len(cup_m), int(min(cp.get(cv2.CAP_PROP_FRAME_COUNT) for cp in caps.values())))
    W, H = 480, 270
    cols = min(len(caps), 3); rows = (len(caps) + cols - 1) // cols
    out = OUT / f"{a.part}_{a.trial}_cupdiag2d.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W * cols, H * rows + 34))
    print(f"{len(caps)} cams, {n} frames -> {out}", flush=True)

    for f in range(n):
        tiles = []
        for c, cap in sorted(caps.items()):
            ok, img = cap.read()
            if not ok:
                img = np.zeros((H, W, 3), np.uint8)
            img = cv2.resize(img, (W, H))
            sx, sy = W / 1920.0, H / 1080.0
            for xyz, col, lab in ((cup_m[f], (0, 0, 255), "cupMMC"), (cup_o[f], (0, 220, 0), "cupOMC"),
                                  (wr_m[f], (255, 160, 0), "wrist")):
                if np.isfinite(xyz).all():
                    uv, in_front = project(cams[f"cam_{c}"], np.asarray(xyz, float))
                    if in_front:
                        x, y = int(uv[0] * sx), int(uv[1] * sy)
                        if -50 < x < W + 50 and -50 < y < H + 50:
                            cv2.circle(img, (x, y), 6, col, 2)
                            cv2.putText(img, lab, (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        col, 1, cv2.LINE_AA)
            # UETrack's own 2D point in THIS view (yellow) -- what the tracker actually has,
            # before any 3D consensus. Distinguishes "tracker lost it" from "cameras disagree".
            e2 = trk2d.get(str(f), {}).get(f"cam_{c}") or {}
            if e2.get("trk"):
                x2, y2 = int(e2["trk"][0] * sx), int(e2["trk"][1] * sy)
                cv2.drawMarker(img, (x2, y2), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(img, "trk2D", (x2 + 8, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 255, 255), 1, cv2.LINE_AA)
            if nan_m[f]:
                cv2.putText(img, "cup NaN", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"cam_{c} (pipeline)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(img)
        while len(tiles) < cols * rows:
            tiles.append(np.zeros((H, W, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        bar = np.zeros((34, grid.shape[1], 3), np.uint8)
        cv2.putText(bar, f"f{f:4d}  cup->mouth {d_cm[f]:6.0f}mm   wrist->mouth {d_wm[f]:6.0f}mm   "
                         f"phase={phase_of[f]}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.vstack([bar, grid]))
        if (f + 1) % 100 == 0:
            print(f"  {f+1}/{n}", flush=True)
    vw.release()
    for cp in caps.values():
        cp.release()
    print(f"wrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
