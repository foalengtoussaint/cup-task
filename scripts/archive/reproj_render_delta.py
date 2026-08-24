"""Diagnostic render: detected wrist (green) vs reprojected RANSAC-consensus wrist (red) per cam.

Lets the eye confirm the miscalib-vs-desync classification:
  green == red always                      -> FINE
  green != red EVEN WHEN STILL (const gap)  -> MISCALIBRATION (geometry wrong)
  green == red still, diverges in MOTION    -> DESYNC (camera shows a different instant)
A yellow line connects the two so the gap is visible; the tile border/label carries the verdict.

Consensus is the RANSAC point (largest mutually-agreeing subset), so the red dot is trustworthy
even when many cameras are bad.

    python scripts/reproj_render_delta.py --part P14 --trial trial_10_R_unaffected
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from pipeline.kalman_3d import project  # noqa: E402
from reaudit_cam_quality import ransac_point  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
               "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad")


def _side(part):
    return "right" if glob.glob(str(C.DELTA / part / "dets" / "*_R_*.pose.json")) else "left"


def _pick_trial(part):
    s = "R" if _side(part) == "right" else "L"
    ts = sorted({Path(f).name.split(".")[0].replace(f"delta_{part}_", "")
                 for f in glob.glob(str(C.DELTA / part / "dets" / f"*_{s}_*.pose.json"))})
    return ts[0] if ts else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--trial", default=None)
    ap.add_argument("--verdicts", default=None, help="path to reaudit_cam_quality.json for labels")
    ap.add_argument("--max-cam", type=int, default=5,
                    help="only render cams 1..MAX (default 5 -- we ignore cams 6-10)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    part = a.part
    side = _side(part)
    trial = a.trial or _pick_trial(part)
    cams = C._load_calib_mm(part)
    fn = C._kp_point(f"{side}_wrist")

    dets = {}
    for pj in sorted(glob.glob(str(C.DELTA / part / "dets" / f"delta_{part}_{trial}.*.pose.json"))):
        c = int(Path(pj).name.split(".")[1])
        if c > a.max_cam:
            continue  # ignore cams 6-10; only the common 1-5 set matters
        dets[c] = json.loads(Path(pj).read_text())["frames"]
    staged = C.DELTA / part / "staged"
    vids = {c: staged / f"delta_{part}_{trial}.{c}.mp4" for c in dets}
    vids = {c: v for c, v in vids.items() if v.exists()}
    if not vids:
        raise SystemExit(f"no staged clips for {part} {trial}")

    verd = {}
    vpath = a.verdicts or (C.DELTA / "reaudit_cam_quality.json")
    if Path(vpath).exists():
        vj = json.loads(Path(vpath).read_text()).get(part, {})
        for c in vj.get("fine", []):
            verd[int(c.split("_")[1])] = ("FINE", (0, 200, 0))
        for c in vj.get("recut", []):
            verd[int(c.split("_")[1])] = ("DESYNC", (0, 180, 255))
        for c in vj.get("recalib", []):
            verd[int(c.split("_")[1])] = ("MISCALIB", (0, 0, 255))

    caps = {c: cv2.VideoCapture(str(vids[c])) for c in sorted(vids)}
    n = min(len(dets[c]) for c in caps)
    ncam = len(caps)
    cols = 5 if ncam > 5 else ncam
    rows = (ncam + cols - 1) // cols
    W, H = 480, 270
    out = Path(a.out or SCRATCH / f"{part}_{trial}_reproj.mp4")
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W * cols, H * rows))

    for f in range(n):
        # RANSAC consensus 3D wrist this frame
        pts_d = {f"cam_{c}": fn(dets[c][f]) for c in dets if fn(dets[c][f]) is not None}
        camsub = {k: cams[k] for k in pts_d if k in cams}
        X, inl = ransac_point(camsub, {k: pts_d[k] for k in camsub}) if len(camsub) >= 2 else (None, set())
        tiles = []
        for c in sorted(caps):
            ok, img = caps[c].read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            sx, sy = W / 1920.0, H / 1080.0
            img = cv2.resize(img, (W, H))
            lbl, col = verd.get(c, ("?", (200, 200, 200)))
            det = fn(dets[c][f])
            gp = None
            if det is not None:
                gp = (int(det[0] * sx), int(det[1] * sy))
            rp = None
            if X is not None and f"cam_{c}" in cams:
                pr = project(cams[f"cam_{c}"], X)[0]
                rp = (int(pr[0] * sx), int(pr[1] * sy))
            if gp and rp:
                cv2.line(img, gp, rp, (0, 255, 255), 1)
            if rp:
                cv2.circle(img, rp, 6, (0, 0, 255), 2)      # reprojected consensus = RED
            if gp:
                cv2.circle(img, gp, 5, (0, 220, 0), -1)     # detected = GREEN
            gap = (int(np.hypot(gp[0] - rp[0], gp[1] - rp[1]) / sx) if (gp and rp) else -1)
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), col, 3)
            cv2.rectangle(img, (0, 0), (W, 20), (0, 0, 0), -1)
            txt = f"cam{c} {lbl}" + (f" gap{gap}px" if gap >= 0 else "")
            cv2.putText(img, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) < cols * rows:
            tiles.append(np.zeros((H, W, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.putText(grid, f"{part} {trial}  f{f}  GREEN=detected RED=consensus-reproj",
                    (8, H * rows - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(grid)
    vw.release()
    for cap in caps.values():
        cap.release()
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
