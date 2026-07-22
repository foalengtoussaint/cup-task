"""Render the pose overlay per camera, labelled GOOD/BAD, so the camera-quality claim is VISIBLE.

The claim under test: P15's cams 6-10 produce a wrist that does not track the reference camera's
wrist motion, and with 5/10 cameras bad we sit exactly on robust_triangulate's 50% breakdown
point -- so the all-cam DLT seed is corrupted and the iterative ejection throws out the GOOD
cameras. (Measured: P15 t10 coverage 25% with all 10 cams -> 100% with cams 1-5 only, and
reproj IMPROVES 10.8 -> 7.5px.)

Draws the SAME point the metric consumes -- compare_pose_omc_delta._kp_point('right_wrist') --
never a re-derived one. A renderer that recomputes the transform can silently disagree with the
number it is supposed to illustrate.

    python scripts/render_cam_quality_delta.py --part P15 --trial trial_10_R_unaffected
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

SK = [("shoulder", "elbow"), ("elbow", "wrist")]
SCRATCH = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
               "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad")


def w2d(frames, side):
    P = []
    for fr in frames:
        k = fr.get("kps", {})
        P.append(k[side][:2] if side in k else [np.nan, np.nan])
    return np.array(P, float)


def speed(P):
    v = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[np.nan], v])


def best_r(a, b, maxlag=12):
    best = -2.0
    for L in range(-maxlag, maxlag + 1):
        bs = np.full_like(b, np.nan)
        if L > 0:
            bs[L:] = b[:-L]
        elif L < 0:
            bs[:L] = b[-L:]
        else:
            bs = b.copy()
        m = np.isfinite(a) & np.isfinite(bs)
        if m.sum() < 60 or np.std(a[m]) < 1e-9 or np.std(bs[m]) < 1e-9:
            continue
        best = max(best, float(np.corrcoef(a[m], bs[m])[0, 1]))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P15")
    ap.add_argument("--trial", default="trial_10_R_unaffected")
    ap.add_argument("--side", default="right")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default="3")
    a = ap.parse_args(argv)

    dets = {}
    for pj in sorted(glob.glob(str(C.DELTA / a.part / "dets"
                                   / f"delta_{a.part}_{a.trial}.*.pose.json"))):
        dets[Path(pj).name.split(".")[1]] = json.loads(Path(pj).read_text())["frames"]

    ref = C._lp(speed(w2d(dets[a.ref], f"{a.side}_wrist")))
    rs = {}
    for c, fr in dets.items():
        rs[c] = 1.0 if c == a.ref else best_r(ref, C._lp(speed(w2d(fr, f"{a.side}_wrist"))))

    # Read the STAGED videos (what detection actually ran on). The scratch copies mixed
    # resolutions -- cam1 was raw 720p while detection ran on the upscaled 1080p staged clip,
    # so scaling keypoints by the source frame's own size (as this once did) drew cam1's points
    # 1.5x off the participant. Keypoints live in DET_RES; scale by DET_RES, never by img.shape.
    staged = C.DELTA / a.part / "staged"
    vids = {c: staged / f"delta_{a.part}_{a.trial}.{c}.mp4" for c in dets}
    vids = {c: v for c, v in vids.items() if v.exists()}
    if not vids:  # fall back to scratch copies if staged not present
        vids = {c: v for c in dets if (v := SCRATCH / f"{a.part}_t10_cam{c}.mp4").exists()}
    caps = {c: cv2.VideoCapture(str(v)) for c, v in sorted(vids.items(), key=lambda kv: int(kv[0]))}
    n = min(len(dets[c]) for c in caps)
    out = Path(a.out or SCRATCH / f"{a.part}_{a.trial}_camquality.mp4")
    W, H = 480, 270
    cols, rows = 5, 2
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W * cols, H * rows))

    for f in range(n):
        tiles = []
        for c, cap in caps.items():
            ok, img = cap.read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            # keypoints are in the DETECTION resolution (1920x1080 for all DELTA cams, incl.
            # upscaled cam1) -- scale by that, NOT by img.shape (which for a raw 720p source
            # would misplace them).
            DET_W, DET_H = 1920, 1080
            sx, sy = W / DET_W, H / DET_H
            img = cv2.resize(img, (W, H))
            good = rs[c] >= 0.65
            col = (0, 220, 0) if good else (0, 0, 255)
            k = dets[c][f].get("kps", {})
            pts = {}
            for j in ("shoulder", "elbow", "wrist"):
                nm = f"{a.side}_{j}"
                if nm in k:
                    pts[j] = (int(k[nm][0] * sx), int(k[nm][1] * sy))
            for p, q in SK:
                if p in pts and q in pts:
                    cv2.line(img, pts[p], pts[q], col, 2)
            if "wrist" in pts:
                cv2.circle(img, pts["wrist"], 7, col, -1)
                cv2.circle(img, pts["wrist"], 9, (255, 255, 255), 1)
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), col, 3)
            tag = f"cam{c}  r={rs[c]:.2f}  {'GOOD' if good else 'BAD'}"
            cv2.rectangle(img, (0, 0), (W, 22), (0, 0, 0), -1)
            cv2.putText(img, tag, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) < cols * rows:
            tiles.append(np.zeros((H, W, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.putText(grid, f"{a.part} {a.trial}  frame {f}/{n}", (8, H * rows - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(grid)
    vw.release()
    for cap in caps.values():
        cap.release()
    print("r vs cam" + a.ref + ": " + "  ".join(
        f"cam{c}={rs[c]:.2f}" for c in sorted(rs, key=int)), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
