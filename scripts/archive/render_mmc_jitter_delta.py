"""Render the MMC pose on one DELTA camera: RAW vs HYBRID, so the jitter fix is directly visible.

Draws the 3D-triangulated right arm (shoulder-elbow-wrist) reprojected into one camera, twice:
  * raw    (red)   -- the plain triangulation; the wrist trail scribbles = the wander/jitter
  * HYBRID (green) -- PBD on the torso quad + chain-lock on the arm (same code path as
                      score_omc_delta.py --hybrid, so the picture matches the numbers)
plus a recent wrist trail for each, so frame-to-frame jitter shows as a scribble (raw) vs a clean
arc (hybrid).

Everything reprojected is the SAME triangulated 3D the measures read (dumb-player rule).

    python scripts/render_mmc_jitter_delta.py --cam 2 -o out/mmc_hybrid_P14_cam2.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import triangulate
from cup_task.kalman_3d import project
from scripts.score_omc_delta import _bonelock_pbd
from scripts.compare_pose_omc_delta import _load_calib_mm, _despike, VIDEO_FPS, DELTA

PART, TRIAL = "P14", "trial_1_R_unaffected"
ARM = ["right_shoulder", "right_elbow", "right_wrist"]
# the hybrid's PBD half needs the torso quad, so triangulate those too
ALL_J = ARM + ["left_shoulder", "right_hip", "left_hip"]
TORSO_BONES = [("right_shoulder", "left_shoulder"), ("right_hip", "left_hip"),
               ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip")]


def _hybrid(J, iters=8):
    """PBD on the torso quad + chain-lock on the arm -- the same code path as
    score_omc_delta.py --hybrid, so the picture matches the numbers."""
    P = _bonelock_pbd({k: v.copy() for k, v in J.items()}, bones=TORSO_BONES, iters=iters)
    sh, el, wr = P["right_shoulder"], P["right_elbow"], P["right_wrist"]
    Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
    Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
    du = el - sh; du = du / (np.linalg.norm(du, axis=1, keepdims=True) + 1e-9)
    el2 = sh + du * Lu
    df = wr - el2; df = df / (np.linalg.norm(df, axis=1, keepdims=True) + 1e-9)
    P["right_elbow"], P["right_wrist"] = el2, el2 + df * Lf
    return P


def _kp_point(name):
    def fn(fr):
        k = fr.get("kps", {})
        return np.array(k[name][:2], float) if name in k else None
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--trail", type=int, default=12, help="wrist trail length (frames)")
    a = ap.parse_args()
    out = a.out or (Path(__file__).resolve().parents[1] / "out" / f"mmc_jitter_{PART}_cam{a.cam}.mp4")

    cams = _load_calib_mm(PART)
    d = DELTA / PART
    per_cam = {}
    for pj in sorted(d.glob("dets/*.pose.json")):
        c = pj.name.split(".")[1]
        per_cam[f"cam_{c}"] = json.loads(pj.read_text())["frames"]
    camkey = f"cam_{a.cam}"
    cam = cams[camkey]
    T = max(len(v) for v in per_cam.values())

    # triangulate every joint the hybrid needs (arm + torso quad), then build the hybrid
    raw = {}
    for j in ALL_J:
        tr = triangulate.triangulate_target(per_cam, cams, _kp_point(j), T)
        raw[j] = _despike(np.array([t["X"] if t.get("X") else [np.nan] * 3 for t in tr]))
    hyb = _hybrid(raw)

    clip = d / "staged" / f"{('delta_' + PART + '_' + TRIAL)}.{a.cam}.mp4"
    cap = cv2.VideoCapture(str(clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (W, H))

    variants = [("raw", raw, (60, 60, 255)), ("HYBRID", hyb, (80, 255, 80))]
    trails = {name: deque(maxlen=a.trail) for name, _, _ in variants}

    def proj(X):
        if not np.isfinite(X).all():
            return None
        p, ok = project(cam, X)
        return (int(p[0]), int(p[1])) if ok else None

    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= T:
            break
        # reprojected 3D arm, three variants
        for name, tr, col in variants:
            pts = [proj(tr[j][f]) for j in ARM]
            for k in range(len(pts) - 1):
                if pts[k] and pts[k + 1]:
                    cv2.line(img, pts[k], pts[k + 1], col, 2)
            if pts[-1]:
                cv2.circle(img, pts[-1], 6, col, 2)
                trails[name].append(pts[-1])
            # wrist trail -- jitter shows as a scribble
            tr_pts = [p for p in trails[name] if p]
            for k in range(len(tr_pts) - 1):
                cv2.line(img, tr_pts[k], tr_pts[k + 1], col, 1)

        # legend
        cv2.rectangle(img, (0, 0), (330, 84), (0, 0, 0), -1)
        cv2.putText(img, f"cam{a.cam}  frame {f}  (reprojected 3D)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, (name, _, col) in enumerate(variants):
            cv2.putText(img, f"3D {name}", (8, 42 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        vw.write(img)
        f += 1
    cap.release(); vw.release()
    print(f"wrote {out}  ({f} frames)", flush=True)


if __name__ == "__main__":
    main()
