"""Render one trial's cameras side by side with the DLT 30px gate drawn on every joint.

Per camera, per frame, for each of the three SCORED-ARM joints:
  * the DETECTED 2D keypoint (circle)
  * the DLT 3D point REPROJECTED into that camera (cross)
  * a line between them = the reprojection residual the gate thresholds
  GREEN = camera is an inlier (<=30px)   RED = outlier
A per-frame banner shows how many cameras are inliers for the arm, and flags RELAXED
(the <3 case, where robust_triangulate abandons filtering and uses every camera).

The gate itself comes from dlt_gate_audit.gate_state -- the SAME function that produced
the audit numbers, so the picture cannot disagree with the table.

    python scripts/render_gate_trial.py --part P12 --trial trial_43_R_affected
    -> out/gate_videos/<part>_<trial>_gate.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                                     # noqa: E402
import compare_pose_omc_delta as H                        # noqa: E402
import gnn_refiner as G                                   # noqa: E402
from dlt_gate_audit import gate_state, REPROJ_PX, MIN_CAMS  # noqa: E402

TILE_W = 480
GREEN, RED, WHITE, YELL = (80, 230, 80), (60, 60, 235), (255, 255, 255), (40, 220, 235)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    trials = T.load_clean(need_reproj=True)
    t = next((x for x in trials if x["part"] == a.part and x["trial"] == a.trial), None)
    if t is None:
        sys.exit(f"trial not found: {a.part}/{a.trial}")

    inl, resid, fin = gate_state(t)                        # (T,C,J), (T,C,J), (T,J)
    Tn, C, _ = inl.shape
    side = t["side"]
    ARM = [i for i, n in enumerate(G.JOINTS)
           if n.startswith(side) and any(k in n for k in ("wrist", "elbow", "shoulder"))]
    names = [G.JOINTS[i].replace(side + "_", "") for i in ARM]

    # reproject the DLT point once per camera (same projection the gate used)
    X = torch.from_numpy(np.nan_to_num(t["mmc"]).astype(np.float32))[None]
    proj = []
    for c in range(C):
        uvp, _ = G.project_torch(X, *[torch.from_numpy(t[k][c].astype(np.float32))
                                      for k in ("K", "dist", "R", "t")])
        proj.append(uvp[0].numpy())                        # (T,J,2)

    # Column c of uv/K/R/t is NOT camera c+1. gnn_build_dataset.py:145 orders columns as
    #   sorted(whitelisted cams, key=numeric)
    # so for a participant whose whitelist skips a camera (P13/P19 = cam_1,3,4; P17 = cam_2,3,4)
    # the mapping shifts. Reading .{c+1}.mp4 overlays one camera's detections on another's video,
    # which looks like a timing error. Resolve the real camera number per column.
    H.use_good_cams()
    cam_names = sorted(H.GOOD_CAMS[a.part], key=lambda s: int(s.split("_")[1]))
    if len(cam_names) != C:
        print(f"WARNING: {len(cam_names)} whitelisted cams but uv has {C} columns", flush=True)
    print(f"column -> camera: " + ", ".join(f"[{i}]->{n}" for i, n in enumerate(cam_names)), flush=True)

    staged = ROOT / "cache" / "delta" / a.part / "staged"
    caps, cams = [], []
    for c in range(min(C, len(cam_names))):
        num = int(cam_names[c].split("_")[1])
        f = staged / f"delta_{a.part}_{a.trial}.{num}.mp4"
        if f.exists():
            caps.append(cv2.VideoCapture(str(f))); cams.append(c)
        else:
            print(f"  missing staged clip for {cam_names[c]}: {f.name}", flush=True)
    if not caps:
        sys.exit(f"no staged video under {staged}")
    print(f"{a.part}/{a.trial}  arm={side}  {len(caps)} cams, {Tn} frames", flush=True)

    outdir = ROOT / "out" / "gate_videos"; outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{a.part}_{a.trial}_gate.mp4"
    vw, n_relaxed = None, 0

    for f in range(Tn):
        tiles = []
        for k, c in enumerate(cams):
            ok, img = caps[k].read()
            if not ok:
                img = np.zeros((360, 640, 3), np.uint8)
            s = TILE_W / img.shape[1]
            img = cv2.resize(img, (TILE_W, int(img.shape[0] * s)))
            for j, jn in zip(ARM, names):
                det = t["uv"][f, c, j] * s
                rep = proj[c][f, j] * s
                if not (np.isfinite(det).all() and np.isfinite(rep).all()):
                    continue
                good = bool(inl[f, c, j])
                col = GREEN if good else RED
                # A miscalibrated / diverged DLT point can reproject arbitrarily far off-image
                # (P19 overflows int64 on cast). Clamp to a margin around the tile so the line
                # still shows the DIRECTION of the error instead of crashing the draw.
                lim = np.array(img.shape[1::-1], float)
                det = np.clip(det, -lim, 2 * lim)
                offscreen = bool((rep < -lim).any() or (rep > 2 * lim).any())
                rep = np.clip(rep, -lim, 2 * lim)
                cv2.line(img, tuple(det.astype(int)), tuple(rep.astype(int)), col, 2)
                if offscreen:
                    cv2.putText(img, "OFF-IMAGE", (int(np.clip(det[0], 0, lim[0]-90)),
                                int(np.clip(det[1], 14, lim[1]-4)) + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)
                cv2.circle(img, tuple(det.astype(int)), 5, col, 2)          # detection
                cv2.drawMarker(img, tuple(rep.astype(int)), col, cv2.MARKER_CROSS, 11, 2)
                r = resid[f, c, j]
                if np.isfinite(r):
                    cv2.putText(img, f"{jn[:3]} {r:4.0f}px", (int(det[0]) + 8, int(det[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            nin = int(inl[f, c, ARM].sum())
            cv2.putText(img, f"{cam_names[c]}  {nin}/3 arm joints inlier", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            tiles.append(img)

        h = max(x.shape[0] for x in tiles)
        tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
        grid = np.hstack(tiles)
        bar = np.zeros((46, grid.shape[1], 3), np.uint8)
        per_j = [int(inl[f, :, j].sum()) for j in ARM]
        relaxed = [p < MIN_CAMS and fin[f, j] for p, j in zip(per_j, ARM)]
        n_relaxed += sum(relaxed)
        txt = "  ".join(f"{n}:{p}cam{'  RELAXED' if r else ''}"
                        for n, p, r in zip(names, per_j, relaxed))
        cv2.putText(bar, f"f{f:4d}  {txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    YELL if any(relaxed) else WHITE, 2, cv2.LINE_AA)
        frame = np.vstack([bar, grid])

        if vw is None:
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                 (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        if (f + 1) % 150 == 0:
            print(f"   [{f+1}/{Tn}]", flush=True)

    for c in caps:
        c.release()
    if vw:
        vw.release()
    print(f"PROCESSING CHECK: {Tn} frames, relaxed arm joint-frames {n_relaxed}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
