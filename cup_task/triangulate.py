"""Fuse per-camera 2D detections of a rep into 3D world points (mm).

For each frame, gather every camera that saw the target (cup center, or a body
keypoint), then triangulate a single 3D point. Detections are NOT trusted
blindly: one camera locking onto the wrong object (the classic failure in this
rig — see drink_study notes on cam_4 / cam_10) would drag a naive all-camera DLT
metres off. So we triangulate the largest geometrically-agreeing subset:

  1. DLT over all available cams -> candidate X
  2. reproject X, drop cams whose reprojection error > REPROJ_PX
  3. re-triangulate on the survivors; require >= MIN_CAMS

This is the same consensus principle the research pipeline uses, distilled to a
single-frame robust triangulation (no temporal KF yet — that's the next stage).

Inputs are the per-camera JSONs from cup_detect.py / pose_keypoints.py. The
clip-suffix `.N.mp4` maps to calib key `cam_N`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task.kalman_3d import CamCalib, load_calibration, project, triangulate_dlt

MIN_CAMS = 3          # >=3 agreeing cams — the hard floor from the robustness study
REPROJ_PX = 30.0      # a cam is an inlier if it reprojects within this many px

# The gate above is load-bearing for the CUP and REDUNDANT for POSE. Measured (11 reps,
# wrists vs MeTRAbs 3D): it fires on 24% of frames and moves the median 16.8 -> 16.6mm,
# never killing a frame. A cup FP is a DIFFERENT OBJECT (the side-desk glass) — it sits
# elsewhere in the world, reprojects far off, and the gate catches it. A pose error is the
# right person's slightly-wrong joint: still roughly in place, so it reprojects plausibly
# and passes the gate anyway, while a 10-cam DLT already averages the jitter out.
# Don't bother tightening this to "improve" pose — and note MIN_CAMS would start DROPPING
# pose frames on a rig with fewer cameras, for no accuracy gain. See docs/WORKLOG.md.


def kf_rts_smooth(cons: np.ndarray, fps: float = 60.0,
                  q: float = 200.0 ** 2, r: float = 30.0 ** 2) -> np.ndarray:
    """Consensus -> constant-velocity KF -> RTS smoother. Fills gaps, kills jitter.

    THIS REPLACES LINEAR INTERPOLATION, and the difference is not cosmetic. The cup is
    OCCLUDED for ~24% of frames -- and not at random: it is occluded precisely at the mouth,
    during the drink, which is the part we are trying to measure. Something has to invent
    those positions. Drawing a straight line through them puts kinks in the trajectory
    exactly where the interesting curvature is.

    Measured in the research pipeline over 355 reps (trajectory-cleanliness):
        consensus + linear interp   : 274/355 clean
        consensus -> KF -> RTS      : 354/355 clean
    The KF wins because a constant-velocity model coasts through a gap the way an arm
    actually moves, and the RTS backward pass then uses the frames AFTER the gap to correct
    the coast -- so the filled segment is consistent with both sides, not just extrapolated.

    It also moves the DWELL BOUNDARIES the right way, checked frame-by-frame on the raw
    video (P07): drink onset KF 2.90s vs linear 3.08s (at 2.90s the cup is already at her
    lips); drink offset KF 3.78s vs linear 4.20s (at 3.78s it has tilted off them). Linear
    is late at both ends because a straight line between the last sighting before the
    occlusion and the first after CUTS THE CORNER of the cup's arc to the mouth -- the
    chord sits farther from the face than the true path, so "near the mouth" fires late
    and clears early.

    NO GATING. An earlier pipeline fed the KF raw 2D detections behind a Mahalanobis gate;
    one bad detection kicked the state, and then the gate rejected every TRUE detection as
    "too far" and it coasted away to metres. Feeding the CONSENSUS in as a direct 3D
    measurement with no gate makes divergence impossible: every consensus frame pulls the
    state back. It was the gating that was harmful, not the filter.

    cons: (T,3), NaN where no consensus. Returns (T,3) smoothed, gaps filled.
    """
    cons = np.asarray(cons, float)
    T = len(cons)
    dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = q * dt ** 3 / 3 * np.eye(3); Q[:3, 3:] = q * dt ** 2 / 2 * np.eye(3)
    Q[3:, :3] = q * dt ** 2 / 2 * np.eye(3); Q[3:, 3:] = q * dt * np.eye(3)
    R = r * np.eye(3)

    valid = np.isfinite(cons).all(1)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((T, 3), np.nan)

    x = np.zeros(6); x[:3] = cons[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0]) ** 2
    xp, Pp, xu, Pu = [], [], [], []
    for t in range(T):
        x = F @ x; P = F @ P @ F.T + Q                     # predict (= the gap fill)
        xp.append(x.copy()); Pp.append(P.copy())
        if valid[t]:                                        # update, ungated
            y = cons[t] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        xu.append(x.copy()); Pu.append(P.copy())

    xs = [None] * T; xs[-1] = xu[-1]                        # RTS backward pass
    for t in range(T - 2, -1, -1):
        C = Pu[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xu[t] + C @ (xs[t + 1] - xp[t + 1])
    return np.array([s[:3] for s in xs])


def _cam_key_from_clip(clip_name: str) -> str | None:
    """`P07_..._142730.4.mp4` -> 'cam_4'. Returns None if no .N. suffix."""
    m = re.search(r"\.(\d+)\.mp4$", clip_name)
    return f"cam_{int(m.group(1))}" if m else None


def robust_triangulate(cams: list[CamCalib], pts: list[np.ndarray]
                       ) -> tuple[np.ndarray | None, list[int], float]:
    """Consensus triangulation. Returns (X_mm | None, inlier_indices, median_reproj_px).

    None if fewer than MIN_CAMS survive the reprojection gate.
    """
    if len(cams) < 2:
        return None, [], float("inf")
    X = triangulate_dlt(cams, pts)                      # 1: all-cam candidate
    errs = np.array([np.linalg.norm(project(c, X)[0] - p)
                     for c, p in zip(cams, pts)])
    keep = [i for i in range(len(cams)) if errs[i] <= REPROJ_PX]  # 2: gate
    if len(keep) < MIN_CAMS:
        return None, [], float("inf")
    if len(keep) < len(cams):                           # 3: refit on inliers
        X = triangulate_dlt([cams[i] for i in keep], [pts[i] for i in keep])
    med = float(np.median([np.linalg.norm(project(cams[i], X)[0] - pts[i])
                           for i in keep]))
    return X, keep, med


def _load_per_cam(json_dir: Path, kind: str) -> dict[str, list]:
    """Map cam_key -> that camera's per-frame list, from *.{kind}.json files."""
    out: dict[str, list] = {}
    for jf in sorted(json_dir.glob(f"*.{kind}.json")):
        payload = json.loads(jf.read_text())
        ckey = _cam_key_from_clip(Path(payload["clip"]).name)
        if ckey:
            out[ckey] = payload["frames"]
    return out


def _cup_point(fr: dict) -> np.ndarray | None:
    c = fr.get("center")
    return np.array(c, dtype=float) if c else None


def _mouth_point(fr: dict) -> np.ndarray | None:
    """Mouth-proxy 2D from a pose frame: nose, else eye/ear centroid."""
    kps = fr.get("kps", {})
    if "nose" in kps:
        return np.array(kps["nose"][:2], dtype=float)
    pts = [kps[k][:2] for k in ("left_eye", "right_eye", "left_ear", "right_ear")
           if k in kps]
    return np.mean(pts, axis=0) if pts else None


def _wrist_point(fr: dict, side: str) -> np.ndarray | None:
    kps = fr.get("kps", {})
    k = f"{side}_wrist"
    return np.array(kps[k][:2], dtype=float) if k in kps else None


def _trunk_point(fr: dict) -> np.ndarray | None:
    """Upper-back proxy = shoulder midpoint. The Murphy trunk-displacement measure wants
    an upper-back site; COCO has no spine, and the shoulder midpoint is the closest rigid
    stand-in that both shoulders vote on (so one bad shoulder can't swing it far)."""
    kps = fr.get("kps", {})
    pts = [kps[k][:2] for k in ("left_shoulder", "right_shoulder") if k in kps]
    return np.mean(pts, axis=0) if len(pts) == 2 else None


POINT_FN = {
    "cup": ("cup", _cup_point),
    "mouth": ("pose", _mouth_point),
    "left_wrist": ("pose", lambda fr: _wrist_point(fr, "left")),
    "right_wrist": ("pose", lambda fr: _wrist_point(fr, "right")),
    "trunk": ("pose", _trunk_point),
}


def triangulate_target(per_cam: dict[str, list], cams: dict[str, CamCalib],
                       point_fn, n_frames: int) -> list[dict]:
    """Per-frame robust 3D for one target. Returns [{frame, X, n_cams, reproj_px}]."""
    track = []
    for f in range(n_frames):
        use_cams, use_pts = [], []
        for ckey, frames in per_cam.items():
            if ckey not in cams or f >= len(frames):
                continue
            p = point_fn(frames[f])
            if p is not None:
                use_cams.append(cams[ckey])
                use_pts.append(p)
        X, keep, med = robust_triangulate(use_cams, use_pts)
        track.append({
            "frame": f,
            "X": None if X is None else [round(float(v), 1) for v in X],
            "n_cams": len(keep),
            "reproj_px": None if X is None else round(med, 2),
        })
    return track


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_dir", type=Path, help="dir of per-cam *.cup.json / *.pose.json")
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--targets", nargs="+",
                    default=["cup", "mouth", "left_wrist", "right_wrist"],
                    choices=list(POINT_FN))
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cams = load_calibration(args.calib, target_size=(1920, 1080))
    print(f"calib: {len(cams)} cams {sorted(cams)}", flush=True)

    result = {"calib": str(args.calib), "targets": {}}
    for tgt in args.targets:
        kind, fn = POINT_FN[tgt]
        per_cam = _load_per_cam(args.json_dir, kind)
        if not per_cam:
            print(f"  {tgt}: no {kind} JSONs found, skipping", flush=True)
            continue
        n = max(len(v) for v in per_cam.values())
        track = triangulate_target(per_cam, cams, fn, n)
        got = sum(1 for t in track if t["X"] is not None)
        med_px = np.median([t["reproj_px"] for t in track if t["reproj_px"] is not None]) \
            if got else float("nan")
        print(f"  {tgt:12} {len(per_cam)} cams, 3D in {got}/{n} frames "
              f"({got/n:.0%}), median reproj {med_px:.1f}px", flush=True)
        result["targets"][tgt] = track

    args.out.write_text(json.dumps(result))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
