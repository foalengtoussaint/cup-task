"""Render the FULL 3D skeleton + cup, reprojected into one camera.

Everything drawn here is a TRIANGULATED 3D point (mm, world frame) projected back through
this camera's calibration -- not that camera's own 2D detection. So the skeleton you see is
the multi-camera fusion's answer, and a joint that sits wrong is a 3D/calibration error, not
a detector error. (Dumb-player rule: the renderer draws the arrays the numbers came from and
re-derives nothing.)

Every joint is triangulated with the same jitter-weighted DLT the pipeline uses, so the
picture is exactly what the measures see.

    python scripts/render_full_skeleton.py --clipdir DIR --stem STEM --calib C.toml --cam 4 -o out.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import pose_keypoints, segment, triangulate
from pipeline.kalman_3d import load_calibration, project

FPS = 60.0

# the joints we keep (pose_keypoints.TASK_KP) and the bones between them
BONES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]
JOINT_COLOR = {
    "nose": (235, 60, 235), "left_eye": (235, 60, 235), "right_eye": (235, 60, 235),
    "left_ear": (200, 80, 200), "right_ear": (200, 80, 200),
    "left_shoulder": (255, 200, 80), "right_shoulder": (255, 200, 80),
    "left_elbow": (120, 220, 255), "right_elbow": (120, 220, 255),
    "left_wrist": (60, 235, 60), "right_wrist": (60, 235, 60),
    "left_hip": (200, 200, 200), "right_hip": (200, 200, 200),
}
PHASE_COLOR = {
    "rest_pre": (150, 150, 150), "reaching": (100, 220, 100),
    "forward_transport": (235, 180, 60), "drinking": (60, 60, 235),
    "back_transport": (235, 120, 200), "returning": (100, 200, 220),
    "rest_post": (150, 150, 150),
}


def _kp_fn(name):
    def f(fr):
        k = fr.get("kps", {}).get(name)
        return np.array(k[:2], float) if k else None
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repdir", type=Path, required=True,
                    help="pipeline output dir (has the per-cam *.pose.json / *.cup.json)")
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(argv)

    n_cam = int(re.search(r"\.(\d+)\.mp4$", a.clip.name).group(1))
    ck = f"cam_{n_cam}"
    cams = load_calibration(a.calib, target_size=(1920, 1080))
    cam = cams[ck]

    pose_per = triangulate._load_per_cam(a.repdir, "pose")
    cup_per = triangulate._load_per_cam(a.repdir, "cup")
    n = max(len(v) for v in pose_per.values())
    print(f"{ck}: triangulating {len(pose_keypoints.TASK_KP)} joints + cup over {n} frames",
          flush=True)

    # every joint, same jitter-weighted DLT the pipeline uses
    J = {}
    for name in pose_keypoints.TASK_KP:
        tr = triangulate.triangulate_target(pose_per, cams, _kp_fn(name), n)
        J[name] = np.array([t["X"] if t["X"] else [np.nan] * 3 for t in tr], float)
        got = np.isfinite(J[name]).all(1).mean()
        print(f"    {name:16s} {got:5.0%}", flush=True)

    cup_tr = triangulate.triangulate_target(cup_per, cams, triangulate._cup_point, n)
    cup, cup_conf = segment.track_confidence(cup_tr)

    # phases, from the same segmenter the measures use
    mouth, _ = segment.track_confidence(
        triangulate.triangulate_target(pose_per, cams, triangulate._mouth_point, n))
    rng = lambda w: (lambda x: float(np.linalg.norm(np.nanmax(x, 0) - np.nanmin(x, 0))))(
        J[w][np.isfinite(J[w]).all(1)])
    side = "right" if rng("right_wrist") >= rng("left_wrist") else "left"
    hand = J[f"{side}_wrist"]
    seg = segment.segment_cup_only(cup)
    seg = segment.refine_grasp_with_pose(seg, cup, hand, mouth)
    phases = segment.to_murphy_phases(seg, hand, cup)
    phase_of = {f: nm for nm, s, e in phases for f in range(s, e)}
    print(f"  phases: " + "  ".join(f"{nm} {s/FPS:.2f}-{e/FPS:.2f}" for nm, s, e in phases),
          flush=True)

    cap = cv2.VideoCapture(str(a.clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    def px(X):
        if X is None or not np.isfinite(X).all():
            return None
        (u, v), infront = project(cam, X)
        return (int(round(u)), int(round(v))) if infront else None

    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ph = phase_of.get(fr, "rest_post")

        pts = {nm: px(J[nm][fr]) for nm in J if fr < len(J[nm])}
        for a_, b_ in BONES:                       # bones first, joints on top
            if pts.get(a_) and pts.get(b_):
                cv2.line(img, pts[a_], pts[b_], (210, 210, 210), 2, cv2.LINE_AA)
        for nm, p in pts.items():
            if p:
                cv2.circle(img, p, 6, JOINT_COLOR[nm], -1, cv2.LINE_AA)
                cv2.circle(img, p, 6, (30, 30, 30), 1, cv2.LINE_AA)

        c = px(cup[fr]) if fr < len(cup) else None
        if c:
            filled = fr < len(cup_conf) and cup_conf[fr] == 0
            cv2.circle(img, c, 12, (60, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(img, c, 3, (60, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "cup" + (" (filled)" if filled else ""), (c[0] + 16, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 255), 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, 0), (W, 44), (0, 0, 0), -1)
        cv2.putText(img, ph.upper(), (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    PHASE_COLOR[ph], 2, cv2.LINE_AA)
        cv2.putText(img, f"{ck}   t={fr/FPS:5.2f}s   3D skeleton + cup (reprojected)",
                    (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 225, 225), 1,
                    cv2.LINE_AA)

        y = H - 24
        cv2.rectangle(img, (0, y - 10), (W, H), (0, 0, 0), -1)
        for nm, s, e in phases:
            cv2.rectangle(img, (int(s / n * W), y - 5), (int(e / n * W), y + 7),
                          PHASE_COLOR[nm], -1)
        cv2.line(img, (int(fr / n * W), y - 10), (int(fr / n * W), y + 12),
                 (255, 255, 255), 2)
        vw.write(img)
        fr += 1

    cap.release(); vw.release()
    print(f"  wrote {a.out} ({fr} frames)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
