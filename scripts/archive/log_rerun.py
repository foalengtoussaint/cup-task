"""Log the whole rep to a Rerun .rrd: 3D skeleton + cup + camera frusta + video, on one timeline.

The 2D overlays answer "does the marker sit on the cup in THIS view". They cannot show you
the thing that actually matters when a joint looks wrong: WHERE IN 3D the estimate is, and
which cameras were even looking at it. This does.

What you get in the viewer:
  * a 3D view -- the skeleton, the cup, and all 10 camera frusta in world space (mm). Orbit
    it. A joint that is triangulated badly is VISIBLY off the body in 3D, and you can see
    which cameras had a grazing view of it.
  * the video from each camera, time-synced.
  * the cup track as a fading trail, so the arc to the mouth is visible as a shape.
  * scalar plots: cup confidence (0 = the KF invented that frame), and the wrist->cup
    distance whose PLATEAU is what defines grasp and release.
  * the phase as a timeline annotation.

Everything logged is the array the measures were computed from -- no re-derivation.

    python scripts/log_rerun.py --repdir DIR --clipdir CLIPS --stem STEM --calib C.toml \
        -o out.rrd
    rerun out.rrd
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import pose_keypoints, segment, triangulate
from pipeline.kalman_3d import load_calibration

FPS = 60.0

BONES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]
PHASE_RGB = {
    "rest_pre": (150, 150, 150), "reaching": (100, 220, 100),
    "forward_transport": (60, 180, 235), "drinking": (235, 60, 60),
    "back_transport": (200, 120, 235), "returning": (220, 200, 100),
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
    ap.add_argument("--repdir", type=Path, required=True)
    ap.add_argument("--clipdir", type=Path, required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--video-cams", type=int, nargs="*", default=[4, 8],
                    help="which cameras' video to embed (all 10 makes a big file)")
    a = ap.parse_args(argv)

    cams = load_calibration(a.calib, target_size=(1920, 1080))
    pose_per = triangulate._load_per_cam(a.repdir, "pose")
    cup_per = triangulate._load_per_cam(a.repdir, "cup")
    n = max(len(v) for v in pose_per.values())
    print(f"{n} frames, {len(cams)} cams", flush=True)

    # ---- triangulate everything (same jitter-weighted DLT the measures use) ----
    J = {}
    for name in pose_keypoints.TASK_KP:
        tr = triangulate.triangulate_target(pose_per, cams, _kp_fn(name), n)
        J[name] = np.array([t["X"] if t["X"] else [np.nan] * 3 for t in tr], float)
    cup_tr = triangulate.triangulate_target(cup_per, cams, triangulate._cup_point, n)
    cup, cup_conf = segment.track_confidence(cup_tr)
    mouth, _ = segment.track_confidence(
        triangulate.triangulate_target(pose_per, cams, triangulate._mouth_point, n))

    def rng(w):
        x = J[w][np.isfinite(J[w]).all(1)]
        return float(np.linalg.norm(x.max(0) - x.min(0))) if len(x) else 0.0
    side = "right" if rng("right_wrist") >= rng("left_wrist") else "left"
    hand = J[f"{side}_wrist"]

    seg = segment.segment_cup_only(cup)
    seg = segment.refine_grasp_with_pose(seg, cup, hand, mouth)
    phases = segment.to_murphy_phases(seg, hand, cup)
    phase_of = {f: nm for nm, s, e in phases for f in range(s, e)}
    print("  phases: " + "  ".join(f"{p} {s/FPS:.2f}-{e/FPS:.2f}" for p, s, e in phases),
          flush=True)

    d_wc = np.linalg.norm(hand - cup, axis=1)      # the plateau signal (grasp + release)

    rr.init("pipeline", spawn=False)
    rr.save(str(a.out))

    # static: camera frusta + world axes
    for ck, cam in cams.items():
        # calibration gives world->camera (X_cam = R @ X_world + t). Rerun's Transform3D
        # under a `world/...` path wants camera->world, so invert: R_inv = R.T, t_inv = -R.T @ t
        R, t = np.asarray(cam.R, float), np.asarray(cam.t, float).ravel()
        rr.log(f"world/{ck}",
               rr.Transform3D(translation=(-R.T @ t), mat3x3=R.T), static=True)
        rr.log(f"world/{ck}/image",
               rr.Pinhole(image_from_camera=np.asarray(cam.K, float),
                          resolution=[1920, 1080]), static=True)

    # video frames for the chosen cams
    caps = {}
    for c in a.video_cams:
        p = a.clipdir / f"{a.stem}.{c}.mp4"
        if p.exists():
            caps[f"cam_{c}"] = cv2.VideoCapture(str(p))
            print(f"  embedding video for cam_{c}", flush=True)

    trail = []
    for f in range(n):
        rr.set_time("frame", sequence=f)
        rr.set_time("time", duration=f / FPS)

        ph = phase_of.get(f, "rest_post")
        col = PHASE_RGB[ph]

        pts, names = [], []
        for nm, X in J.items():
            if f < len(X) and np.isfinite(X[f]).all():
                pts.append(X[f]); names.append(nm)
        if pts:
            rr.log("world/skeleton/joints",
                   rr.Points3D(np.array(pts), radii=14.0, labels=names,
                               colors=[(60, 235, 60) if "wrist" in nm else (235, 235, 60)
                                       for nm in names]))
            segs = [[J[x][f], J[y][f]] for x, y in BONES
                    if f < len(J[x]) and np.isfinite(J[x][f]).all()
                    and np.isfinite(J[y][f]).all()]
            if segs:
                rr.log("world/skeleton/bones",
                       rr.LineStrips3D(segs, colors=(200, 200, 200), radii=4.0))

        if f < len(cup) and np.isfinite(cup[f]).all():
            filled = cup_conf[f] == 0
            rr.log("world/cup", rr.Points3D(
                [cup[f]], radii=26.0,
                colors=[(255, 140, 40) if filled else (60, 220, 255)],
                labels=["cup (KF-filled)" if filled else "cup"]))
            trail.append(cup[f])
            if len(trail) > 90:
                trail.pop(0)
            if len(trail) > 1:
                rr.log("world/cup_trail",
                       rr.LineStrips3D([np.array(trail)], colors=(60, 220, 255),
                                       radii=3.0))

        rr.log("phase", rr.Scalars(float(list(PHASE_RGB).index(ph))))
        rr.log("signals/cup_confidence",
               rr.Scalars(float(cup_conf[f]) if f < len(cup_conf) else 0.0))
        if f < len(d_wc) and np.isfinite(d_wc[f]):
            rr.log("signals/wrist_to_cup_mm", rr.Scalars(float(d_wc[f])))

        for ck, cap in caps.items():
            ok, img = cap.read()
            if ok:
                small = cv2.resize(img, (960, 540))
                rr.log(f"world/{ck}/image",
                       rr.Image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).compress(
                           jpeg_quality=75))

        if f % 100 == 0:
            print(f"  {f}/{n}", flush=True)

    for cap in caps.values():
        cap.release()
    print(f"\nwrote {a.out}  ({a.out.stat().st_size/1e6:.0f} MB)", flush=True)
    print(f"open with:  rerun {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
