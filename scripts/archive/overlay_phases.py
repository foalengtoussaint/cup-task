"""Render one camera with the 3D tracks + phases + Murphy measures drawn on it.

The point of this renderer is to be CHECKABLE BY EYE: everything it draws is the exact
array the numbers were computed from. It does NOT re-derive the tracks, the phases, or
the measures -- it loads them, projects them, and draws them. If the marker is off the
cup, the NUMBER is wrong, not just the picture. (A renderer that recomputes anything the
metric depends on will drift from it, silently, and then the video "proves" a number that
was never measured.)

    python scripts/overlay_phases.py REP_OUT_DIR --clip CLIP.mp4 --calib calib.toml -o out.mp4

REP_OUT_DIR is what pipeline.pipeline wrote (tracks3d.json + the per-cam JSONs).
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
from pipeline import score, segment, triangulate
from pipeline.kalman_3d import load_calibration, project

FPS = 60.0
PHASE_COLOR = {
    "rest_pre":          (150, 150, 150),
    "reaching":          (100, 220, 100),
    "forward_transport": (235, 180, 60),
    "drinking":          (60, 60, 235),
    "back_transport":    (235, 120, 200),
    "returning":         (100, 200, 220),
    "rest_post":         (150, 150, 150),
}


def _draw(img, cam, X, color, r=7, x=False, label=None):
    if X is None or not np.isfinite(X).all():
        return
    (u, v), infront = project(cam, np.asarray(X, float))
    if not infront:
        return
    u, v = int(round(u)), int(round(v))
    if not (-50 < u < img.shape[1] + 50 and -50 < v < img.shape[0] + 50):
        return
    if x:
        cv2.line(img, (u - r, v - r), (u + r, v + r), color, 2, cv2.LINE_AA)
        cv2.line(img, (u - r, v + r), (u + r, v - r), color, 2, cv2.LINE_AA)
    else:
        cv2.circle(img, (u, v), r, color, 2, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (u + r + 3, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_dir", type=Path)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(argv)

    m = re.search(r"\.(\d+)\.mp4$", a.clip.name)
    ckey = f"cam_{int(m.group(1))}"
    cams = load_calibration(a.calib, target_size=(1920, 1080))
    cam = cams[ckey]

    tracks = json.loads((a.rep_dir / "tracks3d.json").read_text())["targets"]
    per = triangulate._load_per_cam(a.rep_dir, "pose")
    n = max(len(v) for v in per.values())
    trunk_tr = triangulate.triangulate_target(per, cams, triangulate._trunk_point, n)

    cup, cup_conf = segment.track_confidence(tracks["cup"])
    trunk, _ = segment.track_confidence(trunk_tr)
    mouth, _ = segment.track_confidence(tracks["mouth"])

    def rng(t):
        x, _ = segment.track_confidence(t)
        x = x[np.isfinite(x).all(1)]
        return float(np.linalg.norm(x.max(0) - x.min(0))) if len(x) else 0.0
    side = "right" if rng(tracks["right_wrist"]) >= rng(tracks["left_wrist"]) else "left"
    hand, _ = segment.track_confidence(tracks[f"{side}_wrist"])

    # THE SAME calls the numbers come from -- not a re-derivation
    seg = segment.segment_cup_only(cup)
    seg = segment.refine_grasp_with_pose(seg, cup, hand, mouth)   # grasp = wrist->cup plateau
    phases = segment.to_murphy_phases(seg, hand, cup)
    meas = score.compute_position_measures(hand, trunk, phases, side)

    phase_of = {}
    for nm, s, e in phases:
        for f in range(s, e):
            phase_of[f] = nm

    cap = cv2.VideoCapture(str(a.clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    print(f"{a.clip.name} -> {a.out}  ({side} hand)", flush=True)
    print(f"  phases: " + "  ".join(f"{n} {s/FPS:.2f}-{e/FPS:.2f}s" for n, s, e in phases),
          flush=True)

    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ph = phase_of.get(fr, "rest_post")
        col = PHASE_COLOR[ph]

        _draw(img, cam, cup[fr] if fr < len(cup) else None, (60, 220, 235), r=9, label="cup")
        _draw(img, cam, mouth[fr] if fr < len(mouth) else None, (235, 60, 235), x=True, label="mouth")
        _draw(img, cam, hand[fr] if fr < len(hand) else None, (60, 235, 60), r=7, label=f"{side} hand")
        _draw(img, cam, trunk[fr] if fr < len(trunk) else None, (255, 255, 255), r=5, label="trunk")

        cv2.rectangle(img, (0, 0), (W, 46), (0, 0, 0), -1)
        cv2.putText(img, f"{ph.upper()}", (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"t={fr/FPS:5.2f}s   cup_conf={cup_conf[fr] if fr < len(cup_conf) else 0:.2f}",
                    (330, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

        # phase timeline bar (so the boundaries are visible, not just the current label)
        y = H - 26
        cv2.rectangle(img, (0, y - 8), (W, H), (0, 0, 0), -1)
        T = max(len(cup), 1)
        for nm, s, e in phases:
            cv2.rectangle(img, (int(s / T * W), y - 4), (int(e / T * W), y + 8),
                          PHASE_COLOR[nm], -1)
        cv2.line(img, (int(fr / T * W), y - 8), (int(fr / T * W), y + 12), (255, 255, 255), 2)
        cv2.putText(img, f"peak_v {meas.peak_velocity:.0f}mm/s  t_peak {meas.time_to_peak_velocity_percent:.0f}%"
                         f"  MU {meas.number_of_movement_units}  trunk {meas.max_trunk_displacement:.0f}mm",
                    (10, H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        vw.write(img)
        fr += 1
    cap.release(); vw.release()
    print(f"  wrote {a.out} ({fr} frames)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
