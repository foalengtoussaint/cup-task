"""Watch WHERE cup-task's phases differ from OMC -- two strips over the footage.

The scalar says cup-task's drink dwell runs ~433ms short of OMC. This lets you SEE it: two
phase timelines stacked on the video -- CUP-TASK (yolo26s pose + refill cup) on top, OMC
(MeTRAbs pose + refill cup) below -- with a moving cursor and a DIFF tag when the two disagree
on the current frame. If the dwell really is short, you watch the top strip's red `drinking`
band start late or end early against the bottom one, at the actual moment her hand is at her
lips.

DUMB-PLAYER: the intervals drawn are the exact ones scripts/compare_phases_omc.py wrote to
cache/phase_compare_omc.json. This script re-segments NOTHING -- it draws the arrays the
numbers came from. (Same rule as the object_tracking render_phase_compare it's modelled on.)

    python scripts/render_phase_compare_omc.py --stem P07_drinking_left_20240124_142730 \
        --clipdir /home/imove/Documents/clips/P07 -o out.mp4
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
OT = Path("/home/imove/Documents/object_tracking/experiments/drink_study/cache")
COMPARE = Path(__file__).resolve().parents[1] / "cache" / "phase_compare_omc.json"
REFILL = OT / "student_dets_clean3d_refill"
FPS = 60.0
PC = {"rest_pre": (190, 190, 190), "reaching": (200, 200, 70),
      "forward_transport": (232, 155, 76), "drinking": (76, 85, 232),
      "back_transport": (59, 162, 240), "returning": (190, 90, 190),
      "rest_post": (140, 140, 140)}


def _phase_at(iv, fi):
    for n, s, e in iv:
        if s <= fi < e:
            return n
    return "?"


def _best_cam(stem, clipdir: Path):
    """Camera whose cup 2D spans the most pixels = best view. Falls back to first mp4."""
    part = stem.split("_")[0]
    det = REFILL / f"{part}_{stem}__clean3d_refill__c0.25.json"
    if det.exists():
        raw = json.loads(det.read_text())
        best, bestspan = None, -1
        for c, v in raw.items():
            pts = np.array([x for x in v if x is not None], float)
            if len(pts) >= 0.3 * len(v):
                span = float(np.hypot(*(pts.max(0) - pts.min(0))))
                if span > bestspan:
                    best, bestspan = c, span
        if best:
            return int(best.split("_")[1])
    mp4s = sorted(clipdir.glob(f"{stem}.*.mp4"))
    return int(re.search(r"\.(\d+)\.mp4$", mp4s[0].name).group(1)) if mp4s else 4


def _strip(canvas, y, label, iv, T, W, fi):
    h, x0 = 24, 130
    bw = W - x0 - 10
    cv2.putText(canvas, label, (6, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for n, s, e in iv:
        cv2.rectangle(canvas, (x0 + int(s / T * bw), y),
                      (x0 + int(e / T * bw), y + h), PC.get(n, (100, 100, 100)), -1)
    cx = x0 + int(fi / T * bw)
    cv2.line(canvas, (cx, y - 2), (cx, y + h + 2), (255, 255, 255), 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--clipdir", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--cam", type=int, default=0, help="0 = auto (best cup view)")
    a = ap.parse_args(argv)

    rec = json.loads(COMPARE.read_text()).get(a.stem)
    if rec is None:
        sys.exit(f"{a.stem} not in {COMPARE} -- run compare_phases_omc.py first")
    def ivs(key):
        r = rec.get(key)
        iv = r["intervals"] if isinstance(r, dict) and "intervals" in r else r
        return [(n, int(s), int(e)) for n, s, e in iv] if iv else []

    omc = ivs("omc")
    same = ivs("same")       # iMOVE segmenter on yolo pose (isolates POSE)
    e2e = ivs("e2e")         # cup-task's own segmenter (whole product)
    strips = [("OMC", omc), ("SAME(pose)", same), ("CUP-TASK", e2e)]
    de_same = (rec.get("same") or {}).get("dwell_err_ms", "?")
    de_e2e = (rec.get("e2e") or {}).get("dwell_err_ms", "?")

    cam = a.cam or _best_cam(a.stem, a.clipdir)
    video = a.clipdir / f"{a.stem}.{cam}.mp4"
    if not video.exists():
        sys.exit(f"no video {video}")
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(3)); H = int(cap.get(4))
    T = max((e for _, iv in strips for _, _, e in iv), default=1)

    sf = 0.6
    vw, vh = int(W * sf), int(H * sf)
    BAN = 108              # room for 3 strips + status line
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (vw, vh + BAN))
    print(f"{a.stem}  cam_{cam}  dwell err: same={de_same}ms  e2e={de_e2e}ms -> {a.out}",
          flush=True)

    fi = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        canvas = np.zeros((vh + BAN, vw, 3), np.uint8)
        canvas[BAN:] = cv2.resize(img, (vw, vh))
        for k, (label, iv) in enumerate(strips):
            _strip(canvas, 4 + k * 28, label, iv, T, vw, fi)
        on = _phase_at(omc, fi); sn = _phase_at(same, fi); en = _phase_at(e2e, fi)
        # colour the tag by whether the WHOLE-PRODUCT (e2e) matches OMC this frame
        diff = en != on
        cv2.putText(canvas, f"t={fi/FPS:4.1f}s  omc:{on}  same:{sn}  cup-task:{en}  "
                            f"[{'DIFF' if diff else '='}]",
                    (6, BAN - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if diff else (0, 220, 0), 1)
        writer.write(canvas)
        fi += 1

    cap.release(); writer.release()
    print(f"  wrote {a.out} ({fi} frames)", flush=True)


if __name__ == "__main__":
    main()
