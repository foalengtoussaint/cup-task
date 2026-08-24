"""The drink boundary, explained in ONE picture: the two signals that actually gate it.

cup-task and OMC place the drink phase using DIFFERENT triggers, and that single difference
is the whole -358ms dwell gap. This shows just those two signals, nothing else:

  * OMC (iMOVE):  HAND -> mouth distance  -- drink fires when the HAND reaches the mouth.
  * cup-task:     CUP  -> mouth distance  -- drink fires when the CUP reaches the mouth.

Each is drawn with its own van-Andel trigger line (closest + 15% of the rest->closest
excursion) and its drink band shaded. Where the line is crossed = where that side starts/ends
drinking. Watch the two bands: the hand arrives and leaves at different frames than the cup,
so OMC's drink band is wider and cup-task's is clipped -- that IS the gap, on one axis.

Two rows only:
  row 1  OMC   : hand->mouth (green), trigger line, OMC drink band shaded, OMC phase strip
  row 2  cuptk : cup->mouth  (cyan),  trigger line, cup-task drink band shaded, its strip

    python scripts/render_drink_gates.py --stem P07_drinking_left_20240124_142730 \
        --clipdir /home/imove/Documents/clips/P07 --cam 2 -o out/gates_P07.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import segment, triangulate
from pipeline.kalman_3d import load_calibration

OT = Path("/home/imove/Documents/object_tracking/experiments/drink_study/cache")
QTM_CUP = Path(__file__).resolve().parents[1] / "cache" / "qtm_cup"
COMPARE = Path(__file__).resolve().parents[1] / "cache" / "phase_compare_omc.json"
FPS = 60.0
DRINK_FRAC = 0.15
DRINK_RGB = (76, 85, 232)          # the drinking phase colour (BGR)


def _norm(s):
    p = s.split("_")
    return "_".join(p[1:]) if len(p) > 1 and p[0] == p[1] else s


def _qtm_cup(stem, T):
    f = QTM_CUP / f"{_norm(stem)}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    cup = np.full((T, 3), np.nan, float)
    for i, p in enumerate(d["cup"]):
        if i < T and p is not None:
            cup[i] = p
    return cup


def _trigger(dist):
    """van-Andel near line: closest + 15% of (steady - closest). Returns (line, near_mask)."""
    finite = np.isfinite(dist)
    if finite.sum() < 5:
        return np.nan, np.zeros(len(dist), bool)
    steady = np.percentile(dist[finite], 90)
    closest = np.percentile(dist[finite], 5)
    line = closest + DRINK_FRAC * (steady - closest)
    return line, (dist < line)


def _drink_span(iv):
    for n, s, e in iv:
        if n == "drinking":
            return int(s), int(e)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--clipdir", type=Path, required=True)
    ap.add_argument("--model", default="s")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(argv)

    part = a.stem.split("_")[0]
    cams = load_calibration(
        Path(f"/home/imove/Documents/object_tracking/data/calib/{part}/calibration.toml"),
        target_size=(1920, 1080))

    per_cam = json.loads((Path(__file__).resolve().parents[1] / "cache" / "pose_models" /
                          a.stem / f"yolo26{a.model}.2d.json").read_text())
    n = max(len(v) for v in per_cam.values())

    def tri(fn):
        tr = triangulate.triangulate_target(per_cam, cams, fn, n)
        return np.array([t["X"] if t["X"] else [np.nan] * 3 for t in tr], float)

    mouth, _ = segment.track_confidence(
        triangulate.triangulate_target(per_cam, cams, triangulate._mouth_point, n))
    lw, rw = tri(lambda f: triangulate._wrist_point(f, "left")), \
        tri(lambda f: triangulate._wrist_point(f, "right"))
    rng = lambda X: np.linalg.norm(np.nanmax(X, 0) - np.nanmin(X, 0)) if np.isfinite(X).any() else 0
    hand = rw if rng(rw) >= rng(lw) else lw
    cup = _qtm_cup(a.stem, n)
    if cup is None:
        sys.exit("no QTM cup for this rep")

    d_hand = np.linalg.norm(hand - mouth, axis=1)      # OMC trigger signal
    d_cup = np.linalg.norm(cup - mouth, axis=1)        # cup-task trigger signal
    hand_line, _ = _trigger(d_hand)
    cup_line, _ = _trigger(d_cup)

    rec = json.loads(COMPARE.read_text()).get(a.stem, {})

    def ivs(k):
        r = rec.get(k)
        iv = r["intervals"] if isinstance(r, dict) and "intervals" in r else r
        return [(x[0], int(x[1]), int(x[2])) for x in iv] if iv else []
    omc_iv, e2e_iv = ivs("omc"), ivs("e2e")
    omc_dr, ct_dr = _drink_span(omc_iv), _drink_span(e2e_iv)

    cap = cv2.VideoCapture(str(a.clipdir / f"{a.stem}.{a.cam}.mp4"))
    W = int(cap.get(3)); H = int(cap.get(4))
    T = n
    sf = 0.5
    vw, vh = int(W * sf), int(H * sf)
    ROW = 96
    TOP = 2 * ROW + 20
    x0 = 128
    bw = vw - x0 - 12
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (vw, vh + TOP))
    print(f"{a.stem}  OMC drink {omc_dr}  cup-task drink {ct_dr}  -> {a.out}", flush=True)

    rows = [
        ("OMC:  HAND -> mouth (mm)", d_hand, hand_line, (100, 235, 100), omc_dr),
        ("CUP-TASK:  CUP -> mouth (mm)", d_cup, cup_line, (255, 210, 60), ct_dr),
    ]

    def xf(f):
        return x0 + int(f / T * bw)

    fi = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        canvas = np.zeros((vh + TOP, vw, 3), np.uint8)
        canvas[TOP:] = cv2.resize(img, (vw, vh))
        for r, (label, sig, line, col, span) in enumerate(rows):
            y0 = 8 + r * ROW
            gh = ROW - 30
            fin = sig[np.isfinite(sig)]
            hi = np.percentile(fin, 98) * 1.05 if len(fin) else 1.0
            lo = 0.0
            yv = lambda v: y0 + gh - int(np.clip((v - lo) / (hi - lo), 0, 1) * gh)
            # drink band shaded
            if span:
                cv2.rectangle(canvas, (xf(span[0]), y0), (xf(span[1]), y0 + gh),
                              tuple(int(c * 0.35) for c in DRINK_RGB), -1)
            # trigger line
            if np.isfinite(line):
                cv2.line(canvas, (x0, yv(line)), (vw - 12, yv(line)), (90, 90, 200), 1)
                cv2.putText(canvas, "drink trigger", (vw - 130, yv(line) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 210), 1)
            # signal curve
            pts = [(xf(f), yv(sig[f])) for f in range(min(len(sig), T)) if np.isfinite(sig[f])]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (6, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1)
            # cursor
            cv2.line(canvas, (xf(fi), y0), (xf(fi), y0 + gh), (255, 255, 255), 1)
            # label the drink span in seconds
            if span:
                cv2.putText(canvas, f"drink {span[0]/FPS:.2f}-{span[1]/FPS:.2f}s "
                                    f"({(span[1]-span[0])/FPS*1000:.0f}ms)",
                            (x0, y0 + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(canvas, f"t={fi/FPS:4.2f}s   dwell err (cup-task - OMC) = "
                            f"{rec.get('e2e',{}).get('dwell_err_ms','?')}ms",
                    (6, TOP - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        writer.write(canvas)
        fi += 1
    cap.release(); writer.release()
    print(f"  wrote {a.out} ({fi} frames)", flush=True)


if __name__ == "__main__":
    main()
