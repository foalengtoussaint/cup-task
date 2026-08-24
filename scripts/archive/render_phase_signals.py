"""Phase strips + the SIGNALS the two segmenters actually decide on, over the video.

Answers "why does each drink boundary land there": a boundary is a signal crossing a
threshold, so we draw the signals and the thresholds. The two segmenters key off DIFFERENT
physics, which is the whole reason they disagree:

  iMOVE (OMC):  drink = hand->MOUTH distance < adaptive near-thr  AND hand slow
  cup-task:     drink = cup displacement-from-rest near peak      AND cup slow (+ cup->head)

So we plot, under the OMC / SAME / CUP-TASK phase strips:
  * cup speed (mm/s) with the 80 (still) and 150 (onset) lines        [both segmenters]
  * cup displacement-from-rest (mm) with the peak-150 pad band        [cup-task drink gate]
  * hand->mouth distance (mm) with its adaptive near line             [iMOVE drink gate]
  * cup->head distance (mm)                                           [cup-task van Andel]

CUP SOURCE: we can show BOTH the research refill cup (what the comparison currently feeds)
and cup-task's OWN cup (if its per-cam *.cup.json exist for this rep, e.g. out640/<stem>/),
so you can see whether swapping to the native cup moves the boundaries.

    python scripts/render_phase_signals.py --stem P07_drinking_left_20240124_142730 \
        --clipdir /home/imove/Documents/clips/P07 \
        --repdir out640/P07_drinking_left_20240124_142730 -o out/signals_P07.mp4
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
COMPARE = Path(__file__).resolve().parents[1] / "cache" / "phase_compare_omc.json"
REFILL = OT / "track3d_clean3d_refill"
FPS = 60.0
PC = {"rest_pre": (190, 190, 190), "reaching": (200, 200, 70),
      "forward_transport": (232, 155, 76), "drinking": (76, 85, 232),
      "back_transport": (59, 162, 240), "returning": (190, 90, 190),
      "rest_post": (140, 140, 140)}


def _norm(s):
    p = s.split("_")
    return "_".join(p[1:]) if len(p) > 1 and p[0] == p[1] else s


def _refill_cup(stem, T):
    for c in REFILL.glob("*__clean3d_refill.json"):
        if _norm(c.name.replace("__clean3d_refill.json", "")) == _norm(stem):
            d = json.loads(c.read_text())
            cup = np.full((T, 3), np.nan, float)
            for fr in d["frames"]:
                if fr["fr"] < T and fr.get("rts"):
                    cup[fr["fr"]] = fr["rts"]
            return cup
    return None


def _speed(xyz, w=7):
    v = np.isfinite(xyz).all(1)
    idx = np.flatnonzero(v)
    if len(idx) < 2:
        return np.zeros(len(xyz))
    x = xyz.copy()
    for a in range(3):
        x[:, a] = np.interp(np.arange(len(x)), idx, xyz[idx, a])
    d = np.linalg.norm(np.gradient(x, axis=0), axis=1) * FPS
    k = np.ones(w) / w
    return np.convolve(d, k, "same")


def _dist(a, b):
    return np.linalg.norm(a - b, axis=1)


def _phase_at(iv, fi):
    for n, s, e in iv:
        if s <= fi < e:
            return n
    return "?"


def _strip(canvas, y, label, iv, T, W, fi):
    h, x0 = 22, 130
    bw = W - x0 - 10
    cv2.putText(canvas, label, (6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for n, s, e in iv:
        cv2.rectangle(canvas, (x0 + int(s / T * bw), y),
                      (x0 + int(e / T * bw), y + h), PC.get(n, (100, 100, 100)), -1)
    cv2.line(canvas, (x0 + int(fi / T * bw), y - 2),
             (x0 + int(fi / T * bw), y + h + 2), (255, 255, 255), 2)


class Graph:
    """One stacked signal panel: several named curves + horizontal threshold lines."""
    def __init__(self, y, h, W, T, title, curves, thr=(), x0=130):
        self.y, self.h, self.W, self.T = y, h, W, T
        self.title, self.curves, self.thr, self.x0 = title, curves, thr, x0
        allv = np.concatenate([c[1][np.isfinite(c[1])] for c in curves if len(c[1])] + [[0.0]])
        self.lo, self.hi = 0.0, max(1.0, float(np.percentile(allv, 99)) * 1.1)

    def _yv(self, v):
        return self.y + self.h - int(np.clip((v - self.lo) / (self.hi - self.lo), 0, 1) * self.h)

    def _xf(self, f):
        return self.x0 + int(f / self.T * (self.W - self.x0 - 10))

    def draw(self, canvas, fi):
        cv2.putText(canvas, self.title, (6, self.y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1)
        for val, col, _lab in self.thr:
            yy = self._yv(val)
            cv2.line(canvas, (self.x0, yy), (self.W - 10, yy), col, 1)
        for name, sig, col in self.curves:
            pts = [(self._xf(f), self._yv(sig[f])) for f in range(min(len(sig), self.T))
                   if np.isfinite(sig[f])]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
        cv2.line(canvas, (self._xf(fi), self.y), (self._xf(fi), self.y + self.h),
                 (255, 255, 255), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--clipdir", type=Path, required=True)
    ap.add_argument("--repdir", type=Path, help="cup-task per-cam *.cup.json dir (native cup)")
    ap.add_argument("--calib", type=Path,
                    default=None)
    ap.add_argument("--model", default="s")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args(argv)

    part = a.stem.split("_")[0]
    calib = a.calib or Path(f"/home/imove/Documents/object_tracking/data/calib/{part}"
                            "/calibration.toml")
    cams = load_calibration(calib, target_size=(1920, 1080))

    rec = json.loads(COMPARE.read_text()).get(a.stem, {})

    def ivs(k):
        r = rec.get(k)
        iv = r["intervals"] if isinstance(r, dict) and "intervals" in r else r
        return [(n, int(s), int(e)) for n, s, e in iv] if iv else []
    omc, same, e2e = ivs("omc"), ivs("same"), ivs("e2e")

    # pose from the cohort cache
    posef = Path(__file__).resolve().parents[1] / "cache" / "pose_models" / a.stem / \
        f"yolo26{a.model}.2d.json"
    per_cam = json.loads(posef.read_text())
    n = max(len(v) for v in per_cam.values())

    def tri(fn):
        tr = triangulate.triangulate_target(per_cam, cams, fn, n)
        return np.array([t["X"] if t["X"] else [np.nan] * 3 for t in tr], float)

    mouth = tri(triangulate._mouth_point)
    lw = tri(lambda fr: triangulate._wrist_point(fr, "left"))
    rw = tri(lambda fr: triangulate._wrist_point(fr, "right"))
    rng = lambda X: np.linalg.norm(np.nanmax(X, 0) - np.nanmin(X, 0)) if np.isfinite(X).any() else 0
    hand = rw if rng(rw) >= rng(lw) else lw

    cup_refill = _refill_cup(a.stem, n)
    # native cup-task cup, if we have per-cam dets
    cup_native = None
    if a.repdir and a.repdir.exists():
        cup_per = triangulate._load_per_cam(a.repdir, "cup")
        if cup_per:
            ctr = triangulate.triangulate_target(cup_per, cams, triangulate._cup_point, n)
            cup_native, _ = segment.track_confidence(ctr)

    # ---- the decision signals ----
    cup_for_disp = cup_refill
    rest = np.nanmedian(cup_for_disp[:30], axis=0)          # cup rest = start median
    cup_disp = _dist(cup_for_disp, rest[None, :])
    graphs_data = {
        "cup_speed": ("cup speed mm/s   [-- 80 still  -- 150 onset]",
                      [("refill", _speed(cup_refill), (60, 220, 255))] +
                      ([("native", _speed(cup_native), (255, 140, 40))] if cup_native is not None else []),
                      [(80, (90, 90, 90), "still"), (150, (130, 130, 130), "onset")]),
        "cup_disp": ("cup displacement-from-rest mm   [cup-task drink: near peak-150]",
                     [("refill", cup_disp, (60, 220, 255))],
                     [(float(np.nanmax(cup_disp)) - 150, (76, 85, 232), "peak-150")]),
        "hand_mouth": ("hand->mouth mm   [iMOVE drink signal]",
                       [("dist", _dist(hand, mouth), (100, 235, 100))], []),
        "cup_head": ("cup->head mm   [cup-task van Andel]",
                     [("dist", _dist(cup_for_disp, mouth), (200, 120, 235))], []),
    }

    cam = a.cam or 4
    video = a.clipdir / f"{a.stem}.{cam}.mp4"
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(3)); H = int(cap.get(4))
    T = max((e for iv in (omc, same, e2e) for _, _, e in iv), default=n)

    sf = 0.5
    vw, vh = int(W * sf), int(H * sf)
    STRIP_H = 3 * 26 + 6
    GH = 58
    GTOT = GH * len(graphs_data)
    TOP = STRIP_H + GTOT + 18
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (vw, vh + TOP))

    graphs = []
    yy = STRIP_H + 14
    for key, (title, curves, thr) in graphs_data.items():
        graphs.append(Graph(yy, GH - 8, vw, T, title, curves, thr))
        yy += GH
    print(f"{a.stem} cam_{cam}  native_cup={'yes' if cup_native is not None else 'no'} "
          f"-> {a.out}", flush=True)

    fi = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        canvas = np.zeros((vh + TOP, vw, 3), np.uint8)
        canvas[TOP:] = cv2.resize(img, (vw, vh))
        _strip(canvas, 2, "OMC", omc, T, vw, fi)
        _strip(canvas, 28, "SAME(pose)", same, T, vw, fi)
        _strip(canvas, 54, "CUP-TASK", e2e, T, vw, fi)
        for g in graphs:
            g.draw(canvas, fi)
        cv2.putText(canvas, f"t={fi/FPS:4.1f}s  omc:{_phase_at(omc,fi)}  "
                            f"cup-task:{_phase_at(e2e,fi)}", (6, TOP - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        writer.write(canvas)
        fi += 1
    cap.release(); writer.release()
    print(f"  wrote {a.out} ({fi} frames)", flush=True)


if __name__ == "__main__":
    main()
