"""Animate the triangulated 3D points over time into an mp4.

Renders each frame of the rep as a 3D scatter (cup, mouth, wrists) with short
motion trails, so you can watch the cup rise to the mouth and back. Cup gaps
(occluded apex frames) are shown as a faded last-known ghost so the dropout is
visible rather than a jump.

    python scripts/animate_3d.py out/rep_P07_3d.json -o out/rep_P07_3d.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
import cv2  # noqa: E402

COLORS = {"cup": "#d62728", "mouth": "#1f77b4",
          "left_wrist": "#2ca02c", "right_wrist": "#ff7f0e"}
TRAIL = 20  # frames of motion trail


def _series(track, n):
    """Dense (n,3) array with NaN where no 3D point that frame."""
    a = np.full((n, 3), np.nan)
    for t in track:
        if t["X"] is not None:
            a[t["frame"]] = t["X"]
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tri_json", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth frame")
    args = ap.parse_args(argv)

    data = json.loads(args.tri_json.read_text())
    targets = data["targets"]
    n = max(max(t["frame"] for t in tr) for tr in targets.values()) + 1
    series = {name: _series(tr, n) for name, tr in targets.items()}

    # global axis limits from all finite points (equal aspect)
    allpts = np.vstack([s[~np.isnan(s).any(1)] for s in series.values()])
    mid = allpts.mean(0)
    rng = (allpts.max(0) - allpts.min(0)).max() / 2 * 1.1
    lims = [(mid[i] - rng, mid[i] + rng) for i in range(3)]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    tmp = args.out.with_suffix(".tmpframe.png")
    vw = None
    rendered = 0
    for f in range(0, n, args.stride):
        ax.clear()
        az = -60 + 30 * np.sin(f / n * np.pi)   # gentle orbit so depth reads
        ax.view_init(elev=18, azim=az)
        for name, s in series.items():
            c = COLORS.get(name, "#888")
            lo = max(0, f - TRAIL)
            seg = s[lo:f + 1]
            m = ~np.isnan(seg).any(1)
            if m.any():
                ax.plot(seg[m, 0], seg[m, 1], seg[m, 2], "-", color=c, lw=1.2, alpha=0.5)
            cur = s[f]
            if not np.isnan(cur).any():
                ax.scatter(*cur, color=c, s=70, label=name, depthshade=True)
            else:  # ghost last-known (esp. cup apex gaps)
                prev = s[:f + 1][~np.isnan(s[:f + 1]).any(1)]
                if len(prev):
                    ax.scatter(*prev[-1], color=c, s=40, alpha=0.25, marker="x")
        ax.set(xlim=lims[0], ylim=lims[1], zlim=lims[2],
               xlabel="X", ylabel="Y", zlabel="Z (mm)")
        ax.set_title(f"frame {f}/{n}")
        ax.legend(loc="upper left", fontsize=8)
        fig.savefig(tmp, dpi=90)

        img = cv2.imread(str(tmp))
        if vw is None:
            h, w = img.shape[:2]
            vw = cv2.VideoWriter(str(args.out),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (w, h))
        vw.write(img)
        rendered += 1
        if rendered % 30 == 0:
            print(f"  {rendered} frames rendered", flush=True)

    vw.release()
    tmp.unlink(missing_ok=True)
    print(f"done: {rendered} frames -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
