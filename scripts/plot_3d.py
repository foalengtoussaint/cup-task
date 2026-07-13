"""Plot the triangulated 3D tracks (cup, mouth, wrists) from triangulate.py output.

Renders a 3-panel figure: XY (top-down), XZ (side), and Z-over-time, plus a
per-target coverage bar. Saves a PNG — no interactive display needed.

    python scripts/plot_3d.py out/rep_P07_3d.json -o out/rep_P07_3d.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"cup": "#d62728", "mouth": "#1f77b4",
          "left_wrist": "#2ca02c", "right_wrist": "#ff7f0e"}


def _arr(track):
    """(frames, XYZ) for the frames that have a 3D point."""
    fs = [t["frame"] for t in track if t["X"] is not None]
    xyz = np.array([t["X"] for t in track if t["X"] is not None], dtype=float)
    return np.array(fs), xyz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tri_json", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    data = json.loads(args.tri_json.read_text())
    targets = data["targets"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_xy, ax_xz, ax_zt = axes
    for name, track in targets.items():
        fs, xyz = _arr(track)
        if len(xyz) == 0:
            continue
        c = COLORS.get(name, "#888")
        ax_xy.plot(xyz[:, 0], xyz[:, 1], ".-", ms=3, lw=0.8, color=c, label=name)
        ax_xz.plot(xyz[:, 0], xyz[:, 2], ".-", ms=3, lw=0.8, color=c, label=name)
        ax_zt.plot(fs, xyz[:, 2], ".-", ms=3, lw=0.8, color=c, label=name)

    ax_xy.set(xlabel="X (mm)", ylabel="Y (mm)", title="top-down (XY)")
    ax_xz.set(xlabel="X (mm)", ylabel="Z (mm)", title="side (XZ)")
    ax_zt.set(xlabel="frame", ylabel="Z (mm)", title="height over time")
    for a in axes:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    ax_xy.set_aspect("equal", "datalim")

    # coverage summary in the title
    cov = {n: sum(1 for t in tr if t["X"] is not None) / max(len(tr), 1)
           for n, tr in targets.items()}
    fig.suptitle("3D triangulation — " + "  ".join(f"{n} {p:.0%}" for n, p in cov.items()),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}", flush=True)
    for n, p in cov.items():
        print(f"  {n:12} {p:.0%} of frames have a 3D point", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
