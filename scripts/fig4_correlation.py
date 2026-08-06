"""Figure 4 remake: MMC-vs-OMC correlation grid (one scatter subpanel per Murphy measure).

Mirrors Unger et al. (2411.14992) Fig 4: OMC on x, MMC on y, y=x diagonal reference, one point per
trial, COLOR = participant, MARKER = arm (affected/unaffected). Uses our fast MMC (BA+SmoothNet) vs
AutoMQ OMC from out/automq/score_vs_automq.csv. Per-panel r + n annotated.

    python scripts/fig4_correlation.py  ->  out/figures/fig4_mmc_vs_omc.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "out" / "automq" / "score_vs_automq.csv"
VARIANT = "BA+smoothnet"

# (measure, pretty label, unit) -- ordered + labelled to mirror Unger et al. Fig 4 (2 rows x 6 cols)
MEASURES = [
    ("peak_velocity",                   "PV",                        "mm/s"),
    ("peak_elbow_angular_velocity",     "Elbow angular PV",          "deg/s"),
    ("time_to_peak_velocity",           "Time to PV",                "s"),
    ("time_to_first_peak_velocity",     "Time to first PV",          "s"),
    ("number_of_movement_units",        "Number of movement units",  "count"),
    ("total_movement_time",             "Total movement time",       "s"),
    ("interjoint_coordination",         "Interjoint coordination",   "r"),
    ("max_trunk_displacement",          "Trunk displacement",        "mm"),
    ("max_shoulder_flexion",            "Shoulder flexion",          "deg"),
    ("max_elbow_angle",                 "Elbow extension",           "deg"),
    ("max_shoulder_abduction",          "Shoulder abduction",        "deg"),
    ("time_to_peak_velocity_percent",   "Time to PV",                "%"),
]


def main():
    d = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
    d = d[d["variant"] == VARIANT].copy()
    d["automq"] = pd.to_numeric(d["automq"], errors="coerce")
    d["mmc"] = pd.to_numeric(d["mmc"], errors="coerce")

    parts = sorted(d["part"].unique())
    cmap = plt.get_cmap("tab20")
    pcol = {p: cmap(i % 20) for i, p in enumerate(parts)}
    arm_marker = {"affected": "o", "unaffected": "^"}

    ncol, nrow = 6, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow))
    axes = axes.ravel()

    for ax, (meas, label, unit) in zip(axes, MEASURES):
        g = d[(d["measure"] == meas) & (d["peak_metric"].isin(["n/a", "max"]))]
        g = g.dropna(subset=["automq", "mmc"])
        if len(g) < 3:
            ax.set_visible(False)
            continue
        for _, row in g.iterrows():
            ax.scatter(row["automq"], row["mmc"], s=14, alpha=0.6,
                       color=pcol.get(row["part"], "gray"),
                       marker=arm_marker.get(row.get("arm", "affected"), "o"),
                       edgecolors="none")
        a = g["automq"].values; m = g["mmc"].values
        lo = float(min(a.min(), m.min())); hi = float(max(a.max(), m.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        xs = np.array([lo - pad, hi + pad])
        # IDENTITY line (solid grey, y=x) + REGRESSION best-fit line (dashed black) -- as in the paper
        ax.plot(xs, xs, "-", color="0.55", lw=1.1, zorder=0)
        if np.std(a) > 1e-9:
            sl, ic = np.polyfit(a, m, 1)
            ax.plot(xs, sl * xs + ic, "k--", lw=1.1, zorder=1)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        rs = spearmanr(a, m).correlation
        ax.set_title(f"{label} [{unit}]", fontsize=10, pad=3)
        ax.text(0.95, 0.06, f"$r_s$ = {rs:.2f}", transform=ax.transAxes,
                fontsize=9.5, va="bottom", ha="right")
        ax.set_xlabel("OMC", fontsize=8.5)
        ax.set_ylabel("MMC", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[len(MEASURES):]:
        ax.set_visible(False)

    # legend: participants (color) + arm (marker)
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=pcol[p],
                      markersize=8, label=p) for p in parts]
    handles += [Line2D([0], [0], marker=arm_marker[a], color="0.3", lw=0,
                       markersize=8, label=a) for a in ("affected", "unaffected")]
    handles += [Line2D([0], [0], color="0.55", lw=1.1, label="identity (y=x)"),
                Line2D([0], [0], color="k", lw=1.1, ls="--", label="regression fit")]
    fig.legend(handles=handles, loc="lower center", ncol=8,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Fast MMC (BA + SmoothNet) vs OMC — Murphy movement-quality measures",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    out = ROOT / "out" / "figures" / "fig4_mmc_vs_omc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
