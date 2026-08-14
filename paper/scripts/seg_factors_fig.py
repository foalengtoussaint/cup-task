"""Heatmap of Spearman r_s: rig/calibration predictors vs (pose error, measure |error|).
Reads paper/seg_factors.csv. -> paper/seg_factors_heatmap.png. Data only."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
df = pd.read_csv(REPO / "paper/seg_factors.csv")
PRED = [("n_cam", "n cameras"), ("calib_err", "calib err (px)"),
        ("cov_maxang", "coverage max ang"), ("cov_meanang", "coverage mean ang"),
        ("reproj_med", "in-task reproj (px)")]
OUT = [("pose_sh", "pose: shoulder mm"), ("pose_el", "pose: elbow mm"), ("pose_wr", "pose: wrist mm")]
OUT += [(c, c.replace("err_", "").replace("_", " ")) for c in sorted(df.columns) if c.startswith("err_")]

M = np.full((len(OUT), len(PRED)), np.nan)
N = np.zeros((len(OUT), len(PRED)), int)
for i, (o, _) in enumerate(OUT):
    for j, (p, _) in enumerate(PRED):
        g = df[[p, o]].dropna()
        N[i, j] = len(g)
        if len(g) > 3:
            M[i, j] = spearmanr(g[p], g[o]).correlation

fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(OUT) + 1.6))
im = ax.imshow(M, cmap="RdBu_r", vmin=-0.65, vmax=0.65, aspect="auto")
ax.set_xticks(range(len(PRED))); ax.set_xticklabels([l for _, l in PRED], rotation=35, ha="right", fontsize=8.5)
ax.set_yticks(range(len(OUT))); ax.set_yticklabels([l for _, l in OUT], fontsize=8.5)
for i in range(len(OUT)):
    for j in range(len(PRED)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.42 else "0.15")
ax.axhline(2.5, color="k", lw=1.4)  # divide pose-error rows from measure-error rows
ax.text(-0.48, 1.0, "3D pose err", rotation=90, va="center", fontsize=7.5, color="0.35")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Spearman $r_s$", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
ax.set_title("Rig / calibration factor  vs  accuracy   (trial-level, all cohort trials)\n"
             "n_cam & calib_err are participant-level; coverage & in-task reproj vary per trial",
             fontsize=9.5)
fig.tight_layout()
out = REPO / "paper/seg_factors_heatmap.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}", flush=True)
