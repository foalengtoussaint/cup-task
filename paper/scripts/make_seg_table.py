"""Emit the phase-boundary agreement table (Table IV) from results/seg_boundaries.csv.

Same segmentation rule run on the markerless cup and on the optical cup, paired per
trial; the reported quantity is the MMC-vs-OMC disagreement, not the error against
AutoMQ -- a boundary bias shared by both sides cancels when MMC measures are compared
against OMC measures, so it is not what limits the comparison.

Trials on which the segmenter is degenerate are excluded and counted. The cup track is
unusable there (no consensus frames survive the >=3 floor), the wrist->cup distance goes
flat, and the grasp collapses onto its floor at reach onset + HOLD. Those boundaries are
not measurements: including them puts a 4 s tail on release (p95 240 frames -> 15).

    python paper/scripts/make_seg_table.py     -> paper/tables/table4_boundaries.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
CUP = "mmc_c3kf_wr2"            # shipped: >=3 floor + KF fill + wrist proxy on cup->mouth only
FPS = 60.0
HOLD = max(int(0.15 * FPS), 3)  # the segmenter's own minimum-run constant
TOL_F = 15                      # 0.25 s at 60 Hz

BOUNDS = ["reach onset", "grasp", "drink onset", "drink offset", "release", "settle"]
LABEL = {"reach onset": "Reach onset", "grasp": "Grasp", "drink onset": "Drink onset",
         "drink offset": "Drink offset", "release": "Release", "settle": "Settle"}


def main() -> None:
    d = pd.read_csv(PAPER / "results" / "seg_boundaries.csv")

    # degenerate = grasp pinned to its floor at reach onset + HOLD
    deg = (d[f"seq_{CUP}_grasp"] - d[f"seq_{CUP}_reach onset"]) == HOLD
    print(f"degenerate: {int(deg.sum())}/{len(d)} ({100 * deg.mean():.1f}%)", flush=True)

    rows = []
    for b in BOUNDS:
        e = (d.loc[~deg, f"seq_{CUP}_{b}"] - d.loc[~deg, f"seq_omc_{b}"]).abs().dropna()
        rows.append((LABEL[b], len(e), np.median(e) / FPS * 1e3,
                     np.percentile(e, 90) / FPS * 1e3, 100 * (e > TOL_F).mean()))
        print(f"  {b:13s} n={len(e):4d}  med {rows[-1][2]:5.0f} ms  "
              f"p90 {rows[-1][3]:5.0f} ms  >0.25s {rows[-1][4]:4.1f}%", flush=True)

    L = [r"\begin{table}[t]",
         r"\caption{Phase-boundary agreement between markerless and optical input to the",
         r"same segmenter, paired per trial. Excludes the $5.8\%$ of trials on which the",
         r"segmenter is degenerate for want of a usable cup track (see text).}",
         r"\label{tab:boundaries}", r"\centering", r"\footnotesize",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Boundary & $n$ & Median & $p_{90}$ & $>0.25$\,s \\",
         r" & & (ms) & (ms) & (\%) \\", r"\midrule"]
    for name, n, med, p90, frac in rows:
        L.append(f"{name} & {n} & {med:.0f} & {p90:.0f} & {frac:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    out = PAPER / "tables" / "table4_boundaries.tex"
    out.write_text("\n".join(L))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
