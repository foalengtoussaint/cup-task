"""Table III (Unger et al.): movement-quality measures MMC vs OMC.
  r_s  = PEARSON across ALL trials (single trials -- unger2024's subscript is "single", not "Spearman")
  r_av = PEARSON on per-(participant,arm) AVERAGED trials
Reads the Fig-4 scorer CSV (out/scoring/score_vs_automq.csv, BA+smoothnet). No OMC computation --
the CSV already holds AutoMQ's stored scalar vs our pose-derived scalar per trial.
Writes out/paper/table3_measures.md (+ .csv).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
# paper/scripts/ -> REPO is two up, PAPER (output folder) one up.
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
import os
CSV = Path(os.environ.get("SCORE_CSV", REPO / "out" / "scoring" / "score_vs_automq.csv"))
VARIANT = "BA+smoothnet"

# (measure key, pretty label) in Unger Table III order; percent + flexion-D excluded (see methods.md)
MEAS = [
    ("peak_velocity",                 "PV [mm/s]"),
    ("peak_elbow_angular_velocity",   "Elbow angular PV [deg/s]"),
    ("time_to_peak_velocity",         "Time to PV [s]"),
    ("time_to_first_peak_velocity",   "Time to first PV [s]"),
    ("number_of_movement_units",      "Number of movement units [n]"),
    ("total_movement_time",           "Total movement time [s]"),
    ("interjoint_coordination",       "Interjoint coordination"),
    ("max_trunk_displacement",        "Trunk displacement [mm]"),
    ("max_shoulder_flexion",          "Shoulder flexion [deg]"),
    ("max_elbow_angle",               "Elbow extension [deg]"),
    ("max_shoulder_abduction",        "Shoulder abduction [deg]"),
]


def main():
    d = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
    d = d[(d["variant"] == VARIANT) & (d["peak_metric"].isin(["n/a", "max"]))].copy()
    d["automq"] = pd.to_numeric(d["automq"], errors="coerce")
    d["mmc"] = pd.to_numeric(d["mmc"], errors="coerce")

    rows = []
    for key, label in MEAS:
        g = d[d["measure"] == key].dropna(subset=["automq", "mmc"])
        if len(g) < 3:
            rows.append((label, np.nan, np.nan, 0)); continue
        rs = pearsonr(g["automq"], g["mmc"])[0]
        # r_av: average per (participant, arm) then correlate
        av = g.groupby(["part", "arm"])[["automq", "mmc"]].mean().reset_index()
        rav = pearsonr(av["automq"], av["mmc"])[0] if len(av) >= 3 else np.nan
        rows.append((label, rs, rav, len(g)))

    out_md = PAPER / "table3_measures.md"
    out_csv = PAPER / "table3_measures.csv"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Table III — Movement-quality measures: MMC (BA + SmoothNet) vs OMC", "",
             "`r_s` = Spearman across all trials; `r_av` = Spearman on per-(participant, arm) averaged trials.",
             "", "| Movement quality measure | r_s | r_av | n |", "|---|---|---|---|"]
    for label, rs, rav, n in rows:
        lines.append(f"| {label} | {rs:.2f} | {rav:.2f} | {n} |")
    out_md.write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows, columns=["measure", "r_s", "r_av", "n"]).to_csv(out_csv, index=False)

    print("\n".join(lines))
    print(f"\nwrote {out_md}\nwrote {out_csv}\nDONE", flush=True)


if __name__ == "__main__":
    main()
