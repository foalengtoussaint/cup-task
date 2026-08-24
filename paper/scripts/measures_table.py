"""Table III: the twelve movement-quality measures, MMC vs OMC, under both phase-window sources.

One measure operator and one segmentation rule on both sides. The optical reference is the OMC
markers put through OUR operator with OUR segmenter (score_own_phases.csv, column `omc_omc`); the
DELTA study's stored measures and its phase windows are not used. Its windows come from end-effector
velocity on the optical wrist and cup markers (unger2024 II-D-2) -- a different rule from the two
distance channels here -- so scoring against them would fold a segmentation difference into what is
meant to read as pose error.

Four columns, following unger2024's pair:
    r_s     Pearson over SINGLE TRIALS         ("s" is Unger's subscript for single, not Spearman)
    r_av    Pearson over per-(participant, arm) AVERAGES
each computed twice -- with the OPTICAL phase windows, and with the MARKERLESS ones. The optical
column is the like-for-like comparison against unger2024, whose phases were classified from optical
data on both sides; the markerless column is the same measures with nothing optical anywhere, which
unger2024's Limitations names as unvalidated.

Excludes the trials the segmenter declines, as Table IV does, so the two tables are scored on the
same trials. `--keep-declined` puts them back.

    python paper/scripts/measures_table.py   -> paper/table3_measures.{md,csv}
    python paper/scripts/make_tables_tex.py  -> paper/tables/table3_measures.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
OWN = REPO / "out" / "scoring" / "score_own_phases.csv"
SEGB = REPO / "out" / "scoring" / "seg_boundaries.csv"
CUP = "mmc_c3kf"
HOLD = 9

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
    ("max_shoulder_flexion_drink",    "Shoulder flexion, drinking [deg]"),
    ("max_elbow_angle",               "Elbow extension [deg]"),
    ("max_shoulder_abduction",        "Shoulder abduction [deg]"),
]


def declined():
    d = pd.read_csv(SEGB)
    m = (d[f"seq_{CUP}_grasp"] - d[f"seq_{CUP}_reach onset"]) == HOLD
    return set(zip(d.loc[m, "part"], d.loc[m, "trial"])), len(d)


def pair(g, ycol):
    """r over single trials and over per-(participant, arm) averages.

    `g` is already restricted to trials finite under BOTH window sources, so the two columns of the
    table are the same trials and their difference is the windows. A measure can be undefined under
    one and not the other -- interjoint is a correlation over the reaching window, so a degenerate
    markerless reach leaves it NaN there but not on the optical side."""
    if len(g) < 3:
        return np.nan, np.nan, 0
    rs = pearsonr(g["omc_omc"], g[ycol])[0]
    av = g.groupby(["part", "arm"])[["omc_omc", ycol]].mean()
    rav = pearsonr(av["omc_omc"], av[ycol])[0] if len(av) >= 3 else np.nan
    return rs, rav, len(g)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-declined", action="store_true")
    ap.add_argument("--own", default=str(OWN),
                    help="scorer CSV; pass the --anat12 one for the landmark-matched table")
    ap.add_argument("--suffix", default="", help="appended to the output filenames")
    a = ap.parse_args(argv)

    d = pd.read_csv(a.own, keep_default_na=False, na_values=[""])
    for c in ("omc_omc", "mmc_omc", "mmc_mmc"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if not a.keep_declined:
        bad, ntot = declined()
        m = pd.Series(list(zip(d["part"], d["trial"]))).isin(bad).values
        print(f"excluding {len(bad)}/{ntot} declined trials ({100*len(bad)/ntot:.1f}%)")
        d = d[~m]

    rows = []
    for key, label in MEAS:
        g = d[d["measure"] == key].dropna(subset=["omc_omc", "mmc_omc", "mmc_mmc"])
        rs_o, rav_o, n = pair(g, "mmc_omc")
        rs_m, rav_m, n_m = pair(g, "mmc_mmc")
        assert n == n_m, f"{key}: paired subset disagrees ({n} vs {n_m})"
        rows.append(dict(measure=label, r_s=rs_o, r_av=rav_o,
                         r_s_mmcwin=rs_m, r_av_mmcwin=rav_m, n=n))

    T = pd.DataFrame(rows)
    T.to_csv(PAPER / f"table3_measures{a.suffix}.csv", index=False)

    hdr = ("| Movement quality measure | r_s | r_av | r_s (MMC win) | r_av (MMC win) | n |")
    lines = ["# Table III — Movement-quality measures, MMC vs OMC", "",
             "Pearson. `r_s` = single trials, `r_av` = per-(participant, arm) averages "
             "(unger2024's pair). The first two columns use OPTICAL phase windows, the next two "
             "MARKERLESS ones; the reference is the optical markers through our own operator and "
             "segmenter throughout.", "", hdr, "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['measure']} | {r['r_s']:.2f} | {r['r_av']:.2f} | "
                     f"{r['r_s_mmcwin']:.2f} | {r['r_av_mmcwin']:.2f} | {r['n']} |")
    (PAPER / f"table3_measures{a.suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {PAPER}/table3_measures{a.suffix}.md and .csv")
    print("DONE_MEASURES_TABLE", flush=True)


if __name__ == "__main__":
    main()
