"""Figure 4, twice: the same measures scored with OPTICAL phase windows and with MARKERLESS ones.

Fig. 4 as it stands hands both systems the optical phase windows, so it reports pose error with
segmentation controlled. A deployed markerless system has no optical windows -- it segments from its
own cup and hand tracks. The difference between those two figures IS the cost of markerless
segmentation, and it is only readable if nothing else moves between them.

So both panels here share one truth on x, one trial set, one measure operator and one set of axis
limits. The single variable is where the phase windows came from:

    fig4a_omcseg.png   y = markerless pose, OPTICAL windows      pose error alone
    fig4b_mmcseg.png   y = markerless pose, MARKERLESS windows   end to end

x is the optical keypoints scored through OUR operator with OUR segmenter (score_own_phases.csv,
column `omc_omc`), so one measure definition and one segmentation rule hold across the whole figure
and a panel's spread is pose and windows, never a difference of definition. The DELTA study's own
phase windows are not used anywhere: they are classified from end-effector velocity on the optical
wrist and cup markers (unger2024 II-D-2), a different rule from the distance channels here, and
scoring against them would mix that difference into the comparison.

Paired throughout: a trial enters only if it has both y values, so the two figures are the same
points and the n in each panel is identical.

    python paper/scripts/fig4_pair.py     -> paper/fig4a_omcseg.png, paper/fig4b_mmcseg.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from scipy.stats import pearsonr                                         # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
OWN = REPO / "out" / "scoring" / "score_own_phases.csv"
SEGB = REPO / "out" / "scoring" / "seg_boundaries.csv"
CUP = "mmc_c3kf"                # the shipped markerless cup, as in paper/scripts/make_seg_table.py
HOLD = 9                        # seg_sequential's minimum-run constant, max(int(0.15*60), 3)
# same set and order as fig4_correlation.py, so the three figures are read the same way
MEASURES = [
    ("peak_velocity",               "PV",                       "mm/s"),
    ("peak_elbow_angular_velocity", "Elbow angular PV",         "deg/s"),
    ("time_to_peak_velocity",       "Time to PV",               "s"),
    ("time_to_first_peak_velocity", "Time to first PV",         "s"),
    ("number_of_movement_units",    "Number of movement units", "count"),
    ("total_movement_time",         "Total movement time",      "s"),
    ("interjoint_coordination",     "Interjoint coordination",  "r"),
    ("max_trunk_displacement",      "Trunk displacement",       "mm"),
    ("max_shoulder_flexion",        "Shoulder flexion",         "deg"),
    ("max_shoulder_flexion_drink",  "Shoulder flexion (drinking)", "deg"),
    ("max_elbow_angle",             "Elbow extension",          "deg"),
    ("max_shoulder_abduction",      "Shoulder abduction",       "deg"),
]
KEY = ["part", "trial", "measure"]


def declined():
    """Trials the segmenter itself declines: no usable cup track, so the wrist-to-cup distance is
    flat and grasp is pinned to its floor at reach onset + HOLD. Detected from the cup track alone,
    with no reference, and already excluded from Table IV -- so excluded here too, or the two tables
    would be scored on different trial sets."""
    d = pd.read_csv(SEGB)
    m = (d[f"seq_{CUP}_grasp"] - d[f"seq_{CUP}_reach onset"]) == HOLD
    return set(zip(d.loc[m, "part"], d.loc[m, "trial"])), len(d)


def load(keep_declined, own=OWN):
    d = pd.read_csv(own, keep_default_na=False, na_values=[""])
    for c in ("omc_omc", "mmc_omc", "mmc_mmc"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[KEY + ["arm", "omc_omc", "mmc_omc", "mmc_mmc"]].rename(columns={"omc_omc": "truth"})
    if not keep_declined:
        bad, ntot = declined()
        m = pd.Series(list(zip(d["part"], d["trial"]))).isin(bad).values
        print(f"  excluding {len(bad)}/{ntot} declined trials ({100*len(bad)/ntot:.1f}%): "
              f"{int(m.sum())} of {len(d)} rows")
        d = d[~m]
    # PAIRED: both figures must plot the same points, or their difference is not the windows
    return d.dropna(subset=["truth", "mmc_omc", "mmc_mmc"])


def panel_limits(d):
    """One set of limits per measure, over BOTH y-columns, so the two figures overlay."""
    lim = {}
    for meas, _, _ in MEASURES:
        g = d[d["measure"] == meas]
        if len(g) < 3:
            continue
        v = np.concatenate([g["truth"].values, g["mmc_omc"].values, g["mmc_mmc"].values])
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        pad = 0.05 * (hi - lo + 1e-9)
        lim[meas] = (lo - pad, hi + pad)
    return lim


def render(d, ycol, lim, title, out):
    parts = sorted(d["part"].unique())
    cmap = plt.get_cmap("tab20")
    pcol = {p: cmap(i % 20) for i, p in enumerate(parts)}
    arm_marker = {"affected": "o", "unaffected": "^"}

    ncol, nrow = 6, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.6 * nrow),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.42})
    axes = axes.ravel()
    for ax, (meas, label, unit) in zip(axes, MEASURES):
        g = d[d["measure"] == meas]
        if len(g) < 3 or meas not in lim:
            ax.set_visible(False)
            continue
        for _, row in g.iterrows():
            ax.scatter(row["truth"], row[ycol], s=14, alpha=0.6,
                       color=pcol.get(row["part"], "gray"),
                       marker=arm_marker.get(row.get("arm", "affected"), "o"),
                       edgecolors="none")
        a, m = g["truth"].values, g[ycol].values
        lo, hi = lim[meas]
        xs = np.array([lo, hi])
        ax.plot(xs, xs, "-", color="0.55", lw=1.1, zorder=0)
        if np.std(a) > 1e-9:
            sl, ic = np.polyfit(a, m, 1)
            ax.plot(xs, sl * xs + ic, "k--", lw=1.1, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r = pearsonr(a, m)[0] if np.std(a) > 1e-9 and np.std(m) > 1e-9 else np.nan
        ax.set_title(f"{label} [{unit}]", fontsize=10, pad=3)
        ax.text(0.95, 0.06, f"$r$ = {r:.2f}  $n$ = {len(g)}", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="right")
        ax.set_xlabel("OMC", fontsize=8.5); ax.set_ylabel("MMC", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(MEASURES):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=pcol[p],
                      markersize=8, label=p) for p in parts]
    handles += [Line2D([0], [0], marker=arm_marker[a], color="0.3", lw=0,
                       markersize=8, label=a) for a in ("affected", "unaffected")]
    handles += [Line2D([0], [0], color="0.55", lw=1.1, label="identity (y=x)"),
                Line2D([0], [0], color="k", lw=1.1, ls="--", label="regression fit")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    fig.text(0.5, 0.015, "The 12 measures of Unger et al.; percent-timing variants omitted as in "
             "Unger.", ha="center", fontsize=7.5, color="0.35")
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.15)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own", default=str(OWN),
                    help="scorer CSV; pass the --anat12 one to see the landmark-matched figures")
    ap.add_argument("--suffix", default="", help="appended to the output filenames")
    ap.add_argument("--outdir", default=str(PAPER))
    ap.add_argument("--keep-declined", action="store_true",
                    help="keep the trials the segmenter declines (Table IV excludes them)")
    a = ap.parse_args(argv)

    d = load(a.keep_declined, Path(a.own))
    lim = panel_limits(d)
    npair = d.drop_duplicates(["part", "trial"]).shape[0]
    print(f"{npair} paired trials, {d['part'].nunique()} participants, "
          f"{d['measure'].nunique()} measures", flush=True)

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sfx = a.suffix + ("_withdeclined" if a.keep_declined else "")
    render(d, "mmc_omc", lim,
           "(a) Optical phase windows — pose error alone",
           outdir / f"fig4a_omcseg{sfx}.png")
    render(d, "mmc_mmc", lim,
           "(b) Markerless phase windows — end to end",
           outdir / f"fig4b_mmcseg{sfx}.png")

    # the number the pair exists to produce
    print(f"\n{'measure':28s}{'r (OMC win)':>13s}{'r (MMC win)':>13s}{'delta':>9s}{'n':>7s}")
    rows = []
    for meas, label, _ in MEASURES:
        g = d[d["measure"] == meas]
        if len(g) < 3:
            continue
        ra = pearsonr(g["truth"], g["mmc_omc"])[0]
        rb = pearsonr(g["truth"], g["mmc_mmc"])[0]
        print(f"{label:28s}{ra:>13.3f}{rb:>13.3f}{rb-ra:>9.3f}{len(g):>7d}")
        rows.append(dict(measure=meas, label=label, r_omc_win=ra, r_mmc_win=rb,
                         delta=rb - ra, n=len(g)))
    csv = outdir / f"fig4_pair{sfx}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nwrote {csv}")
    print("DONE_FIG4_PAIR", flush=True)


if __name__ == "__main__":
    main()
