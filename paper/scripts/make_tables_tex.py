"""Emit the LaTeX result tables from the generated CSVs.

Tables I/II come from ``trajectory_agreement.csv`` (per-trial per-trajectory
agreement, written by ``paper_trajectories.py``); Table III from
``table3_measures.csv`` (written by ``paper_table3.py``).

The point of this script is that the numbers in the PDF are never typed by
hand -- ``main.tex`` \\input{}s what this writes, so the paper cannot drift from
the caches. Re-run it after any pipeline change:

    conda activate object_tracking
    python paper/scripts/make_tables_tex.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "tables"

# Row order and unit label per trajectory, matching Table I of Unger et al.
TRAJ_ORDER = [
    ("eev", "End-effector vel.", "mm/s"),
    ("eav", "Elbow ang.\\ vel.", "deg/s"),
    ("elbow", "Elbow extension", "deg"),
    ("flex", "Shoulder flexion", "deg"),
    ("abd", "Shoulder abduction", "deg"),
    ("trunk", "Trunk displacement", "mm"),
]
ARMS = ["unaffected", "affected"]


def _iqr(s: pd.Series) -> float:
    return float(np.subtract(*np.percentile(s.dropna(), [75, 25])))


def _med_iqr(s: pd.Series, dec: int) -> str:
    """Format as ``median [q25, q75]`` -- the convention used in the source paper."""
    s = s.dropna()
    if s.empty:
        return "--"
    q25, med, q75 = np.percentile(s, [25, 50, 75])
    return f"{med:.{dec}f} [{q25:.{dec}f}, {q75:.{dec}f}]"


def table1(df: pd.DataFrame) -> str:
    # Single IEEEtran column: fits at \scriptsize with tight column separation.
    L = [
        r"\begin{table}[t]",
        r"\caption{Medians and [IQRs] across trials for the six kinematic trajectories, by arm.",
        r"Bias is markerless minus optical and is removed before RMSE; the Bias IQR row is the",
        r"mean across participants of each participant's own IQR of it.}",
        r"\label{tab:trajectories}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Kinematic & Value & Unaffected arm & Affected arm \\",
        r"\midrule",
    ]
    for i, (key, label, unit) in enumerate(TRAJ_ORDER):
        sub = df[df.traj == key]
        if sub.empty:
            continue
        # r gets 2 decimals; the dimensioned rows get 2 as well, lag gets 2.
        rows = [
            ("$r$", "r", 2),
            (f"RMSE ({unit})", "rmse", 2),
            (f"Bias ({unit})", "bias", 2),
        ]   # the lag is one value per TRIAL, identical across trajectories -- stated once in prose
        for j, (rowlabel, col, dec) in enumerate(rows):
            first = rowlabel if j == 0 else rowlabel
            cells = [_med_iqr(sub[sub.arm == a][col], dec) for a in ARMS]
            name = label if j == 0 else ""
            L.append(f"{name} & {first} & {cells[0]} & {cells[1]} \\\\")
        # former Table II, folded in as a fifth row: the within-participant variability of the
        # bias above. One table instead of two, and it sits next to the bias it qualifies.
        cells = []
        for a in ARMS:
            per_part = sub[sub.arm == a].groupby("part")["bias"].apply(_iqr)
            cells.append(f"{per_part.mean():.2f}" if len(per_part) else "--")
        L.append(f" & Bias IQR ({unit}) & {cells[0]} & {cells[1]} \\\\")
        if i != len(TRAJ_ORDER) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table2(df: pd.DataFrame) -> str:
    L = [
        r"\begin{table}[t]",
        r"\caption{Mean IQR of the static bias across participants, i.e.\ the",
        r"within-participant trial-to-trial variability of the offset between MMC and OMC.}",
        r"\label{tab:biasiqr}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Kinematic & Unaffected arm & Affected arm \\",
        r"\midrule",
    ]
    for key, label, unit in TRAJ_ORDER:
        sub = df[df.traj == key]
        if sub.empty:
            continue
        cells = []
        for a in ARMS:
            per_part = sub[sub.arm == a].groupby("part")["bias"].apply(_iqr)
            cells.append(f"{per_part.mean():.2f}" if len(per_part) else "--")
        L.append(f"{label} ({unit}) & {cells[0]} & {cells[1]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# Shortened row labels for the seven-column version of this table; the full names are in the
# text. Only the rows that set the table width need one.
_SHORT = {
    "Number of movement units [n]": "Movement units [n]",
    "Shoulder flexion, drinking [deg]": "Shoulder flexion, drink [deg]",
}


def table3(df: pd.DataFrame) -> str:
    """Full width (`table*`): the CIs do not fit one IEEE column.

    Point estimates alone cannot be read here -- interjoint coordination's r_s spans 0.27-0.98
    across random subsamples of the same trials -- so every r carries a bootstrap interval from
    ``measure_cis.py``. Falls back to the bare numbers if that file is absent.
    """
    ci_f = PAPER / "table3_cis.csv"
    ci = pd.read_csv(ci_f).set_index("measure") if ci_f.exists() else None
    f30 = PAPER / "table3_fps30.csv"
    hz30 = pd.read_csv(f30).set_index("measure")["r_s_30"].to_dict() if f30.exists() else {}

    L = [
        r"\begin{table*}[t]",
        r"\caption{Movement-quality measures, MMC versus OMC: $r_s$ over single trials and",
        r"$r_{av}$ over per-participant, per-arm averages, with the phase windows taken from the",
        r"optical and from the markerless recording in turn, and the markerless condition",
        r"repeated at $30$\,Hz. Brackets are $95\%$ bootstrap intervals, resampling trials for",
        r"$r_s$ and the $21$ participant--arm groups for $r_{av}$. $n$ is the $60$\,Hz pair count.}",
        r"\label{tab:measures}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Optical windows} & \multicolumn{2}{c}{Markerless windows} & 30\,Hz & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Measure & $r_s$ & $r_{av}$ & $r_s$ & $r_{av}$ & $r_s$ & $n$ \\",
        r"\midrule",
    ]

    def cell(point, lo, hi):
        if lo is None or not np.isfinite(lo):
            return f"{point:.2f}"
        return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"

    for _, r in df.iterrows():
        raw = str(r["measure"])
        measure = _SHORT.get(raw, raw).replace("_", r"\_")
        c = ci.loc[raw] if ci is not None and raw in ci.index else None
        if c is None:
            cells = [f"{r['r_s']:.2f}", f"{r['r_av']:.2f}",
                     f"{r['r_s_mmcwin']:.2f}", f"{r['r_av_mmcwin']:.2f}"]
        else:
            cells = [cell(c.r_s_opt, c.r_s_opt_lo, c.r_s_opt_hi),
                     cell(c.r_av_opt, c.r_av_opt_lo, c.r_av_opt_hi),
                     cell(c.r_s_mmc, c.r_s_mmc_lo, c.r_s_mmc_hi),
                     cell(c.r_av_mmc, c.r_av_mmc_lo, c.r_av_mmc_hi)]
        # 30 Hz from the CI file when present, so all six columns come from one source
        if c is not None and np.isfinite(c.get("r_s_30", np.nan)):
            c30 = cell(c.r_s_30, c.r_s_30_lo, c.r_s_30_hi)
        else:
            v30 = hz30.get(raw)
            c30 = f"{v30:.2f}" if v30 is not None else "---"
        L.append(f"{measure} & " + " & ".join(cells) + f" & {c30} & {int(r['n'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    traj_csv = PAPER / "trajectory_agreement.csv"
    # M3_CSV selects the measures table. Default is the landmark-matched one, which is what Fig 4
    # plots; table3_measures.csv (no suffix) is the uncorrected version.
    m_csv = PAPER / __import__("os").environ.get("M3_CSV", "table3_measures_anat12.csv")

    traj = pd.read_csv(traj_csv)
    meas = pd.read_csv(m_csv)

    # Processing check -- what actually went into the tables.
    n_part = traj["part"].nunique()
    # `part` counts RECORDING UNITS, not people: P25 was recorded twice under different
    # calibrations and appears as P251 and P252. `trial` is a per-unit index and is reused
    # across units, so its nunique() is a label count -- the trial count is len(traj)/6.
    n_trial = len(traj) // len(TRAJ_ORDER)
    print(f"trajectory_agreement.csv: {len(traj)} rows, "
          f"{n_part} recording units, {n_trial} trials, "
          f"trajectories={sorted(traj.traj.unique())}", flush=True)
    for col in ("r", "rmse", "bias", "lag_s"):
        n_bad = int((~np.isfinite(traj[col])).sum())
        print(f"  {col}: non-finite {n_bad}/{len(traj)}", flush=True)
    missing = [k for k, _, _ in TRAJ_ORDER if k not in set(traj.traj)]
    if missing:
        print(f"  WARNING: no rows for {missing}", flush=True)
    print(f"table3_measures.csv: {len(meas)} measures, "
          f"n range {meas.n.min()}-{meas.n.max()}", flush=True)

    for name, text in [
        ("table1_trajectories.tex", table1(traj)),
        ("table3_measures.tex", table3(meas)),
    ]:
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name}", flush=True)


if __name__ == "__main__":
    main()
