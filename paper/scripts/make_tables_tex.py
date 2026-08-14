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
    ("eev", "End-effector velocity", "mm/s"),
    ("eav", "Elbow angular velocity", "deg/s"),
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
    # Spans both columns: the median[IQR] cells are too wide for one IEEEtran column.
    L = [
        r"\begin{table*}[t]",
        r"\caption{Medians and [IQRs] for RMSE, $r$, bias, and time lag of kinematic",
        r"trajectories, calculated across all participants and differentiating between",
        r"the affected and unaffected arms.}",
        r"\label{tab:trajectories}",
        r"\centering",
        r"\footnotesize",
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
            ("Time lag (s)", "lag_s", 2),
        ]
        for j, (rowlabel, col, dec) in enumerate(rows):
            first = rowlabel if j == 0 else rowlabel
            cells = [_med_iqr(sub[sub.arm == a][col], dec) for a in ARMS]
            name = label if j == 0 else ""
            L.append(f"{name} & {first} & {cells[0]} & {cells[1]} \\\\")
        if i != len(TRAJ_ORDER) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
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


def table3(df: pd.DataFrame) -> str:
    L = [
        r"\begin{table}[t]",
        r"\caption{Movement-quality measures: Spearman correlation between MMC and OMC",
        r"across all trials ($r_s$) and on per-participant, per-arm averaged trials ($r_{av}$).}",
        r"\label{tab:measures}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Movement quality measure & $r_s$ & $r_{av}$ & $n$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        measure = (
            str(r["measure"]).replace("[", r"[").replace("_", r"\_")
        )
        L.append(f"{measure} & {r['r_s']:.2f} & {r['r_av']:.2f} & {int(r['n'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    traj_csv = PAPER / "trajectory_agreement.csv"
    m_csv = PAPER / "table3_measures.csv"

    traj = pd.read_csv(traj_csv)
    meas = pd.read_csv(m_csv)

    # Processing check -- what actually went into the tables.
    n_part = traj["part"].nunique()
    n_trial = traj["trial"].nunique()
    print(f"trajectory_agreement.csv: {len(traj)} rows, "
          f"{n_part} participants, {n_trial} trials, "
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
        ("table2_bias_iqr.tex", table2(traj)),
        ("table3_measures.tex", table3(meas)),
    ]:
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name}", flush=True)


if __name__ == "__main__":
    main()
