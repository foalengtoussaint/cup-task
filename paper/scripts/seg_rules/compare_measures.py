"""Compare the twelve movement-quality measures across boundary-rule variants.

This is the script that settled the question, and it should be run BEFORE trusting any boundary
metric: a rule that looks strictly better on boundary agreement can still destroy a measure.

Reads the scored CSVs written by `score_own_phases.py` under different OT_SEG_* settings (see the
README for the loop) and emits `paper/seg_rule_measures.csv` with r_s and r_av per measure per
variant, under both the optical and the markerless phase windows.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from fps_ablation import declined                                    # noqa: E402

OUT = ROOT / "paper" / "seg_rule_measures.csv"
VARIANTS = [
    ("shipped",     "score_own_phases_anat12.csv"),      # onset=pos    settle=end
    ("onset-speed", "score_own_phases_ONSETONLY.csv"),   # onset=speed  settle=end
    ("settle-speed", "score_own_phases_SETTLEONLY.csv"),  # onset=pos    settle=speed
    ("both-speed",  "score_own_phases_SPEED010.csv"),    # onset=speed  settle=speed
]
ORDER = ["peak_velocity", "peak_elbow_angular_velocity", "time_to_peak_velocity",
         "time_to_first_peak_velocity", "number_of_movement_units", "total_movement_time",
         "interjoint_coordination", "max_trunk_displacement", "max_shoulder_flexion",
         "max_shoulder_flexion_drink", "max_elbow_angle", "max_shoulder_abduction"]


def score(path, bad):
    d = pd.read_csv(path)
    d = d[[(p, t) not in bad for p, t in zip(d.part, d.trial)]]
    rows = []
    for m, g in d.groupby("measure"):
        rec = {"measure": m}
        for lbl, col in (("opt", "mmc_omc"), ("mmc", "mmc_mmc")):
            s = g.dropna(subset=["omc_omc", col])
            if len(s) < 5:
                rec[f"r_s_{lbl}"] = rec[f"r_av_{lbl}"] = np.nan
                rec[f"n_{lbl}"] = 0
                continue
            av = s.groupby(["part", "arm"])[["omc_omc", col]].mean()
            rec[f"r_s_{lbl}"] = pearsonr(s.omc_omc, s[col])[0]
            rec[f"r_av_{lbl}"] = pearsonr(av.omc_omc, av[col])[0] if len(av) >= 3 else np.nan
            rec[f"n_{lbl}"] = len(s)
        rec["omc_mean"] = g.omc_omc.mean()
        rows.append(rec)
    return pd.DataFrame(rows).set_index("measure")


def main():
    bad = declined()
    tabs = {}
    for name, fn in VARIANTS:
        p = ROOT / "out" / "scoring" / fn
        if not p.exists():
            print(f"  MISSING {fn} -- see the README for how to regenerate", flush=True)
            continue
        tabs[name] = score(p, bad)
    if "shipped" not in tabs:
        raise SystemExit("shipped baseline missing")
    out = []
    for m in ORDER:
        for name, T in tabs.items():
            if m not in T.index:
                continue
            r = T.loc[m]
            out.append(dict(measure=m, variant=name,
                            r_s_opt=r.r_s_opt, r_av_opt=r.r_av_opt,
                            r_s_mmc=r.r_s_mmc, r_av_mmc=r.r_av_mmc,
                            n_mmc=int(r.n_mmc), omc_mean=r.omc_mean,
                            delta_r_s_mmc=r.r_s_mmc - tabs["shipped"].loc[m].r_s_mmc))
    R = pd.DataFrame(out)
    R[["r_s_opt", "r_av_opt", "r_s_mmc", "r_av_mmc", "delta_r_s_mmc", "omc_mean"]] = \
        R[["r_s_opt", "r_av_opt", "r_s_mmc", "r_av_mmc", "delta_r_s_mmc", "omc_mean"]].round(4)
    R.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(R)} rows)\n")
    piv = R.pivot(index="measure", columns="variant", values="delta_r_s_mmc").reindex(ORDER)
    cols = [c for c in ("onset-speed", "settle-speed", "both-speed") if c in piv.columns]
    print("change in r_s (markerless windows) against the shipped rule:")
    print(piv[cols].to_string())
    print("\nAnything beyond -0.02 is a measure being damaged, not noise.")


if __name__ == "__main__":
    main()
