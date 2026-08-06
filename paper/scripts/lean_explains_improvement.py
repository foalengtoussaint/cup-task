"""Does POSTURE (per-trial lean / yaw) explain WHY body-referencing improves the shoulder-angle agreement?

Claim under test: MMC matches body-referenced OMC better than world-frame OMC BECAUSE the world-frame
definition is contaminated by trunk posture, trial-by-trial. If true, then per trial:
  the world-frame definition should sit FURTHER from MMC exactly on the HIGH-posture trials, i.e.
  |mmc - omc_world| - |mmc - omc_body|   should GROW with lean (flexion) / yaw (abduction).
And the OMC-internal definition gap (omc_world - omc_body) should track posture too.

If posture does NOT predict the per-trial improvement, then body-referencing helps for some OTHER reason
(e.g. a constant per-participant offset) and 'different leans' is NOT the mechanism. Read-only OMC.

Reuses the per-trial CSVs written by mmc_vs_flexion_defs.py + mmc_vs_abduction_defs.py, joined to the
OMC posture (lean/yaw) from the automq_*_ref_test.py CSVs. All by (part, trial, side).
    python paper/scripts/lean_explains_improvement.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
PAPER = Path(__file__).resolve().parents[1]


def _load(name, cols):
    f = PAPER / name
    if not f.exists():
        print(f"MISSING {f} -- run its generator first"); sys.exit(1)
    d = pd.read_csv(f)
    return d[["part", "trial", "side"] + cols]


def _report(tag, df, world, body, mmc, posture, pname):
    df = df.dropna(subset=[world, body, mmc, posture]).copy()
    df["err_world"] = (df[mmc] - df[world]).abs()
    df["err_body"] = (df[mmc] - df[body]).abs()
    df["improve"] = df["err_world"] - df["err_body"]      # how much body-ref HELPS this trial
    df["omc_gap"] = df[world] - df[body]                  # OMC-internal definition gap
    print(f"\n=== {tag}  (n={len(df)}) ===")
    print(f"  median improvement from body-ref: {df['improve'].median():+.1f} deg  "
          f"(err_world {df['err_world'].median():.1f} -> err_body {df['err_body'].median():.1f})")
    r1 = pearsonr(df[posture], df["improve"])[0]
    r2 = pearsonr(df[posture], df["omc_gap"])[0]
    rs1 = spearmanr(df[posture], df["improve"]).correlation
    print(f"  per-trial IMPROVEMENT vs {pname:5}: pearson r={r1:+.2f}  spearman={rs1:+.2f}")
    print(f"  OMC world-vs-body GAP  vs {pname:5}: pearson r={r2:+.2f}")
    # within-participant (remove per-participant offset) -- is it STILL posture-driven per trial?
    wr = []
    for p, g in df.groupby("part"):
        if len(g) >= 8 and g[posture].std() > 1e-6 and g["improve"].std() > 1e-6:
            wr.append(pearsonr(g[posture], g["improve"])[0])
    print(f"  WITHIN-participant improvement-vs-{pname}: mean r={np.nanmean(wr):+.2f}  (n={len(wr)} parts)")
    print(f"    -> if pooled r is high but WITHIN is ~0, the effect is BETWEEN-participant offset, NOT per-trial lean")
    return df


def main():
    # FLEXION: join MMC defs (mmc_flex, omc_vert, omc_trunk) + posture (lean) from the OMC ref test
    fl = _load("mmc_vs_flexion_defs.csv", ["mmc_flex", "omc_vert", "omc_trunk"])
    flp = _load("automq_flexion_ref_test.csv", ["lean_med"]).rename(columns={"lean_med": "lean"})
    fj = fl.merge(flp, on=["part", "trial", "side"], how="inner")
    _report("FLEXION: world(vert) vs trunk-ref, posture=LEAN", fj,
            "omc_vert", "omc_trunk", "mmc_flex", "lean", "lean")

    # ABDUCTION: join MMC defs (omc_world, omc_body) + posture (yaw, lean)
    ab = _load("mmc_vs_abduction_defs.csv", ["mmc_abd", "omc_world", "omc_body"])
    abp = _load("automq_abduction_ref_test.csv", ["yaw", "lean"])
    aj = ab.merge(abp, on=["part", "trial", "side"], how="inner")
    _report("ABDUCTION: world vs body-ref, posture=YAW", aj,
            "omc_world", "omc_body", "mmc_abd", "yaw", "yaw")
    _report("ABDUCTION: world vs body-ref, posture=LEAN", aj,
            "omc_world", "omc_body", "mmc_abd", "lean", "lean")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
