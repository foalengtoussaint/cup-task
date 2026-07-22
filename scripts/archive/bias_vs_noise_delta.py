"""Is the MMC measure error a CONSTANT BIAS or trial-to-trial NOISE? (the question n=1 cannot answer)

A constant bias is benign: it cancels in the comparisons clinicians actually make (affected vs
unaffected arm, pre vs post therapy), and it is calibratable. Trial-to-trial NOISE does not cancel
-- it degrades every use.

For a SCALAR measure this is only definable across trials:
    bias  = mean(MMC - OMC)   over trials   -> the systematic offset
    noise = SD(MMC - OMC)     over trials   -> the part that does NOT cancel
With n=1 every error looks like pure bias, which is why this needs the multi-trial run.

Reference for "is that noise big?": the within-condition SD of the measure itself (from the 85-trial
OMC-only run) and the affected-vs-unaffected clinical effect.

    python scripts/bias_vs_noise_delta.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRIALS = [f"trial_{i}_R_unaffected" for i in [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

# clinical effect (affected - unaffected) + within-condition SD, from omc_effect_size_delta.py
CLIN = {
    "total_movement_time": (0.6, 0.4), "peak_velocity": (-59.1, 70.6),
    "time_to_peak_velocity": (0.03, 0.3), "number_of_movement_units": (-0.2, 0.8),
    "max_trunk_displacement": (-17.9, 8.9), "elbow_extension_reaching (max)": (-8.5, 3.1),
    "shoulder_flexion_reaching (max)": (4.1, 1.7), "shoulder_flexion_drinking (max)": (-1.8, 3.5),
    "shoulder_abduction_drinking (max)": (1.1, 2.3), "peak_elbow_ang_vel (deg/s)": (-25.4, 13.0),
}


def run_trial(trial, variant):
    """Returns {measure: (mmc, omc)} for one trial."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_omc_delta.py"),
         "--part", "P14", "--trial", trial] + variant,
        capture_output=True, text=True, cwd=str(ROOT)).stdout
    d = {}
    for ln in out.splitlines():
        m = re.match(r'\s*([a-z_]+(?:\s*\([^)]*\))?)\s+(-?[\d.]+|nan)\s+(-?[\d.]+|nan)\s+'
                     r'([+-]?[\d.]+|nan)', ln)
        if m:
            try:
                d[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
            except ValueError:
                pass
    return d


def main():
    variant = ["--hybrid"]
    rows = {}
    for t in TRIALS:
        if not list((ROOT / "cache" / "delta" / "P14" / "dets").glob(f"*{t}*.2.pose.json")):
            print(f"  {t}: no pose cache, skip", flush=True)
            continue
        r = run_trial(t, variant)
        if not r:
            print(f"  {t}: no output, skip", flush=True)
            continue
        rows[t] = r
        print(f"  {t}: ok", flush=True)
    if len(rows) < 3:
        print("need >=3 trials"); return

    keys = [k for k in next(iter(rows.values())) if k in CLIN]
    print(f"\n=== BIAS vs NOISE across {len(rows)} trials (HYBRID) ===\n")
    print(f"{'measure':34}{'BIAS':>8}{'NOISE':>8}{'bias%':>7}{'noise/SD':>10}{'noise/eff':>10}")
    print("-" * 80)
    for k in keys:
        errs = np.array([rows[t][k][0] - rows[t][k][1] for t in rows
                         if np.isfinite(rows[t][k][0]) and np.isfinite(rows[t][k][1])])
        if len(errs) < 3:
            continue
        bias, noise = float(np.mean(errs)), float(np.std(errs, ddof=1))
        eff, sd = CLIN[k]
        pct = abs(bias) / (abs(bias) + noise) * 100 if (abs(bias) + noise) > 1e-9 else np.nan
        # the part that does NOT cancel, vs the natural spread and vs the clinical effect
        n_sd = noise / sd * 100 if sd > 1e-9 else np.nan
        n_eff = noise / abs(eff) * 100 if abs(eff) > 1e-9 else np.inf
        verdict = "BIAS (cancels)" if pct > 65 else ("noise" if pct < 35 else "mixed")
        print(f"{k:34}{bias:+8.2f}{noise:8.2f}{pct:6.0f}%{n_sd:9.0f}%{n_eff:9.0f}%  {verdict}")
    print("\nbias%      = |bias|/(|bias|+noise). >65% = mostly a constant offset -> CANCELS in")
    print("             affected-vs-unaffected and pre-vs-post comparisons, and is calibratable.")
    print("noise/SD   = the non-cancelling part vs the measure's own trial-to-trial spread.")
    print("noise/eff  = the non-cancelling part vs the clinical effect. THIS is what limits use.")


if __name__ == "__main__":
    main()
