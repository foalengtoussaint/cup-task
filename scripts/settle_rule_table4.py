"""Table IV (phase-boundary agreement) under BOTH settle rules, plus the censoring flag.

Recomputes exactly what `paper/scripts/make_seg_table.py` reports -- |seq_mmc_c3kf_wr2 - seq_omc| per
boundary, excluding the degenerate trials -- but runs the segmenter twice, with settle_rule="pos"
(shipped) and settle_rule="still", and additionally reports the trials whose movement END WAS NEVER
OBSERVED inside the clip.

That flag is a CENSORING flag, not a quality flag: `total_movement_time` is
(returning_end - reaching_start), so when the settle is a timeout the value is a lower bound and
cannot be recovered by a better estimator. `number_of_movement_units` sums over `returning` and is
partially affected. Every reaching-windowed measure (peak velocity, both timings, interjoint, the
shoulder angles, elbow angular velocity) is untouched.

Nothing is written into paper/. Data only.

    OT_SEG_INPUTS_DIR=seg_inputs_ship OT_NCAMS_DIR=cup_ncams_26x \\
        python scripts/settle_rule_table4.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import cache_seg_inputs as CSI                                          # noqa: E402
from seg_sequential import segment_sequential, HOLD, FPS                # noqa: E402
from pipeline.triangulate import fill_cup_from_wrist, kf_fill_gaps      # noqa: E402

NCAMS = ROOT / "cache" / (os.environ.get("OT_NCAMS_DIR") or "cup_ncams")
RULES = ("pos", "still", "end", "still_end")
BOUNDS = [("reach onset", "reaching", 0), ("grasp", "reaching", 1),
          ("drink onset", "drinking", 0), ("drink offset", "drinking", 1),
          ("release", "back_transport", 1), ("settle", "returning", 1)]
TOL_F = 15                      # 0.25 s


def edges(seg):
    d = {nm: (s, e) for nm, s, e in seg}
    return {lab: (d[ph][i] if ph in d else np.nan) for lab, ph, i in BOUNDS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "out/scoring/settle_rule_table4.csv"))
    a = ap.parse_args(argv)

    rows = []
    for r in CSI.load_all():
        # the shipped MMC cup: >=3-camera floor, KF fill, wrist proxy on cup->mouth only
        z = np.load(NCAMS / f"{r['part']}__{r['trial']}.npz")
        cup = np.asarray(r["cup_mmc"], float).copy()
        cup[z["n_cams"][:len(cup)] < 3] = np.nan
        cup = kf_fill_gaps(cup)
        cup_mouth, _ = fill_cup_from_wrist(cup, r["wrist_mmc"])

        row = dict(part=r["part"], trial=r["trial"], arm=r["arm"])
        for rule in RULES:
            so, fo = segment_sequential(r["cup_omc"], r["wrist_omc"], r["nose_omc"],
                                        settle_rule=rule, return_flags=True)
            sm, fm = segment_sequential(cup, r["wrist_mmc"], r["nose_mmc"],
                                        cup_mouth_xyz=cup_mouth,
                                        settle_rule=rule, return_flags=True)
            eo, em = edges(so), edges(sm)
            for lab, _, _ in BOUNDS:
                row[f"{rule}_omc_{lab}"] = eo[lab]
                row[f"{rule}_mmc_{lab}"] = em[lab]
            row[f"{rule}_obs_omc"] = fo["settle_observed"]
            row[f"{rule}_obs_mmc"] = fm["settle_observed"]
        w = r["amq_returning"]
        row["amq_settle"] = float(w[1]) if w[0] >= 0 else float("nan")
        rows.append(row)

    d = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(d)} trials)\n")

    deg = (d["pos_mmc_grasp"] - d["pos_mmc_reach onset"]) == HOLD
    print(f"degenerate (excluded, as in make_seg_table): {int(deg.sum())}/{len(d)} "
          f"({100*deg.mean():.1f}%)\n")

    print("SETTLE, both targets at once.  MMC-vs-OMC is what Table IV reports;")
    print("vs-AutoMQ is definitional accuracy.  A rule must not win one by losing the other.\n")
    print(f"  {'rule':10s}{'MMC-vs-OMC med':>16s}{'p90':>7s}{'>0.25s':>9s}"
          f"{'vs-AutoMQ med':>16s}{'p90':>7s}{'censored':>10s}{'mixed':>7s}")
    for rule in RULES:
        e = (d.loc[~deg, f"{rule}_mmc_settle"] - d.loc[~deg, f"{rule}_omc_settle"]).abs().dropna()
        g = d.dropna(subset=["amq_settle"])
        a = (g[f"{rule}_omc_settle"] - g["amq_settle"]).abs()
        both = d[f"{rule}_obs_omc"] & d[f"{rule}_obs_mmc"]
        mixed = d[f"{rule}_obs_omc"] != d[f"{rule}_obs_mmc"]
        print(f"  {rule:10s}{np.median(e)/FPS*1e3:16.0f}{np.percentile(e,90)/FPS*1e3:7.0f}"
              f"{100*(e>TOL_F).mean():8.1f}%"
              f"{a.median()/FPS*1e3:16.0f}{a.quantile(.9)/FPS*1e3:7.0f}"
              f"{100*(~both).mean():9.1f}%{100*mixed.mean():6.1f}%")

    print("\n  other boundaries are unaffected by the settle rule (sanity check):")
    for lab, _, _ in BOUNDS[:-1]:
        e = (d.loc[~deg, f"pos_mmc_{lab}"] - d.loc[~deg, f"pos_omc_{lab}"]).abs().dropna()
        print(f"    {lab:14s} med {np.median(e)/FPS*1e3:5.0f} ms  p90 {np.percentile(e,90)/FPS*1e3:5.0f} ms")

    print("\n  -> total_movement_time should be reported on the 'observed on both sides' subset;")
    print("     number_of_movement_units is partially affected (it sums over `returning`).")


if __name__ == "__main__":
    main()
