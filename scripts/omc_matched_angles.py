"""Matched-definition OMC ground truth for the three ANGLE measures (flexion, abduction, interjoint).

WHY. Table III / Fig 4 score our angles against AutoMQ's STORED scalars, but AutoMQ's angles use
WORLD reference axes -- flexion = arm projected on the world Y-Z plane vs global Z, abduction = X-Z
plane vs Z, interjoint = corr of that flexion with elbow (verified in their notebook source; NO IK,
no OpenSim). Ours are body/trunk-referenced. Those three rows therefore mix a DEFINITION difference
into what reads as pose error. This recomputes the OMC side from OMC markers using OUR operator, so
the residual is pose/keypoint error.

HOW (this is the part that must not be hand-rolled). The OMC markers are put on the VIDEO timebase
(`H._load_omc` + the same `_find_lag` shift the scorer uses) and then pushed through the SAME
functions the MMC side uses -- `_murphy_signals`, `_elbow_series`, `_lp`, `_win`, `_seg_reduce`,
`automq_phases_to_video`. An earlier version hand-rolled the angles on raw 100Hz markers with no
low-pass; the CONTROL caught it (see below).

  flexion    : PER-FRAME trunk axis (user's call 2026-08-13) -- the trunk really does move and OMC
               hips are sub-mm, so there is no reason to freeze the reference.
               ⚠ASYMMETRY ON PURPOSE: our MMC flexion axis IS fully constant
               (`_murphy_signals_right` takes the per-trial MEDIAN of the per-frame
               hip_mid->trunk_mid direction) because our participants are SEATED with OCCLUDED HIPS,
               where a per-frame axis is jitter (trial-corr 0.16 vs 0.44, 2026-08-05). The
               constant-axis OMC variant is emitted as `*_constaxis` so the choice stays visible.
  abduction  : our shoulder-line construction (already per-frame on both sides).
  interjoint : corr(our flexion on OMC, our elbow on OMC) over Reaching -- AutoMQ's reduction, our
               angles.
  elbow      : CONTROL. Same three-point construction on both sides, so `r_s vs matched OMC` must
               land close to `r_s vs AutoMQ`. If it does not, this script is wrong -- that is
               exactly how the hand-rolled version was caught (0.808 vs 0.863).

The MMC column is JOINED FROM out/automq/score_vs_automq.csv (BA+smoothnet, peak max/n-a), NOT
recomputed, so the MMC numbers are literally the ones in the current table and the ONLY thing that
changes is the ground truth.

    python scripts/omc_matched_angles.py   ->  out/automq/omc_matched_angles.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import compare_pose_omc_delta as H                                    # noqa: E402
import gnn_train as GT                                                # noqa: E402
import results_v3_delta as R                                          # noqa: E402
from compare_pose_omc_delta import _murphy_signals, _lp               # noqa: E402
from score_vs_automq import (load_automq, automq_part, _elbow_series,  # noqa: E402
                             automq_phases_to_video, _win, _seg_reduce, _planar_body_angles)

SCORER = ROOT / "out/automq/score_vs_automq.csv"
OUT = ROOT / "out/automq/omc_matched_angles.csv"
GRID = R._GRID_JOINTS
JN = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
      "right_wrist", "left_wrist", "right_hip", "left_hip", "nose"]


def _flex_perframe(P, side):
    """Our flexion construction but with the PER-FRAME trunk axis (no median freeze)."""
    other = "right" if side == "left" else "left"
    sh, el = P[f"{side}_shoulder"], P[f"{side}_elbow"]
    trunk_mid = (P[f"{side}_shoulder"] + P[f"{other}_shoulder"]) / 2.0
    hip_mid = (P["right_hip"] + P["left_hip"]) / 2.0
    down = hip_mid - trunk_mid
    down = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    arm = el - sh
    arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    return _lp(np.degrees(np.arccos(np.clip((arm * down).sum(1), -1, 1))))


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    H.use_good_cams()
    amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    rows = []
    trials = GT.load_clean(need_reproj=False)
    n_ok = 0
    print(f"matched-definition OMC angles over {len(trials)} trials", flush=True)

    for i, t in enumerate(trials):
        m = pat.search(t["trial"])
        if not m:
            continue
        p, tn, sd = t["part"], int(m.group(1)), m.group(2)
        rec = amq.get((automq_part(p), tn, sd))
        if rec is None or rec.get("phases") is None:
            continue
        side = t["side"]
        n = t["mmc"].shape[0]
        omc = H._load_omc(p, t["trial"], n)
        wr = f"{side}_wrist"
        if wr not in omc or not all(j in omc for j in JN):
            continue
        if not np.isfinite(omc[wr]).any():
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        pose_omc = {j: R._shift(omc[j], lag) for j in JN}          # OMC on the VIDEO timebase

        other = "right" if side == "left" else "left"
        try:
            # PLANAR body-frame angles -- the SAME function the scorer uses for the MMC side
            # (score_vs_automq._planar_body_angles). Pairing planar-MMC with a hip-down-axis OMC was
            # my bug: that comparison is NOT matched, and it read 0.67 for flexion.
            flex_pf, abd = _planar_body_angles(pose_omc, side, other)
            sig = _murphy_signals(pose_omc, side=side)              # old hip-down, kept as secondary
            flex_const = _lp(sig["shoulder_flexion"])
            elb = _elbow_series(pose_omc, side)
        except (IndexError, ValueError, KeyError):
            continue

        ph = automq_phases_to_video(rec["phases"], lag, n)          # same windows as the scorer
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        allw = _win(ph, "reaching", "forward_transport", "drinking", "back_transport", "returning")
        if rd is None or reach is None:
            continue

        rows.append(dict(
            part=p, trial=t["trial"], side=sd,
            max_shoulder_flexion=_seg_reduce(flex_pf, rd, "max"),
            max_shoulder_abduction=_seg_reduce(abd, rd, "max"),
            max_elbow_angle=_seg_reduce(elb, allw or rd, "max"),
            interjoint_coordination=_corr(flex_pf[reach[0]:reach[1]], elb[reach[0]:reach[1]]),
            flexion_constaxis=_seg_reduce(flex_const, rd, "max"),
            interjoint_constaxis=_corr(flex_const[reach[0]:reach[1]], elb[reach[0]:reach[1]]),
        ))
        n_ok += 1
        if (i + 1) % 150 == 0:
            print(f"  [{i+1}/{len(trials)}] kept {n_ok}", flush=True)

    omc_df = pd.DataFrame(rows)
    print(f"\nPROCESSING CHECK: trials {len(trials)}, matched-OMC computed {n_ok}, "
          f"non-finite flexion {int(omc_df.max_shoulder_flexion.isna().sum())}, "
          f"non-finite interjoint {int(omc_df.interjoint_coordination.isna().sum())}", flush=True)

    sc = pd.read_csv(SCORER, keep_default_na=False, na_values=[""])
    sc = sc[(sc.variant == "BA+smoothnet") & (sc.peak_metric.isin(["n/a", "max"]))].copy()
    sc["mmc"] = pd.to_numeric(sc["mmc"], errors="coerce")
    sc["automq"] = pd.to_numeric(sc["automq"], errors="coerce")

    MEAS = ["max_shoulder_flexion", "max_shoulder_abduction", "interjoint_coordination",
            "max_elbow_angle"]
    long = omc_df.melt(id_vars=["part", "trial", "side"], value_vars=MEAS,
                       var_name="measure", value_name="omc_matched")
    j = sc.merge(long, on=["part", "trial", "measure"], how="inner")
    j.to_csv(OUT, index=False)

    print(f"\n{'measure':26}{'n':>5}{'vs AutoMQ':>11}{'vs MATCHED OMC':>16}   note")
    for k in MEAS:
        g = j[j.measure == k].dropna(subset=["mmc", "omc_matched"])
        ga = g.dropna(subset=["automq"])
        ra = spearmanr(ga.automq, ga.mmc).correlation if len(ga) > 3 else np.nan
        rm = spearmanr(g.omc_matched, g.mmc).correlation if len(g) > 3 else np.nan
        note = "<- CONTROL: these two must be close" if k == "max_elbow_angle" else ""
        print(f"{k:26}{len(g):>5}{ra:>11.3f}{rm:>16.3f}   {note}", flush=True)

    alt = omc_df[["part", "trial", "flexion_constaxis", "interjoint_constaxis"]]
    for meas, col in (("max_shoulder_flexion", "flexion_constaxis"),
                      ("interjoint_coordination", "interjoint_constaxis")):
        g = sc[sc.measure == meas].merge(alt, on=["part", "trial"], how="inner").dropna(
            subset=["mmc", col])
        if len(g) > 3:
            print(f"  [secondary] {meas} vs CONSTANT-axis OMC: "
                  f"r_s {spearmanr(g[col], g.mmc).correlation:.3f}  n={len(g)}", flush=True)
    print(f"\nwrote {OUT}\nDONE", flush=True)


if __name__ == "__main__":
    main()
