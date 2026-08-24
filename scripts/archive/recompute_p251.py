"""Recompute P251's measure rows with the per-trial LINEAR time-warp (drift-corrected OMC), then
report the effect and (if it helps) splice the corrected P251 rows into the cohort CSV.

Warp aligns OMC (pose + cup) onto the MMC video timeline per trial: where a real linear drift is
detected (fit r2>=0.7, span>=5fr) resample OMC at t-(a+b*t); else fall back to the constant lag --
identical to the shipped path. Measures/phases then computed exactly as score_e2e_seq.py.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import score_vs_automq as S
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from seg_sequential import segment_sequential
from score_e2e_seq import measures
from desync_v3 import _pairs
from desync_correct import fit_lagline, warp
GRID = S.GRID_JOINTS; FPS = S.FPS
PART = "P251"
JN = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
      "right_wrist", "left_wrist", "right_hip", "left_hip", "nose"]


def align(joint, lag_const, ab):
    """Warp by a+b*t if ab given (real linear drift), else constant shift -- both put OMC on video time."""
    return warp(joint, ab[0], ab[1], len(joint)) if ab is not None else R._shift(joint, lag_const)


def main():
    S.H.use_good_cams(); amq = S.load_automq(); ba = R._ba_traj_cache()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] == PART]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    rows = []; n = 0; n_warp = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((S.automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        nfr = t["mmc"].shape[0]
        pose = S._pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        omc = H._load_omc(part, trial, nfr); wr = f"{side}_wrist"
        if wr not in omc or not np.isfinite(omc[wr]).any() or not all(j in omc for j in JN):
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        # decide warp
        pairs = _pairs(t, part, trial, side, nfr, omc)
        fl = fit_lagline(pairs, nfr) if len(pairs) >= 3 else None
        ab = (fl[0], fl[1]) if (fl and fl[2] >= 0.7 and fl[3] >= 5) else None
        if ab is not None:
            n_warp += 1
        ph_amq = S.automq_phases_to_video(rec["phases"], lag, nfr)   # AutoMQ column keeps constant lag
        if not ph_amq:
            continue
        # OMC pose + cup, aligned (warp where drift real)
        pose_omc = {j: align(omc[j], lag, ab) for j in JN}
        ocup = R._omc_cup(part, trial, nfr)
        seq_omc = segment_sequential(align(ocup, lag, ab), pose_omc[wr], pose_omc["nose"]) \
            if np.isfinite(ocup).any() else None
        mcup = R._cup_v3(part, trial, R._calib(part), nfr)
        seq_mmc = segment_sequential(R._smooth_joint(mcup), R._smooth_joint(t["mmc"][:, GRID.index(wr)]),
                                     R._smooth_joint(t["mmc"][:, GRID.index("nose")])) \
            if np.isfinite(mcup).any() else None
        srcs = [("automq", pose, ph_amq), ("seq_omc", pose, seq_omc), ("seq_mmc", pose, seq_mmc)]
        if seq_omc is not None:
            srcs.append(("omc_full", pose_omc, seq_omc))
        for src, ps, ph in srcs:
            if ph is None:
                continue
            for meas, (mv, gt) in measures(ps, ph, side, rec).items():
                rows.append(dict(part=part, trial=trial, phase_src=src, measure=meas, mmc=mv, automq=gt))
        n += 1
    new = pd.DataFrame(rows)
    new.to_csv(ROOT / "out/scoring/p251_warp_rows.csv", index=False)
    print(f"P251 recomputed: {n} trials ({n_warp} warped), rows {len(new)}\n", flush=True)

    MEAS = ["max_shoulder_flexion", "max_shoulder_abduction", "max_elbow_angle",
            "peak_elbow_angular_velocity", "interjoint_coordination", "max_trunk_displacement",
            "peak_velocity", "total_movement_time", "time_to_peak_velocity",
            "time_to_first_peak_velocity", "number_of_movement_units"]

    def agree(df):
        o = df[df.phase_src == "omc_full"][["trial", "measure", "mmc"]].rename(columns={"mmc": "o"})
        mm = df[df.phase_src == "seq_mmc"][["trial", "measure", "mmc"]].rename(columns={"mmc": "m"})
        j = o.merge(mm, on=["trial", "measure"]); out = {}
        for meas in MEAS:
            g = j[j.measure == meas].dropna()
            out[meas] = (spearmanr(g.o, g.m).correlation if len(g) > 3 else np.nan,
                         (g.m - g.o).abs().median() if len(g) else np.nan)
        return out

    old = pd.read_csv(ROOT / "out/scoring/score_e2e_seq.csv"); old = old[old.part == PART]
    ao, an = agree(old), agree(new)
    print("P251 MMC-vs-OMC agreement (both our method):  CONSTANT-lag -> WARP")
    print(f"{'measure':<30}{'rs const':>9}{'rs warp':>9}{'d':>7}{'err const':>10}{'err warp':>10}")
    for meas in MEAS:
        rc, ec = ao[meas]; rw, ew = an[meas]
        print(f"{meas:<30}{rc:>9.2f}{rw:>9.2f}{rw-rc:>+7.2f}{ec:>10.1f}{ew:>10.1f}")
    print("\nwrote out/scoring/p251_warp_rows.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
