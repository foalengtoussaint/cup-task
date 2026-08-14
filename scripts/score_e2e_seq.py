"""How much does the PHASE SEGMENTATION alone move the Murphy measures?

Fixed pose (MMC BA+SmoothNet -- the product pose), measures scoped THREE ways so only the phase
windows differ:
  automq   -> AutoMQ ground-truth phases (mapped to video)
  seq_omc  -> SEQUENTIAL segmenter on the OMC cup+wrist+mouth  (clean-input phases)
  seq_mmc  -> SEQUENTIAL segmenter on the markerless MMC cup+pose (the product's own phases)

seq_omc vs seq_mmc = the cost of markerless PHASE segmentation on the score (pose held fixed).
automq vs seq_omc  = the definitional gap between our rule and AutoMQ's.
Reuses the scorer's own measure functions + AutoMQ truth. Data only.
"""
from __future__ import annotations
import sys, re, time
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
from cup_task.score import compute_position_measures
GRID = S.GRID_JOINTS; FPS = S.FPS


def measures(pose, ph, side, rec):
    out = {}
    wr = f"{side}_wrist"; other = "right" if side == "left" else "left"
    ang = S.angle_measures_automq(pose, ph, side, "max")
    for k in ("max_shoulder_flexion", "max_shoulder_abduction", "max_elbow_angle",
              "peak_elbow_angular_velocity", "interjoint_coordination", "max_trunk_displacement"):
        gt = rec.get(k)
        if gt is not None and np.isfinite(gt) and np.isfinite(ang.get(k, np.nan)):
            out[k] = (ang[k], float(gt))
    pv = S.peak_velocity_reduce(pose, ph, side, "max")
    g = rec.get("peak_velocity_wrist")
    if g is not None and np.isfinite(g) and np.isfinite(pv):
        out["peak_velocity"] = (pv, float(g))
    try:
        trunk = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
        mm = compute_position_measures(pose[wr], trunk, ph, side, fps=FPS)
    except Exception:
        mm = None
    if mm is not None:
        for k in ("total_movement_time", "time_to_peak_velocity", "time_to_first_peak_velocity",
                  "number_of_movement_units"):
            gt = rec.get(k); mv = getattr(mm, k, None)
            if gt is not None and mv is not None and np.isfinite(gt) and np.isfinite(mv):
                out[k] = (float(mv), float(gt))
    return out


def _cup_mmc(part, trial, nfr, wrist=None):
    """The MMC cup the segmenter sees.

    OT_CUP_STRICT_KF=1: apply the >=3-agreeing-camera floor strictly (a 2-camera cup is not weak
    evidence, it is none), then fill the resulting holes with the KF+RTS smoother rather than letting
    the segmenter's _interp_nan_xyz draw a straight line through them -- a chord cuts the corner of
    the cup's arc to the mouth, so "near the mouth" fires late and clears early.
    """
    import os
    X = R._cup_v3(part, trial, R._calib(part), nfr)
    if os.environ.get("OT_CUP_STRICT_KF") != "1":
        return R._smooth_joint(X)
    nc_dir = ROOT / "cache" / (os.environ.get("OT_NCAMS_DIR") or "cup_ncams")
    f = nc_dir / f"{part}__{trial}.npz"
    if f.exists():
        nc = np.load(f)["n_cams"]
        X = np.asarray(X, float).copy()
        X[nc[:len(X)] < 3] = np.nan
    from cup_task.triangulate import fill_cup_from_wrist, kf_fill_gaps
    X = kf_fill_gaps(X)
    # OT_CUP_WRIST_PROXY=1: whatever the guarded KF would not fill (long gaps / low coverage) is taken
    # from the WRIST, which still carries the cup -- cup ~= wrist + median(cup-wrist) over the frames
    # that DO have consensus. Without this those frames are NaN and _interp_nan_xyz draws a chord.
    if os.environ.get("OT_CUP_WRIST_PROXY") == "1" and wrist is not None:
        X, _info = fill_cup_from_wrist(X, wrist)
    return R._smooth_joint(X)


def main():
    S.H.use_good_cams()
    amq = S.load_automq(); ba = R._ba_traj_cache()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in S.COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    print(f"cohort {len(trials)}; BA {len(ba) if ba else 0}", flush=True)
    rows = []; n = 0; t0 = time.time()
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((S.automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        nfr = t["mmc"].shape[0]
        pose = S._pose_variant_cached(t, "BA", "smoothnet", ba)   # FIXED product pose
        if pose is None:
            continue
        omc = H._load_omc(part, trial, nfr); wr = f"{side}_wrist"
        if wr not in omc or not np.isfinite(omc[wr]).any() or "nose" not in omc:
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        ph_amq = S.automq_phases_to_video(rec["phases"], lag, nfr)
        if not ph_amq:
            continue
        # seq phases from OMC cup and from MMC cup
        ocup = R._omc_cup(part, trial, nfr)
        seq_omc = segment_sequential(R._shift(ocup, lag), R._shift(omc[wr], lag), R._shift(omc["nose"], lag)) \
            if np.isfinite(ocup).any() else None
        mcup = _cup_mmc(part, trial, nfr, wrist=t["mmc"][:, GRID.index(wr)])
        seq_mmc = segment_sequential(mcup, R._smooth_joint(t["mmc"][:, GRID.index(wr)]),
                                     R._smooth_joint(t["mmc"][:, GRID.index("nose")])) \
            if np.isfinite(mcup).any() else None
        # fully-OMC pose (mocap markers, shifted onto the video timeline)
        JN = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
              "right_wrist", "left_wrist", "right_hip", "left_hip", "nose"]
        pose_omc = {j: R._shift(omc[j], lag) for j in JN} if all(j in omc for j in JN) else None
        # (source, POSE, PHASES): first three hold pose=MMC (phase isolation); omc_full = OMC pose + OMC phases
        srcs = [("automq", pose, ph_amq), ("seq_omc", pose, seq_omc), ("seq_mmc", pose, seq_mmc)]
        if pose_omc is not None and seq_omc is not None:
            srcs.append(("omc_full", pose_omc, seq_omc))
        for src, ps, ph in srcs:
            if ph is None:
                continue
            for meas, (mv, gt) in measures(ps, ph, side, rec).items():
                rows.append(dict(part=part, trial=trial, phase_src=src, measure=meas, mmc=mv, automq=gt))
        n += 1
        if n % 60 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    import os
    out = ROOT / "out/automq" / (os.environ.get("OT_E2E_OUT") or "score_e2e_seq.csv")
    df.to_csv(out, index=False)
    print(f"\nPROCESSING CHECK: trials {n}, rows {len(df)}", flush=True)

    MEAS = ["max_shoulder_flexion", "max_shoulder_abduction", "max_elbow_angle",
            "peak_elbow_angular_velocity", "interjoint_coordination", "max_trunk_displacement",
            "peak_velocity", "total_movement_time", "time_to_peak_velocity",
            "time_to_first_peak_velocity", "number_of_movement_units"]
    SRCS = ["automq", "omc_full", "seq_omc", "seq_mmc"]
    print("\n== r_s vs AutoMQ, by PHASE SOURCE (pose fixed = MMC BA+SN) ==")
    print(f"{'measure':<30}" + "".join(f"{s:>10}" for s in SRCS))
    for meas in MEAS:
        cells = []
        for s in SRCS:
            g = df[(df.measure == meas) & (df.phase_src == s)].dropna(subset=["mmc", "automq"])
            cells.append(spearmanr(g.automq, g.mmc).correlation if len(g) > 3 else np.nan)
        print(f"{meas:<30}" + "".join(f"{c:>10.2f}" for c in cells))
    print("\n== median |error| (measure units), by PHASE SOURCE ==")
    print(f"{'measure':<30}" + "".join(f"{s:>10}" for s in SRCS))
    for meas in MEAS:
        cells = []
        for s in SRCS:
            g = df[(df.measure == meas) & (df.phase_src == s)].dropna(subset=["mmc", "automq"])
            cells.append((g.mmc - g.automq).abs().median() if len(g) else np.nan)
        print(f"{meas:<30}" + "".join(f"{c:>10.2f}" for c in cells))
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
