"""END-TO-END Murphy scoring: measures scoped by OUR OWN segmenter's phases (whole product), vs the
same measures scoped by AutoMQ's OMC phases (isolated pose). Same measure functions, same AutoMQ
truth, same BA+SmoothNet pose -- the ONLY thing that changes is where the phase windows come from.

  phase_src = "automq"  -> AutoMQ OMC phases mapped to video (what score_vs_automq.py uses)
  phase_src = "ours"    -> segment_cup_only -> refine_grasp_with_pose -> to_murphy_phases on the
                           markerless v3+SN cup + BA/SN wrist (the real pipeline segmentation)

The gap between the two columns = the cost of using our segmentation instead of ground-truth phases.
Trials with no cup track can't produce "ours" -> scored on "automq" only. Saves per-row CSV + prints
per-measure r_s and median|error| for each phase source. Data only.
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
from pipeline import segment as SEG
from pipeline.score import compute_position_measures
GRID = S.GRID_JOINTS
FPS = S.FPS


def our_phases(cup_v3sn, hand, nose):
    try:
        seg = SEG.segment_cup_only(R._fill(cup_v3sn), fps=FPS)
        seg = SEG.refine_grasp_with_pose(seg, R._fill(cup_v3sn), R._fill(hand),
                                         None if nose is None else R._fill(nose), fps=FPS)
        return SEG.to_murphy_phases(seg, R._fill(hand), R._fill(cup_v3sn), fps=FPS)  # list (name,s,e)
    except Exception:
        return None


def measures(pose, ph, side, rec):
    """{measure: (mmc, automq_truth)} for the paper measures, using the scorer's own functions."""
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


def main():
    S.H.use_good_cams()
    amq = S.load_automq()
    ba_cache = R._ba_traj_cache()
    print(f"BA cache {'loaded '+str(len(ba_cache)) if ba_cache else 'MISSING'}; AutoMQ {len(amq)}", flush=True)
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in S.COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([LR])_")
    rows = []; t0 = time.time(); n = 0; n_noours = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((S.automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        nfr = t["mmc"].shape[0]
        pose = S._pose_variant_cached(t, "BA", "smoothnet", ba_cache)
        if pose is None:
            continue
        omc = H._load_omc(part, trial, nfr)
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph_automq = S.automq_phases_to_video(rec["phases"], lag, nfr)
        if not ph_automq:
            continue
        arm = "affected" if rec.get("condition") == "affected" else "unaffected"
        # our segmentation phases from the markerless cup track
        calib = R._calib(part)
        cup = R._cup_v3(part, trial, calib, nfr)
        ph_ours = None
        if np.isfinite(cup).any():
            ph_ours = our_phases(R._smooth_joint(cup), pose[f"{side}_wrist"], pose.get("nose"))
        if ph_ours is None:
            n_noours += 1

        for src, ph in [("automq", ph_automq), ("ours", ph_ours)]:
            if ph is None:
                continue
            for meas, (mv, gt) in measures(pose, ph, side, rec).items():
                rows.append(dict(part=part, trial=trial, arm=arm, phase_src=src,
                                 measure=meas, mmc=mv, automq=gt))
        n += 1
        if n % 60 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s rows={len(rows)} no-ours={n_noours}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "out/scoring/score_vs_automq_e2e.csv"
    df.to_csv(out, index=False)
    print(f"\nPROCESSING CHECK: trials {n}, rows {len(df)}, trials-without-our-phases {n_noours}", flush=True)

    MEAS = ["max_shoulder_flexion", "max_shoulder_abduction", "max_elbow_angle",
            "peak_elbow_angular_velocity", "interjoint_coordination", "max_trunk_displacement",
            "peak_velocity", "total_movement_time", "time_to_peak_velocity",
            "time_to_first_peak_velocity", "number_of_movement_units"]
    print("\n== r_s vs AutoMQ  +  median|error|,  by PHASE SOURCE  (AutoMQ phases = isolated pose;"
          " ours = whole product) ==")
    print(f"{'measure':<30}{'rs_automq':>10}{'rs_ours':>9}{'|err|_automq':>14}{'|err|_ours':>12}{'n_ours':>8}")
    for meas in MEAS:
        cells = []
        for src in ("automq", "ours"):
            g = df[(df.measure == meas) & (df.phase_src == src)].dropna(subset=["mmc", "automq"])
            rs = spearmanr(g.automq, g.mmc).correlation if len(g) > 3 else np.nan
            err = (g.mmc - g.automq).abs().median() if len(g) else np.nan
            cells.append((rs, err, len(g)))
        (ra, ea, na), (ro, eo, no) = cells
        print(f"{meas:<30}{ra:>10.2f}{ro:>9.2f}{ea:>14.2f}{eo:>12.2f}{no:>8}")
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
