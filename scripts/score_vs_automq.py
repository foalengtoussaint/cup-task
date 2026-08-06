"""Score the markerless Murphy measures against the DELTA study's AutoMQ ground truth.

AutoMQ (`~/Documents/AutoMQ/<P>/`) is the study's OWN OMC processing: `murphy_measures_df.pkl`
(17 measures) + `phases_df.pkl` (its 5 phase windows, at 100Hz mocap frames) + validity flag.
It is the AUTHORITATIVE scoring target -- NOT the self-computed OMC in results_v3_delta.murphy_grid.

⚠ AutoMQ does NOT use IK. Its angle measures (elbow_angle, shoulder_flexion, shoulder_abduction) are
GEOMETRIC three-point angles from raw optical markers -- the SAME construction as our _angle_scalars.
So the angle measures ARE fairly comparable; the gap is pose error + calib/scale floor, not IK-vs-raw.

WHAT THIS DOES, per scorable trial (gnn_train.load_clean, 826 trials, 693 with an AutoMQ row):
  1. Build the MMC pose for each VARIANT (pipeline / +savgol / +smoothnet / BA+smoothnet).
  2. Take AutoMQ's OWN phase windows (isolates pose -- removes segmentation as a confound; these are
     the exact windows that produced the ground-truth numbers), resample 100Hz->60Hz and lag-shift
     onto the MMC timebase (same _find_lag sync used everywhere).
  3. Compute the 8 POSITION measures (compute_position_measures) + the 5 comparable ANGLE/trunk
     measures with AutoMQ's EXACT reductions:
        max_shoulder_flexion/abduction = max over Reaching[0]..Drinking[1]
        max_elbow_angle                = max over Reaching..Returning (whole movement)
        peak_elbow_angular_velocity    = max |d(elbow_angle)|*fps over Reaching   (max AND p95)
        interjoint_coordination        = corr(shoulder_flexion, elbow_angle) over Reaching
        max_trunk_displacement         = max |trunk disp about rest|
  4. Peak measures (peak_velocity, peak_elbow_ang_vel) emitted with BOTH `max` (matches AutoMQ) and
     `p95` (our robust choice) -- as separate measure rows tagged with the peak metric.
  5. Join to AutoMQ, write a tidy per-trial CSV + an npz of per-trial arrays (KEEP ALL DATA).

Watch:  tail -f out/gnn/score_vs_automq.log
"""
from __future__ import annotations
import sys, csv, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R              # reuse _pose_variant, _angle_scalars helpers, POSITION_MEASURES
from cup_task.score import (compute_position_measures, _smoothed_xyz, _hand_speed_mmps,
                            _butter_lowpass, DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER)

FPS = H.VIDEO_FPS                          # 60
C3D_RATE = H.C3D_RATE                      # 100
AUTOMQ = Path.home() / "Documents" / "AutoMQ"
GRID_JOINTS = R._GRID_JOINTS               # gnn_refiner.JOINTS order

# AutoMQ 5-phase name  ->  container 7-phase name that compute_position_measures / our windows expect
PHASE_MAP = {"Reaching": "reaching", "Forward Transport": "forward_transport",
             "Drinking": "drinking", "Back Transport": "back_transport", "Returning": "returning"}

# our internal measure name  ->  AutoMQ column
POS_TO_AUTOMQ = {
    "total_movement_time": "total_movement_time",
    "peak_velocity": "peak_velocity",
    "time_to_peak_velocity": "time_to_peak_velocity",
    "time_to_peak_velocity_percent": "time_to_peak_velocity_percent",
    "time_to_first_peak_velocity": "time_to_first_peak_velocity",
    "time_to_first_peak_velocity_percent": "time_to_first_peak_velocity_percent",
    "number_of_movement_units": "number_of_movement_units",
    # max_trunk_displacement is computed the ANATOMICAL way (forward-normal projection) in
    # angle_measures_automq, NOT via compute_position_measures' world-Y axis (which assumed the
    # participant faces world-Y and gave a spurious -23mm bias). Moved to ANG_TO_AUTOMQ. (2026-08-05)
}
ANG_TO_AUTOMQ = {
    "max_shoulder_flexion": "max_shoulder_flexion",
    "max_shoulder_abduction": "max_shoulder_abduction",
    "max_elbow_angle": "max_elbow_angle",
    "peak_elbow_angular_velocity": "peak_elbow_angular_velocity",
    "interjoint_coordination": "interjoint_coordination",
    "max_trunk_displacement": "max_trunk_displacement",
}
# which measures get emitted twice, once per peak metric
PEAK_MEASURES = {"peak_velocity", "peak_elbow_angular_velocity"}


def automq_part(part):
    return "P25" if part in ("P251", "P252") else part


def _amq_marker_peak(markers_df, marker, ph, fps=C3D_RATE):
    """AutoMQ's EXACT peak_velocity operator on a chosen marker series (from combined_data markers).

    Reproduces `peak_velocity = max(butter6Hz(|diff(pos)|*100)[Reaching])` -- verified to reproduce the
    PUBLISHED hand-based scalar to 0.00 mm/s. We call it with the WRIST marker so the ground truth uses
    the SAME landmark as our markerless wrist keypoint (the published measure is hand-based; our pipeline
    has no hand keypoint, so scoring wrist-vs-hand injected a spurious ~landmark bias -- wrist_L-vs-wrist
    collapses P07 peak-vel to bias -9 / err 16 mm/s). Denoised (6Hz), reaching-window, marker-matched."""
    from scipy.signal import butter, filtfilt
    lv = markers_df.index.get_level_values("marker")
    if marker not in lv:
        return float("nan")
    pos = markers_df.xs(marker, level="marker")[["x", "y", "z"]].to_numpy(float)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    fin = np.isfinite(v)
    if fin.sum() < 10:
        return float("nan")
    vi = np.interp(np.arange(len(v)), np.flatnonzero(fin), v[fin])
    b, a = butter(2, 6.0 / (fps / 2), btype="low")
    vf = filtfilt(b, a, vi)
    rs, re = ph["Reaching"]
    seg = vf[rs:re] if re > rs else vf
    return float(np.nanmax(seg)) if len(seg) else float("nan")


def load_automq():
    """{(automq_part, trial_number, side): {measures..., 'phases': dict, 'peak_velocity_wrist': float}}.

    peak_velocity_wrist = AutoMQ's own operator recomputed from the WRIST marker (landmark-matched to
    our markerless wrist keypoint); the published `peak_velocity` (hand-based) is kept for reference."""
    out = {}
    parts = sorted({automq_part(p) for p in COHORT_PARTS})
    for p in parts:
        mdf = pd.read_pickle(AUTOMQ / p / "murphy_measures_df.pkl")
        pdf = pd.read_pickle(AUTOMQ / p / "phases_df.pkl")
        cdf = pd.read_pickle(AUTOMQ / p / "combined_data_with_kinematics.pkl")
        # markers lookup keyed (trial_number, side) from combined_data's 5-tuple index
        markers_lookup = {}
        for fk in cdf.index:
            try:
                markers_lookup[(int(fk[1]), fk[2])] = cdf.loc[fk, "markers"]
            except Exception:
                continue
        # phases_df index is (participant trial side condition) tuple-string rows; align by position
        # via the same (trial_number, side) the measures df carries. Build a phase lookup keyed the
        # same way. pdf rows are indexed by a MultiIndex we can't trust to match; instead re-key from
        # the measures df order -- BUT phases_df has its OWN index. Safer: key phases by parsing index.
        ph_lookup = {}
        for idx, prow in pdf.iterrows():
            # idx like 'P07 2 L unaffected' (space-joined) OR a tuple; normalise
            key = idx if isinstance(idx, tuple) else tuple(str(idx).split())
            if len(key) >= 3:
                try:
                    tn, sd = int(key[1]), key[2]
                except (ValueError, IndexError):
                    continue
                ph_lookup[(tn, sd)] = (prow["phases"], int(prow.get("validity", 1)))
        for _, mrow in mdf.iterrows():
            tn, sd = int(mrow["trial_number"]), mrow["side"]
            rec = {c: mrow[c] for c in mdf.columns}
            phv = ph_lookup.get((tn, sd))
            rec["phases"] = phv[0] if phv else None
            rec["validity"] = phv[1] if phv else 1
            # landmark-matched peak_velocity: recompute from the wrist marker (same op, same window)
            rec["peak_velocity_wrist"] = float("nan")
            mdw = markers_lookup.get((tn, sd))
            if mdw is not None and rec["phases"] is not None:
                rec["peak_velocity_wrist"] = _amq_marker_peak(mdw, f"wrist_{sd}", rec["phases"])
            out[(p, tn, sd)] = rec
    return out


def automq_phases_to_video(ph_dict, lag, n_video):
    """AutoMQ phase dict (100Hz mocap frames) -> container 7-phase list on the 60Hz MMC timebase.

    100Hz->60Hz by *FPS/C3D_RATE, then + lag (the OMC->MMC shift from _find_lag). Adds synthesised
    rest_pre=(0, reach_start) and rest_post=(return_end, n_video). Clamps to [0, n_video] and drops
    any zero/negative-width window. Returns None if the core Reaching/Drinking windows are unusable.
    """
    s = FPS / C3D_RATE
    conv = {}
    for amq_name, (a, b) in ph_dict.items():
        cn = PHASE_MAP.get(amq_name)
        if cn is None:
            continue
        av = int(round(a * s)) + lag
        bv = int(round(b * s)) + lag
        av = max(0, min(av, n_video)); bv = max(0, min(bv, n_video))
        if bv > av:
            conv[cn] = (av, bv)
    if "reaching" not in conv or "drinking" not in conv:
        return None
    out = []
    reach_s = conv["reaching"][0]
    if reach_s > 0:
        out.append(("rest_pre", 0, reach_s))
    for cn in ("reaching", "forward_transport", "drinking", "back_transport", "returning"):
        if cn in conv:
            out.append((cn, conv[cn][0], conv[cn][1]))
    ret = conv.get("returning")
    if ret and ret[1] < n_video:
        out.append(("rest_post", ret[1], n_video))
    return out


def _elbow_series(P, side):
    """Per-frame elbow angle (deg) from 3 points -- SAME geometry as _angle_scalars/AutoMQ."""
    sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    u, v = P[sh] - P[el], P[wr] - P[el]
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return H._lp(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _win(ph, *names):
    sel = [(s, e) for (n, s, e) in ph if n in names]
    if not sel:
        return None
    return min(s for s, _ in sel), max(e for _, e in sel)


def _reduce(vals, how):
    """max / p95 / p99 of a finite 1-D array. AutoMQ's GT uses max; p95/p99 are robustness references."""
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return float("nan")
    if how == "max":
        return float(np.max(v))
    return float(np.nanpercentile(v, 99 if how == "p99" else 95))


def _seg_reduce(a, w, how="max"):
    if not (w and w[1] > w[0]):
        return float("nan")
    seg = a[w[0]:w[1]]
    if not np.isfinite(seg).any():
        return float("nan")
    return _reduce(seg, how)


def angle_measures_automq(pose, ph, side, peak="max"):
    """The 5 comparable angle/trunk measures with AutoMQ's EXACT windows + reductions."""
    from compare_pose_omc_delta import _murphy_signals
    elb = _elbow_series(pose, side)
    try:
        sig = _murphy_signals(pose, side=side)          # from OUR pose keypoints, not OMC
        flex, abd = H._lp(sig["shoulder_flexion"]), H._lp(sig["shoulder_abduction"])
    except Exception:
        flex = abd = np.full(len(elb), np.nan)
    # TRUNK: robust-reference 3D EXCURSION of the shoulder midpoint, NOT the anatomical forward
    # projection. The forward normal (side x down) uses the occluded seated hips, so it collapsed to
    # ~0 on ~half of some participants' trials (P07: fwd-proj rs 0.06 while raw motion matched OMC).
    # The 3D excursion (drink-task trunk motion is dominated by forward lean) with a median-rest
    # reference tracks AutoMQ: cohort rs 0.65 -> 0.93, P07 0.06 -> 0.98. (2026-08-06)
    other = "right" if side == "left" else "left"
    sh_mid = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2.0
    finm = np.isfinite(sh_mid).all(1)
    if finm.sum() >= 15:
        smf = sh_mid[finm]
        trunk_disp = float(np.percentile(np.linalg.norm(smf - np.median(smf[:15], 0), axis=1), 98))
    else:
        trunk_disp = float("nan")
    eav = np.abs(np.gradient(elb)) * FPS                         # deg/s

    reach = _win(ph, "reaching")
    drink = _win(ph, "drinking")
    # AutoMQ: max_shoulder_* over Reaching[0]..Drinking[1]
    rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
    whole = (0, len(elb))
    out = {
        "max_shoulder_flexion": _seg_reduce(flex, rd, "max"),
        "max_shoulder_abduction": _seg_reduce(abd, rd, "max"),
        "max_elbow_angle": _seg_reduce(elb, whole, "max"),
        "peak_elbow_angular_velocity": _seg_reduce(eav, reach, peak),
        "interjoint_coordination": float("nan"),
        "max_trunk_displacement": trunk_disp,   # robust-ref 3D shoulder-mid excursion (see above)
    }
    if reach and reach[1] - reach[0] >= 10:
        # AutoMQ: corr(flexion, elbow) over the INNER 80% of reaching (crop 10% each end -- the
        # leading/trailing 10% straddle the pre-reach and grasp pauses = zero-variance/opposite-trend
        # samples). Verified: reproduces AutoMQ's stored interjoint 84% within 0.05. Was full window.
        rr0, rr1 = reach
        mg = int(0.1 * (rr1 - rr0))
        a1, b1 = flex[rr0 + mg:rr1 - mg], elb[rr0 + mg:rr1 - mg]
        m = np.isfinite(a1) & np.isfinite(b1)
        if m.sum() >= 10 and np.std(a1[m]) > 1e-6 and np.std(b1[m]) > 1e-6:
            out["interjoint_coordination"] = float(np.corrcoef(a1[m], b1[m])[0, 1])
    return out


def peak_velocity_reduce(pose, ph, side, how="max"):
    """peak wrist speed (mm/s) with max OR p95, over the REACHING window only.

    AutoMQ: peak_velocity = max(filtered_hand_velocity[Reaching[0]:Reaching[1]]) -- REACHING ONLY, not
    the whole trial. Scoping to the whole trial was the +238mm/s max-to-max inflation bug: a fast
    hand motion in back_transport/returning became a spurious 'peak' AutoMQ never sees. This matches
    both AutoMQ AND compute_position_measures (which already scopes its own peak_velocity to reaching).
    OPERATOR = AutoMQ's own: differentiate the RAW wrist first, then low-pass the SPEED at 6 Hz
    (butter, zero-phase). This replaced the previous low-pass-POSITION-at-4Hz-then-differentiate:
    a filter sweep (peakvel_filter_sweep.py, BA+SmoothNet, 685 trials) showed AutoMQ's operator is
    STRICTLY better on all three -- bias -26.8->-17.9, median|err| 46.2->44.4, pearson r 0.81->0.90 --
    because low-passing position at 4Hz over-smoothed the velocity peak. Using AutoMQ's exact operator
    is the fair apples-to-apples definition, not a tuned offset. The residual ~-18 bias is the genuine
    markerless-vs-marker + 60-vs-100Hz effect. (2026-08-05)"""
    wr = pose[f"{side}_wrist"]
    sp = _butter_lowpass(_hand_speed_mmps(wr, FPS), FPS, 6.0, DEFAULT_BUTTER_ORDER)
    reach = _win(ph, "reaching")
    if reach and reach[1] > reach[0]:
        sp = sp[reach[0]:reach[1]]
    return _reduce(np.asarray(sp, float), how)


_SMCACHE = ROOT / "cache" / "pose_smoothed"


def _pose_variant_cached(t, triangulation, smoother, ba_cache):
    """Cache-first _pose_variant: for the two SmoothNet variants, read the cached smoothed pose
    (cache/pose_smoothed/<part>__<trial>.npz -- the expensive GPU SOLVE OUTPUT, saved ONCE) instead of
    re-running SmoothNet on the GPU every scoring run. Falls back to R._pose_variant (GPU) if the trial
    isn't cached. Non-SmoothNet variants (none/savgol) are cheap CPU ops -> compute directly.

    Applies the SAME physical-plausibility guard as _pose_variant: a blown-up BA trajectory (|coord|>1e5,
    6/826 trials) is treated as unavailable so it can't poison a metric with an infinity."""
    part, trial = t["part"], t["trial"]
    if smoother == "smoothnet":
        p = _SMCACHE / f"{part}__{trial}.npz"
        if p.exists():
            z = np.load(str(p), allow_pickle=True)
            joints = list(z["joints"])
            key = "ba_sn" if triangulation == "BA" else "pipeline_sn"
            arr = np.asarray(z[key], float)
            if not np.isfinite(arr).any() or np.nanmax(np.abs(arr)) > 1e5:
                return None
            return {j: arr[:, i, :] for i, j in enumerate(joints)}
    return R._pose_variant(t, triangulation, smoother, ba_cache)


# populated in main() from the requested cohort
COHORT_PARTS = ["P07", "P08", "P10", "P12", "P13", "P14", "P15", "P17", "P19", "P251", "P252"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/automq/score_vs_automq.csv")
    ap.add_argument("--parts", nargs="*", default=None, help="subset of parts (default: all 11)")
    ap.add_argument("--variants", nargs="*",
                    default=["pipeline", "pipeline+savgol", "pipeline+smoothnet", "BA+smoothnet"])
    args = ap.parse_args(argv)

    global COHORT_PARTS
    if args.parts:
        COHORT_PARTS = args.parts

    H.use_good_cams()
    VSPEC = {
        "pipeline":          {"triangulation": "pipeline", "smoother": "none"},
        "pipeline+savgol":   {"triangulation": "pipeline", "smoother": "savgol"},
        "pipeline+smoothnet":{"triangulation": "pipeline", "smoother": "smoothnet"},
        "BA+smoothnet":      {"triangulation": "BA",       "smoother": "smoothnet"},
    }
    variants = [(v, VSPEC[v]) for v in args.variants]
    ba_cache = R._ba_traj_cache()
    print(f"BA cache: {'loaded '+str(len(ba_cache))+' trajs' if ba_cache else 'MISSING'}", flush=True)

    amq = load_automq()
    print(f"AutoMQ rows loaded: {len(amq)}", flush=True)

    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    print(f"scorable trials in cohort: {len(trials)}   variants: {[v for v,_ in variants]}", flush=True)

    import re
    pat = re.compile(r"trial_(\d+)_([LR])_")
    rows = []
    n_join = 0; n_nomatch = 0; n_nophase = 0; n_ok = 0
    per_var = {v: 0 for v, _ in variants}
    from tqdm import tqdm
    t0 = time.time()
    pbar = tqdm(trials, ncols=95, file=sys.stdout, mininterval=3)
    for ti, t in enumerate(pbar):
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            n_nomatch += 1; continue
        tn, sd = int(m.group(1)), m.group(2)
        rec = amq.get((automq_part(part), tn, sd))
        if rec is None:
            n_nomatch += 1; continue
        n_join += 1
        if rec.get("phases") is None:
            n_nophase += 1; continue

        n = t["mmc"].shape[0]
        # sync: same wrist-speed lag path as the grid (OMC->MMC). AutoMQ phases live in OMC time.
        omc = H._load_omc(part, trial, n)
        wr = f"{side}_wrist"
        lag, _ = H._find_lag(t["mmc"][:, GRID_JOINTS.index(wr)], omc[wr])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if ph is None:
            n_nophase += 1; continue

        arm = "affected" if rec.get("condition") == "affected" else "unaffected"
        scored_any = False
        for vname, spec in variants:
            pose = _pose_variant_cached(t, spec["triangulation"], spec["smoother"], ba_cache)
            if pose is None:
                continue
            trunk = None
            other = "right" if side == "left" else "left"
            try:
                trunk_xyz = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
                mm = compute_position_measures(pose[wr], trunk_xyz, ph, side, fps=FPS)
            except Exception:
                mm = None
            # POSITION measures
            for our, col in POS_TO_AUTOMQ.items():
                gt = rec.get(col)
                if our in PEAK_MEASURES:
                    continue  # handled below with both metrics
                mv = getattr(mm, our, None) if mm is not None else None
                if gt is None or mv is None or not np.isfinite(gt) or not np.isfinite(mv):
                    continue
                rows.append((vname, part, trial, arm, side, our, "n/a", float(gt), float(mv)))
                scored_any = True
            # peak_velocity: score against the LANDMARK-MATCHED wrist ground truth (peak_velocity_wrist),
            # NOT the published hand-based scalar -- our pipeline has a wrist keypoint, no hand marker, so
            # wrist-vs-hand injects a spurious landmark bias. Both peak metrics (max/p95).
            gtpv = rec.get("peak_velocity_wrist")
            if gtpv is not None and np.isfinite(gtpv):
                for pk in ("max", "p95", "p99"):
                    mv = peak_velocity_reduce(pose, ph, side, pk)
                    if np.isfinite(mv):
                        rows.append((vname, part, trial, arm, side, "peak_velocity", pk,
                                     float(gtpv), float(mv)))
                        scored_any = True
            # ANGLE measures -- peak_elbow_angular_velocity gets all 3 metrics; rest are metric-invariant
            am_by_peak = {pk: angle_measures_automq(pose, ph, side, peak=pk)
                          for pk in ("max", "p95", "p99")}
            am_max = am_by_peak["max"]
            for our, col in ANG_TO_AUTOMQ.items():
                gt = rec.get(col)
                if gt is None or not np.isfinite(gt):
                    continue
                if our == "peak_elbow_angular_velocity":
                    for pk in ("max", "p95", "p99"):
                        mv = am_by_peak[pk][our]
                        if np.isfinite(mv):
                            rows.append((vname, part, trial, arm, side, our, pk, float(gt), float(mv)))
                            scored_any = True
                else:
                    mv = am_max[our]
                    if np.isfinite(mv):
                        rows.append((vname, part, trial, arm, side, our, "n/a", float(gt), float(mv)))
                        scored_any = True
            if scored_any:
                per_var[vname] += 1
        if scored_any:
            n_ok += 1
        if (ti + 1) % 40 == 0:
            print(f"    [{ti+1}/{len(trials)}] {time.time()-t0:5.0f}s  joined {n_join} "
                  f"scored {n_ok}  rows {len(rows)}", flush=True)
        pbar.set_postfix(join=n_join, scored=n_ok, rows=len(rows))

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "part", "trial", "arm", "side", "measure", "peak_metric",
                    "automq", "mmc"])
        w.writerows(rows)
    # per-item npz (KEEP ALL DATA): arrays for re-analysis without re-scoring
    arr = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in rows], dtype=object)
    np.savez(str(outp.with_suffix(".npz")), rows=arr,
             cols=np.array(["variant","part","trial","arm","side","measure","peak_metric","automq","mmc"]))

    print(f"\nPROCESSING CHECK: {len(trials)} cohort trials, joined-to-AutoMQ {n_join}, "
          f"no-match {n_nomatch}, no/failed-phase {n_nophase}, scored {n_ok}", flush=True)
    print(f"per-variant trials scored: {per_var}", flush=True)
    print(f"tidy rows: {len(rows)}   wrote {outp} (+ .npz)  [{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
