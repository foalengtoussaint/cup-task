"""Score MMC against OMC with OUR phase segmenter on BOTH sides -- no AutoMQ phases anywhere.

WHY. Table III currently gives both systems AutoMQ's phase windows. That controls segmentation, but
the windows are a third party's definition: they are built at 100Hz from the mocap and mapped onto the
video, so on 70/706 trials the window runs past the end of the clip and is silently truncated, and the
angle definitions had to be reconciled separately (omc_matched_angles). Running OUR segmenter on the
OMC channels removes both problems -- one segmentation rule, one measure operator, applied to each
system's own 3D.

Three columns, all with our operator and our segmenter:
  omc_pose / omc_win   the TRUTH: OMC keypoints, windows from our segmenter on OMC
  mmc_pose / omc_win   POSE ISOLATION: markerless pose, same windows -> residual is pose error alone
  mmc_pose / mmc_win   END-TO-END: markerless pose, windows from our segmenter on MMC

  (omc,omc) vs (mmc,omc) = pose error.  (mmc,omc) vs (mmc,mmc) = the cost of markerless segmentation.

Measures come from the scorer's OWN functions (compute_position_measures, angle_measures_automq,
peak_velocity_reduce) so the two sides cannot drift apart -- the same rule CLAUDE.md states for a
metric and its renderer.

    total_movement_time is NaN where `settle_observed` is False -- the recording stopped while the
    hand was still moving, so the movement end is not in the video and the value would be a lower
    bound. 24% of trials. Every reaching-windowed measure keeps the full set.

    --anat12 <theta.csv> applies the fitted landmark offsets to the OPTICAL keypoints before the
    reference measures are computed, per participant x arm (paper/scripts/anat12/anat12_lopo.py).
    This is a landmark match, not a correction of our estimate: neither system is positionally
    authoritative. Outputs go to a separate file so the uncorrected run is never overwritten.

    python scripts/score_own_phases.py      -> out/scoring/score_own_phases.{csv,npz}
    python scripts/score_own_phases.py --anat12 out/scoring/anat12_wv1wa1_theta.csv \
        --out out/scoring/score_own_phases_anat12.csv
    tail -f out/scoring/score_own_phases.log
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr    # Pearson everywhere since 2026-08-20, as in Table III

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H          # noqa: E402
import results_v3_delta as R                # noqa: E402
import score_vs_automq as S                 # noqa: E402
from seg_sequential import segment_sequential   # noqa: E402
from pipeline.score import compute_position_measures  # noqa: E402
from pipeline.triangulate import kf_fill_gaps  # noqa: E402
sys.path.insert(0, str(ROOT / "paper" / "scripts" / "anat12"))
import anat_frame as AF                        # noqa: E402  (the fitter; apply() lives there)

SEG = ROOT / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs_ship")
NCAMS = ROOT / "cache" / (__import__("os").environ.get("OT_NCAMS_DIR") or "cup_ncams_26x")
GRID = S.GRID_JOINTS
FPS = S.FPS


def _body_basis(pose, side, other):
    """down / fwd / lat, medianed over finite frames -- the frame anat_frame fitted theta in."""
    nn = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    sh, shO = pose[f"{side}_shoulder"], pose[f"{other}_shoulder"]
    down = nn((pose["right_hip"] + pose["left_hip"]) / 2.0 - (sh + shO) / 2.0)
    fwd = nn(np.cross(nn(sh - shO), down))
    lat = nn(np.cross(down, fwd))
    f = np.isfinite(down).all(1) & np.isfinite(fwd).all(1)
    if not f.any():
        return None
    u = lambda v: v / (np.linalg.norm(v) + 1e-9)
    return np.stack([u(np.nanmedian(down[f], 0)), u(np.nanmedian(fwd[f], 0)),
                     u(np.nanmedian(lat[f], 0))])


def _trunk_basis(pose):
    """PER-FRAME torso triad (down, fwd, lat), rows of B(t) are the axes.

    Deliberately not the frozen basis `_body_basis` returns. Trunk displacement is a distance from
    the trial's own rest position, so an offset held constant in a frozen frame is a constant world
    vector and cancels exactly; only an offset that rotates with the torso changes the arc the point
    sweeps during a forward lean, which is the sternum-vs-shoulder-midpoint discrepancy being fitted.
    Same construction as paper/scripts/anat12/trunk_offset_fit.basis, so fit and application agree.
    """
    nn = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    sh, shO = pose["right_shoulder"], pose["left_shoulder"]
    down = nn((pose["right_hip"] + pose["left_hip"]) / 2.0 - (sh + shO) / 2.0)
    fwd = nn(np.cross(nn(sh - shO), down))
    lat = nn(np.cross(down, fwd))
    return np.stack([down, fwd, lat], axis=1)


def _apply_trunk(pose, d):
    """Displace the OPTICAL sternum by `d` expressed in the per-frame torso triad."""
    if "trunk" not in pose:
        return pose
    try:
        B = _trunk_basis(pose)
    except KeyError:
        return pose
    out = dict(pose)
    out["trunk"] = pose["trunk"] + np.einsum("tij,i->tj", B, np.asarray(d, float))
    return out


def _apply_anat12(pose, side, other, th):
    """Displace the optical shoulder/elbow/wrist by the fitted offsets, via anat_frame.apply so the
    frames here and in the fitter cannot drift apart. Returns the pose unchanged if the body frame
    is undefined for this trial."""
    B = _body_basis(pose, side, other)
    if B is None:
        return pose
    rec = {f"omc_{j}": pose[j] for j in AF.pose_of.__globals__["JN"] if j in pose}
    if len(rec) < 9:
        return pose
    rec.update(B=B, side=side, other=other)
    out = dict(pose)
    out.update(AF.apply(rec, th, "anat12"))
    return out


def _ship_cup(part, trial, cup):
    """The SHIPPED markerless cup: below the three-camera floor is not a measurement, and the
    surviving gaps are filled by the Kalman smoother. Same preprocessing as `mmc_c3kf` in
    score_seg_boundaries.py, so the end-to-end windows here and Table IV's are the same windows."""
    cup = np.asarray(cup, float).copy()
    f = NCAMS / f"{part}__{trial}.npz"
    if f.exists():
        cup[np.load(f)["n_cams"][:len(cup)] < 3] = np.nan
    return kf_fill_gaps(cup)


def _measures(pose, ph, side, tag):
    """Every measure, via the scorer's own operators. Returns {measure: value}."""
    other = "right" if side == "left" else "left"
    out = {}
    try:
        # sternum marker where the source has one (optical); the markerless pose has no sternum
        # keypoint and uses the shoulder midpoint. NB the REPORTED max_trunk_displacement comes from
        # angle_measures_automq below, which makes the same choice itself -- this argument only feeds
        # compute_position_measures, whose trunk outputs are not harvested here.
        trunk = pose.get("trunk")
        if trunk is None or np.isfinite(trunk).all(1).sum() < 15:
            trunk = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
        pm = compute_position_measures(pose[f"{side}_wrist"], trunk, ph, side, fps=FPS)
        for nm in ("total_movement_time", "time_to_peak_velocity", "time_to_first_peak_velocity",
                   "number_of_movement_units"):
            v = getattr(pm, nm, None)
            if v is not None and np.isfinite(v):
                out[nm] = float(v)
    except Exception:
        pass
    try:
        out["peak_velocity"] = float(S.peak_velocity_reduce(pose, ph, side, "max"))
    except Exception:
        pass
    try:
        am = S.angle_measures_automq(pose, ph, side, peak="max")
        for k, v in am.items():
            if v is not None and np.isfinite(v):
                out[k] = float(v)
    except Exception:
        pass
    return {k: v for k, v in out.items() if np.isfinite(v)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anat12", default=None,
                    help="theta CSV from anat12_lopo.py; applied to the OPTICAL keypoints")
    ap.add_argument("--trunk-theta", default=None,
                    help="per-group sternum offsets from anat12/trunk_theta.py, applied to the "
                         "OPTICAL trunk landmark in the PER-FRAME torso frame")
    ap.add_argument("--out", default=str(ROOT / "out/scoring/score_own_phases.csv"))
    args = ap.parse_args(argv)
    theta = {}
    if args.anat12:
        t = pd.read_csv(args.anat12)
        cols = [f"th{i}" for i in range(12)]
        theta = {(r["part"], r["arm"]): np.array([r[c] for c in cols], float)
                 for _, r in t.iterrows() if not str(r["part"]).startswith("LOPO_")}
        print(f"anat12: {len(theta)} participant x arm offsets from {args.anat12}", flush=True)
    trunk_theta = {}
    if args.trunk_theta:
        t = pd.read_csv(args.trunk_theta)
        trunk_theta = {(r["part"], r["arm"]): np.array([r.d0, r.d1, r.d2], float)
                       for _, r in t.iterrows()}
        print(f"trunk: {len(trunk_theta)} participant x arm sternum offsets from "
              f"{args.trunk_theta}", flush=True)
    H.use_good_cams()
    import gnn_train as GT
    ba = R._ba_traj_cache()
    files = sorted(SEG.glob("*.npz"))
    trials = {f"{t['part']}/{t['trial']}": t for t in GT.load_clean(need_reproj=False)}
    print(f"score_own_phases: {len(files)} trials with segmenter inputs ({SEG.name})", flush=True)
    rows, t0 = [], time.time()
    n_noseg = n_nopose = 0
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        part, trial, side = str(z["part"]), str(z["trial"]), str(z["side"])
        t = trials.get(f"{part}/{trial}")
        if t is None:
            continue
        n = int(z["n"])
        # ---- windows from OUR segmenter, on each system's own channels
        try:
            # settle_observed=False means the recording stopped while the hand was still moving, so
            # the movement END is not in the video. total_movement_time is then a LOWER BOUND, not a
            # measurement, and is emitted as NaN rather than scored (24% of trials; see VERIFY.md
            # II-C). Every reaching-windowed measure is unaffected and keeps the full set.
            # NB: the 4th positional of segment_sequential is FPS, not the frame count -- an
            # earlier revision passed `n` there, which set the low-pass cutoff from a frame
            # count. Leave it at the default.
            ph_omc, fl_omc = segment_sequential(z["cup_omc"], z["wrist_omc"], z["nose_omc"],
                                                return_flags=True)
            ph_mmc, fl_mmc = segment_sequential(_ship_cup(part, trial, z["cup_mmc"]),
                                                z["wrist_mmc"], z["nose_mmc"],
                                                return_flags=True)
            tmt_observed = fl_omc["settle_observed"] and fl_mmc["settle_observed"]
        except Exception:
            n_noseg += 1; continue
        if not ph_omc or not ph_mmc:
            n_noseg += 1; continue
        # ---- poses: OMC keypoints on the video timebase, and the product MMC pose
        omc_raw = H._load_omc(part, trial, n)
        lag, _, _ = H.find_lag_best({j: t["mmc"][:, k] for k, j in enumerate(GRID)}, omc_raw, side)
        omc_pose = {j: R._shift(omc_raw[j], lag) for j in omc_raw}
        mmc_pose = S._pose_variant_cached(t, "BA", "smoothnet", ba)
        if mmc_pose is None or not all(j in omc_pose for j in GRID):
            n_nopose += 1; continue
        arm0 = str(z["arm"])
        if trunk_theta:
            dt = trunk_theta.get((part, arm0))
            if dt is not None:
                omc_pose = _apply_trunk(omc_pose, dt)
        if theta:
            other = "right" if side == "left" else "left"
            th = theta.get((part, arm0))
            if th is None:
                n_nopose += 1; continue
            omc_pose = _apply_anat12(omc_pose, side, other, th)
        cols = {"omc_omc": _measures(omc_pose, ph_omc, side, "omc"),
                "mmc_omc": _measures(mmc_pose, ph_omc, side, "mmc"),
                "mmc_mmc": _measures(mmc_pose, ph_mmc, side, "mmc")}
        arm = arm0
        for meas in set().union(*[set(c) for c in cols.values()]):
            vals = {k: cols[k].get(meas, np.nan) for k in cols}
            if meas == "total_movement_time" and not tmt_observed:
                vals = {k: np.nan for k in vals}      # right-censored: end of movement not recorded
            rows.append(dict(part=part, trial=trial, arm=arm, side=side, measure=meas,
                             settle_observed=tmt_observed, **vals))
        if (i + 1) % 40 == 0 or (i + 1) == len(files):
            el = time.time() - t0
            print(f"  [{i+1}/{len(files)}] {el:5.0f}s ({el/(i+1):.2f}s/trial)  rows {len(rows)}  "
                  f"noseg {n_noseg} nopose {n_nopose}", flush=True)
    d = pd.DataFrame(rows)
    out_csv = Path(args.out)
    d.to_csv(out_csv, index=False)
    np.savez(out_csv.with_suffix(".npz"), **{c: d[c].values for c in d.columns})
    print(f"\nPROCESSING CHECK: {d.trial.nunique()} trial-names, {len(d)} rows, "
          f"no-seg {n_noseg}, no-pose {n_nopose}, "
          f"non-finite omc {int(d.omc_omc.isna().sum())}", flush=True)
    print(f"\n{'measure':38s} {'n':>5s} {'POSE r':>9s} {'E2E r':>8s} | "
          f"{'medAE pose':>11s} {'medAE e2e':>10s}")
    for m, g in d.groupby("measure"):
        a = g.dropna(subset=["omc_omc", "mmc_omc"])
        b = g.dropna(subset=["omc_omc", "mmc_mmc"])
        if len(a) < 20:
            continue
        print(f"{m:38s} {len(a):5d} {pearsonr(a.omc_omc, a.mmc_omc)[0]:9.3f} "
              f"{pearsonr(b.omc_omc, b.mmc_mmc)[0]:8.3f} | "
              f"{(a.mmc_omc-a.omc_omc).abs().median():11.3f} "
              f"{(b.mmc_mmc-b.omc_omc).abs().median():10.3f}")
    print("DONE_OWNPHASES", flush=True)


if __name__ == "__main__":
    main()
