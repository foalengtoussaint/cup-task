"""Reproduce Unger et al. Table I + Table II + Fig 3 -- TRAJECTORY-level MMC-vs-OMC agreement.

Per trial: build the six kinematic trajectories (end-effector velocity, elbow angular velocity, elbow
extension, shoulder flexion, shoulder abduction, trunk displacement) from our BA+SmoothNet pose AND
from the optical markers, through the SAME function; align by the stacked multi-signal lag; remove the
per-trial static bias as unger2024 does; then compute per-frame RMSE, Pearson r, bias and lag.

The optical side is `_our_trajectories` applied to the C3D keypoints (2026-08-21). It used to read the
DELTA study's own stored kinematics (combined_data_with_kinematics.pkl), which made these two tables
the last place its processing was the ground truth -- and gated them on its coverage, holding them at
~705 trials while every other table moved to 794. Deriving both sides with one operator removes a
difference of definition from what should read as pose error, and is the same change already made for
the measures (scripts/score_own_phases.py).

Trials come from cache/seg_inputs_ship, so Tables I-IV are scored on one trial set.

  Table I  : median [IQR] of RMSE / r / bias / lag per trajectory, split unaffected vs affected arm.
  Table II : mean IQR of bias across patients (within-patient trial variability of the static offset).
  Fig 3    : one participant (default P07), one trial per arm, all 6 trajectories overlaid MMC vs OMC,
             with phase boundaries + movement-unit x-marks on the velocity trace.

    python scripts/paper_trajectories.py                 # tables + fig for default participant
    python scripts/paper_trajectories.py --fig-part P14 --fig-trial-r trial_1_R_unaffected
"""
from __future__ import annotations
import sys, re, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
# This script lives in paper/scripts/. REPO = the cup-task root (two up); PAPER = the paper/ folder
# (one up). Import the repo's core modules from REPO/scripts, write all outputs into PAPER.
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
ROOT = REPO
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(PAPER / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from compare_pose_omc_delta import _murphy_signals, _lp, _movement_units
from score_vs_automq import _pose_variant_cached, _elbow_series, FPS
from seg_sequential import segment_sequential
GRID = R._GRID_JOINTS
SEG = REPO / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs_ship")

# our trajectory name -> (pretty label, unit, _unused: was the AutoMQ column)
TRAJ = [
    ("eev",   "End-effector velocity", "mm/s", "hand_R_velocity"),
    ("eav",   "Elbow angular velocity", "deg/s", None),          # d/dt elbow_angle
    ("elbow", "Elbow extension",       "deg",  "elbow_angle"),
    ("flex",  "Shoulder flexion",      "deg",  "shoulder_flexion"),
    ("abd",   "Shoulder abduction",    "deg",  "shoulder_abduction"),
    ("trunk", "Trunk displacement",    "mm",   "trunk_displacement"),
]


def _resample_1d(y, n_out):
    """resample a 1-D series to n_out samples by linear interp on a normalized axis."""
    y = np.asarray(y, float)
    if len(y) < 2 or n_out < 2:
        return np.full(n_out, np.nan)
    xi = np.linspace(0, 1, len(y)); xo = np.linspace(0, 1, n_out)
    return np.interp(xo, xi, y)


def _trunk_excursion(pose, side, n):
    """Trunk displacement: 3D excursion of the trunk point from the trial's own rest position.

    POINT. The STERNUM marker where one exists -- `_load_omc` supplies it as `trunk` (C3D label
    `chest`), the same marker AutoMQ derives its trunk_displacement from and the same landmark
    unger2024 use. COCO has no sternum keypoint, so the markerless side falls back to the midpoint of
    the two shoulders. Because this is a distance from rest, a constant offset between the two points
    cancels exactly; they can differ only through trunk ROTATION.

    GEOMETRY. 3D norm, not a signed single axis. AutoMQ stores -(trunk_y - trunk_y[0]) in its own
    frame -- the sternum's travel along ONE lab axis (its -y is the C3D's +x), signed, referenced to
    a single frame. That ignores trunk motion in the other two directions, cannot be compared against
    an unsigned norm without a sign flip, and takes its reference from one possibly-noisy frame.
    """
    pt = pose.get("trunk")
    if pt is None or np.isfinite(pt).all(1).sum() < 15:
        other = "right" if side == "left" else "left"
        pt = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2.0
    fin = np.isfinite(pt).all(1)
    if fin.sum() < 15:
        return np.full(n, np.nan)
    return np.linalg.norm(pt - np.median(pt[fin][:15], 0), axis=1)


def _our_trajectories(pose, side):
    """the 6 MMC kinematic series from OUR pose (mm/s, deg/s, deg, deg, deg, mm)."""
    wr = pose[f"{side}_wrist"]
    eev = _lp(H._speed(wr))                                   # end-effector speed
    elb = _elbow_series(pose, side)                           # elbow angle (already low-passed)
    eav = np.abs(np.gradient(elb)) * FPS                      # elbow angular velocity
    try:
        sig = _murphy_signals(pose, side=side)               # flexion / abduction / trunk (our pose)
        flex, abd = _lp(sig["shoulder_flexion"]), _lp(sig["shoulder_abduction"])
    except (IndexError, ValueError):                          # no finite shoulder/hip frames this trial
        flex = abd = np.full(len(elb), np.nan)
    trunk = _trunk_excursion(pose, side, len(elb))
    return {"eev": eev, "eav": eav, "elbow": elb, "flex": flex, "abd": abd, "trunk": trunk}


def _agree(mmc, omc):
    """bias-removed per-frame agreement: (rmse, pearson_r, bias) on shared finite frames."""
    m = np.isfinite(mmc) & np.isfinite(omc)
    if m.sum() < 20:
        return None
    a, b = mmc[m], omc[m]
    bias = float(np.mean(a - b))                # static offset (added to OMC in the paper; here a-b)
    rmse = float(np.sqrt(np.mean(((a - bias) - b) ** 2)))
    r = float(np.corrcoef(a, b)[0, 1]) if (np.std(a) > 1e-9 and np.std(b) > 1e-9) else np.nan
    return rmse, r, bias


def _shift(v, lag):
    out = np.full_like(v, np.nan, dtype=float)
    if lag >= 0:
        out[lag:] = v[:len(v) - lag] if lag else v
    else:
        out[:lag] = v[-lag:]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-part", default="P07")
    ap.add_argument("--fig-trial-r", default=None, help="right/unaffected trial stem for the Fig-3 panel")
    ap.add_argument("--fig-trial-l", default=None, help="left/affected trial stem for the Fig-3 panel")
    a = ap.parse_args(argv)

    H.use_good_cams(); ba = R._ba_traj_cache()
    rows = []            # per-trial per-trajectory agreement
    fig_store = {}       # trial -> (mmc, omc, phases, side, arm) for the fig participant
    n_ok = n_skip = 0

    files = sorted(SEG.glob("*.npz"))
    lookup = {f"{t['part']}/{t['trial']}": t for t in GT.load_clean(need_reproj=False)}
    trials = [(f, lookup[k]) for f in files
              if (k := f"{str(np.load(f)['part'])}/{str(np.load(f)['trial'])}") in lookup]
    print(f"{len(trials)} trials in {SEG.name}; scoring trajectories (BA+SmoothNet)...", flush=True)
    for i, (f, t) in enumerate(trials):
        z = np.load(f)
        part = t["part"]
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            n_skip += 1; continue
        side = t["side"]; n = t["mmc"].shape[0]
        # STACKED multi-signal lag (same path as the kinematics scorer) -- see H.find_lag_best
        omc_kp = H._load_omc(part, t["trial"], n)
        lag, _, _ = H.find_lag_best({j: t["mmc"][:, k] for k, j in enumerate(GRID)}, omc_kp, side)
        # shift the KEYPOINTS onto the video timebase, then derive -- so both sides go through one
        # operator and the optical series is never a differently-defined quantity
        omc_pose = {j: R._shift(v, lag) for j, v in omc_kp.items()}

        our = _our_trajectories(pose, side)
        try:
            omc = _our_trajectories(omc_pose, side)
        except Exception:
            n_skip += 1; continue
        # phase marks for Fig 3 come from OUR segmenter on the optical channels, as in Table IV
        ph = segment_sequential(z["cup_omc"], z["wrist_omc"], z["nose_omc"])
        arm = str(z["arm"])
        for key, label, unit, _c in TRAJ:
            res = _agree(our[key], omc[key])
            if res is None:
                continue
            rmse, r, bias = res
            rows.append({"part": part, "trial": t["trial"], "arm": arm, "traj": key,
                         "label": label, "unit": unit, "rmse": rmse, "r": r,
                         "bias": bias, "lag_s": lag / FPS})
        n_ok += 1
        if part == a.fig_part:
            fig_store[t["trial"]] = (our, omc, ph, side, arm)
        if (i + 1) % 80 == 0:
            print(f"  {i+1}/{len(trials)}  ok={n_ok} skip={n_skip}", flush=True)

    df = pd.DataFrame(rows)
    outdir = PAPER; outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "trajectory_agreement.csv", index=False)
    print(f"\nPROCESSING CHECK: trials ok={n_ok}, skipped={n_skip}, rows={len(df)}, "
          f"non-finite r={int((~np.isfinite(df['r'])).sum())}", flush=True)

    _table1(df, outdir)
    _table2(df, outdir)
    _fig3(fig_store, a.fig_part, a.fig_trial_r, a.fig_trial_l, outdir)
    print("DONE", flush=True)


def _fmt(v):
    return f"{np.median(v):.2f} [{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]"


def _table1(df, outdir):
    lines = ["# Table I — Kinematic-trajectory agreement (MMC BA+SmoothNet vs OMC)", "",
             "Median [IQR] across all trials, per trajectory, split by arm. "
             "RMSE/bias in the trajectory's unit; r = Pearson; lag from the stacked multi-signal "
             "criterion. Both sides derived from keypoints by the same function.",
             "", "| Kinematic | Metric | Unaffected arm | Affected arm |", "|---|---|---|---|"]
    for key, label, unit, _c in TRAJ:
        for metric, col, u in [("r", "r", ""), ("RMSE", "rmse", f" ({unit})"),
                               ("Bias", "bias", f" ({unit})"), ("Time lag", "lag_s", " (s)")]:
            un = df[(df.traj == key) & (df.arm == "unaffected")][col].dropna().values
            af = df[(df.traj == key) & (df.arm == "affected")][col].dropna().values
            lab = label if metric == "r" else ""
            lines.append(f"| {lab} | {metric}{u} | {_fmt(un)} | {_fmt(af)} |")
    (outdir / "table1_trajectories.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nwrote {outdir/'table1_trajectories.md'}", flush=True)


def _table2(df, outdir):
    """mean, across patients, of each patient's per-arm IQR of bias (within-patient variability)."""
    lines = ["\n# Table II — Mean IQR of bias across patients", "",
             "Within-patient trial-to-trial variability of the static bias (mean over patients of "
             "each patient's IQR).", "", "| Kinematic | Unaffected arm | Affected arm |", "|---|---|---|"]
    for key, label, unit, _c in TRAJ:
        vals = {}
        for arm in ("unaffected", "affected"):
            iqrs = []
            for p, g in df[(df.traj == key) & (df.arm == arm)].groupby("part"):
                b = g["bias"].dropna().values
                if len(b) >= 3:
                    iqrs.append(np.percentile(b, 75) - np.percentile(b, 25))
            vals[arm] = np.mean(iqrs) if iqrs else np.nan
        lines.append(f"| {label} ({unit}) | {vals['unaffected']:.2f} | {vals['affected']:.2f} |")
    (outdir / "table2_bias_iqr.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nwrote {outdir/'table2_bias_iqr.md'}", flush=True)


def _fig3(store, part, tr_r, tr_l, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not store:
        print(f"[fig3] no trials for {part}; skipping figure", flush=True); return
    # pick one unaffected(R) and one affected(L) trial
    def pick(want_arm):
        cands = [k for k, v in store.items() if v[4] == want_arm]
        return cands[0] if cands else None
    tR = tr_r or pick("unaffected") or next(iter(store))
    tL = tr_l or pick("affected")
    cols = [c for c in [tR, tL] if c]
    # four of the six trajectories, at a shorter row height: the figure is illustrative and the
    # full set is quantified in Table I, so it does not need a third of a page.
    TRAJF = [t for t in TRAJ if t[0] in ("eev", "elbow", "flex", "trunk")]
    fig, axes = plt.subplots(len(TRAJF), len(cols), figsize=(5.2 * len(cols), 1.25 * len(TRAJF)),
                             squeeze=False)
    for ci, trial in enumerate(cols):
        our, omc, ph, side, arm = store[trial]
        for ri, (key, label, unit, _c) in enumerate(TRAJF):
            ax = axes[ri][ci]
            tsec = np.arange(len(our[key])) / FPS
            ax.plot(tsec, our[key], color="#c1440e", lw=1.1, label="MMC (end-to-end)")
            ax.plot(tsec, omc[key], color="#e0a200", lw=1.1, ls="-", label="OMC (two-stage)")
            # phase boundaries
            for (nm, s, e) in (ph or []):
                if nm in ("reaching", "drinking", "forward_transport", "back_transport", "returning"):
                    ax.axvline(s / FPS, color="0.75", lw=0.6, zorder=0)
            # movement-unit x-marks on the velocity trace
            if key == "eev":
                v = our[key]
                mins = np.where((np.r_[v[1:], v[-1]] > v) & (np.r_[v[0], v[:-1]] > v))[0]
                for mi in mins[::max(1, len(mins)//8)]:
                    ax.plot(mi / FPS, v[mi], "kx", ms=4)
            if ci == 0:
                ax.set_ylabel(f"{label}\n[{unit}]", fontsize=8)
            if ri == 0:
                ax.set_title(f"{part}  {arm}", fontsize=10)
            if ri == len(TRAJ) - 1:
                ax.set_xlabel("time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, loc="upper right", frameon=False)
    fig.suptitle(f"Fig 3 — Exemplary kinematic trajectories ({part}): MMC vs OMC", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = outdir / "fig3_trajectories.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
