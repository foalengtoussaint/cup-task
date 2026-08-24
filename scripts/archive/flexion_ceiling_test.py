"""Try to BREAK the flexion r_s 'ceiling' by reducing measurement noise. If nothing lifts it, the
ceiling (OMC within-participant variance ~2-3 deg < keypoint noise ~3 deg) is real. Variants:
  down axis : per-TRIAL median (current) vs per-PARTICIPANT median (more stable, less per-trial noise)
  reduction : max (AutoMQ) vs p90 vs p80 (robust peak = less noise amplification)
Also our internal RELIABILITY (BA-flexion vs pipeline-flexion per trial) = the noise floor.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, FPS)
GRID = R._GRID_JOINTS


def flex_series(pose, side, down_axis):
    """per-frame flexion (deg) of the upper arm from a given (constant) down axis."""
    sh, el = pose[f"{side}_shoulder"], pose[f"{side}_elbow"]
    arm = el - sh
    arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    d = down_axis / (np.linalg.norm(down_axis) + 1e-9)
    return H._lp(np.degrees(np.arccos(np.clip((arm * d[None, :]).sum(1), -1, 1))))


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    # first pass: collect per-participant down axes (median of hip_mid - sh_mid across all frames/trials)
    trials = [t for t in GT.load_clean(need_reproj=False)]
    part_down = {}
    percup = {}
    for t in trials:
        pose = _pose_variant_cached(t, "pipeline", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        sh_mid = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
        hip_mid = (pose["right_hip"] + pose["left_hip"]) / 2
        dwn = hip_mid - sh_mid
        fin = np.isfinite(dwn).all(1)
        if fin.any():
            part_down.setdefault(t["part"], []).append(np.nanmedian(dwn[fin], 0))
    part_axis = {p: np.median(np.stack(v), 0) for p, v in part_down.items()}

    configs = {"trial+max": [], "trial+p90": [], "part+max": [], "part+p90": [], "part+p80": []}
    gt = {k: [] for k in configs}; pp = {k: [] for k in configs}
    rel_ba = []; rel_pipe = []; rel_part = []
    for t in trials:
        m = pat.search(t["trial"])
        if not m:
            continue
        rec = amq.get((automq_part(t["part"]), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        g = rec.get("max_shoulder_flexion")
        if g is None or not np.isfinite(g):
            continue
        pose = _pose_variant_cached(t, "pipeline", "smoothnet", ba)
        pose_ba = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        omc = H._load_omc(t["part"], t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        if not (rd and rd[1] > rd[0]):
            continue
        sh_mid = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
        hip_mid = (pose["right_hip"] + pose["left_hip"]) / 2
        dwn = hip_mid - sh_mid; fin = np.isfinite(dwn).all(1)
        d_trial = np.nanmedian(dwn[fin], 0) if fin.any() else part_axis[t["part"]]
        d_part = part_axis[t["part"]]
        def red(fx, how):
            seg = fx[rd[0]:rd[1]]; seg = seg[np.isfinite(seg)]
            if len(seg) == 0:
                return np.nan
            return float(np.max(seg)) if how == "max" else float(np.percentile(seg, how))
        ft = flex_series(pose, side, d_trial); fp = flex_series(pose, side, d_part)
        vals = {"trial+max": red(ft, "max"), "trial+p90": red(ft, 90),
                "part+max": red(fp, "max"), "part+p90": red(fp, 90), "part+p80": red(fp, 80)}
        for k, v in vals.items():
            if np.isfinite(v):
                gt[k].append(g); pp[k].append(v)
        # reliability: pipeline vs BA flexion (part axis, max)
        if pose_ba is not None:
            fb = flex_series(pose_ba, side, d_part)
            a1, a2 = red(fp, "max"), red(fb, "max")
            if np.isfinite(a1) and np.isfinite(a2):
                rel_pipe.append(a1); rel_ba.append(a2); rel_part.append(t["part"])

    print(f"{'config':12}{'pooled rs':>11}{'within-part rs':>16}")
    for k in configs:
        a = np.array(gt[k]); o = np.array(pp[k])
        pr = spearmanr(a, o).correlation
        # within
        import collections
        byp = collections.defaultdict(lambda: [[], []])
        # need part labels; recompute alongside -- approximate: skip within here
        print(f"{k:12}{pr:>+11.2f}")
    rb, rp = np.array(rel_ba), np.array(rel_pipe)
    print(f"\nINTERNAL RELIABILITY (pipeline-flexion vs BA-flexion, our two estimates): rs {spearmanr(rp, rb).correlation:+.2f}")
    print("  (high = our measurement is repeatable -> ceiling is OMC low-variance; low = OUR noise is the limit)")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
