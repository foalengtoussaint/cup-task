"""Flexion down-axis bake-off: which trunk-down reference best matches AutoMQ shoulder_flexion?

The flexion formula (angle of upper arm from the trunk-down axis) is verified correct on clean
markers (corr 1.00), but on OUR pose it's weak (trial-corr 0.16) because the participant is SEATED
and the HIPS are occluded, so the per-frame hip-based down axis is noisy. This tests down-axis
variants that reduce reliance on per-frame hips -- ALL from our pose / rig only, NO OMC:

  perframe   : down = hip_mid(t) - shoulder_mid(t)          [current baseline]
  medhip     : down = median(hip_mid) - shoulder_mid(t)     [seated hips ~constant -> denoise]
  fullconst  : down = median(hip_mid - shoulder_mid)        [constant trunk axis over the trial]
  lphip      : down = lowpass(hip_mid) - shoulder_mid(t)    [smoothed hips]

Reports, per variant, trial-level corr / bias / median|err| of max_shoulder_flexion vs AutoMQ.
"""
from __future__ import annotations
import sys, re, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, FPS)
GRID = R._GRID_JOINTS


def _down_variants(shmid, hipmid):
    """dict name -> down-axis (T,3), from shoulder/hip midpoints (our pose)."""
    fin = np.isfinite(hipmid).all(1)
    med_hip = np.nanmedian(hipmid[fin], 0) if fin.any() else np.array([0, 0, 1.0])
    lp_hip = np.stack([H._lp(hipmid[:, k]) for k in range(3)], 1)
    const_axis = np.nanmedian(hipmid[fin] - shmid[fin], 0) if fin.any() else np.array([0, 0, 1.0])
    T = len(shmid)
    return {
        "perframe":  hipmid - shmid,
        "medhip":    med_hip[None, :] - shmid,
        "fullconst": np.broadcast_to(const_axis, (T, 3)),
        "lphip":     lp_hip - shmid,
    }


def _flex_max(arm, down, w):
    a = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    d = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    fx = H._lp(np.degrees(np.arccos(np.clip((a * d).sum(1), -1, 1))))
    if not (w and w[1] > w[0]):
        return np.nan
    seg = fx[w[0]:w[1]]
    return float(np.max(seg[np.isfinite(seg)])) if np.isfinite(seg).any() else np.nan


def main():
    H.use_good_cams()
    ba = R._ba_traj_cache()
    amq = load_automq()
    trials = GT.load_clean(need_reproj=False)
    pat = re.compile(r"trial_(\d+)_([LR])_")
    VARIANTS = ["perframe", "medhip", "fullconst", "lphip"]
    acc = {v: {"gt": [], "ours": []} for v in VARIANTS}
    n_ok = 0
    t0 = time.time()
    for ti, t in enumerate(trials):
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        gt = rec.get("max_shoulder_flexion")
        if gt is None or not np.isfinite(gt):
            continue
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        n = t["mmc"].shape[0]
        omc = H._load_omc(part, trial, n)
        wr = f"{side}_wrist"
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        shmid = (pose[f"{side}_shoulder"] + pose[f"{'right' if side=='left' else 'left'}_shoulder"]) / 2
        hipmid = (pose["right_hip"] + pose["left_hip"]) / 2
        arm = pose[f"{side}_elbow"] - pose[f"{side}_shoulder"]
        downs = _down_variants(shmid, hipmid)
        got = False
        for v in VARIANTS:
            fv = _flex_max(arm, downs[v], rd)
            if np.isfinite(fv):
                acc[v]["gt"].append(gt); acc[v]["ours"].append(fv); got = True
        n_ok += got
        if (ti + 1) % 100 == 0:
            print(f"  [{ti+1}/{len(trials)}] {time.time()-t0:.0f}s  scored {n_ok}", flush=True)

    print(f"\n=== flexion down-axis bake-off (BA+SmoothNet, {n_ok} trials) vs AutoMQ ===")
    print(f"{'variant':12} {'n':>4} {'corr':>6} {'bias':>7} {'med|err|':>9}")
    for v in VARIANTS:
        a = np.array(acc[v]["gt"]); o = np.array(acc[v]["ours"])
        mask = np.isfinite(a) & np.isfinite(o)
        a, o = a[mask], o[mask]
        corr = np.corrcoef(a, o)[0, 1] if len(a) > 5 else np.nan
        print(f"{v:12} {len(a):>4} {corr:+6.2f} {np.median(o-a):+7.1f} {np.median(np.abs(o-a)):9.1f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
