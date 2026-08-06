"""peak_velocity filter sweep: reduce the -27 mm/s under-read WITHOUT hurting r or median|err|.

The bias is a filter asymmetry: we low-pass POSITION at 4Hz then differentiate; AutoMQ low-passes the
VELOCITY at 6Hz. This sweeps cutoff + operator-order on OUR BA+SmoothNet pose (wrist), max over the
reaching window, vs AutoMQ's wrist peak_velocity. Reports bias / pearson r / median|err| per config so
we pick a config that cuts bias but keeps r>=current and |err| not worse. NO OMC in the computation.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from cup_task.score import _smoothed_xyz, _hand_speed_mmps, _butter_lowpass
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, FPS)
GRID = R._GRID_JOINTS


def peak_pos(wr, w, hz):        # low-pass POSITION @hz, then diff (our operator, varying cutoff)
    sp = _hand_speed_mmps(_smoothed_xyz(wr, FPS, hz, 2), FPS)
    seg = sp[w[0]:w[1]]; return float(np.max(seg[np.isfinite(seg)])) if np.isfinite(seg).any() else np.nan

def peak_vel(wr, w, hz):        # diff RAW, then low-pass the SPEED @hz (AutoMQ operator order)
    raw = _hand_speed_mmps(wr, FPS)
    sp = _butter_lowpass(raw, FPS, hz, 2)
    seg = sp[w[0]:w[1]]; return float(np.max(seg[np.isfinite(seg)])) if np.isfinite(seg).any() else np.nan


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    trials = GT.load_clean(need_reproj=False)
    pat = re.compile(r"trial_(\d+)_([LR])_")
    CONFIGS = {"pos_4hz(cur)": ("pos", 4.0), "pos_5hz": ("pos", 5.0), "pos_6hz": ("pos", 6.0),
               "pos_8hz": ("pos", 8.0), "vel_6hz(AMQ)": ("vel", 6.0), "vel_8hz": ("vel", 8.0),
               "vel_10hz": ("vel", 10.0)}
    acc = {c: {"gt": [], "ours": []} for c in CONFIGS}
    for ti, t in enumerate(trials):
        m = pat.search(t["trial"])
        if not m: continue
        rec = amq.get((automq_part(t["part"]), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None: continue
        gt = rec.get("peak_velocity_wrist")
        if gt is None or not np.isfinite(gt): continue
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None: continue
        n = t["mmc"].shape[0]; side = t["side"]; wrj = f"{side}_wrist"
        omc = H._load_omc(t["part"], t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wrj)], omc[wrj])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph: continue
        reach = _win(ph, "reaching")
        if not (reach and reach[1] > reach[0]): continue
        wr = pose[wrj]
        for c, (op, hz) in CONFIGS.items():
            v = peak_pos(wr, reach, hz) if op == "pos" else peak_vel(wr, reach, hz)
            if np.isfinite(v): acc[c]["gt"].append(gt); acc[c]["ours"].append(v)
        if (ti + 1) % 150 == 0: print(f"  [{ti+1}/{len(trials)}]", flush=True)

    print(f"\n{'config':16}{'n':>5}{'bias':>8}{'med|err|':>10}{'pearson r':>11}")
    for c in CONFIGS:
        a = np.array(acc[c]["gt"]); o = np.array(acc[c]["ours"])
        mask = np.isfinite(a) & np.isfinite(o); a, o = a[mask], o[mask]
        r = np.corrcoef(a, o)[0, 1] if len(a) > 5 else np.nan
        print(f"{c:16}{len(a):>5}{np.median(o-a):+8.1f}{np.median(np.abs(o-a)):10.1f}{r:+11.2f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
