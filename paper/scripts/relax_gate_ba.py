"""Gate relaxation at the SHIPPED level: BA + SmoothNet on the forced (no-gate) pose vs the gated pose.

The raw-DLT comparison (relax_gate_all.py) isolated the gate but is below the shipped BA+SmoothNet level.
This runs the FULL pipeline on both:
  GATED  = _pose_variant(t, "BA", "smoothnet")           # shipped: cached BA on gated mmc + SmoothNet
  FORCED = refine_trial_ba(forced-DLT init) + SmoothNet  # no reproj gate, then same BA + SmoothNet
Then the 11 measures on both vs AutoMQ, on common trials + all (incl newly-added P19).
    python paper/scripts/relax_gate_ba.py    (BACKGROUND, GPU, ~826 trials BA+SmoothNet)
"""
from __future__ import annotations
import sys, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
import ba_refine
from compare_pose_omc_delta import JOINTS
from score_vs_automq import load_automq, automq_phases_to_video, automq_part
# reuse forced triangulation + the 11-measure operator from the raw-DLT script
from relax_gate_all import _forced, _measures, _posed, JLIST


def _forced_ba_sn(t, forced_pose_dict):
    """BA-refine the forced pose (with the trial's reproj sidecar) then SmoothNet. Returns pose-dict or None."""
    arr = np.stack([forced_pose_dict[j] for j in JLIST], 1).astype(np.float32)
    tt = dict(t); tt["mmc"] = arr; tt["valid"] = np.isfinite(arr).all(2)
    try:
        Xba, _ = ba_refine.refine_trial_ba(tt, lam_bone=0.0, huber_px=20.0, iters=40, trial_guard_mm=150.0)
    except Exception:
        return None
    if not np.isfinite(Xba).any() or np.nanmax(np.abs(Xba)) > 1e5:      # BA blow-up -> unavailable
        return None
    return {j: R._smooth_joint(Xba[:, i]) for i, j in enumerate(JLIST)}


def main():
    H.use_good_cams(); amq = load_automq(); ba = R._ba_traj_cache()
    pat = re.compile(r"trial_(\d+)_([RL])_")
    trials = GT.load_clean(need_reproj=True)
    print(f"{len(trials)} trials; BA+SmoothNet on gated vs forced...", flush=True)
    rows = []; t0 = time.time(); nba_fail = 0
    for i, t in enumerate(trials):
        m = pat.search(t["trial"])
        if not m:
            continue
        p = t["part"]; rec = amq.get((automq_part(p), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"; n = t["mmc"].shape[0]
        omc = H._load_omc(p, t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, JLIST.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        # GATED BA+SmoothNet (cached BA)
        gp = R._pose_variant(t, "BA", "smoothnet", ba)
        gm = _measures(gp, side, other, ph) if gp else {k: np.nan for k in
              ("flex", "abd", "elbow", "peakvel", "ij", "eav", "ttp", "ttfp", "munits", "tmt", "trunk")}
        # FORCED BA+SmoothNet
        fr = _forced(p, t["trial"])
        if fr:
            fp = _forced_ba_sn(t, fr[0])
            if fp is None:
                nba_fail += 1
                fp = {j: R._smooth_joint(fr[0][j]) for j in JLIST}     # fallback: forced DLT + SmoothNet
            fm = _measures(fp, side, other, ph)
        else:
            fm = {k: np.nan for k in gm}
        rows.append({"part": p, "trial": t["trial"],
                     **{f"g_{k}": v for k, v in gm.items()}, **{f"f_{k}": v for k, v in fm.items()},
                     "a_flex": rec.get("max_shoulder_flexion"), "a_abd": rec.get("max_shoulder_abduction"),
                     "a_elbow": rec.get("max_elbow_angle"), "a_peakvel": rec.get("peak_velocity"),
                     "a_ij": rec.get("interjoint_coordination"), "a_eav": rec.get("peak_elbow_angular_velocity"),
                     "a_ttp": rec.get("time_to_peak_velocity"), "a_ttfp": rec.get("time_to_first_peak_velocity"),
                     "a_munits": rec.get("number_of_movement_units"), "a_tmt": rec.get("total_movement_time"),
                     "a_trunk": rec.get("max_trunk_displacement")})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(trials)}  elapsed {time.time()-t0:.0f}s  rows={len(rows)}  ba_fail={nba_fail}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(PAPER / "relax_gate_ba.csv", index=False)
    amap = {"peakvel": "a_peakvel", "eav": "a_eav", "ttp": "a_ttp", "ttfp": "a_ttfp",
            "munits": "a_munits", "tmt": "a_tmt", "ij": "a_ij", "trunk": "a_trunk",
            "flex": "a_flex", "elbow": "a_elbow", "abd": "a_abd"}
    print(f"\nPROCESSING CHECK: rows={len(d)}  forced-BA-blowup fallbacks={nba_fail}", flush=True)
    print(f"\n{'measure':10}{'GATED BA+SN rs (n)':>20}{'FORCED common (n)':>21}{'FORCED all (n)':>18}{'+trials':>9}")
    for k, acol in amap.items():
        g, f, a = d[f"g_{k}"], d[f"f_{k}"], d[acol]
        gmask = np.isfinite(g) & np.isfinite(a); common = gmask & np.isfinite(f); fmask = np.isfinite(f) & np.isfinite(a)
        added = int((~np.isfinite(g) & np.isfinite(f) & np.isfinite(a)).sum())
        rg = spearmanr(g[gmask], a[gmask]).correlation if gmask.sum() >= 5 else np.nan
        rfc = spearmanr(f[common], a[common]).correlation if common.sum() >= 5 else np.nan
        rfa = spearmanr(f[fmask], a[fmask]).correlation if fmask.sum() >= 5 else np.nan
        print(f"{k:10}{f'{rg:+.2f} ({gmask.sum()})':>20}{f'{rfc:+.2f} ({common.sum()})':>21}{f'{rfa:+.2f} ({fmask.sum()})':>18}{added:>9}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
