"""Measure the ACTUAL per-frame flexion noise (don't assume 0.5deg), for MMC vs OMC-through-our-method
vs AutoMQ-native -- then rerun the interjoint perturbation at the REAL MMC noise level.

Two things to answer:
  1. What IS the frame-to-frame flexion noise in each? Estimate as the high-frequency residual: the std
     of (flex - savgol_smooth(flex)) over the reach -- the jitter a smoother would remove. Report per
     source. (0.5deg was a guess; MMC is likely far noisier.)
  2. Given the REAL MMC noise, redo the per-trial interjoint perturbation -- does the flat-signal
     instability actually bite at MMC noise levels (even if it didn't at 0.5deg)?
Also decompose our-method's EXCESS interjoint variance vs AutoMQ into: per-frame noise vs a genuinely
different (trunk-referenced) trajectory shape.
    python paper/scripts/measure_flexion_noise.py
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from compare_pose_omc_delta import _lp, _murphy_signals
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, AUTOMQ)
GRIDJ = R._GRID_JOINTS
rng = np.random.default_rng(0)


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _hf_noise(x):
    """high-frequency residual std = jitter a light smoother removes (the real per-frame noise)."""
    x = np.asarray(x, float); m = np.isfinite(x)
    if m.sum() < 11:
        return np.nan
    xf = x.copy()
    xf[m] = savgol_filter(x[m], min(11, (m.sum() // 2) * 2 - 1), 2)
    return float(np.nanstd(x[m] - xf[m]))


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def _pearson(f, e):
    if np.std(f) < 1e-9 or np.std(e) < 1e-9:
        return np.nan
    return float(np.corrcoef(f, e)[0, 1])


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    mk_cache = {}
    noise = {"MMC_ourmethod": [], "OMC_ourmethod": [], "OMC_native": []}
    pert = []   # (fstd_mmc, noise_mmc, interjoint_mmc, pert_std_at_realnoise)
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        p = t["part"]; tn = int(m.group(1)); sd = m.group(2)
        rec = amq.get((automq_part(p), tn, sd))
        if rec is None or rec.get("phases") is None:
            continue
        rr = rec["phases"].get("Reaching")
        if rr is None:
            continue
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; n = t["mmc"].shape[0]
        # MMC flexion (our method) + reach window on MMC timeline
        try:
            flex_mmc = _lp(_murphy_signals(pose, side=side)["shoulder_flexion"])
        except (IndexError, ValueError):
            continue
        # ALSO the UNsmoothed MMC flexion, to measure raw jitter (bypass _lp)
        other = "right" if side == "left" else "left"
        arm = pose[f"{side}_elbow"] - pose[f"{side}_shoulder"]
        down = (pose["right_hip"] + pose["left_hip"]) / 2 - (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
        flex_mmc_raw = _ang(arm, down)
        omc = H._load_omc(p, t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, GRIDJ.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        reach = _win(ph, "reaching") if ph else None
        if not reach:
            continue
        r0, r1 = reach
        noise["MMC_ourmethod"].append(_hf_noise(flex_mmc_raw[r0:r1]))
        # elbow (mmc) for perturbation
        u = pose[f"{side}_shoulder"] - pose[f"{side}_elbow"]; v = pose[f"{side}_wrist"] - pose[f"{side}_elbow"]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        elb_mmc = _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))
        f = flex_mmc[r0:r1]; e = elb_mmc[r0:r1]
        mm = np.isfinite(f) & np.isfinite(e)
        # OMC through our method + native
        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        if key is not None:
            mkk = cdf.loc[key, "markers"]; kin = cdf.loc[key, "kinematics"]
            sh = _marker(mkk, f"shoulder_{sd}"); el = _marker(mkk, f"elbow_{sd}")
            shO = _marker(mkk, "shoulder_L" if sd == "R" else "shoulder_R")
            hR = _marker(mkk, "hip_R"); hL = _marker(mkk, "hip_L")
            rr0, rr1 = int(rr[0]), int(rr[1])
            if all(x is not None for x in (sh, el, shO, hR, hL)):
                T = min(map(len, (sh, el, shO, hR, hL))); a2, b2 = min(rr0, T), min(rr1, T)
                fo = _ang((el - sh), (hR + hL) / 2 - (sh + shO) / 2)[a2:b2]
                noise["OMC_ourmethod"].append(_hf_noise(fo))
            if "shoulder_flexion" in kin.columns:
                noise["OMC_native"].append(_hf_noise(kin["shoulder_flexion"].to_numpy()[rr0:min(rr1, len(kin))]))
        # perturbation at REAL mmc noise
        if mm.sum() >= 12 and np.std(e[mm]) > 1e-6:
            realn = noise["MMC_ourmethod"][-1]
            if np.isfinite(realn) and realn > 0:
                r_raw = _pearson(f[mm], e[mm])
                rs = [_pearson(f[mm] + rng.normal(0, realn, size=mm.sum()), e[mm]) for _ in range(40)]
                rs = np.array([x for x in rs if np.isfinite(x)])
                pert.append((float(np.nanstd(f[mm])), realn, r_raw, float(np.std(rs)) if len(rs) > 5 else np.nan))

    print("=== ACTUAL per-frame flexion NOISE (high-freq residual std, deg) ===")
    for k, v in noise.items():
        v = np.array([x for x in v if np.isfinite(x)])
        print(f"  {k:16} median {np.median(v):.2f}  IQR [{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]  n={len(v)}")
    print("  (0.5 deg was my GUESS; compare to reality above)")

    pert = np.array([r for r in pert if all(np.isfinite(r))])
    fstd, realn, r_raw, pstd = pert.T
    print(f"\n=== interjoint perturbation at REAL MMC noise (not 0.5deg) ===  n={len(pert)}")
    lo = fstd <= np.percentile(fstd, 25)
    print(f"  FLAT trials (bottom-quartile flex std): median real-noise = {np.median(realn[lo]):.2f} deg, "
          f"per-trial interjoint spread = {np.median(pstd[lo]):.3f}")
    print(f"  normal trials:                          median real-noise = {np.median(realn[~lo]):.2f} deg, "
          f"per-trial interjoint spread = {np.median(pstd[~lo]):.3f}")
    print(f"  spearman(flex_std, pert_spread) = {spearmanr(fstd, pstd).correlation:+.2f}")
    print(f"  -> if FLAT pert_spread is now LARGE (e.g. >0.1), the instability DOES bite at real MMC noise")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
