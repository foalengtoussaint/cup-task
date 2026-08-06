"""Separate METHOD from TRACKING for interjoint: run OUR flexion method on OMC's CLEAN markers.

Prior test compared interjoint from OUR pose (bad) vs OMC's OWN stored series (0.97). That does NOT
isolate whether OUR METHOD (hip-down axis flexion) is the problem or our KEYPOINT NOISE is. This does:
feed OMC's sub-mm shoulder/elbow/hip markers THROUGH OUR flexion construction, compute interjoint, and
compare to AutoMQ's interjoint scalar.

  our_method_on_MMC : our flexion (hip-down axis) from OUR pose, elbow from OUR pose      [= shipped]
  our_method_on_OMC : our flexion (hip-down axis) from OMC MARKERS, elbow from OMC markers [METHOD only]
  omc_native        : AutoMQ's own stored flexion+elbow series                            [CEILING]

If our_method_on_OMC ~ omc_native (0.97): our METHOD is fine, the limiter is keypoint NOISE (tracking).
If our_method_on_OMC is LOW like MMC: our METHOD (hip-down flexion) intrinsically breaks interjoint.
All read-only OMC; nothing changes scoring.
    python paper/scripts/interjoint_ourmethod_on_omc.py
"""
from __future__ import annotations
import sys, re
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
from compare_pose_omc_delta import _murphy_signals, _lp
from score_vs_automq import (load_automq, _pose_variant_cached, automq_part, _elbow_series, AUTOMQ)
GRIDJ = R._GRID_JOINTS


def _elbow_from_pts(sh, el, wr):
    u, v = sh - el, wr - el
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _our_flexion_from_pts(sh, el, shO, hipR, hipL):
    """OUR flexion recipe applied to arbitrary points: arm(el-sh) vs per-trial-const hip-down axis."""
    trunk_mid = (sh + shO) / 2.0
    hip_mid = (hipR + hipL) / 2.0
    down = hip_mid - trunk_mid
    down = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    fin = np.isfinite(down).all(1)
    down_c = np.nanmedian(down[fin], 0) if fin.any() else down[0]
    down_c = down_c / (np.linalg.norm(down_c) + 1e-9)
    arm = el - sh
    arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    return _lp(np.degrees(np.arccos(np.clip((arm * down_c[None, :]).sum(1), -1, 1))))


def _interj(flex, elb, r0, r1):
    if r1 - r0 < 10:
        return np.nan
    mg = int(0.1 * (r1 - r0))
    a, b = flex[r0 + mg:r1 - mg], elb[r0 + mg:r1 - mg]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-6 or np.std(b[m]) < 1e-6:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    mk_cache = {}
    keys = ["our_on_MMC", "our_on_OMC", "omc_native"]
    gt = {k: [] for k in keys}; ours = {k: [] for k in keys}
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        p = t["part"]; tn = int(m.group(1)); sd = m.group(2)
        rec = amq.get((automq_part(p), tn, sd))
        if rec is None or rec.get("phases") is None:
            continue
        g = rec.get("interjoint_coordination")
        if g is None or not np.isfinite(g):
            continue
        rr = rec["phases"].get("Reaching")
        if rr is None:
            continue
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        # ---- our method on OUR pose (shipped) ----
        try:
            flex_mmc = _lp(_murphy_signals(pose, side=side)["shoulder_flexion"])
        except (IndexError, ValueError):
            continue
        elb_mmc = _elbow_series(pose, side)
        # reach window on the MMC timeline: reuse AutoMQ 100Hz frames scaled to 60Hz (no lag needed for
        # a within-signal correlation; both flex_mmc & elb_mmc share the SAME timeline)
        from score_vs_automq import automq_phases_to_video, _win
        omc = H._load_omc(p, t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, GRIDJ.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        reach = _win(ph, "reaching") if ph else None
        v_mmc = _interj(flex_mmc, elb_mmc, reach[0], reach[1]) if reach else np.nan

        # ---- our method on OMC markers (100Hz, native reach frames) ----
        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        v_omc = v_native = np.nan
        if key is not None:
            mk = cdf.loc[key, "markers"]
            sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}"); wr = _marker(mk, f"hand_{sd}")
            shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
            hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
            r0, r1 = int(rr[0]), int(rr[1])
            if all(x is not None for x in (sh, el, wr, shO, hR, hL)):
                T = min(map(len, (sh, el, wr, shO, hR, hL)))
                sh, el, wr, shO, hR, hL = (a[:T] for a in (sh, el, wr, shO, hR, hL))
                flex_o = _our_flexion_from_pts(sh, el, shO, hR, hL)     # OUR method, OMC points
                elb_o = _elbow_from_pts(sh, el, wr)
                v_omc = _interj(flex_o, elb_o, min(r0, T), min(r1, T))
            kin = cdf.loc[key, "kinematics"]
            if {"shoulder_flexion", "elbow_angle"} <= set(kin.columns):
                fO = kin["shoulder_flexion"].to_numpy(); eO = kin["elbow_angle"].to_numpy()
                v_native = _interj(fO, eO, r0, min(r1, len(fO)))
        for k, v in [("our_on_MMC", v_mmc), ("our_on_OMC", v_omc), ("omc_native", v_native)]:
            if np.isfinite(v):
                gt[k].append(g); ours[k].append(v)

    print(f"{'interjoint from ...':22}{'n':>5}{'rs vs AutoMQ':>14}{'median ours':>13}")
    for k in keys:
        a = np.array(gt[k]); o = np.array(ours[k])
        if len(a) < 5:
            print(f"{k:22}{len(a):>5}  too few"); continue
        print(f"{k:22}{len(a):>5}{spearmanr(a, o).correlation:>+14.2f}{np.median(o):>13.2f}")
    print("\nINTERPRETATION:")
    print("  our_on_OMC ~ omc_native (~0.9) -> our METHOD is fine; MMC's low score = keypoint NOISE (tracking)")
    print("  our_on_OMC LOW like our_on_MMC -> our hip-down flexion METHOD itself breaks interjoint")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
