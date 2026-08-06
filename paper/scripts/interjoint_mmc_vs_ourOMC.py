"""Does MMC interjoint agree better with OUR-METHOD-on-OMC than with AutoMQ's own interjoint?

Parallel to the flexion/abduction scalar tests: AutoMQ's interjoint uses ITS flexion (world-vertical).
Ours uses the hip-down trunk-referenced flexion. So compare MMC interjoint against TWO ground truths,
matched per trial:
  vs AutoMQ_native : interjoint from OMC's OWN stored flexion+elbow series   [world-vertical def]
  vs our_on_OMC    : interjoint from OUR method (hip-down flexion + 3pt elbow) on OMC's CLEAN markers
                     [same DEFINITION as our MMC, just clean inputs]
If MMC agrees better with our_on_OMC, then like the scalars, the interjoint 'disagreement' with AutoMQ
is largely DEFINITIONAL (their world-vertical flexion vs our trunk-referenced one), not tracking.
    python paper/scripts/interjoint_mmc_vs_ourOMC.py
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
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, _elbow_series, AUTOMQ)
GRIDJ = R._GRID_JOINTS


def _elbow_from_pts(sh, el, wr):
    u, v = sh - el, wr - el
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _our_flexion_from_pts(sh, el, shO, hipR, hipL):
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
    mmc_ij, tgt_native, tgt_ourOMC = [], [], []
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
        # ---- MMC interjoint (our method, our pose) ----
        try:
            flex_mmc = _lp(_murphy_signals(pose, side=side)["shoulder_flexion"])
        except (IndexError, ValueError):
            continue
        elb_mmc = _elbow_series(pose, side)
        omc = H._load_omc(p, t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, GRIDJ.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        reach = _win(ph, "reaching") if ph else None
        if not reach:
            continue
        v_mmc = _interj(flex_mmc, elb_mmc, reach[0], reach[1])
        # ---- two ground truths from OMC ----
        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        if key is None:
            continue
        r0, r1 = int(rr[0]), int(rr[1])
        v_native = v_our = np.nan
        kin = cdf.loc[key, "kinematics"]
        if {"shoulder_flexion", "elbow_angle"} <= set(kin.columns):
            fO = kin["shoulder_flexion"].to_numpy(); eO = kin["elbow_angle"].to_numpy()
            v_native = _interj(fO, eO, r0, min(r1, len(fO)))
        mk = cdf.loc[key, "markers"]
        sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}"); wr = _marker(mk, f"hand_{sd}")
        shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
        hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
        if all(x is not None for x in (sh, el, wr, shO, hR, hL)):
            T = min(map(len, (sh, el, wr, shO, hR, hL)))
            sh, el, wr, shO, hR, hL = (a[:T] for a in (sh, el, wr, shO, hR, hL))
            flex_o = _our_flexion_from_pts(sh, el, shO, hR, hL)
            elb_o = _elbow_from_pts(sh, el, wr)
            v_our = _interj(flex_o, elb_o, min(r0, T), min(r1, T))
        if np.isfinite(v_mmc) and np.isfinite(v_native) and np.isfinite(v_our):
            mmc_ij.append(v_mmc); tgt_native.append(v_native); tgt_ourOMC.append(v_our)

    mmc_ij = np.array(mmc_ij); tgt_native = np.array(tgt_native); tgt_ourOMC = np.array(tgt_ourOMC)
    n = len(mmc_ij)
    print(f"\nn = {n} matched trials  (all three interjoint values finite)\n")
    print(f"{'MMC interjoint vs ...':38}{'rs':>7}{'bias':>8}{'med|err|':>10}")
    for lab, tgt in [("AutoMQ native (world-vertical flexion)", tgt_native),
                     ("OUR method on OMC markers (trunk-ref)", tgt_ourOMC)]:
        rs = spearmanr(tgt, mmc_ij).correlation
        print(f"{lab:38}{rs:>+7.2f}{np.median(mmc_ij - tgt):>+8.2f}{np.median(np.abs(mmc_ij - tgt)):>10.2f}")
    print(f"\n  (the two GROUND TRUTHS agree with each other: rs={spearmanr(tgt_native, tgt_ourOMC).correlation:+.2f})")
    print("\n  If MMC agrees BETTER with our-method-on-OMC, the interjoint gap to AutoMQ is DEFINITIONAL")
    print("  (their world-vertical flexion vs our trunk-referenced one), same story as the scalar angles.")

    pd.DataFrame({"mmc": mmc_ij, "omc_native": tgt_native, "omc_ourmethod": tgt_ourOMC}).to_csv(
        PAPER / "interjoint_mmc_vs_ourOMC.csv", index=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
