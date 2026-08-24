"""Does a CLEANER shoulder-flexion series improve INTERJOINT COORDINATION?

interjoint = corr(shoulder_flexion, elbow_angle) over the inner-80% of reaching. Our shipped interjoint
scores rs=0.25 vs AutoMQ. Hypothesis: our flexion series is degraded by the reference frame, so a better
flexion would lift interjoint. BUT correlation is invariant to constant offset/scale, so the reference
contamination (mostly level shifts) may not touch it. Test several flexion SERIES, same elbow, same window:

  ship_hipdown : flexion vs per-trial-const hip-down axis   [shipped]
  vert_omcZ    : flexion vs OMC lab vertical (borrowed)      [the 'best' scalar variant]
  three_pt     : 3-pt elbow-shoulder-hip angle
  elbow_only   : shuffle-control -> interjoint from elbow vs elbow-shifted (sanity)
Also the CEILING: interjoint computed from OMC's OWN flexion+elbow series (what AutoMQ stores) -> the
max any MMC flexion could achieve. And OMC interjoint variance (is there headroom at all?).

vs AutoMQ interjoint_coordination scalar. OMC read-only.
    python paper/scripts/interjoint_flexion_variants.py
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
from compare_pose_omc_delta import _murphy_signals
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, _elbow_series, AUTOMQ)
GRIDJ = R._GRID_JOINTS
Z = np.array([0.0, 0.0, 1.0])


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _interjoint(flex, elb, reach):
    """corr(flex, elb) over inner-80% of reaching -- AutoMQ's recipe."""
    if not (reach and reach[1] - reach[0] >= 10):
        return np.nan
    rr0, rr1 = reach
    mg = int(0.1 * (rr1 - rr0))
    a, b = flex[rr0 + mg:rr1 - mg], elb[rr0 + mg:rr1 - mg]
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
    keys = ["ship_hipdown", "vert_omcZ", "three_pt", "omc_ceiling"]
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
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        omc = H._load_omc(p, t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, GRIDJ.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach = _win(ph, "reaching")
        elb = _elbow_series(pose, side)
        # flexion variants (series)
        try:
            flex_ship = H._lp(_murphy_signals(pose, side=side)["shoulder_flexion"])
        except (IndexError, ValueError):
            continue
        sh, el = pose[f"{side}_shoulder"], pose[f"{side}_elbow"]
        upper = sh - el
        # borrowed OMC vertical via arm+shoulder Kabsch (elbow-excluded to avoid circularity)
        FIT = [f"{other}_shoulder", "nose", "right_hip", "left_hip"]
        A = np.vstack([omc[j] for j in FIT]); B = np.vstack([t["mmc"][:, GRIDJ.index(j)] for j in FIT])
        fin = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        flex_vert = np.full(n, np.nan)
        if fin.sum() >= 30:
            Rm, _, _ = H._kabsch(A[fin], B[fin]); zc = Rm @ Z
            flex_vert = H._lp(_ang(upper, zc[None, :]))
        hip_mid = (pose["right_hip"] + pose["left_hip"]) / 2.0
        flex_3p = H._lp(_ang(el - sh, hip_mid - sh))
        # interjoint per variant
        vals = {"ship_hipdown": _interjoint(flex_ship, elb, reach),
                "vert_omcZ": _interjoint(flex_vert, elb, reach),
                "three_pt": _interjoint(flex_3p, elb, reach)}
        # CEILING: OMC's own flexion + elbow series
        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        omc_ij = np.nan
        if key is not None:
            kin = cdf.loc[key, "kinematics"]
            if {"shoulder_flexion", "elbow_angle"} <= set(kin.columns):
                fO = kin["shoulder_flexion"].to_numpy(); eO = kin["elbow_angle"].to_numpy()
                rrO = rec["phases"].get("Reaching")
                if rrO is not None:
                    r0, r1 = int(rrO[0]), int(rrO[1]); mg = int(0.1 * (r1 - r0))
                    aa, bb = fO[r0 + mg:r1 - mg], eO[r0 + mg:r1 - mg]
                    mm = np.isfinite(aa) & np.isfinite(bb)
                    if mm.sum() >= 10 and np.std(aa[mm]) > 1e-6 and np.std(bb[mm]) > 1e-6:
                        omc_ij = float(np.corrcoef(aa[mm], bb[mm])[0, 1])
        vals["omc_ceiling"] = omc_ij
        for k, v in vals.items():
            if np.isfinite(v):
                gt[k].append(g); ours[k].append(v)

    print(f"{'flexion variant -> interjoint':30}{'n':>5}{'rs vs AutoMQ':>14}{'median ours':>13}{'median GT':>12}")
    for k in keys:
        a = np.array(gt[k]); o = np.array(ours[k])
        if len(a) < 5:
            print(f"{k:30}{len(a):>5}  too few"); continue
        rs = spearmanr(a, o).correlation
        print(f"{k:30}{len(a):>5}{rs:>+14.2f}{np.median(o):>13.2f}{np.median(a):>12.2f}")
    # headroom check
    a = np.array(gt["ship_hipdown"])
    print(f"\nOMC interjoint (AutoMQ GT) distribution: median {np.median(a):.3f}  "
          f"IQR [{np.percentile(a,25):.3f}, {np.percentile(a,75):.3f}]  min {a.min():.3f}")
    print("  (if IQR ~0.03, there is a VARIANCE CEILING and no flexion fix can lift rs)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
