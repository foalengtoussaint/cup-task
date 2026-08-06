"""Confirm the interjoint 'extra spread' from our trunk-referenced method is TAIL INSTABILITY, not signal.

Prediction: interjoint = corr(flexion, elbow) is numerically unstable when the flexion series has low
VARIANCE over the reach (near-flat -> a few noisy frames flip corr from +1 to large negative). If the
divergence between our-method-on-OMC and AutoMQ-native interjoint concentrates on LOW-flexion-variance
trials, the extra spread is an artifact, not real coordination structure.

Per trial (OMC markers only): compute both interjoints + our flexion's std over the reach window, then
check whether |our - native| grows as flexion-variance shrinks.
    python paper/scripts/interjoint_instability_check.py
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
from compare_pose_omc_delta import _lp
from score_vs_automq import load_automq, automq_part, AUTOMQ


def _our_flexion(sh, el, shO, hR, hL):
    down = (hR + hL) / 2.0 - (sh + shO) / 2.0
    down = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    fin = np.isfinite(down).all(1)
    dc = np.nanmedian(down[fin], 0) if fin.any() else down[0]
    dc = dc / (np.linalg.norm(dc) + 1e-9)
    arm = el - sh; arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    return _lp(np.degrees(np.arccos(np.clip((arm * dc[None, :]).sum(1), -1, 1))))


def _elbow(sh, el, wr):
    u, v = sh - el, wr - el
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _ij(flex, elb, r0, r1):
    if r1 - r0 < 10:
        return np.nan, np.nan
    mg = int(0.1 * (r1 - r0)); a, b = flex[r0 + mg:r1 - mg], elb[r0 + mg:r1 - mg]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan, np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1]), float(np.std(a[m]))   # (interjoint, flexion std)


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def main():
    amq = load_automq()
    rows = []
    for p in ["P07", "P08", "P10", "P12", "P13", "P14", "P15", "P17", "P19", "P25"]:
        try:
            cdf = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        except Exception:
            continue
        for fk in cdf.index:
            tn, sd = int(fk[1]), fk[2]
            rec = amq.get((automq_part(p), tn, sd))
            if rec is None or rec.get("phases") is None:
                continue
            rr = rec["phases"].get("Reaching")
            if rr is None:
                continue
            r0, r1 = int(rr[0]), int(rr[1])
            mk = cdf.loc[fk, "markers"]; kin = cdf.loc[fk, "kinematics"]
            sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}"); wr = _marker(mk, f"hand_{sd}")
            shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
            hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
            if any(x is None for x in (sh, el, wr, shO, hR, hL)):
                continue
            if not {"shoulder_flexion", "elbow_angle"} <= set(kin.columns):
                continue
            T = min(map(len, (sh, el, wr, shO, hR, hL)))
            sh, el, wr, shO, hR, hL = (a[:T] for a in (sh, el, wr, shO, hR, hL))
            rr1 = min(r1, T)
            our_ij, our_std = _ij(_our_flexion(sh, el, shO, hR, hL), _elbow(sh, el, wr), min(r0, T), rr1)
            fO = kin["shoulder_flexion"].to_numpy(); eO = kin["elbow_angle"].to_numpy()
            nat_ij, nat_std = _ij(fO, eO, r0, min(r1, len(fO)))
            if np.isfinite(our_ij) and np.isfinite(nat_ij):
                rows.append({"part": p, "our_ij": our_ij, "nat_ij": nat_ij,
                             "our_flex_std": our_std, "nat_flex_std": nat_std,
                             "disagree": abs(our_ij - nat_ij)})
    d = pd.DataFrame(rows)
    print(f"n={len(d)}")
    # split by our flexion variance over the reach
    lo = d[d.our_flex_std < d.our_flex_std.median()]
    hi = d[d.our_flex_std >= d.our_flex_std.median()]
    print(f"\nflexion std over reach: median {d.our_flex_std.median():.2f} deg")
    print(f"{'group':22}{'n':>5}{'median disagree':>17}{'our_ij median':>15}{'frac our_ij<0.5':>17}")
    for lab, g in [("LOW flexion-variance", lo), ("HIGH flexion-variance", hi)]:
        print(f"{lab:22}{len(g):>5}{g.disagree.median():>17.3f}{g.our_ij.median():>15.2f}{(g.our_ij<0.5).mean():>17.2f}")
    r = spearmanr(d.our_flex_std, d.disagree).correlation
    print(f"\n*** |our - native| interjoint vs flexion-variance: spearman = {r:+.2f} ***")
    print("   (strong NEGATIVE => disagreement concentrates at LOW flexion variance = INSTABILITY ARTIFACT)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
