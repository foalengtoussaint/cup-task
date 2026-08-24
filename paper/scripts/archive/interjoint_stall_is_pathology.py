"""Is low-flexion-variance / negative interjoint STROKE PATHOLOGY (stalled flexion) or NUMERICAL NOISE?

User's hypothesis: a stalled/flat shoulder flexion during the reach IS impaired stroke coordination -- so
the low-flexion-variance trials our trunk-ref method flags are SIGNAL, not artifact. Tests:

  1. AFFECTED vs UNAFFECTED arm: if stall = pathology, low flexion-variance + low/negative interjoint
     should be MUCH more common on the AFFECTED arm. If numerical noise, ~equal across arms.
  2. STABILITY: does AutoMQ-native interjoint ALSO dip on those trials (real), or does only our unstable
     estimate flip (noise)? And is the negative value reproducible under a robust re-estimate?

All OMC markers only. condition (affected/unaffected) comes from the AutoMQ trial index.
    python paper/scripts/interjoint_stall_is_pathology.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from compare_pose_omc_delta import _lp
from score_vs_automq import load_automq, automq_part, AUTOMQ


def _our_flexion(sh, el, shO, hR, hL):
    down = (hR + hL) / 2.0 - (sh + shO) / 2.0
    down = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    fin = np.isfinite(down).all(1); dc = np.nanmedian(down[fin], 0) if fin.any() else down[0]
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
    return float(np.corrcoef(a[m], b[m])[0, 1]), float(np.std(a[m]))


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
            tn, sd, cond = int(fk[1]), fk[2], fk[3]
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
            our_ij, our_std = _ij(_our_flexion(sh, el, shO, hR, hL), _elbow(sh, el, wr), min(r0, T), min(r1, T))
            fO = kin["shoulder_flexion"].to_numpy(); eO = kin["elbow_angle"].to_numpy()
            nat_ij, nat_std = _ij(fO, eO, r0, min(r1, len(fO)))
            if np.isfinite(our_ij) and np.isfinite(nat_ij):
                rows.append({"part": p, "cond": cond, "our_ij": our_ij, "nat_ij": nat_ij,
                             "our_flex_std": our_std, "nat_flex_std": nat_std})
    d = pd.DataFrame(rows)
    aff = d[d.cond == "affected"]; un = d[d.cond == "unaffected"]
    print(f"n={len(d)}  (affected {len(aff)}, unaffected {len(un)})\n")

    print("=== TEST 1: does the low-variance / low-interjoint concentrate on the AFFECTED arm? ===")
    print(f"{'metric':30}{'affected':>12}{'unaffected':>12}")
    print(f"{'our flexion std (deg)':30}{aff.our_flex_std.median():>12.2f}{un.our_flex_std.median():>12.2f}")
    print(f"{'OMC flexion std (deg)':30}{aff.nat_flex_std.median():>12.2f}{un.nat_flex_std.median():>12.2f}")
    print(f"{'our interjoint median':30}{aff.our_ij.median():>12.2f}{un.our_ij.median():>12.2f}")
    print(f"{'AutoMQ interjoint median':30}{aff.nat_ij.median():>12.2f}{un.nat_ij.median():>12.2f}")
    print(f"{'frac our_ij < 0.5':30}{(aff.our_ij<0.5).mean():>12.2f}{(un.our_ij<0.5).mean():>12.2f}")
    print(f"{'frac AutoMQ_ij < 0.9':30}{(aff.nat_ij<0.9).mean():>12.2f}{(un.nat_ij<0.9).mean():>12.2f}")
    for col in ["our_flex_std", "nat_flex_std", "our_ij", "nat_ij"]:
        u = mannwhitneyu(aff[col], un[col], alternative="two-sided").pvalue
        print(f"   Mann-Whitney affected vs unaffected on {col:16}: p={u:.3g}")

    print("\n=== TEST 2: on the trials our method flags (our_ij<0.5), does AutoMQ ALSO dip? ===")
    flagged = d[d.our_ij < 0.5]
    print(f"   n flagged (our_ij<0.5) = {len(flagged)}")
    print(f"   on these, AutoMQ-native interjoint: median {flagged.nat_ij.median():.2f}  "
          f"IQR [{flagged.nat_ij.quantile(.25):.2f}, {flagged.nat_ij.quantile(.75):.2f}]")
    print(f"   (if AutoMQ ALSO low -> real stall both methods see; if AutoMQ still ~0.97 -> our flag is noise)")
    print(f"   our flexion std on flagged: median {flagged.our_flex_std.median():.2f} deg  "
          f"vs OMC flexion std on same trials: {flagged.nat_flex_std.median():.2f} deg")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
