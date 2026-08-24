"""On flat-shoulder (impaired) trials, is AutoMQ's flexion motion actually TRUNK LEAN, not shoulder flexion?

Hypothesis (user): AutoMQ flexion = arm-vs-world-vertical has no trunk reference, so when the SHOULDER
stalls but the patient LEANS FORWARD, the arm still moves relative to vertical -> AutoMQ reads it as
flexion rising. That trunk-driven 'flexion' correlates with the elbow -> AutoMQ interjoint stays high on
exactly the impaired trials where the true shoulder-elbow coordination has broken down.

Test, per trial, OMC markers only:
  amq_flex_series   : AutoMQ arm-vs-vertical flexion (its shipped series)
  our_flex_series   : our trunk-referenced flexion (arm vs shoulder->hip)
  trunk_lean_series : trunk tilt from vertical over time (hip->shoulder axis vs Z)
For the FLAT-SHOULDER trials (our flexion std small = shoulder barely flexes), check:
  - does AutoMQ flexion still MOVE? (its std)
  - does AutoMQ's flexion motion CORRELATE with the trunk lean? (per-frame corr over reach)
If AutoMQ flexion tracks trunk lean on flat-shoulder trials -> AutoMQ is measuring trunk, not shoulder.
    python paper/scripts/automq_flexion_is_trunk.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from compare_pose_omc_delta import _lp
from score_vs_automq import load_automq, automq_part, AUTOMQ
Z = np.array([0.0, 0.0, 1.0])


def _ang(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-6 or np.std(b[m]) < 1e-6:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


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
            mk = cdf.loc[fk, "markers"]
            sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}")
            shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
            hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
            if any(x is None for x in (sh, el, shO, hR, hL)):
                continue
            T = min(map(len, (sh, el, shO, hR, hL)))
            sh, el, shO, hR, hL = (a[:T] for a in (sh, el, shO, hR, hL))
            r1 = min(r1, T)
            if r1 - r0 < 12:
                continue
            arm = sh - el
            # AutoMQ flexion (arm vs world-Z, sagittal projection -- but the pure angle vs Z is the driver)
            amq_flex = _lp(_ang(arm, Z[None, :]))
            # our trunk-referenced flexion (arm vs shoulder->hip, per-frame here to see the true motion)
            down = (hR + hL) / 2.0 - (sh + shO) / 2.0
            our_flex = _lp(_ang(arm, down))
            # trunk lean over time (hip->shoulder axis vs vertical)
            up_trunk = (sh + shO) / 2.0 - (hR + hL) / 2.0
            trunk_lean = _lp(_ang(up_trunk, Z[None, :]))
            sl = slice(r0, r1)
            rows.append({"part": p, "cond": cond,
                         "amq_std": float(np.nanstd(amq_flex[sl])),
                         "our_std": float(np.nanstd(our_flex[sl])),
                         "lean_std": float(np.nanstd(trunk_lean[sl])),
                         "amq_vs_lean": _corr(amq_flex[sl], trunk_lean[sl]),
                         "our_vs_lean": _corr(our_flex[sl], trunk_lean[sl])})
    d = pd.DataFrame(rows)
    d.to_csv(PAPER / "automq_flexion_is_trunk.csv", index=False)
    # FLAT-SHOULDER trials = our (trunk-referenced) flexion std is small
    thr = d.our_std.quantile(0.25)
    flat = d[d.our_std <= thr]; norm = d[d.our_std > thr]
    print(f"n={len(d)}  flat-shoulder (our_std<= {thr:.2f} deg, bottom quartile): n={len(flat)}\n")
    print(f"{'group':16}{'our_flex_std':>13}{'AMQ_flex_std':>13}{'trunk_lean_std':>15}"
          f"{'corr(AMQ,lean)':>16}{'corr(our,lean)':>16}")
    for lab, g in [("FLAT-shoulder", flat), ("normal", norm)]:
        print(f"{lab:16}{g.our_std.median():>13.2f}{g.amq_std.median():>13.2f}{g.lean_std.median():>15.2f}"
              f"{g.amq_vs_lean.median():>16.2f}{g.our_vs_lean.median():>16.2f}")
    print("\n*** KEY: on FLAT-shoulder trials, if AMQ_flex_std >> our_flex_std AND corr(AMQ,lean) is HIGH,")
    print("    AutoMQ's 'flexion' is TRUNK LEAN, not shoulder motion -- confirming the user's hypothesis. ***")
    print(f"\n  frac of FLAT trials that are AFFECTED arm: {(flat.cond=='affected').mean():.2f} "
          f"(vs {(d.cond=='affected').mean():.2f} overall)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
