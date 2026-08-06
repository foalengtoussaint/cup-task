"""OMC-ONLY: does trunk-referenced flexion give a different (and more clinically sensitive) interjoint
than AutoMQ's world-vertical flexion -- and is the difference a real TRAJECTORY-SHAPE difference?

No MMC, no tracking noise. Both flexions from AutoMQ's clean markers over the reach window:
  flex_world  : angle(arm, world-Z)                     [AutoMQ shipped]
  flex_trunk  : angle(arm, shoulder->hip, per-frame)    [trunk-referenced, removes lean]
For each trial: interjoint (corr flex,elbow) under each, the two flexion SERIES' mutual correlation
(shape agreement), and condition (affected/unaffected). Then:
  - do the two interjoints differ, and does the difference track the flexion-shape difference?
  - does trunk-referenced interjoint separate affected vs unaffected BETTER than world-vertical?
    python paper/scripts/interjoint_omc_only.py
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
Z = np.array([0.0, 0.0, 1.0])


def _ang(u, v):
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
            mk = cdf.loc[fk, "markers"]; kin = cdf.loc[fk, "kinematics"]
            sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}"); wr = _marker(mk, f"hand_{sd}")
            shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
            hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
            if any(x is None for x in (sh, el, wr, shO, hR, hL)):
                continue
            T = min(map(len, (sh, el, wr, shO, hR, hL))); r1 = min(r1, T)
            if r1 - r0 < 12:
                continue
            sh, el, wr, shO, hR, hL = (a[:T] for a in (sh, el, wr, shO, hR, hL))
            arm = el - sh
            flex_world = _lp(_ang(arm, Z[None, :]))
            flex_trunk = _lp(_ang(arm, (hR + hL) / 2 - (sh + shO) / 2))
            u, v = sh - el, wr - el
            elb = _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))
            sl = slice(r0, r1)
            ij_w = _corr(flex_world[sl], elb[sl])
            ij_t = _corr(flex_trunk[sl], elb[sl])
            shape = _corr(flex_world[sl], flex_trunk[sl])     # do the two flexion series agree in shape?
            if np.isfinite(ij_w) and np.isfinite(ij_t):
                rows.append({"part": p, "cond": cond, "ij_world": ij_w, "ij_trunk": ij_t,
                             "flex_shape_corr": shape,
                             "world_std": float(np.nanstd(flex_world[sl])),
                             "trunk_std": float(np.nanstd(flex_trunk[sl]))})
    d = pd.DataFrame(rows)
    d.to_csv(PAPER / "interjoint_omc_only.csv", index=False)
    print(f"n={len(d)}\n")

    print("=== the two OMC interjoint definitions ===")
    for lab, c in [("world-vertical (AutoMQ)", "ij_world"), ("trunk-referenced", "ij_trunk")]:
        s = d[c]
        print(f"  {lab:26} median {s.median():.3f}  IQR {s.quantile(.75)-s.quantile(.25):.3f}  "
              f"std {s.std():.3f}  min {s.min():+.2f}  frac<0.8 {(s<0.8).mean():.2f}")
    print(f"  the two agree with each other: rs={spearmanr(d.ij_world, d.ij_trunk).correlation:+.2f}")

    print("\n=== is the interjoint difference a TRAJECTORY-SHAPE difference? ===")
    d["ij_diff"] = (d.ij_world - d.ij_trunk).abs()
    r = spearmanr(d.flex_shape_corr, d.ij_diff).correlation
    print(f"  flex-series shape agreement (world vs trunk): median {d.flex_shape_corr.median():.2f}")
    print(f"  interjoint |difference| vs flex-shape agreement: spearman = {r:+.2f}")
    print("  (strong NEGATIVE => where the two flexion SHAPES diverge, the interjoints diverge = shape-driven, real)")

    print("\n=== clinical separation: affected vs unaffected ===")
    aff = d[d.cond == "affected"]; un = d[d.cond == "unaffected"]
    print(f"{'metric':26}{'aff med':>10}{'un med':>10}{'MWU p':>10}{'rank-biserial':>15}")
    for lab, c in [("world-vertical", "ij_world"), ("trunk-referenced", "ij_trunk")]:
        a = aff[c].dropna(); u = un[c].dropna()
        mw = mannwhitneyu(a, u, alternative="two-sided"); rb = 1 - 2 * mw.statistic / (len(a) * len(u))
        print(f"  {lab:24}{a.median():>10.3f}{u.median():>10.3f}{mw.pvalue:>10.1e}{rb:>15.2f}")
    print("\n  (higher rank-biserial = better affected/unaffected separation = more clinically sensitive)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
