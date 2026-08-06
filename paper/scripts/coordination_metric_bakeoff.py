"""Fix interjoint's flat-signal instability: bake off coordination metrics that keep MAGNITUDE info.

Pearson corr(flexion, elbow) is scale-invariant -> when flexion is flat it normalizes away the 'flat'
and reports the direction of noise (random +-1). Alternatives that DON'T discard magnitude:
  pearson   : corr(flex, elb)                              [current -- unstable when flat]
  cov_n     : covariance / (robust global scale)           [flat flex -> ~0, not random]
  slope     : OLS slope d(flex)/d(elb) over the reach      [flat shoulder -> slope~0 = abnormal]
  rangeratio: (flex p95-p5) / (elb p95-p5)                 [magnitude ratio, immune to flat]
Flexion is TRUNK-REFERENCED (per-frame, OMC clean hips) so trunk lean is removed. Judge each by:
  (A) SEPARATION: affected vs unaffected arm (Mann-Whitney effect size / medians) -- clinical sensitivity
  (B) STABILITY : add small noise (0.5 deg) to flexion, recompute, correlate perturbed vs clean across
                  trials. High = stable. Pearson should be WORST here on the flat trials.
All OMC markers.
    python paper/scripts/coordination_metric_bakeoff.py
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
rng = np.random.default_rng(0)


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def _metrics(flex, elb):
    m = np.isfinite(flex) & np.isfinite(elb)
    if m.sum() < 10 or np.std(elb[m]) < 1e-6:
        return {k: np.nan for k in ("pearson", "cov_n", "slope", "rangeratio")}
    f, e = flex[m], elb[m]
    pear = np.corrcoef(f, e)[0, 1] if np.std(f) > 1e-9 else np.nan
    cov_n = np.mean((f - f.mean()) * (e - e.mean())) / 100.0        # /100 just to scale to ~O(1)
    slope = np.polyfit(e, f, 1)[0]                                  # d flex / d elb
    fr = np.percentile(f, 95) - np.percentile(f, 5)
    er = np.percentile(e, 95) - np.percentile(e, 5)
    rr = fr / er if er > 1e-6 else np.nan
    return {"pearson": pear, "cov_n": cov_n, "slope": slope, "rangeratio": rr}


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
            down = (hR + hL) / 2.0 - (sh + shO) / 2.0                 # per-frame trunk-ref (clean OMC hips)
            flex = _lp(_ang(arm, down))
            u, v = sh - el, wr - el
            elbow = _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))
            sl = slice(r0, r1)
            met = _metrics(flex[sl], elbow[sl])
            # perturbed flexion for stability
            metp = _metrics(flex[sl] + rng.normal(0, 0.5, size=len(flex))[sl], elbow[sl])
            row = {"part": p, "cond": cond}
            row.update({f"{k}": met[k] for k in met})
            row.update({f"{k}_p": metp[k] for k in metp})
            rows.append(row)
    d = pd.DataFrame(rows)
    aff = d[d.cond == "affected"]; un = d[d.cond == "unaffected"]
    print(f"n={len(d)} (aff {len(aff)}, un {len(un)})\n")
    print(f"{'metric':12}{'aff median':>12}{'un median':>12}{'MWU p':>10}{'rank-biserial':>15}{'stability r':>13}")
    for k in ("pearson", "cov_n", "slope", "rangeratio"):
        a = aff[k].dropna(); u = un[k].dropna()
        mw = mannwhitneyu(a, u, alternative="two-sided")
        # rank-biserial effect size = 1 - 2U/(n1 n2)
        rb = 1 - 2 * mw.statistic / (len(a) * len(u))
        # stability: corr(clean, perturbed) across all trials
        cc = d[[k, f"{k}_p"]].dropna()
        stab = spearmanr(cc[k], cc[f"{k}_p"]).correlation
        print(f"{k:12}{a.median():>12.2f}{u.median():>12.2f}{mw.pvalue:>10.1e}{rb:>15.2f}{stab:>13.2f}")
    print("\n  SEPARATION (rank-biserial, higher=better affected/unaffected split) = clinical sensitivity")
    print("  STABILITY (corr clean vs +0.5deg-noise, higher=better) -- pearson should be LOWEST")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
