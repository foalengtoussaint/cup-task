"""DIRECTLY test the flat-signal mechanism (not just correlate with variance) + try confidence-weighting.

Claim to verify: when flexion is flat over the reach, the interjoint (Pearson corr flex,elb) VALUE is
determined by noise, so it JUMPS under tiny perturbation. Earlier I only showed cross-trial rank
stability (0.99) -- that can HIDE per-trial instability. Real test: PER TRIAL, add small independent
noise to flexion N times, measure the SPREAD (std) of the resulting interjoint. If flat trials have
large per-trial spread and high-variance trials don't, the flat-signal mechanism is confirmed.

Then CONFIDENCE-WEIGHTING candidates to tame it:
  raw_pearson : corr(flex, elb)                                  [current]
  shrink_var  : corr * (flex_std^2 / (flex_std^2 + k^2))         [shrink toward 0 when flex is flat]
  fisher_se   : corr but flag/attenuate when its Fisher-z SE is large (n small or |r| high)
Judge shrink by: does it REDUCE per-trial noise-spread on flat trials while KEEPING affected/unaffected
separation? Flexion is trunk-referenced (per-frame, clean OMC hips).
    python paper/scripts/interjoint_perturbation_test.py
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
NOISE_DEG = 0.5          # realistic per-frame flexion jitter
NREP = 60                # perturbations per trial
K_SHRINK = 3.0           # shrink scale (deg) -- flex_std below ~3deg gets pulled toward 0


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


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
            arm = el - sh; down = (hR + hL) / 2.0 - (sh + shO) / 2.0
            flex = _lp(_ang(arm, down))
            u, v = sh - el, wr - el
            elb = _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))
            sl = slice(r0, r1); f = flex[sl]; e = elb[sl]
            m = np.isfinite(f) & np.isfinite(e)
            if m.sum() < 12 or np.std(e[m]) < 1e-6:
                continue
            f, e = f[m], e[m]
            fstd = float(np.std(f))
            r_raw = _pearson(f, e)
            # PER-TRIAL perturbation spread
            rs = [_pearson(f + rng.normal(0, NOISE_DEG, size=len(f)), e) for _ in range(NREP)]
            rs = np.array([x for x in rs if np.isfinite(x)])
            pert_std = float(np.std(rs)) if len(rs) > 5 else np.nan
            # confidence-weighted (shrink toward 0 by flexion variance)
            w = fstd**2 / (fstd**2 + K_SHRINK**2)
            r_shrink = r_raw * w if np.isfinite(r_raw) else np.nan
            rows.append({"part": p, "cond": cond, "fstd": fstd, "r_raw": r_raw,
                         "pert_std": pert_std, "r_shrink": r_shrink})
    d = pd.DataFrame(rows)
    d.to_csv(PAPER / "interjoint_perturbation.csv", index=False)
    thr = d.fstd.quantile(0.25)
    flat = d[d.fstd <= thr]; norm = d[d.fstd > thr]

    print(f"n={len(d)}   flat = bottom-quartile flexion std (<= {thr:.2f} deg), n={len(flat)}\n")
    print("=== DIRECT MECHANISM TEST: per-trial interjoint SPREAD under {:.1f}deg noise x{} ===".format(NOISE_DEG, NREP))
    print(f"{'group':16}{'flex std':>10}{'interjoint':>12}{'PER-TRIAL pert_std':>20}")
    for lab, g in [("FLAT flexion", flat), ("normal", norm)]:
        print(f"{lab:16}{g.fstd.median():>10.2f}{g.r_raw.median():>12.2f}{g.pert_std.median():>20.3f}")
    rr = spearmanr(d.fstd, d.pert_std).correlation
    print(f"\n  per-trial pert_std vs flex_std: spearman = {rr:+.2f}")
    print("  *** if FLAT pert_std >> normal pert_std, the flat-signal mechanism is DIRECTLY confirmed ***")
    print("  (this is the real test; the earlier 0.99 was cross-trial RANK stability, which hides this)")

    print("\n=== CONFIDENCE-WEIGHTING (shrink by flex variance, k={:.0f}deg) ===".format(K_SHRINK))
    for lab, col in [("raw pearson", "r_raw"), ("shrink_var", "r_shrink")]:
        a = d[d.cond == "affected"][col].dropna(); un = d[d.cond == "unaffected"][col].dropna()
        mw = mannwhitneyu(a, un, alternative="two-sided")
        rb = 1 - 2 * mw.statistic / (len(a) * len(un))
        # per-trial noise-spread of the SHRUNK metric on flat trials (recompute quickly is heavy; approximate:
        # shrink multiplies r by w, so its pert_std on flat trials ~ w * pert_std_raw -> report expected)
        print(f"  {lab:12} aff median={a.median():+.2f}  un median={un.median():+.2f}  "
              f"MWU p={mw.pvalue:.1e}  separation(rank-biserial)={rb:+.2f}")
    # how much does shrink cut the flat-trial noise? w on flat trials:
    wflat = (flat.fstd**2 / (flat.fstd**2 + K_SHRINK**2)).median()
    print(f"\n  shrink weight w on FLAT trials (median) = {wflat:.2f}  -> flat interjoint (and its noise) cut to ~{wflat:.0%}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
