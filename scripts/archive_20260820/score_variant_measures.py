"""Score any cached BA variant through the REAL scorer, by pointing its cache at the variant.

score_vs_automq reads cache/pose_smoothed/<part>__<trial>.npz. Rather than reimplement its measure
reductions (the shared-code rule -- the number and its source must be the same operator), this
materialises a per-variant cache directory in the same layout and monkeypatches _SMCACHE at it, then
calls score_vs_automq.main() unchanged.

    python scripts/score_variant_measures.py --tags fix__g150__sn fix__sn bonesub_0.05__sn
    -> out/scoring/variant_measures/<tag>.csv  + a per-participant summary per measure
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import score_vs_automq as S              # noqa: E402

VAR = ROOT / "cache" / "ba_variants"
OUT = ROOT / "out" / "automq" / "variant_measures"
TMP = ROOT / "cache" / "_variant_sm"
REAL = ROOT / "cache" / "pose_smoothed"


def materialise(tag):
    """Write <part>__<trial>.npz with ba_sn = this variant, pipeline_sn = the real one (unused)."""
    z = np.load(VAR / f"{tag}.npz", allow_pickle=True)
    ids, traj = [str(i) for i in z["ids"]], z["traj"]
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    n_written = 0
    for tid, P in zip(ids, traj):
        part, trial = tid.split("/", 1)
        src = REAL / f"{part}__{trial}.npz"
        if not src.exists():
            continue
        r = np.load(str(src), allow_pickle=True)
        np.savez(str(TMP / f"{part}__{trial}.npz"), joints=r["joints"],
                 pipeline_sn=r["pipeline_sn"], ba_sn=np.asarray(P, np.float32))
        n_written += 1
    return n_written, len(ids)


def summarise(csv, tag):
    d = pd.read_csv(csv)
    d = d[(d.variant == "BA+smoothnet") & (d.peak_metric.isin(["n/a", "max"]) | d.peak_metric.isna())]
    rows = []
    for m, g in d.groupby("measure"):
        g = g.dropna(subset=["automq", "mmc"])
        if len(g) < 20:
            continue
        rest = g[g.part != "P19"]
        p19 = g[g.part == "P19"]
        ratio = (p19.mmc / p19.automq).replace([np.inf, -np.inf], np.nan)
        rows.append(dict(tag=tag, measure=m, n=len(g),
                         r_all=spearmanr(g.automq, g.mmc).statistic,
                         r_noP19=spearmanr(rest.automq, rest.mmc).statistic,
                         r_P19=spearmanr(p19.automq, p19.mmc).statistic if len(p19) > 10 else np.nan,
                         P19_over2x=int((ratio > 2).sum()), P19_n=len(p19),
                         medAE=float((g.mmc - g.automq).abs().median())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    orig = S._SMCACHE
    allsum = []
    for k, tag in enumerate(a.tags):
        t0 = time.time()
        n, tot = materialise(tag)
        print(f"\n[{k+1}/{len(a.tags)}] {tag}: materialised {n}/{tot} trials ({time.time()-t0:.0f}s)",
              flush=True)
        S._SMCACHE = TMP
        csv = OUT / f"{tag}.csv"
        S.main(["--out", str(csv), "--variants", "BA+smoothnet"])
        s = summarise(csv, tag)
        s.to_csv(OUT / f"{tag}__summary.csv", index=False)
        allsum.append(s)
        print(f"  [{k+1}/{len(a.tags)}] {tag} done ({time.time()-t0:.0f}s)", flush=True)
    S._SMCACHE = orig
    d = pd.concat(allsum, ignore_index=True)
    d.to_csv(OUT / "ALL_summary.csv", index=False)
    print(f"\nPROCESSING CHECK: {len(a.tags)} tags, {d.measure.nunique()} measures, "
          f"{int(d[['r_all','r_noP19']].isna().values.sum())} non-finite", flush=True)
    for m in ["peak_elbow_angular_velocity", "peak_velocity", "max_elbow_angle",
              "max_shoulder_flexion", "max_shoulder_abduction"]:
        g = d[d.measure == m]
        if not len(g):
            continue
        print(f"\n=== {m} ===")
        print(f"{'tag':28s} {'n':>4s} {'r_all':>7s} {'r_noP19':>8s} {'r_P19':>7s} "
              f"{'P19>2x':>7s} {'medAE':>8s}")
        for _, r in g.iterrows():
            print(f"{r.tag:28s} {int(r.n):4d} {r.r_all:7.3f} {r.r_noP19:8.3f} {r.r_P19:7.3f} "
                  f"{int(r.P19_over2x):3d}/{int(r.P19_n):<3d} {r.medAE:8.2f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
