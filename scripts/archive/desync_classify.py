"""Classify within-trial desync SHAPE: linear drift (rate mismatch) vs step (dropped-frame block) vs
erratic. Finer sliding-window multi-signal lag profile per flagged trial; fit a line; a high-R^2
monotonic profile = linear drift, a single dominant jump = step, else erratic. For linear trials the
SLOPE is a clock-rate error -- constant across a participant => a sampling-rate mismatch.
"""
from __future__ import annotations
import sys, re, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from desync_drift import _sig_pairs, _best_window_lag, MIN_CORR
from score_vs_automq import COHORT_PARTS, FPS
GRID = R._GRID_JOINTS
NWIN = 9
DRIFT_FLAG = 3


def classify(prof, T):
    """prof: list of (center_frame, lag|nan). Returns (klass, slope_fr_per_fr, span, r2)."""
    pts = [(c, l) for c, l in prof if np.isfinite(l)]
    if len(pts) < 4:
        return "unknown", np.nan, np.nan, np.nan
    x = np.array([c for c, _ in pts], float); y = np.array([l for _, l in pts], float)
    span = float(y.max() - y.min())
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - yhat) ** 2).sum()) / ss if ss > 1e-9 else 0.0
    if span < DRIFT_FLAG:
        return "stable", m, span, r2
    if r2 >= 0.80:
        return "linear", m, span, r2
    # step: one consecutive jump dominates and the rest is flat-ish
    dif = np.abs(np.diff(y))
    if len(dif) >= 2 and dif.max() >= 0.7 * dif.sum():
        return "step", m, span, r2
    return "erratic", m, span, r2


def main():
    H.use_good_cams()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    print(f"cohort {len(trials)}; {NWIN}-window multi-signal profiles", flush=True)
    rows = []; t0 = time.time(); n = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        if not pat.search(trial):
            continue
        nfr = t["mmc"].shape[0]
        omc = H._load_omc(part, trial, nfr)
        mmc = {j: t["mmc"][:, GRID.index(j)] for j in GRID}
        pairs = _sig_pairs(mmc, omc, side)
        if not pairs:
            continue
        prof = []
        for k in range(NWIN):
            lo, hi = int(k * nfr / NWIN), int((k + 1) * nfr / NWIN)
            L, c, _ = _best_window_lag(pairs, lo, hi)
            prof.append(((lo + hi) / 2, L if c >= MIN_CORR else np.nan))
        klass, slope, span, r2 = classify(prof, nfr)
        rate = slope / (nfr / NWIN)                      # frames of drift PER frame of trial
        rows.append(dict(part=part, trial=trial, n=nfr, klass=klass, span=round(span, 1),
                         slope=round(slope, 2), rate_pct=round(100 * rate, 2), r2=round(r2, 2),
                         prof="|".join("--" if not np.isfinite(l) else str(int(l)) for _, l in prof)))
        n += 1
        if n % 100 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "out/scoring/desync_classify.csv", index=False)
    drifty = df[~df.klass.isin(["stable", "unknown"])]
    print(f"\nPROCESSING CHECK: trials {n}; stable {int((df.klass=='stable').sum())}, "
          f"unknown {int((df.klass=='unknown').sum())}, drifty {len(drifty)}", flush=True)

    print("\n== drift-shape class by participant (count) ==")
    piv = df.pivot_table(index="part", columns="klass", values="trial", aggfunc="count", fill_value=0)
    print(piv.to_string())

    print("\n== LINEAR-drift trials: clock-rate error (frames of drift per 100 frames) by participant ==")
    lin = df[df.klass == "linear"]
    for p, g in lin.groupby("part"):
        rr = g.rate_pct.abs()
        print(f"  {p:<6} n={len(g):>3}  median |rate| {rr.median():.2f}%  (range {rr.min():.2f}-{rr.max():.2f}%)  "
              f"signed median {g.rate_pct.median():+.2f}%")
    print("\n  (a CONSISTENT rate within a participant => sampling-rate mismatch; "
          "e.g. 100 vs 120Hz resampled as 100 = ~16.7%)")

    print("\n== worst 12 drifty trials (class, span, rate, profile) ==")
    for _, r in drifty.reindex(drifty.span.abs().sort_values(ascending=False).index).head(12).iterrows():
        print(f"  {r.part}/{r['trial']:<26} {r.klass:<8} span={r.span:>5}fr rate={r.rate_pct:+.1f}% r2={r.r2}  [{r.prof}]")
    print(f"\nwrote out/scoring/desync_classify.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
