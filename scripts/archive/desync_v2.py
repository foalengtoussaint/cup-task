"""Within-trial desync v2: MULTIVARIATE lag (all joint signals combined, not best-of) + WIDER
overlapping windows. Verifies the P13 96Hz fix and classifies drift shape robustly.

Multivariate lag: z-score each signal (wrist/elbow/shoulder x speed/disp) per window, then at each
candidate lag SUM the cross-correlation across all signals -- a noisy signal adds noise but can't
hijack the estimate, and agreement across joints reinforces. Far more robust than argmax-of-one.
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
from desync_drift import _sig_pairs
from score_vs_automq import COHORT_PARTS, FPS
GRID = R._GRID_JOINTS
MAX_LAG = 60
MIN_SCORE = 0.35        # combined cross-corr score to trust a window
DRIFT_FLAG = 3


def _z(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-9 else None


def mv_lag(pairs, lo, hi, max_lag=MAX_LAG):
    """Multivariate lag over [lo,hi): sum z-scored cross-corr across all signal pairs. Returns (lag,score)."""
    idx = np.arange(lo, hi)
    best_s, best_l = -2.0, 0
    for L in range(-max_lag, max_lag + 1):
        j = idx - L
        ok0 = (j >= 0) & (j < 10 ** 9)
        tot = 0.0; nsig = 0
        for _, a, b in pairs:
            jj = idx - L
            ok = (jj >= 0) & (jj < len(b))
            if ok.sum() < 40:
                continue
            xa, xb = a[idx[ok]], b[jj[ok]]
            m = np.isfinite(xa) & np.isfinite(xb)
            if m.sum() < 40:
                continue
            za, zb = _z(xa[m]), _z(xb[m])
            if za is None or zb is None:
                continue
            tot += float(np.mean(za * zb)); nsig += 1
        if nsig:
            s = tot / nsig
            if s > best_s:
                best_s, best_l = s, L
    return best_l, best_s


def classify(prof):
    pts = [(c, l) for c, l in prof if np.isfinite(l)]
    if len(pts) < 4:
        return "unknown", np.nan, np.nan, np.nan
    x = np.array([c for c, _ in pts], float); y = np.array([l for _, l in pts], float)
    span = float(y.max() - y.min())
    m, b = np.polyfit(x, y, 1); yhat = m * x + b
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - yhat) ** 2).sum()) / ss if ss > 1e-9 else 0.0
    if span < DRIFT_FLAG:
        return "stable", m, span, r2
    if r2 >= 0.80:
        return "linear", m, span, r2
    d = np.abs(np.diff(y))
    if len(d) >= 2 and d.max() >= 0.7 * d.sum():
        return "step", m, span, r2
    return "erratic", m, span, r2


def main():
    H.use_good_cams()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    print(f"cohort {len(trials)}; multivariate lag, wide overlapping windows", flush=True)
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
        W = max(150, nfr // 4); step = max(W // 2, 40)      # wide, 50% overlap
        prof = []
        lo = 0
        while lo + W <= nfr + step:
            hi = min(lo + W, nfr)
            if hi - lo >= 80:
                L, s = mv_lag(pairs, lo, hi)
                prof.append(((lo + hi) / 2, L if s >= MIN_SCORE else np.nan))
            lo += step
        mid = nfr // 2
        l1, c1 = mv_lag(pairs, 0, mid); l2, c2 = mv_lag(pairs, mid, nfr)
        klass, slope, span, r2 = classify(prof)
        rate = slope / (step)                                # frames drift per frame (approx)
        rows.append(dict(part=part, trial=trial, n=nfr, klass=klass, span=round(span, 1),
                         rate_pct=round(100 * rate, 2), r2=round(r2, 2),
                         h1_lag=l1, h1_sc=round(c1, 2), h2_lag=l2, h2_sc=round(c2, 2),
                         half_drift=l2 - l1,
                         prof="|".join("--" if not np.isfinite(l) else str(int(l)) for _, l in prof)))
        n += 1
        if n % 100 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s", flush=True)

    df = pd.DataFrame(rows); df.to_csv(ROOT / "out/scoring/desync_v2.csv", index=False)
    rel = df[(df.h1_sc >= MIN_SCORE) & (df.h2_sc >= MIN_SCORE)].copy()
    rel["ad"] = rel.half_drift.abs()
    print(f"\nPROCESSING CHECK: trials {n}, both-halves-reliable {len(rel)}", flush=True)

    print("\n== half-to-half |drift| by participant (AFTER P13 96Hz fix) ==")
    print(f"{'part':<6}{'n':>5}{'med|drift|':>11}{'p90':>6}{'flagged>=3':>11}")
    for p, g in rel.groupby("part"):
        print(f"{p:<6}{len(g):>5}{g.ad.median():>11.0f}{g.ad.quantile(.9):>6.0f}"
              f"{f'{int((g.ad>=DRIFT_FLAG).sum())}/{len(g)}':>11}")

    print("\n== drift-shape class by participant ==")
    print(df.pivot_table(index="part", columns="klass", values="trial", aggfunc="count", fill_value=0).to_string())

    print("\n== worst 12 by half-drift (class, profile) ==")
    for _, r in rel.sort_values("ad", ascending=False).head(12).iterrows():
        print(f"  {r.part}/{r['trial']:<26} {r.klass:<8} half_drift={int(r.half_drift):+3d} "
              f"span={r.span:>5} r2={r.r2}  [{r.prof}]")
    print(f"\nwrote out/scoring/desync_v2.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
