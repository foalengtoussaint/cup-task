"""WITHIN-TRIAL desync: a single constant lag (the sync 'translation') can't fix a trial whose MMC<->OMC
offset DRIFTS -- synced in the first half, off in the second, or a different lag in each half.

For every cohort trial: build a motion signal present in both streams (wrist/elbow speed + displacement),
find the best lag in the FIRST half vs the SECOND half (and a 5-window sliding profile) via proper
(non-circular) shifted cross-correlation. Flag trials where the two halves disagree while BOTH halves
are well-correlated (so it's a real drift, not a dead/low-motion window).

Saves per-trial CSV + prints per-participant drift counts + worst offenders + their sliding profiles.
Data only.
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
from score_vs_automq import COHORT_PARTS, FPS
GRID = R._GRID_JOINTS
MAX_LAG = 45          # frames (+-0.75s) -- within-trial drift is small vs the global offset
MIN_CORR = 0.55       # a window's lag is only trusted above this
DRIFT_FLAG = 3        # frames of half-to-half change to flag (>=50ms)


def _sig_pairs(mmc, omc, side):
    """Candidate (name, a, b) motion-signal pairs present in both streams."""
    out = []
    for j in ("wrist", "elbow", "shoulder"):
        jn = f"{side}_{j}"
        if jn in mmc and jn in omc:
            out.append((f"{j}_spd", H._lp(H._speed(mmc[jn])), H._lp(H._speed(omc[jn]))))
            out.append((f"{j}_disp", H._lp(H._disp_from_start(mmc[jn])), H._lp(H._disp_from_start(omc[jn]))))
    return out


def _local_lag(a, b, lo, hi, max_lag=MAX_LAG):
    """Best lag aligning a[lo:hi] to a shifted b, NON-circular. Returns (lag, corr)."""
    idx = np.arange(lo, hi)
    best_c, best_l = -2.0, 0
    for L in range(-max_lag, max_lag + 1):
        j = idx - L
        ok = (j >= 0) & (j < len(b))
        if ok.sum() < 40:
            continue
        x, y = a[idx[ok]], b[j[ok]]
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < 40 or np.std(x[mm]) < 1e-9 or np.std(y[mm]) < 1e-9:
            continue
        c = np.corrcoef(x[mm], y[mm])[0, 1]
        if c > best_c:
            best_c, best_l = c, L
    return best_l, best_c


def _best_window_lag(pairs, lo, hi):
    """Across all signal pairs, the lag from the best-correlating one in [lo,hi]."""
    bl, bc, bs = 0, -2.0, "none"
    for name, a, b in pairs:
        L, c = _local_lag(a, b, lo, hi)
        if c > bc:
            bl, bc, bs = L, c, name
    return bl, bc, bs


def main():
    H.use_good_cams()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    print(f"cohort {len(trials)}", flush=True)
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
        T = nfr; mid = T // 2
        gl, gc, gs = _best_window_lag(pairs, 0, T)
        l1, c1, s1 = _best_window_lag(pairs, 0, mid)
        l2, c2, s2 = _best_window_lag(pairs, mid, T)
        # 5-window sliding profile (each ~T/5)
        prof = []
        for k in range(5):
            lo, hi = int(k * T / 5), int((k + 1) * T / 5)
            L, c, _ = _best_window_lag(pairs, lo, hi)
            prof.append(L if c >= MIN_CORR else np.nan)
        rows.append(dict(part=part, trial=trial, n=T, glag=gl, gcorr=round(gc, 2),
                         h1_lag=l1, h1_corr=round(c1, 2), h2_lag=l2, h2_corr=round(c2, 2),
                         drift=l2 - l1, prof="|".join("--" if np.isnan(x) else str(int(x)) for x in prof),
                         prof_range=(np.nanmax(prof) - np.nanmin(prof)) if np.isfinite(prof).any() else np.nan))
        n += 1
        if n % 80 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "out/scoring/desync_drift.csv", index=False)
    good = df[(df.h1_corr >= MIN_CORR) & (df.h2_corr >= MIN_CORR)].copy()
    good["absdrift"] = good["drift"].abs()
    flagged = good[good.absdrift >= DRIFT_FLAG]
    print(f"\nPROCESSING CHECK: trials {n}, both-halves-reliable {len(good)}, "
          f"flagged |drift|>={DRIFT_FLAG}fr: {len(flagged)} ({len(flagged)/max(len(good),1):.0%})", flush=True)

    print(f"\n== within-trial half-to-half DRIFT by participant (reliable trials only) ==")
    print(f"{'part':<6}{'n':>5}{'med|drift|':>11}{'p90':>6}{'flagged':>9}")
    for p, g in good.groupby("part"):
        fl = (g.absdrift >= DRIFT_FLAG).sum()
        print(f"{p:<6}{len(g):>5}{g.absdrift.median():>11.0f}{g.absdrift.quantile(.9):>6.0f}"
              f"{f'{fl}/{len(g)}':>9}")

    print(f"\n== worst 15 trials by |drift| (h1_lag -> h2_lag, sliding profile) ==")
    for _, r in flagged.sort_values("absdrift", ascending=False).head(15).iterrows():
        print(f"  {r.part}/{r['trial']:<26} drift {int(r.drift):+3d}fr  h1={int(r.h1_lag):+d}({r.h1_corr}) "
              f"h2={int(r.h2_lag):+d}({r.h2_corr})  prof[{r.prof}]")
    print(f"\nwrote out/scoring/desync_drift.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
