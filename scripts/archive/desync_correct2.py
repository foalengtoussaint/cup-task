"""Correct ERRATIC (non-linear) within-trial drift with a FLEXIBLE per-trial time-warp: dense
sliding-window multivariate lags -> robust median-smoothed lag(t) curve -> resample OMC by lag(t).
Generalises the affine warp (recovers the line when the drift is linear; follows the curve when it
isn't). Compares CONSTANT vs LINEAR vs FLEXIBLE warp per drift class, and MUST NOT hurt the clean
controls (P07/P13) -- a flexible warp that damages well-synced trials is too aggressive.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import medfilt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
from desync_v3 import _pairs
from desync_v2 import mv_lag, MIN_SCORE
from desync_correct import fit_lagline, warp, _kab
GRID = R._GRID_JOINTS
TARGET = ["P251", "P10", "P252", "P07", "P13"]


def flex_lag(pairs, T):
    """Per-frame lag(t) from dense sliding windows, robustly smoothed. None if too few reliable."""
    W = max(120, T // 6); step = max(W // 4, 20)
    xs, ys = [], []
    lo = 0
    while lo < T:
        hi = min(lo + W, T)
        if hi - lo >= 80:
            L, s = mv_lag(pairs, lo, hi)
            if s >= MIN_SCORE:
                xs.append((lo + hi) / 2); ys.append(L)
        if hi >= T:
            break
        lo += step
    if len(xs) < 5:
        return None
    x = np.array(xs, float); y = np.array(ys, float)
    o = np.argsort(x); x, y = x[o], y[o]
    y = medfilt(y, 3 if len(y) < 7 else 5)                 # kill single-window outliers
    return np.interp(np.arange(T), x, y)                   # per-frame, constant-extrapolated ends


def main():
    H.use_good_cams()
    kl = pd.read_csv(ROOT / "out/scoring/desync_v2.csv")[["part", "trial", "klass"]]
    klass = {(r.part, r.trial): r.klass for _, r in kl.iterrows()}
    pat = re.compile(r"trial_(\d+)_([RL])_")
    rows = []
    for part in TARGET:
        for t in [x for x in GT.load_clean(need_reproj=False) if x["part"] == part]:
            trial, side = t["trial"], t["side"]
            if not pat.search(trial):
                continue
            nfr = t["mmc"].shape[0]; wr = f"{side}_wrist"
            omc = H._load_omc(part, trial, nfr)
            if wr not in omc or not np.isfinite(omc[wr]).any():
                continue
            mw = t["mmc"][:, GRID.index(wr)]
            pairs = _pairs(t, part, trial, side, nfr, omc)
            if len(pairs) < 3:
                continue
            clag, _ = H._find_lag(mw, omc[wr])
            e_const = _kab(mw, R._shift(omc[wr], clag))
            fl = fit_lagline(pairs, nfr)
            e_lin = _kab(mw, warp(omc[wr], fl[0], fl[1], nfr)) if fl else np.nan
            flag = flex_lag(pairs, nfr)
            if flag is not None:
                src = np.arange(nfr) - flag
                ow = np.full((nfr, 3), np.nan); xp = np.arange(len(omc[wr]))
                for k in range(3):
                    ow[:, k] = np.interp(src, xp, omc[wr][:, k], left=np.nan, right=np.nan)
                e_flex = _kab(mw, ow)
            else:
                e_flex = np.nan
            rows.append(dict(part=part, trial=trial, klass=klass.get((part, trial), "?"),
                             e_const=round(e_const, 1) if np.isfinite(e_const) else np.nan,
                             e_lin=round(e_lin, 1) if np.isfinite(e_lin) else np.nan,
                             e_flex=round(e_flex, 1) if np.isfinite(e_flex) else np.nan))
    df = pd.DataFrame(rows); df.to_csv(ROOT / "out/scoring/desync_correct2.csv", index=False)
    print(f"trials {len(df)}\n")
    print("median wrist 3D err (mm): CONSTANT -> LINEAR-warp -> FLEXIBLE-warp\n")
    print(f"{'part':<6}{'class':<9}{'n':>4}{'const':>8}{'linear':>8}{'flex':>8}")
    for (part, kls), g in df.groupby(["part", "klass"]):
        gg = g.dropna(subset=["e_const"])
        if len(gg) < 2:
            continue
        print(f"{part:<6}{kls:<9}{len(gg):>4}{gg.e_const.median():>8.1f}"
              f"{gg.e_lin.median():>8.1f}{gg.e_flex.median():>8.1f}")
    print("\n== per participant, ALL trials ==")
    for part, g in df.groupby("part"):
        gg = g.dropna(subset=["e_const"])
        print(f"  {part:<6} n={len(gg):>3}  const {gg.e_const.median():.1f}  "
              f"linear {gg.e_lin.median():.1f}  flex {gg.e_flex.median():.1f} mm")
    print("\n(controls P07/P13 must NOT worsen under flex; targets P251/P10 erratic should improve)")
    print("wrote out/scoring/desync_correct2.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
