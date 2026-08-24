"""Correct LINEAR within-trial drift with a per-trial affine time-warp, and verify it beats the
constant-lag sync on the drift-heavy participants (P251/P10/P252; their C3D is 100Hz so the drift is
video-side, not fixable by the rate patch).

Estimate lag(t)=a+b*t from multivariate (wrist/elbow/shoulder/head/cup x spd/disp) sliding-window
lags; if the fit is linear enough and the drift real, resample OMC at index t-(a+b*t). Compare
MMC<->OMC wrist speed-corr + 3D error: CONSTANT-lag vs LINEAR-warp.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
from desync_v3 import _pairs
from desync_v2 import mv_lag, MIN_SCORE
GRID = R._GRID_JOINTS
TARGET = ["P251", "P10", "P252", "P13", "P07"]      # linear-drift cases + P13/P07 controls


def fit_lagline(pairs, T, nwin=7):
    W = max(150, T // 4); step = max((T - W) // (nwin - 1), 30)
    xs, ys = [], []
    lo = 0
    while lo + W <= T + step and lo < T:
        hi = min(lo + W, T)
        if hi - lo >= 80:
            L, s = mv_lag(pairs, lo, hi)
            if s >= MIN_SCORE:
                xs.append((lo + hi) / 2); ys.append(L)
        lo += step
    if len(xs) < 4:
        return None
    x = np.array(xs); y = np.array(ys)
    b, a = np.polyfit(x, y, 1)
    yh = a + b * x; ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yh) ** 2).sum() / ss if ss > 1e-9 else 0.0
    return a, b, r2, float(y.max() - y.min())


def warp(joint, a, b, T):
    src = np.arange(T) - (a + b * np.arange(T))
    out = np.full((T, 3), np.nan)
    xp = np.arange(len(joint))
    for k in range(3):
        out[:, k] = np.interp(src, xp, joint[:, k], left=np.nan, right=np.nan)
    return out


def _kab(mw, ow):
    m = np.isfinite(mw).all(1) & np.isfinite(ow).all(1)
    if m.sum() < 10:
        return np.nan
    A, B = mw[m], ow[m]; ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb)); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rk = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.median(np.linalg.norm((Rk @ (A - ca).T).T + cb - B, axis=1)))


def main():
    H.use_good_cams()
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
            # CONSTANT lag baseline
            clag, ccorr = H._find_lag(mw, omc[wr])
            c_err = _kab(mw, R._shift(omc[wr], clag))
            # LINEAR warp
            fl = fit_lagline(pairs, nfr)
            if fl is None:
                w_err, wcorr, r2, span = np.nan, np.nan, np.nan, np.nan
            else:
                a, b, r2, span = fl
                ow = warp(omc[wr], a, b, nfr)
                wcorr = H._find_lag(mw, ow)[1]          # residual speed-corr after warp (should be higher)
                w_err = _kab(mw, ow)
            rows.append(dict(part=part, trial=trial, drift_span=round(span, 1) if fl else np.nan,
                             r2=round(r2, 2) if fl else np.nan, const_corr=round(ccorr, 3),
                             warp_corr=round(wcorr, 3) if fl else np.nan,
                             const_err=round(c_err, 1) if np.isfinite(c_err) else np.nan,
                             warp_err=round(w_err, 1) if fl and np.isfinite(w_err) else np.nan))
    df = pd.DataFrame(rows); df.to_csv(ROOT / "out/scoring/desync_correct.csv", index=False)
    print(f"trials {len(df)}\n")
    print(f"{'part':<6}{'n':>4}{'linear(r2>=.7,span>=5)':>22}{'corr const->warp':>20}{'err const->warp(mm)':>22}")
    for part, g in df.groupby("part"):
        lin = g[(g.r2 >= 0.7) & (g.drift_span >= 5)]
        allg = g.dropna(subset=["warp_corr"])
        cc, wc = allg.const_corr.median(), allg.warp_corr.median()
        ce, we = allg.const_err.median(), allg.warp_err.median()
        # on the linear subset specifically
        lc, lw = (lin.const_corr.median(), lin.warp_corr.median()) if len(lin) else (np.nan, np.nan)
        le, lwe = (lin.const_err.median(), lin.warp_err.median()) if len(lin) else (np.nan, np.nan)
        print(f"{part:<6}{len(g):>4}{f'{len(lin)} trials':>22}"
              f"{f'{lc:.2f}->{lw:.2f}':>20}{f'{le:.1f}->{lwe:.1f}':>22}")
    print("\n(corr/err shown on each part's LINEAR-drift subset; higher corr / lower err = warp helped)")
    print("wrote out/scoring/desync_correct.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
