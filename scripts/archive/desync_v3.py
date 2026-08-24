"""Which within-trial drifts are REAL? Add CUP + HEAD to the signal set, then judge each trial by
whether the independent signals AGREE on the drift.

Per half, estimate the lag from EACH signal independently (wrist/elbow/shoulder/cup/head x speed/disp).
median across signals = the half's lag; the SPREAD across signals = the uncertainty. A drift is REAL
only if |h2_lag - h1_lag| exceeds that spread (the shift is bigger than the disagreement) AND enough
signals are reliable in both halves. A trial where signals scatter more than they shift = NOT real.
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
from desync_drift import _local_lag
from score_vs_automq import COHORT_PARTS, FPS
GRID = R._GRID_JOINTS
MIN_CORR = 0.5
MIN_SIG = 3          # need >=3 reliable signals per half


def _pairs(t, part, trial, side, nfr, omc):
    """(name, a_mmc, b_omc) for wrist/elbow/shoulder + CUP + HEAD, speed & displacement."""
    mmc = {j: t["mmc"][:, GRID.index(j)] for j in GRID}
    out = []
    for j in ("wrist", "elbow", "shoulder"):
        jn = f"{side}_{j}"
        if jn in mmc and jn in omc:
            out.append((f"{j}_spd", H._lp(H._speed(mmc[jn])), H._lp(H._speed(omc[jn]))))
            out.append((f"{j}_disp", H._lp(H._disp_from_start(mmc[jn])), H._lp(H._disp_from_start(omc[jn]))))
    # HEAD (nose)
    if "nose" in mmc and "nose" in omc:
        out.append(("head_spd", H._lp(H._speed(mmc["nose"])), H._lp(H._speed(omc["nose"]))))
        out.append(("head_disp", H._lp(H._disp_from_start(mmc["nose"])), H._lp(H._disp_from_start(omc["nose"]))))
    # CUP (independent of pose keypoints): MMC v3 cup vs OMC cup
    mcup = R._cup_v3(part, trial, R._calib(part), nfr); ocup = R._omc_cup(part, trial, nfr)
    if np.isfinite(mcup).any() and np.isfinite(ocup).any():
        out.append(("cup_spd", H._lp(H._speed(mcup)), H._lp(H._speed(ocup))))
        out.append(("cup_disp", H._lp(H._disp_from_start(mcup)), H._lp(H._disp_from_start(ocup))))
    return out


def _half_lags(pairs, lo, hi):
    """Per-signal lag over [lo,hi) for signals above MIN_CORR. Returns list of (name, lag)."""
    res = []
    for name, a, b in pairs:
        L, c = _local_lag(a, b, lo, hi)
        if c >= MIN_CORR:
            res.append((name, L))
    return res


def main():
    H.use_good_cams()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    print(f"cohort {len(trials)}; signals = wrist/elbow/shoulder/head/cup x speed/disp", flush=True)
    rows = []; t0 = time.time(); n = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        if not pat.search(trial):
            continue
        nfr = t["mmc"].shape[0]
        omc = H._load_omc(part, trial, nfr)
        if f"{side}_wrist" not in omc:
            continue
        pairs = _pairs(t, part, trial, side, nfr, omc)
        if len(pairs) < 3:
            continue
        mid = nfr // 2
        h1 = _half_lags(pairs, 0, mid); h2 = _half_lags(pairs, mid, nfr)
        if len(h1) < MIN_SIG or len(h2) < MIN_SIG:
            rows.append(dict(part=part, trial=trial, n1=len(h1), n2=len(h2), real=0, reason="too_few_sig"))
            continue
        L1 = np.array([l for _, l in h1]); L2 = np.array([l for _, l in h2])
        m1, m2 = np.median(L1), np.median(L2)
        # robust spread across signals (MAD -> ~std)
        s1 = 1.4826 * np.median(np.abs(L1 - m1)); s2 = 1.4826 * np.median(np.abs(L2 - m2))
        spread = (s1 + s2) / 2 + 1e-6
        drift = m2 - m1
        # REAL if the shift exceeds the signal disagreement (variance)
        real = int(abs(drift) >= max(3.0, 1.5 * spread))
        rows.append(dict(part=part, trial=trial, n1=len(h1), n2=len(h2),
                         h1_lag=round(m1, 1), h2_lag=round(m2, 1), drift=round(drift, 1),
                         spread1=round(s1, 1), spread2=round(s2, 1), spread=round(spread, 1),
                         ratio=round(abs(drift) / spread, 2), real=real,
                         cup="cup_spd" in dict(h1) or "cup_disp" in dict(h1)))
        n += 1
        if n % 100 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s", flush=True)

    df = pd.DataFrame(rows); df.to_csv(ROOT / "out/scoring/desync_v3.csv", index=False)
    ok = df[df.get("drift").notna()].copy() if "drift" in df else df
    print(f"\nPROCESSING CHECK: trials {n}, with-estimate {len(ok)}, "
          f"too-few-signal {int((df.get('reason')=='too_few_sig').sum()) if 'reason' in df else 0}", flush=True)

    print("\n== REAL vs spurious drift by participant (real = |drift| > 1.5x signal-spread & >=3fr) ==")
    print(f"{'part':<6}{'n':>5}{'REAL':>6}{'med|drift|':>11}{'med spread':>11}{'med ratio':>10}")
    for p, g in ok.groupby("part"):
        gr = g.dropna(subset=["drift"])
        print(f"{p:<6}{len(gr):>5}{int(gr.real.sum()):>6}{gr['drift'].abs().median():>11.1f}"
              f"{gr['spread'].median():>11.1f}{gr['ratio'].median():>10.2f}")

    print("\n== REAL drifts, worst 15 (drift vs spread) ==")
    real = ok[ok.real == 1].dropna(subset=["drift"])
    for _, r in real.reindex(real['drift'].abs().sort_values(ascending=False).index).head(15).iterrows():
        print(f"  {r.part}/{r['trial']:<26} drift={r.drift:+6.1f}  spread={r.spread:>4.1f}  "
              f"ratio={r.ratio:>4.1f}  ({int(r.n1)}/{int(r.n2)} sig)")
    print(f"\ntotal REAL drifts: {int(ok.real.sum())}/{len(ok.dropna(subset=['drift']))}", flush=True)
    print("wrote out/scoring/desync_v3.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
