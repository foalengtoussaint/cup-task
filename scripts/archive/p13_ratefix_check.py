"""Does the P13 96Hz fix improve accuracy (not just the drift metric)? Load OMC the BUGGY way
(resample as 100Hz) vs the FIXED way (file's real 96Hz), and compare MMC<->OMC agreement per trial:
whole-trial wrist-speed correlation at the best single global lag, and per-frame wrist 3D error after
lag + Kabsch. A drifting OMC can't be aligned by one lag, so the fix should raise corr / lower error.
P07 (already 100Hz) is the control -- should not change.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import ezc3d, compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
GRID = R._GRID_JOINTS


def load_omc_at(part, trial, n, rate):
    c = ezc3d.c3d(str(H.DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]; P = c["data"]["points"]; T = P.shape[2]
    def marker(nm):
        return np.full((T, 3), np.nan) if nm not in L else P[:3, L.index(nm), :].T
    out = {}
    import warnings
    for joint, mks in H.JOINTS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = np.nanmean([marker(m) for m in mks], axis=0)
        grid = H._despike(H._resample(raw, rate, H.VIDEO_FPS))
        if len(grid) < n:
            grid = np.vstack([grid, np.full((n - len(grid), 3), np.nan)])
        out[joint] = grid[:n]
    return out


def _kabsch_err(mmc_w, omc_w):
    m = np.isfinite(mmc_w).all(1) & np.isfinite(omc_w).all(1)
    if m.sum() < 10:
        return np.nan
    A, B = mmc_w[m], omc_w[m]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T)); Rk = Vt.T @ np.diag([1, 1, d]) @ U.T
    pred = (Rk @ (A - ca).T).T + cb
    return float(np.median(np.linalg.norm(pred - B, axis=1)))


def eval_part(part, rate_label_pairs):
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] == part]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    res = {lbl: {"corr": [], "err": []} for lbl, _ in rate_label_pairs}
    for t in trials:
        trial, side = t["trial"], t["side"]
        if not pat.search(trial):
            continue
        nfr = t["mmc"].shape[0]; wr = f"{side}_wrist"
        mw = t["mmc"][:, GRID.index(wr)]
        for lbl, rate in rate_label_pairs:
            omc = load_omc_at(part, trial, nfr, rate)
            ow = omc[wr]
            lag, corr = H._find_lag(mw, ow)               # best single global lag + its speed-corr
            res[lbl]["corr"].append(corr)
            res[lbl]["err"].append(_kabsch_err(mw, R._shift(ow, lag)))
    return res


def main():
    H.use_good_cams()
    print("Per-trial MMC<->OMC agreement, OMC loaded at each rate. corr = wrist-speed corr at best lag; "
          "err = wrist 3D error (mm) after lag+Kabsch.\n")
    for part, pairs in [("P13", [("bug_100Hz", 100.0), ("fixed_96Hz", 96.0)]),
                        ("P07", [("100Hz", 100.0), ("as_96Hz", 96.0)])]:   # P07 control (truly 100Hz)
        r = eval_part(part, pairs)
        print(f"== {part} ==")
        for lbl in r:
            c = np.array(r[lbl]["corr"], float); e = np.array(r[lbl]["err"], float)
            print(f"  {lbl:<12} median speed-corr {np.nanmedian(c):.3f}   median wrist err {np.nanmedian(e):.1f}mm   n={np.isfinite(e).sum()}")
        print()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
