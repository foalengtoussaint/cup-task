"""Pure-pixel sync check: cross-correlate each camera's MOTION ENERGY vs a reference, per trial.

Motion energy (mean |frame[t]-frame[t-1]|) is view-INVARIANT and calibration-INDEPENDENT: the
pattern of movement bursts lines up across viewpoints even though the images (and the geometry)
don't. So this separates the two failure modes cleanly, which the reprojection test cannot:

    lag ~= 0, high corr, consistent across trials   -> SYNCED   (any reproj error is pure MISCALIB)
    lag != 0, consistent across trials, high corr    -> DESYNC by that many frames (re-cut fixes)
    low corr at every lag                            -> too little motion / clip window mismatch

Unlike the reprojection lag-sweep, a camera that is BOTH desynced and miscalibrated still reveals
its timing lag here (geometry never enters), so this is the authoritative sync test.

    python scripts/motion_sync_delta.py --part P13 --ref 1 --maxlag 250
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from delta_recut import gray_frames, motion_signal  # noqa: E402
import compare_pose_omc_delta as C  # noqa: E402
from reaudit_cam_quality import _side, _trials  # noqa: E402


def xcorr_lag(a, b, maxlag):
    """Best integer lag L (b shifted by L to match a) and its normalized corr, over |L|<=maxlag.
    Positive L => b lags a (b's event happens L frames later)."""
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0, 0.0
    best_l, best_c = 0, -2.0
    n = len(a)
    for L in range(-maxlag, maxlag + 1):
        if L >= 0:
            x, y = a[L:], b[:n - L]
        else:
            x, y = a[:n + L], b[-L:]
        if len(x) < 10:
            continue
        c = float(np.dot(x - x.mean(), y - y.mean()) /
                  (np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()) + 1e-9))
        if c > best_c:
            best_c, best_l = c, L
    return best_l, best_c


def run(part, cam_ids, ref, maxlag, n_trials):
    side = _side(part)
    trials = _trials(part, n_trials)
    staged = C.DELTA / part / "staged"
    # per cam: list of (lag, corr) over trials
    acc = {c: [] for c in cam_ids}
    for t in trials:
        vids = {c: staged / f"delta_{part}_{t}.{c}.mp4" for c in set(cam_ids) | {ref}}
        vids = {c: v for c, v in vids.items() if v.exists()}
        if ref not in vids:
            continue
        sig = {}
        for c, v in vids.items():
            try:
                sig[c] = motion_signal(gray_frames(v))   # full fps
            except Exception:
                pass
        if ref not in sig:
            continue
        L0 = min(len(s) for s in sig.values())
        for c in cam_ids:
            if c not in sig or c == ref:
                continue
            lag, cor = xcorr_lag(sig[ref][:L0], sig[c][:L0], maxlag)
            acc[c].append((lag, cor))
        print(f"  {t}: " + "  ".join(
            f"cam{c}(L={acc[c][-1][0]:+d},r={acc[c][-1][1]:.2f})"
            for c in cam_ids if acc[c] and c != ref), flush=True)

    print(f"\n{part}  (ref=cam{ref}, side={side}, {len(trials)} trials)")
    print(f"{'cam':5}{'medLag':>8}{'lagSD':>7}{'medR':>7}   verdict")
    verdicts = {}
    for c in cam_ids:
        if c == ref:
            print(f"cam{c:<3}{'--':>8}{'--':>7}{'--':>7}   REFERENCE")
            continue
        if not acc[c]:
            print(f"cam{c:<3}{'n/a':>8}"); continue
        lags = np.array([x[0] for x in acc[c]]); cors = np.array([x[1] for x in acc[c]])
        ml, sd, mr = int(np.median(lags)), float(np.std(lags)), float(np.median(cors))
        if mr < 0.4:
            v = "LOW-CORR (can't judge: little motion / window mismatch)"
        elif abs(ml) <= 1 and sd <= 3:
            v = "SYNCED"
        elif sd <= 5:
            v = f"DESYNC ~{ml:+d} frames (consistent)"
        else:
            v = f"INCONSISTENT lag (sd={sd:.0f}) -- suspect bad detection/clip, not clean desync"
        print(f"cam{c:<3}{ml:>8}{sd:>7.1f}{mr:>7.2f}   {v}")
        verdicts[c] = (ml, sd, mr, v)
    return verdicts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cams", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--ref", type=int, default=1)
    ap.add_argument("--maxlag", type=int, default=250)
    ap.add_argument("--n-trials", type=int, default=8)
    a = ap.parse_args(argv)
    run(a.part, a.cams, a.ref, a.maxlag, a.n_trials)


if __name__ == "__main__":
    main()
