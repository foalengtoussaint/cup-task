"""Is `l1` NON-INFERIOR to the LOO consensus? (not "is it better" -- a different question)

WHY THIS AND NOT fuser_validate.py's TABLE. The preference here is mechanistic, not numeric: `l1`
carries ZERO tuned constants, while `loo` carries `tol=20.0` AND a `max_drop` cap whose own docstring
says the optimum DIFFERS BY TARGET (wrist 2, cup 3). A per-target constant is a fork in the code that
has to be justified per target. So if `l1` merely TIES, it is preferable -- and "ties" is a claim
about a confidence interval, not about which median is smaller.

Three things fuser_validate.py did not do:

  1. BOOTSTRAP CI on the paired per-peak difference (l1 - loo), clustered BY TRIAL. Peaks within a
     trial share a camera geometry and a participant, so treating 24 peaks as 24 independent samples
     overstates precision. Resampling trials, not peaks, is the honest unit.

  2. loo AT ITS DOCUMENTED CAP as well as uncapped. Uncapped is what production runs, but the
     docstring's measured optimum is per-target; comparing against only one of those is comparing
     against a moving target.

  3. WORST-CASE, not just central tendency. Non-inferiority for a peak measure means the TAIL does
     not get worse -- p90 and max matter more than the median.

    python scripts/fuser_noninferior.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from flow_3d_survival import jacobian, solve_velocity

MOVING = 50.0
BOOT = 10000


def _tracks(px, fl, calib, n, variants, fps=60.0):
    """Per-frame 3D speed for each named variant, from ONE pass (shared triangulation + Jacobians)."""
    from cup_task import flow_speed
    from cup_task.kalman_3d import triangulate_dlt

    out = {k: np.full(n, np.nan) for k in variants}
    cams = [c for c in fl if c in px and c in calib]
    for f in range(n):
        obs = {c: px[c][f] for c in cams
               if f < len(px[c]) and f < len(fl[c])
               and np.isfinite(px[c][f]).all() and np.isfinite(fl[c][f]).all()}
        if len(obs) < 2:
            continue
        X = triangulate_dlt([calib[c] for c in obs], [np.asarray(obs[c]) for c in obs])
        if X is None:
            continue
        rows, use = [], []
        for c in obs:
            J = jacobian(calib[c], X)
            if J is not None:
                rows.append((J, np.asarray(fl[c][f]))); use.append(c)
        if len(rows) < 2:
            continue
        P = {c: obs[c] for c in use}
        PV = {c: obs[c] + fl[c][f] for c in use}
        for k, (kind, cap) in variants.items():
            if kind == "loo":
                keep = (flow_speed.flow_consensus_cams(P, PV, calib, max_drop=cap)
                        if len(P) >= 3 else list(P))
                sub = [rows[i] for i, c in enumerate(use) if c in keep]
                v = solve_velocity(sub if len(sub) >= 2 else rows, "plain", fps)
            else:
                v = solve_velocity(rows, kind, fps)
            if v is not None:
                out[k][f] = float(np.linalg.norm(v))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args(argv)

    from scipy.signal import find_peaks
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    import flow_velocity_probe as F

    H.use_good_cams()
    # (solver mode, max_drop). Both loo settings, because the docstring's optimum is per-target
    # while production ships uncapped.
    VAR = {"loo_uncapped": ("loo", None), "loo_cap2": ("loo", 2), "loo_cap3": ("loo", 3),
           "l1": ("l1", None), "trimmed": ("trimmed", None)}
    names = list(VAR)
    # peak errors keyed by trial, so the bootstrap can resample TRIALS
    peaks = {t: {k: {} for k in names} for t in ("wrist", "cup")}
    mv = {t: {k: [] for k in names} for t in ("wrist", "cup")}
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            ow = R._shift(omc[f"{side}_wrist"], lag)
            oc = R._shift(R._omc_cup(part, trial, n), lag)
            key = f"{part}_{trial}"

            for tname, (px, cdir, truth) in {
                    "wrist": (F.load_wrist_px(part, trial, f"{side}_wrist"), "flow_vel", ow),
                    "cup":   (C.cup_px(part, trial, n), "cup_flow_vel", oc)}.items():
                fl = {}
                for cam in px:
                    p = (ROOT / "cache" / cdir /
                         f"delta_{part}_{trial}.{cam.split('_')[1]}__pyrlk.npy")
                    if p.exists() and cam in calib:
                        fl[cam] = np.load(p)
                if not fl:
                    continue
                so = H._lp(H._speed(truth))
                tr = _tracks(px, fl, calib, n, VAR)
                opk = find_peaks(so, height=300, distance=30, prominence=150)[0]
                for k in names:
                    s = H._lp(tr[k])
                    m = np.isfinite(s) & np.isfinite(so) & (so > MOVING)
                    if m.sum() > 20:
                        mv[tname][k].append(float(np.median(np.abs(s[m] - so[m]))))
                    pk, _ = find_peaks(s, height=200, distance=25, prominence=100)
                    ers = []
                    for p_ in opk:
                        if len(pk):
                            j = pk[np.argmin(np.abs(pk - p_))]
                            if abs(j - p_) <= 20:
                                ers.append(float(abs(s[j] - so[p_])))
                    peaks[tname][k][key] = ers
            print(f"  {part}_{trial.split('_')[1]:>10}  ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    for tname in ("wrist", "cup"):
        keys = sorted(k for k in peaks[tname]["l1"] if peaks[tname]["l1"][k])
        npk = sum(len(peaks[tname]["l1"][k]) for k in keys)
        print(f"\n=== {tname.upper()}  ({len(keys)} trials, {npk} peaks) ===")
        print(f"{'variant':14} {'moving':>7} | {'peak med':>9} {'p90':>7} {'max':>8}")
        print("-" * 52)
        for k in names:
            allp = np.array([e for kk in keys for e in peaks[tname][k][kk]])
            if not len(allp):
                continue
            print(f"{k:14} {np.median(mv[tname][k]):7.2f} | {np.median(allp):9.2f} "
                  f"{np.percentile(allp,90):7.2f} {allp.max():8.2f}")

        # ---- paired bootstrap, resampling TRIALS (the independent unit) ----
        print(f"\n  paired l1 - <baseline>, bootstrap over TRIALS (n={len(keys)}), {BOOT} draws.")
        print(f"  Negative = l1 better. Non-inferiority = the CI's UPPER end is small.")
        print(f"  {'baseline':14} {'d median':>9} {'95% CI':>20} {'d p90':>8} {'d max':>8}")
        for base in ("loo_uncapped", "loo_cap2", "loo_cap3"):
            per = {}
            for kk in keys:
                x = np.array(peaks[tname]["l1"][kk]); y = np.array(peaks[tname][base][kk])
                if len(x) and len(x) == len(y):
                    per[kk] = x - y
            kk2 = sorted(per)
            if not kk2:
                continue
            allp = np.concatenate([per[k] for k in kk2])
            bs = np.empty(BOOT)
            for i in range(BOOT):
                pick = rng.choice(len(kk2), len(kk2), replace=True)
                bs[i] = np.median(np.concatenate([per[kk2[j]] for j in pick]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"  {base:14} {np.median(allp):+9.2f}   [{lo:+7.2f},{hi:+7.2f}] "
                  f"{np.percentile(allp,90):+8.2f} {allp.max():+8.2f}")

    print("\n  A CI straddling 0 means the two fusers are NOT DISTINGUISHABLE on this cohort.")
    print("  In that case prefer l1: it carries no `tol` and no per-target `max_drop`.")


if __name__ == "__main__":
    main()
