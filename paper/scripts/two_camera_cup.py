"""How bad is a two-camera cup, actually?

Three places in this repository assert that a 2-camera cup "is not weak evidence, it is not
evidence", citing a robustness study that reported >1 m errors. Neither half of that survives
measurement, so this script is what should be cited instead.

METHOD. The markerless and optical cups live in different world frames, so their raw positional
difference is meaningless (it is ~1-2 m and grows with camera count, which is the giveaway). Fit one
rigid transform per trial by Kabsch, USING ONLY the >=3-camera frames, then measure the residual on
the 2-camera frames, which are held out of the fit. The >=3 numbers are therefore optimistic -- they
are in-sample -- and the 2-camera ones are not.

    python paper/scripts/two_camera_cup.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "cache" / "seg_inputs_ship"
NCAMS = ROOT / "cache" / "cup_ncams_26x"


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def main() -> None:
    by = {}
    n_trials = 0
    for f in sorted(glob.glob(str(SEG / "*.npz"))):
        nf = NCAMS / Path(f).name
        if not nf.exists():
            continue
        z, d = np.load(f, allow_pickle=True), np.load(nf)
        cm, co, nc = z["cup_mmc"], z["cup_omc"], d["n_cams"]
        T = min(len(cm), len(co), len(nc))
        cm, co, nc = cm[:T], co[:T], nc[:T]
        ok = np.isfinite(cm).all(1) & np.isfinite(co).all(1)
        fit = ok & (nc >= 3)
        if fit.sum() < 50:
            continue
        R, t = kabsch(cm[fit], co[fit])
        e = np.linalg.norm((R @ cm.T).T + t - co, axis=1)
        n_trials += 1
        for k in (2, 3, 4, 5):
            m = ok & (nc == k)
            if m.any():
                by.setdefault(k, []).append(e[m])

    print(f"{n_trials} trials; transform fitted on >=3-camera frames, 2-camera frames held out\n")
    print(f"{'cams':>5} {'frames':>9} {'median':>10} {'p90':>9} {'p99':>9} "
          f"{'>100mm':>8} {'>500mm':>8} {'>1m':>7}")
    for k in sorted(by):
        e = np.concatenate(by[k])
        print(f"{k:>5} {len(e):>9} {np.median(e):>9.1f}mm {np.percentile(e, 90):>8.1f} "
              f"{np.percentile(e, 99):>8.1f} {100 * (e > 100).mean():>7.2f}% "
              f"{100 * (e > 500).mean():>7.2f}% {100 * (e > 1000).mean():>6.2f}%")

    a = np.concatenate(by[2])
    b = np.concatenate([x for k in (3, 4, 5) for x in by.get(k, [])])
    print(f"\n2-camera frames are {(a > 100).mean() / max((b > 100).mean(), 1e-12):.0f}x more likely "
          f"to be >100mm out than >=3-camera ones.")
    print(f"Frames beyond 1 m: 2-camera {100 * (a > 1000).mean():.2f}%, "
          f"reference claim was '>1m errors'.")


if __name__ == "__main__":
    main()
