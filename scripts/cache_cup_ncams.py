"""Cache the cup 3D together with its per-frame CONSENSUS SIZE, so unconfirmed frames can be dropped.

cup_track returns a point whenever it can, including frames where only TWO cameras agree -- and a
2-camera cup is the regime whose errors cannot be detected. It is not that the point carries
nothing -- measured against the optical cup it is 60mm at the median -- but 21% of those frames are
>100mm out against 0.02% at >=3 cams, and with two views nothing identifies which. On P13 trial_34
they are 161 of 408 frames and they are what the segmenter then reads as a cup.
See paper/scripts/two_camera_cup.py; the older '>1m errors' claim does not reproduce.

This stores, per trial, the cup 3D and n_cams per frame, so a consumer can apply the >=3 floor
strictly. Cheap (~0.22 s/trial) but worth caching because every segmenter variant re-reads it.

    python scripts/cache_cup_ncams.py     # ~3 min for the 636-trial cohort
    -> cache/cup_ncams/<part>__<trial>.npz  {cup3d (T,3), n_cams (T,)}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "cache" / (__import__("os").environ.get("OT_NCAMS_DIR") or "cup_ncams")


def main():
    import compare_pose_omc_delta as H
    H.use_good_cams()
    import cache_seg_inputs as CSI
    import results_v3_delta as R
    from pipeline import cup_track
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = CSI.load_all()
    print(f"{len(recs)} trials -> {CACHE}", flush=True)
    t0 = time.time(); n_ok = n_skip = n_miss = 0
    for i, r in enumerate(recs):
        out = CACHE / f"{r['part']}__{r['trial']}.npz"
        if out.exists():
            n_skip += 1
        else:
            p = R.TRACKS / f"{r['part']}__{r['trial']}__uetrack__fs1.json"
            if not p.exists():
                n_miss += 1
            else:
                tr = cup_track.track_cup_3d_from_cache(p, R._calib(r["part"]))
                T = len(r["cup_mmc"])
                X = np.full((T, 3), np.nan); nc = np.zeros(T, int)
                for t in tr:
                    f = int(t["frame"])
                    if f < T:
                        nc[f] = int(t.get("n_cams") or 0)
                        if t.get("X"):
                            X[f] = t["X"]
                np.savez_compressed(out, cup3d=X, n_cams=nc)
                n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(recs)}] {time.time()-t0:4.0f}s  new {n_ok} skip {n_skip} "
                  f"missing-track {n_miss}", flush=True)
    print(f"\nPROCESSING CHECK: trials {len(recs)}, new {n_ok}, skipped {n_skip}, "
          f"missing track file {n_miss}, files {len(list(CACHE.glob('*.npz')))}", flush=True)

    # coverage summary: how much of the cup is 2-camera (i.e. below the >=3 floor)
    tot = np.zeros(6, int); per = []
    for f in sorted(CACHE.glob("*.npz")):
        nc = np.load(f)["n_cams"]
        for k in range(6):
            tot[k] += int((nc == k).sum())
        per.append(dict(trial=f.stem, frac_lt3=float((nc < 3).mean()),
                        frac_eq2=float((nc == 2).mean())))
    import pandas as pd
    P = pd.DataFrame(per)
    P.to_csv(ROOT / "out/scoring/cup_ncams_coverage.csv", index=False)
    print("frames by n_cams: " + "  ".join(f"{k}:{v}" for k, v in enumerate(tot)))
    print(f"per-trial fraction below the >=3 floor: median {P.frac_lt3.median():.3f}  "
          f"p90 {P.frac_lt3.quantile(.9):.3f}  trials >30% {int((P.frac_lt3 > .3).sum())}")
    print("\nwrote out/scoring/cup_ncams_coverage.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
