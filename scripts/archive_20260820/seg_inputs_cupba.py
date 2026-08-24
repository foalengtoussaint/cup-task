"""Derive a segmenter-input cache whose MMC cup is the BUNDLE-ADJUSTED cup, everything else identical.

seg_inputs stores cup_mmc = _smooth_joint(consensus cup). This rewrites ONLY that channel with
_smooth_joint(ba_all cup) from cache/cup_ba, leaving the OMC channels, the lag, the phase windows and
the wrist untouched -- so a Table IV difference is attributable to the cup solve and nothing else.

    python scripts/seg_inputs_cupba.py --variant ba_all   -> cache/seg_inputs_cupba/
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H     # noqa: E402
import results_v3_delta as R           # noqa: E402

SRC = ROOT / "cache" / "seg_inputs_final"
DST = ROOT / "cache" / "seg_inputs_cupba"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ba_all")
    a = ap.parse_args()
    H.use_good_cams()
    z = np.load(ROOT / "cache" / "cup_ba" / f"{a.variant}.npz", allow_pickle=True)
    cup = {str(i): np.asarray(t, float) for i, t in zip(z["ids"], z["traj"])}
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    n_sub = n_keep = 0
    files = sorted(SRC.glob("*.npz"))
    for i, f in enumerate(files):
        d = dict(np.load(f, allow_pickle=True))
        key = f"{str(d['part'])}/{str(d['trial'])}"
        X = cup.get(key)
        if X is not None and X.shape == d["cup_mmc"].shape:
            d["cup_mmc"] = R._smooth_joint(X).astype(d["cup_mmc"].dtype)
            n_sub += 1
        else:
            n_keep += 1
        np.savez_compressed(DST / f.name, **d)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] substituted {n_sub} kept-consensus {n_keep}", flush=True)
    print(f"\nPROCESSING CHECK: {len(files)} trials, cup replaced {n_sub}, "
          f"left as consensus {n_keep} -> {DST}", flush=True)
    print("DONE_SUB", flush=True)


if __name__ == "__main__":
    main()
