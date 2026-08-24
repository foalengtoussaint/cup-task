"""Run SmoothNet on cached BA variants, so in-solve terms and SmoothNet can be combined.

The shipped pipeline is BA -> SmoothNet. This applies the same SmoothNet pass to the
free-bone / in-solve-smoothed variants, to test whether a rigid-bone solve plus SmoothNet
beats either alone.

    python scripts/apply_smoothnet_variants.py --tags freebone_0.05_s0.3 fix
    -> cache/ba_variants/<tag>__sn.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_refiner as G                   # noqa: E402
from pipeline import pose_smooth as PS    # noqa: E402

JOINTS = G.JOINTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    VAR = ROOT / "cache" / "ba_variants"
    for tag in a.tags:
        z = np.load(VAR / f"{tag}.npz", allow_pickle=True)
        ids, traj = list(z["ids"]), list(z["traj"])
        out, n_skip = [], 0
        t0 = time.time()
        for P in tqdm(traj, mininterval=3, ncols=90, file=sys.stdout, desc=tag):
            P = np.asarray(P, float)
            if P.ndim != 3 or not np.isfinite(P).any():
                out.append(P.astype(np.float32)); n_skip += 1; continue
            try:
                d = PS.smooth_joints_batched({j: P[:, k] for k, j in enumerate(JOINTS)})
                out.append(np.stack([d[j] for j in JOINTS], 1).astype(np.float32))
            except Exception as e:
                print(f"  [fail] {type(e).__name__}: {str(e)[:60]}", flush=True)
                out.append(P.astype(np.float32)); n_skip += 1
        p = VAR / f"{tag}__sn.npz"
        np.savez(p, ids=np.array(ids), traj=np.array(out, dtype=object))
        fin = sum(np.isfinite(x).any() for x in out)
        print(f"PROCESSING CHECK [{tag}__sn]: {len(out)} trials, {fin} finite, "
              f"{n_skip} skipped ({time.time()-t0:.0f}s)", flush=True)
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
