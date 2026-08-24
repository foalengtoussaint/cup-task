"""Apply the pipeline's fallback_mm guard to cached (unguarded) BA variants.

cache/ba_variants/*.npz are stored UNGUARDED so any threshold can be swept post hoc.
The shipped pipeline applies fallback_mm=150 inside refine_trial_ba, BEFORE SmoothNet.
This reproduces that so variants compare like-for-like with ba_sn.

    python scripts/apply_guard_variants.py --tags freebone_0.05 fix --mm 150
    -> cache/ba_variants/<tag>__g150.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--mm", type=float, default=150.0)
    a = ap.parse_args()
    VAR = ROOT / "cache" / "ba_variants"
    mmc = {f"{t['part']}/{t['trial']}": t["mmc"] for t in T.load_clean(need_reproj=True)}
    for tag in a.tags:
        z = np.load(VAR / f"{tag}.npz", allow_pickle=True)
        ids, traj = list(z["ids"]), list(z["traj"])
        out, n_rev, n_cell, n_tot = [], 0, 0, 0
        for tid, P in zip(ids, traj):
            P = np.asarray(P, float).copy()
            M = mmc.get(tid)
            if M is None or M.shape != P.shape:
                out.append(P.astype(np.float32)); continue
            # float64 + treat non-finite `mv` as a revert: at float32 a 1e20 blow-up squares to
            # 1e40 and overflows to inf, and the old `np.isfinite(mv) &` term read that as "keep".
            # Same escape as ba_refine's in-solve guard; see the note there.
            mv = np.linalg.norm(P.astype(np.float64) - M.astype(np.float64), axis=-1)
            fin = np.isfinite(M).all(-1)
            bad = (~np.isfinite(P).all(-1) | ~np.isfinite(mv) | (mv > a.mm)) & fin
            P[bad] = M[bad]
            n_cell += int(bad.sum()); n_tot += int(fin.sum()); n_rev += int(bad.any())
            out.append(P.astype(np.float32))
        p = VAR / f"{tag}__g{int(a.mm)}.npz"
        np.savez(p, ids=np.array(ids), traj=np.array(out, dtype=object))
        print(f"PROCESSING CHECK [{tag}__g{int(a.mm)}]: {len(out)} trials, "
              f"{n_rev} with >=1 revert, {n_cell:,}/{n_tot:,} joint-frames reverted "
              f"({100*n_cell/max(n_tot,1):.3f}%)", flush=True)
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
