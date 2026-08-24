"""Solve BA ONCE per config over the cohort and cache the refined trajectories.

Every downstream question -- revert rate, run lengths, OMC agreement, gate overlap,
guard-threshold sweeps -- is arithmetic on these arrays. Re-solving per question wastes
minutes of GPU and is what CLAUDE.md's cache-the-solve-outputs rule exists to prevent
(this file was written after re-solving the same 836 trials five times).

Configs:
  base : the ORIGINAL (pre-2026-08-20) unclamped residual, kept here so the bug stays reproducible
  fix  : the shipped ba_refine._reproj_residual -- projected uv clamped to +-UVMAX before the norm,
         so a behind-camera point gives a large-but-finite residual instead of inf (inf * w=0 -> NaN
         kills the line search). This is now the default in ba_refine.py.

Both are stored UNGUARDED (no fallback_mm). The guard is a post-hoc threshold, so any
value of it can be applied to the cache without re-solving.

    python scripts/cache_ba_variants.py            -> cache/ba_variants/<cfg>.npz
    python scripts/cache_ba_variants.py --cfg fix
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                     # noqa: E402
import gnn_refiner as G                   # noqa: E402
import ba_refine as BA                    # noqa: E402

OUT = ROOT / "cache" / "ba_variants"
_shipped = BA._reproj_residual                # now ALREADY clamped -- this is the "fix" config


def _unclamped(X, S, huber_px):
    """The pre-fix residual: no clamp on uvp, so a behind-camera point overflows to inf and
    inf*0 -> NaN. Retained ONLY to reproduce the bug for the base-vs-fix comparison."""
    B, Tn, J, _ = X.shape
    C = S["uv"].shape[1]
    Xp = X.unsqueeze(2).expand(B, Tn, C, J, 3)
    tot = torch.zeros((), device=X.device); wsum = torch.zeros((), device=X.device)
    for c in range(C):
        uvp, infront = G.project_torch(Xp[:, :, c], S["K"][c], S["dist"][c], S["R"][c], S["tt"][c])
        res = torch.linalg.norm(uvp - S["uv"][None, :, c], dim=-1)
        rob = torch.where(res <= huber_px, 0.5 * res * res / huber_px, res - 0.5 * huber_px)
        w = S["uvv"][None, :, c].float() * S["uvc"][None, :, c] * infront.float()
        tot = tot + (rob * w).sum(); wsum = wsum + w.sum()
    return tot, wsum


CFG = {"base": _unclamped, "fix": _shipped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", nargs="*", default=["base", "fix"], choices=list(CFG))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    trials = T.load_clean(need_reproj=True)
    print(f"{len(trials)} trials, configs {a.cfg} -> {OUT}", flush=True)

    for cfg in a.cfg:
        BA._reproj_residual = CFG[cfg]
        traj, ids, n_fail = [], [], 0
        t0 = time.time()
        for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
            ids.append(f"{t['part']}/{t['trial']}")
            try:
                P, _ = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=0, fallback_mm=None)
            except Exception as e:
                print(f"  [solve fail] {ids[-1]}: {type(e).__name__}: {e}", flush=True)
                P = np.full_like(t["mmc"], np.nan); n_fail += 1
            traj.append(P.astype(np.float32))
            if (i + 1) % 40 == 0:
                fin = sum(np.isfinite(p).any() for p in traj)
                print(f"    [{cfg}] [{i+1}/{len(trials)}] {time.time()-t0:5.0f}s  "
                      f"finite {fin}/{i+1}  fail {n_fail}", flush=True)
        p = OUT / f"{cfg}.npz"
        np.savez(p, ids=np.array(ids), traj=np.array(traj, dtype=object))
        fin = sum(np.isfinite(x).any() for x in traj)
        print(f"\nPROCESSING CHECK [{cfg}]: {len(trials)} trials, {fin} finite, {n_fail} solve-fail",
              flush=True)
        print(f"wrote {p}  ({time.time()-t0:.0f}s)", flush=True)
    BA._reproj_residual = _shipped
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
