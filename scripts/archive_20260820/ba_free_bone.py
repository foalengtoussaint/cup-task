"""Bone CONSISTENCY without a target length.

The shipped prior pins each bone to L_uv = the per-trial median RAW DLT length, so it
imports whatever bias that estimate has. This variant penalises only the VARIANCE of each
bone length over the trial -- the optimizer may choose any length, it just may not change
during the trial:

    shipped : sum_t ( d(t) - L_uv )^2          L_uv fixed, from DLT
    free    : sum_t ( d(t) - mean_t d )^2      length free, only variance penalised

Same bones (LIMBS), same lam scale, so results are directly comparable to bonesub_*.

    python scripts/ba_free_bone.py --limit 120 --lams 0.01 0.05 0.2
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

UVMAX = 1e6
_orig_res, _orig_bone_e = BA._reproj_residual, BA._bone_energy


def _fixed(X, S, huber_px):
    B, Tn, J, _ = X.shape
    C = S["uv"].shape[1]
    Xp = X.unsqueeze(2).expand(B, Tn, C, J, 3)
    tot = torch.zeros((), device=X.device); wsum = torch.zeros((), device=X.device)
    for c in range(C):
        uvp, infront = G.project_torch(Xp[:, :, c], S["K"][c], S["dist"][c], S["R"][c], S["tt"][c])
        uvp = uvp.clamp(-UVMAX, UVMAX)
        res = torch.linalg.norm(uvp - S["uv"][None, :, c], dim=-1)
        rob = torch.where(res <= huber_px, 0.5 * res * res / huber_px, res - 0.5 * huber_px)
        w = S["uvv"][None, :, c].float() * S["uvc"][None, :, c] * infront.float()
        tot = tot + (rob * w).sum(); wsum = wsum + w.sum()
    return tot, wsum


def _free_bone(X, L):
    """Variance of each bone length. L is used only for its KEYS (which bones)."""
    e = torch.zeros((), device=X.device)
    for (u, v) in L:
        d = torch.linalg.norm(X[:, :, u] - X[:, :, v], dim=-1)        # (B,T)
        e = e + ((d - d.mean(dim=1, keepdim=True)) ** 2).sum()
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.01, 0.05, 0.2])
    ap.add_argument("--smooth", type=float, default=0.0)
    a = ap.parse_args()
    BA._reproj_residual = _fixed
    BA._bone_energy = _free_bone
    allt = T.load_clean(need_reproj=True)
    trials = allt[::max(1, len(allt) // a.limit)][:a.limit]
    OUT = ROOT / "cache" / "ba_variants"; OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(trials)} trials, FREE bone (variance only), lams {a.lams}, "
          f"smooth_w={a.smooth}", flush=True)
    for lam in a.lams:
        traj, ids, n_fail = [], [], 0
        t0 = time.time()
        for t in tqdm(trials, mininterval=3, ncols=90, file=sys.stdout):
            ids.append(f"{t['part']}/{t['trial']}")
            try:
                P, _ = BA.refine_trial_ba(t, lam, iters=60, smooth_w=a.smooth, fallback_mm=None)
            except Exception as e:
                print(f"  [fail] {ids[-1]}: {type(e).__name__}: {e}", flush=True)
                P = np.full_like(t["mmc"], np.nan); n_fail += 1
            traj.append(P.astype(np.float32))
        tag = f"freebone_{lam}" + (f"_s{a.smooth}" if a.smooth else "")
        np.savez(OUT / f"{tag}.npz", ids=np.array(ids), traj=np.array(traj, dtype=object))
        print(f"PROCESSING CHECK [{tag}]: {len(trials)} trials, "
              f"{sum(np.isfinite(x).any() for x in traj)} finite, {n_fail} fail "
              f"({time.time()-t0:.0f}s)", flush=True)
    BA._reproj_residual = _orig_res; BA._bone_energy = _orig_bone_e
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
