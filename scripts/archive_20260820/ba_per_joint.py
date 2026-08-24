"""Solve each joint with its OWN optimizer instead of one L-BFGS over the whole body.

The shipped loss has NO term linking joints (lam_bone=0, smooth_w=0, anchor_mm_w=0) --
verified: perturbing one joint changes no other joint's gradient. So the problem is
already separable; the only thing coupling joints is that all 11,718 parameters share one
L-BFGS step length and one curvature history. This solves each joint independently,
which should leave well-fitted joints (the shoulders) alone.

Includes the uv clamp fix (behind-camera -> finite residual, not inf*0=NaN).

    python scripts/ba_per_joint.py [--limit N] [--parts P15 P19]
    -> cache/ba_variants/perjoint.npz
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

UVMAX, HUBER, ITERS = 1e6, 20.0, 60


def solve_joint(S, X0j, j):
    """One joint: X (1,T,1,3), its own L-BFGS. Returns (T,3)."""
    X = X0j.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([X], lr=1.0, max_iter=ITERS, history_size=20,
                            line_search_fn="strong_wolfe")
    C = S["uv"].shape[1]

    def closure():
        opt.zero_grad()
        tot = torch.zeros((), device=X.device); wsum = torch.zeros((), device=X.device)
        for c in range(C):
            uvp, infront = G.project_torch(X, S["K"][c], S["dist"][c], S["R"][c], S["tt"][c])
            uvp = uvp.clamp(-UVMAX, UVMAX)
            res = torch.linalg.norm(uvp - S["uv"][None, :, c, j:j+1], dim=-1)
            rob = torch.where(res <= HUBER, 0.5 * res * res / HUBER, res - 0.5 * HUBER)
            w = S["uvv"][None, :, c, j:j+1].float() * S["uvc"][None, :, c, j:j+1] * infront.float()
            tot = tot + (rob * w).sum(); wsum = wsum + w.sum()
        loss = tot / wsum.clamp(min=1.0)
        loss.backward()
        return loss

    opt.step(closure)
    return X.detach()[0, :, 0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--out", default="perjoint")
    a = ap.parse_args()
    trials = T.load_clean(need_reproj=True)
    if a.parts:
        trials = [t for t in trials if t["part"] in a.parts]
    if a.limit:
        trials = trials[:a.limit]
    print(f"{len(trials)} trials, {len(G.JOINTS)} joints each, own optimizer per joint", flush=True)

    OUT = ROOT / "cache" / "ba_variants"; OUT.mkdir(parents=True, exist_ok=True)
    traj, ids, n_fail = [], [], 0
    t0 = time.time()
    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        ids.append(f"{t['part']}/{t['trial']}")
        mmc = t["mmc"]
        try:
            S = BA._to_dev(t)
            X0 = torch.from_numpy(np.nan_to_num(mmc).astype(np.float32)).cuda()
            P = np.empty_like(mmc, dtype=np.float32)
            for j in range(mmc.shape[1]):
                P[:, j] = solve_joint(S, X0[None, :, j:j+1], j)
        except Exception as e:
            print(f"  [fail] {ids[-1]}: {type(e).__name__}: {e}", flush=True)
            P = np.full_like(mmc, np.nan); n_fail += 1
        traj.append(P.astype(np.float32))
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(trials)}] {time.time()-t0:5.0f}s  fail {n_fail}  "
                  f"({(time.time()-t0)/(i+1):.2f}s/trial)", flush=True)
    p = OUT / f"{a.out}.npz"
    np.savez(p, ids=np.array(ids), traj=np.array(traj, dtype=object))
    fin = sum(np.isfinite(x).any() for x in traj)
    print(f"\nPROCESSING CHECK: {len(trials)} trials, {fin} finite, {n_fail} fail", flush=True)
    print(f"wrote {p}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
