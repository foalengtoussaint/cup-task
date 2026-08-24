"""Solve BA minimising METRIC (mm) reprojection instead of PIXEL reprojection.

res_mm = res_px * z / f   -- converts each camera's residual to mm at that point's depth,
which is equivalent to weighting the squared pixel residual by z^2. The Huber knee is
converted with it (20 px at the cohort's median depth) so the robustness is comparable.

Whether px or mm is the right target is an empirical question about where the detector's
noise is constant; this produces the mm variant so it can be scored against OMC alongside
the cached px variant (cache/ba_variants/fix.npz).

    python scripts/ba_mm_objective.py     -> cache/ba_variants/mm.npz
"""
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

UVMAX, HUBER_PX, Z_REF = 1e6, 20.0, 1770.0     # Z_REF = cohort median depth (mm)


def _mm_residual(X, S, huber_px):
    B, Tn, J, _ = X.shape
    C = S["uv"].shape[1]
    Xp = X.unsqueeze(2).expand(B, Tn, C, J, 3)
    tot = torch.zeros((), device=X.device); wsum = torch.zeros((), device=X.device)
    for c in range(C):
        R, tt, K = S["R"][c], S["tt"][c], S["K"][c]
        uvp, infront = G.project_torch(Xp[:, :, c], K, S["dist"][c], R, tt)
        uvp = uvp.clamp(-UVMAX, UVMAX)
        res = torch.linalg.norm(uvp - S["uv"][None, :, c], dim=-1)          # px
        f = K[0, 0]
        z = (torch.einsum("btjk,mk->btjm", Xp[:, :, c], R) + tt)[..., 2].clamp(min=1.0)
        res = res * z / f                                                    # -> mm
        h = huber_px * Z_REF / f                                             # knee in mm
        rob = torch.where(res <= h, 0.5 * res * res / h, res - 0.5 * h)
        w = S["uvv"][None, :, c].float() * S["uvc"][None, :, c] * infront.float()
        tot = tot + (rob * w).sum(); wsum = wsum + w.sum()
    return tot, wsum


def main() -> None:
    BA._reproj_residual = _mm_residual
    trials = T.load_clean(need_reproj=True)
    OUT = ROOT / "cache" / "ba_variants"; OUT.mkdir(parents=True, exist_ok=True)
    traj, ids, n_fail = [], [], 0
    t0 = time.time()
    print(f"{len(trials)} trials, MM objective (huber knee {HUBER_PX}px @ z={Z_REF}mm)", flush=True)
    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        ids.append(f"{t['part']}/{t['trial']}")
        try:
            P, _ = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=0, fallback_mm=None)
        except Exception as e:
            print(f"  [fail] {ids[-1]}: {type(e).__name__}: {e}", flush=True)
            P = np.full_like(t["mmc"], np.nan); n_fail += 1
        traj.append(P.astype(np.float32))
        if (i + 1) % 40 == 0:
            fin = sum(np.isfinite(p).any() for p in traj)
            print(f"    [{i+1}/{len(trials)}] {time.time()-t0:5.0f}s  finite {fin}/{i+1}  "
                  f"fail {n_fail}", flush=True)
    p = OUT / "mm.npz"
    np.savez(p, ids=np.array(ids), traj=np.array(traj, dtype=object))
    print(f"\nPROCESSING CHECK: {len(trials)} trials, "
          f"{sum(np.isfinite(x).any() for x in traj)} finite, {n_fail} fail", flush=True)
    print(f"wrote {p}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
