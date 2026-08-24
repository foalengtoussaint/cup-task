"""Re-test the bone-length prior with the FIXED solver.

The prior was ruled harmful ("slides joints along the unobservable depth ray") on evidence
collected before the 0*inf NaN bug was found -- when 60/836 trials were diverging to
non-finite and reverting to DLT. Two reasons to re-check:

  1. That evaluation was contaminated by the solver bug.
  2. The stated mechanism assumes depth is unobservable. Measured on this rig it is not:
     normal-matrix condition numbers are 2.3-3.8 and the weakest eigenvector is only 0.488
     aligned with the camera ray -- close to isotropic.

Same sweep range as ba_refine's own CLI default. Judged against OMC (alignment-free bone
SD + excursion), not against the energy, per the note in ba_refine.py:227.

    python scripts/ba_bone_resweep.py     -> cache/ba_variants/bone_<lam>.npz
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
_orig = BA._reproj_residual


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", nargs="+", type=float, default=[0.01, 0.05, 0.2, 1.0])
    ap.add_argument("--limit", type=int, default=None,
                    help="score a SUBSET first -- a prior that hurts will show up on ~120 trials")
    ap.add_argument("--tag", default="bone")
    a = ap.parse_args()
    BA._reproj_residual = _fixed                      # the fix, for every lam
    trials = T.load_clean(need_reproj=True)
    if a.limit:
        # spread across participants, not the first N (which would be P07 only)
        trials = trials[::max(1, len(trials) // a.limit)][:a.limit]
    OUT = ROOT / "cache" / "ba_variants"; OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(trials)} trials, lam_bone sweep {a.lams} (fixed solver)", flush=True)

    for lam in a.lams:
        traj, ids, n_fail = [], [], 0
        t0 = time.time()
        for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
            ids.append(f"{t['part']}/{t['trial']}")
            try:
                P, _ = BA.refine_trial_ba(t, lam, iters=60, smooth_w=0, fallback_mm=None)
            except Exception as e:
                print(f"  [fail] {ids[-1]}: {type(e).__name__}: {e}", flush=True)
                P = np.full_like(t["mmc"], np.nan); n_fail += 1
            traj.append(P.astype(np.float32))
            if (i + 1) % 100 == 0:
                print(f"    [lam={lam}] [{i+1}/{len(trials)}] {time.time()-t0:5.0f}s "
                      f"fail {n_fail}", flush=True)
        p = OUT / f"{a.tag}_{lam}.npz"
        np.savez(p, ids=np.array(ids), traj=np.array(traj, dtype=object))
        print(f"\nPROCESSING CHECK [lam={lam}]: {len(trials)} trials, "
              f"{sum(np.isfinite(x).any() for x in traj)} finite, {n_fail} fail", flush=True)
        print(f"wrote {p}  ({time.time()-t0:.0f}s)", flush=True)
    BA._reproj_residual = _orig
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
