"""Two prior variants on a subset: per-PARTICIPANT bone lengths, and in-solve smoothing.

(a) per-participant bones. The shipped prior targets the per-TRIAL median raw length. OMC says
    a forearm varies 1.05mm across a person's trials; MMC's per-trial estimate varies 6.57mm --
    so ~5-6mm of the target is noise, re-estimated every trial. Using the participant's median
    across all their trials removes that (still self-supervised, no OMC).

(b) smooth_w, calibrated against OMC rather than guessed: OMC mean|acc|^2 = 0.115 mm^2,
    BA = 19.36 (168x), data term ~13.1 -> the smooth term equals the data term at
    smooth_w~0.68, so 0.3-10 brackets "barely bites" to "dominates".

    python scripts/ba_prior_tests.py --limit 120
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
_orig_res = BA._reproj_residual
_orig_bone = BA.trial_bone_lengths
_PART_L = {}          # part -> {(u,v): L}
_CUR = {"part": None}


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


def _part_bones(mmc, valid):
    """Per-participant reference, falling back to the per-trial one if unavailable."""
    L = _PART_L.get(_CUR["part"])
    return L if L else _orig_bone(mmc, valid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    a = ap.parse_args()
    BA._reproj_residual = _fixed
    allt = T.load_clean(need_reproj=True)

    # per-participant reference from ALL of that participant's trials
    per = {}
    for t in allt:
        for k, v in _orig_bone(t["mmc"], t["valid"]).items():
            per.setdefault(t["part"], {}).setdefault(k, []).append(v)
    for p, d in per.items():
        _PART_L[p] = {k: float(np.median(v)) for k, v in d.items()}
    print(f"per-participant bone refs built for {len(_PART_L)} participants", flush=True)

    trials = allt[::max(1, len(allt) // a.limit)][:a.limit]
    OUT = ROOT / "cache" / "ba_variants"; OUT.mkdir(parents=True, exist_ok=True)
    CFGS = [(f"combo_l{l}_s{s}", l, s, False) for l, s in
            [(0.01, 3.0), (0.01, 10.0), (0.05, 10.0)]]
    for tag, lam, sw, usepart in CFGS:
        BA.trial_bone_lengths = _part_bones if usepart else _orig_bone
        traj, ids, n_fail = [], [], 0
        t0 = time.time()
        for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
            _CUR["part"] = t["part"]; ids.append(f"{t['part']}/{t['trial']}")
            try:
                P, _ = BA.refine_trial_ba(t, lam, iters=60, smooth_w=sw, fallback_mm=None)
            except Exception as e:
                print(f"  [fail] {ids[-1]}: {type(e).__name__}: {e}", flush=True)
                P = np.full_like(t["mmc"], np.nan); n_fail += 1
            traj.append(P.astype(np.float32))
        np.savez(OUT / f"{tag}.npz", ids=np.array(ids), traj=np.array(traj, dtype=object))
        print(f"PROCESSING CHECK [{tag}] lam={lam} smooth_w={sw}: {len(trials)} trials, "
              f"{sum(np.isfinite(x).any() for x in traj)} finite, {n_fail} fail "
              f"({time.time()-t0:.0f}s)", flush=True)
    BA._reproj_residual = _orig_res; BA.trial_bone_lengths = _orig_bone
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
