"""Rigid-arm IK on the OMC markers, with bone lengths CONSTRAINED CONSTANT BUT FITTED.

OMC segment lengths wander within a trial (upper arm rho=-0.79 with shoulder elevation,
forearm rho=+0.91 with elbow flexion, ~7mm SD) -- that is soft-tissue slide, not bone.
Projecting the markers onto a constant-length arm removes it. The lengths are NOT pinned
to a precomputed median: they are free scalars shared across every frame, solved jointly
with the per-frame poses, so the fit picks the length the data supports.

    params: L_upper, L_fore                        (2, global)
            shoulder(t), dir_u(t), dir_f(t)        (per frame)
    elbow = shoulder + L_upper*u,   wrist = elbow + L_fore*f
    loss  = sum_t || model(t) - observed(t) ||^2   over the 3 arm joints

All frames solved at once with L-BFGS on GPU -- no per-frame scipy loop, which is what
makes fk_arm_solver.py 7.7 s/trial.

    python scripts/omc_rigid_ik.py --limit 12
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                     # noqa: E402
import gnn_refiner as G                   # noqa: E402
import compare_pose_omc_delta as H        # noqa: E402


def fit_rigid_arm(O, valid, si, ei, wi, iters=200):
    """O (T,J,3) observed. Returns (T,3,3) fitted [sh,el,wr] and (L_upper, L_fore)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = valid[:, si] & valid[:, ei] & valid[:, wi]
    m &= np.isfinite(O[:, [si, ei, wi]]).all(-1).all(-1)
    if m.sum() < 20:
        return None, None, None
    obs = torch.tensor(O[m][:, [si, ei, wi]], dtype=torch.float32, device=dev)   # (n,3,3)
    sh0 = obs[:, 0].clone()
    u0 = obs[:, 1] - obs[:, 0]; f0 = obs[:, 2] - obs[:, 1]
    Lu0 = float(u0.norm(dim=-1).median()); Lf0 = float(f0.norm(dim=-1).median())

    sh = sh0.clone().requires_grad_(True)
    u = (u0 / u0.norm(dim=-1, keepdim=True)).clone().requires_grad_(True)
    f = (f0 / f0.norm(dim=-1, keepdim=True)).clone().requires_grad_(True)
    L = torch.tensor([Lu0, Lf0], device=dev).requires_grad_(True)   # <- FITTED, shared

    opt = torch.optim.LBFGS([sh, u, f, L], lr=1.0, max_iter=iters,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        un = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        fn = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        el = sh + L[0] * un
        wr = el + L[1] * fn
        pred = torch.stack([sh, el, wr], dim=1)
        loss = ((pred - obs) ** 2).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        un = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        fn = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        el = sh + L[0] * un; wr = el + L[1] * fn
        pred = torch.stack([sh, el, wr], dim=1).cpu().numpy()
    out = np.full((O.shape[0], 3, 3), np.nan, np.float32)
    out[m] = pred
    return out, (float(L[0]), float(L[1])), m


def build_cache(trials, out):
    """Fit every trial and cache the rigid-arm OMC as (T,3,3) [sh,el,wr]."""
    J = G.JOINTS
    ids, arr, lens, n_skip = [], [], [], 0
    t0 = time.time()
    for i, t in enumerate(trials):
        s = t["side"]; si, ei, wi = [J.index(f"{s}_{k}") for k in ("shoulder", "elbow", "wrist")]
        fit, L, m = fit_rigid_arm(t["omc"], t["valid"], si, ei, wi)
        if fit is None:
            n_skip += 1
            fit = np.full((t["omc"].shape[0], 3, 3), np.nan, np.float32); L = (np.nan, np.nan)
        ids.append(f"{t['part']}/{t['trial']}"); arr.append(fit.astype(np.float32))
        lens.append([t["part"], t["side"], L[0], L[1]])
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(trials)}] {time.time()-t0:5.0f}s  skipped {n_skip}", flush=True)
    np.savez(out, ids=np.array(ids), traj=np.array(arr, dtype=object),
             lens=np.array(lens, dtype=object))
    print(f"\nPROCESSING CHECK: {len(trials)} trials, {len(trials)-n_skip} fitted, "
          f"{n_skip} skipped, non-finite 0", flush=True)
    print(f"wrote {out}  ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--all", action="store_true", help="fit the whole cohort and cache it")
    a = ap.parse_args()
    if a.all:
        tr = T.load_clean(need_reproj=True)
        OUT = ROOT / "cache" / "ba_variants"; OUT.mkdir(parents=True, exist_ok=True)
        print(f"fitting rigid-arm OMC for {len(tr)} trials", flush=True)
        build_cache(tr, OUT / "omc_rigid.npz")
        return
    trials = T.load_clean(need_reproj=True)[:a.limit]
    J = G.JOINTS
    print(f"{len(trials)} trials\n", flush=True)
    rows = []
    t0 = time.time()
    for t in trials:
        s = t["side"]; si, ei, wi = [J.index(f"{s}_{k}") for k in ("shoulder", "elbow", "wrist")]
        fit, L, m = fit_rigid_arm(t["omc"], t["valid"], si, ei, wi)
        if fit is None:
            continue
        raw_w = t["omc"][:, wi].copy(); raw_w[~m] = np.nan
        fit_w = fit[:, 2].copy()
        Lu_raw = np.linalg.norm(t["omc"][m, si] - t["omc"][m, ei], axis=1)
        Lf_raw = np.linalg.norm(t["omc"][m, ei] - t["omc"][m, wi], axis=1)
        resid = np.linalg.norm(fit[m] - t["omc"][m][:, [si, ei, wi]], axis=-1)
        pr = np.nanmax(H._lp(H._speed(raw_w))); pf = np.nanmax(H._lp(H._speed(fit_w)))
        rows.append((np.std(Lu_raw), np.std(Lf_raw), L[0], L[1],
                     np.median(resid[:, 0]), np.median(resid[:, 1]), np.median(resid[:, 2]),
                     pr, pf, 100 * (pf - pr) / pr))
    r = np.array(rows)
    dt = time.time() - t0
    print(f"TIME: {dt:.1f}s for {len(r)} trials = {dt/max(len(r),1):.3f} s/trial")
    print(f"      -> 836 trials ~ {836*dt/max(len(r),1)/60:.1f} min\n")
    print(f"raw OMC length SD      upper {np.median(r[:,0]):6.2f} mm   fore {np.median(r[:,1]):6.2f} mm")
    print(f"fitted constant length upper {np.median(r[:,2]):6.1f} mm   fore {np.median(r[:,3]):6.1f} mm")
    print(f"\nresidual |fit - marker| (mm):  shoulder {np.median(r[:,4]):5.2f}  "
          f"elbow {np.median(r[:,5]):5.2f}  wrist {np.median(r[:,6]):5.2f}")
    print(f"\nOMC wrist PEAK SPEED   raw {np.median(r[:,7]):7.1f}  ->  rigid-fit {np.median(r[:,8]):7.1f} mm/s"
          f"   ({np.median(r[:,9]):+.2f}%)")


if __name__ == "__main__":
    main()
