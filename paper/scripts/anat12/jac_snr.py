"""Which directions are TRUSTWORTHY -- calibrated empirically, not by the formal standard error.

jac_svd.py's sigma/s assumes independent residuals; at 60Hz they are not, so its mm are ~10x too
small (it called 12/12 identified while the two split-half folds disagreed by 143mm on the wrist).
So calibrate against the folds themselves:

  for each participant x arm, take the SHARED direction basis V (from the p0 fold's J^T J), project
  BOTH folds' theta onto it, and per direction i compare
      signal_i = |mean of the two projections|      -- how much offset the fit puts there
      noise_i  = |difference of the two|            -- how much the SAME arm disagrees with itself
      SNR_i    = signal / noise
A direction is trustworthy when the value it carries is large compared with how much it moves when
you swap which half of the trials you fit on. Also reports how each direction loads on the four
blocks, so the answer is anatomical and not just an index.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # anat12 modules live beside this file
import prep_cache
from anat_frame import blocks, NPAR
from joint_fit import speed_series

MODEL = "anat12"; STEP = 1.0
MAGS = ROOT/"out/scoring/anat_frame_mags_a12w0.csv"
BLOCKS = [("sh_trunk", 0), ("sh_seg", 3), ("elb_seg", 6), ("wr_seg", 9)]


def ang_resid(recs, th):
    return np.concatenate([blocks(r, th, MODEL)[0] for r in recs])


def jacobian(tr, th, npar):
    r0 = ang_resid(tr, th)
    J = np.zeros((len(r0), npar))
    for k in range(npar):
        tp = th.copy(); tp[k] += STEP
        rk = ang_resid(tr, tp)
        J[:, k] = (rk - r0)/STEP if len(rk) == len(r0) else np.nan
    ok = np.isfinite(J).all(1)
    return J[ok], float(np.sqrt(np.mean(r0**2)))


if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    M = pd.read_csv(MAGS); npar = NPAR[MODEL]
    cols = [f"th{i}" for i in range(npar)]
    t0 = time.time(); sig, noi, loads, unc = [], [], [], []
    for (p, a), g in M.groupby(["part", "arm"]):
        if len(g) != 2: continue
        g = g.sort_values("parity")
        th0, th1 = (g.iloc[k][cols].values.astype(float) for k in (0, 1))
        rs = groups[(p, a)]
        tr0 = [r for i, r in enumerate(rs) if i % 2 != 0]
        J, s0 = jacobian(tr0, th0, npar)
        ev, V = np.linalg.eigh(J.T @ J)
        s = np.sqrt(np.clip(ev, 0, None))[::-1]; V = V[:, ::-1]
        p0, p1 = V.T @ th0, V.T @ th1
        sig.append(np.abs((p0 + p1)/2)); noi.append(np.abs(p0 - p1))
        unc.append(s0/np.where(s > 0, s, np.nan)); loads.append(V)
        print(f"  [{time.time()-t0:4.0f}s] {p} {a}", flush=True)

    S, N, U = np.array(sig), np.array(noi), np.array(unc)
    L = np.abs(np.array(loads))                      # (folds, param, direction)
    print(f"\nPROCESSING CHECK: arms {len(S)}, directions {npar}, non-finite "
          f"{int(np.sum(~np.isfinite(S)))}\n")
    print(f"{'dir':>4}{'sigma/s mm':>12}{'signal mm':>11}{'split noise':>13}{'SNR':>7}   "
          f"{'|loading| sh_trunk  sh_seg  elb_seg  wr_seg':>44}")
    for i in range(npar):
        ld = [np.linalg.norm(np.median(L[:, j:j+3, i], axis=0)) for _, j in BLOCKS]
        print(f"{i+1:>4}{np.nanmedian(U[:, i]):>12.2f}{np.median(S[:, i]):>11.1f}"
              f"{np.median(N[:, i]):>13.1f}{np.median(S[:, i])/(np.median(N[:, i])+1e-9):>7.2f}   "
              + "".join(f"{v:>10.2f}" for v in ld), flush=True)
    np.savez(ROOT/"out/scoring/jac_snr.npz", signal=S, noise=N, unc=U, loadings=L)
    print("\nSNR>1 = the fit puts more offset in that direction than it moves when you swap halves.")
    print("DONE", flush=True)
