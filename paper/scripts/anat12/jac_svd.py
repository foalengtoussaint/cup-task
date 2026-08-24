"""Which combinations of the 12 anat12 offsets does the geometry actually MEASURE?

J = d(angle residual in DEG) / d(theta in MM), evaluated at the fitted theta of each split-half fold
(read from the PERSISTED fit -- no refit). SVD via the 12x12 normal matrix J^T J:
  right singular vectors V = directions in offset space
  singular values s        = how hard the data pushes back along each
  sigma / s_i              = mm of uncertainty admitted along direction i   (sigma = residual RMS, deg)

Then theta is split into the part the data pins (uncertainty below a threshold) and the part it does
not (the 199.6mm axial wrist). Fold-to-fold |d1-d2| said SOME directions are unidentified; this says
WHICH, in mm.

Angle-only residual (w=0), because that is the fit whose theta was persisted.
    nohup python jac_svd.py > out/scoring/jac_svd.log 2>&1 &
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

MODEL = "anat12"
MAGS = ROOT/"out/scoring/anat_frame_mags_a12w0.csv"
STEP = 1.0            # mm, finite-difference step
TOL_MM = 20.0         # a direction counts as IDENTIFIED if it admits < 20mm of uncertainty
BLOCKS = [("shoulder_trunk", 0), ("shoulder_seg", 3), ("elbow_seg", 6), ("wrist_seg", 9)]


def ang_resid(recs, th):
    return np.concatenate([blocks(r, th, MODEL)[0] for r in recs])


if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    M = pd.read_csv(MAGS)
    npar = NPAR[MODEL]
    print(f"{len(M)} persisted folds, {npar} params, step {STEP}mm", flush=True)

    rows, allS, load = [], [], []
    t0 = time.time()
    for _, m in M.iterrows():
        rs = groups[(m["part"], m["arm"])]
        parity = int(m["parity"])
        tr = [r for i, r in enumerate(rs) if i % 2 != parity]
        th = m[[f"th{i}" for i in range(npar)]].values.astype(float)
        r0 = ang_resid(tr, th)
        sigma = float(np.sqrt(np.mean(r0**2)))
        J = np.zeros((len(r0), npar))
        for k in range(npar):
            tp = th.copy(); tp[k] += STEP
            rk = ang_resid(tr, tp)
            if len(rk) != len(r0):            # NaN pattern moved; skip this column rather than misalign
                J[:, k] = np.nan; continue
            J[:, k] = (rk - r0) / STEP
        ok = np.isfinite(J).all(1)
        JtJ = J[ok].T @ J[ok]
        ev, V = np.linalg.eigh(JtJ)           # ascending
        ev = np.clip(ev, 0, None)
        s = np.sqrt(ev)[::-1]; V = V[:, ::-1]  # descending
        unc = sigma / np.where(s > 0, s, np.nan)          # mm admitted along each direction
        proj = V.T @ th                                    # theta in the direction basis
        idf = np.isfinite(unc) & (unc < TOL_MM)
        rows.append(dict(part=m["part"], arm=m["arm"], parity=parity, sigma_deg=sigma,
                         n_identified=int(idf.sum()),
                         theta_norm=float(np.linalg.norm(th)),
                         theta_identified=float(np.linalg.norm(proj[idf])),
                         theta_unidentified=float(np.linalg.norm(proj[~idf]))))
        allS.append(unc)
        load.append(np.abs(V[:, -1]))          # loadings of the WORST-constrained direction
        print(f"  [{time.time()-t0:5.0f}s] {m['part']} {m['arm']} p{parity}  sigma {sigma:5.2f}deg  "
              f"identified {idf.sum()}/{npar}  |th| {np.linalg.norm(th):6.1f} -> "
              f"{np.linalg.norm(proj[idf]):6.1f} identified", flush=True)

    D = pd.DataFrame(rows)
    U = np.array(allS); L = np.array(load)
    print(f"\nPROCESSING CHECK: folds {len(M)}, analysed {len(D)}, non-finite sigma "
          f"{int(D.sigma_deg.isna().sum())}")

    print(f"\nUNCERTAINTY PER DIRECTION (mm), median over {len(D)} folds, best-constrained first:")
    print("  " + "  ".join(f"{v:7.2f}" for v in np.nanmedian(U, axis=0)))
    print(f"\nHow much of theta the data actually pins (threshold {TOL_MM}mm):")
    print(f"  |theta| total          {D.theta_norm.median():7.1f} mm")
    print(f"  |theta| IDENTIFIED     {D.theta_identified.median():7.1f} mm")
    print(f"  |theta| UNIDENTIFIED   {D.theta_unidentified.median():7.1f} mm")
    print(f"  directions identified  {D.n_identified.median():.0f} / {npar}")
    print(f"\nWorst-constrained direction -- median |loading| per block "
          f"(1.0 = that direction is entirely this block):")
    for nm, i in BLOCKS:
        print(f"  {nm:16}{np.linalg.norm(np.median(L, axis=0)[i:i+3]):.2f}")
    D.to_csv(ROOT/"out/scoring/jac_svd.csv", index=False)
    np.savez(ROOT/"out/scoring/jac_svd_dirs.npz", unc=U, worst_loadings=L)
    print("\nDONE", flush=True)
