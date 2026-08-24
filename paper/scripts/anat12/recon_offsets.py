"""Reconstruct the MEASURABLE part of each landmark offset, in mm, in its own anatomical frame.

theta is kept only in the directions the split-halves agree on (rank-9 truncation from jac_snr.py:
dirs 1-9 have SNR 2.3-36, dirs 10-12 have SNR ~1.2 and split noise 52-84mm). The discarded part is
NOT set to zero because it is zero -- it is unknown, and zero is the minimum-norm choice. So read the
result as "the component of the real marker displacement that these angles can see", with the
per-component error bar = the split-half disagreement after truncation.

Frames (offsets live in the frame they were fitted in):
  shoulder trunk : B = (down, forward, lateral), frozen per trial
  shoulder/elbow : upper-arm triad e1 = along shoulder->elbow, e2/e3 perpendicular
  wrist          : forearm triad  e1 = along elbow->wrist
Reference point: the acromion-vs-COCO-shoulder gap measured earlier was 55-105mm.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # anat12 modules live beside this file
import prep_cache
from anat_frame import NPAR
from jac_snr import jacobian, MAGS, MODEL
from joint_fit import speed_series

import os
RANK = int(os.environ.get("RANK", 9))
BLOCKS = [("shoulder_trunk", 0, ("down", "fwd", "lat")),
          ("shoulder_seg",   3, ("e1_alongUA", "e2", "e3")),
          ("elbow_seg",      6, ("e1_alongUA", "e2", "e3")),
          ("wrist_seg",      9, ("e1_alongFA", "e2", "e3"))]

if __name__ == "__main__":
    recs = prep_cache.load_all()
    for r in recs: r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    M = pd.read_csv(MAGS); npar = NPAR[MODEL]; cols = [f"th{i}" for i in range(npar)]
    rows = []
    for (p, a), g in M.groupby(["part", "arm"]):
        if len(g) != 2: continue
        g = g.sort_values("parity")
        th0, th1 = (g.iloc[k][cols].values.astype(float) for k in (0, 1))
        tr0 = [r for i, r in enumerate(groups[(p, a)]) if i % 2 != 0]
        J, _ = jacobian(tr0, th0, npar)
        ev, V = np.linalg.eigh(J.T @ J)
        V = V[:, ::-1][:, :RANK]                     # keep the trusted subspace only
        P = V @ V.T                                  # projector onto it
        t0, t1 = P @ th0, P @ th1
        rows.append(dict(part=p, arm=a, th=(t0+t1)/2, err=np.abs(t0-t1),
                         full=np.linalg.norm((th0+th1)/2),
                         kept=np.linalg.norm((t0+t1)/2)))
    print(f"PROCESSING CHECK: arms {len(rows)}, rank kept {RANK}/{npar}\n")
    print("MEASURABLE offset per arm (mm), +- split-half disagreement:")
    for r in rows:
        print(f"\n{r['part']} {r['arm']}   (|theta| {r['full']:.0f}mm full -> {r['kept']:.0f}mm measurable)")
        for nm, i, ax in BLOCKS:
            v, e = r["th"][i:i+3], r["err"][i:i+3]
            print(f"   {nm:16}" + "  ".join(f"{ax[k]} {v[k]:7.1f}+-{e[k]:5.1f}" for k in range(3))
                  + f"   |d| {np.linalg.norm(v):6.1f}")
    print("\n\nMEDIAN over arms (mm):")
    T = np.array([r["th"] for r in rows]); E = np.array([r["err"] for r in rows])
    for nm, i, ax in BLOCKS:
        med = np.median(T[:, i:i+3], axis=0); err = np.median(E[:, i:i+3], axis=0)
        print(f"  {nm:16}" + "  ".join(f"{ax[k]} {med[k]:7.1f}+-{err[k]:5.1f}" for k in range(3))
              + f"   |d| {np.linalg.norm(med):6.1f}")
    pd.DataFrame([{**{"part": r["part"], "arm": r["arm"]},
                   **{f"d{i}": r["th"][i] for i in range(npar)},
                   **{f"e{i}": r["err"][i] for i in range(npar)}} for r in rows]
                 ).to_csv(ROOT/f"out/scoring/recon_offsets_r{RANK}.csv", index=False)
    print("\nDONE", flush=True)
