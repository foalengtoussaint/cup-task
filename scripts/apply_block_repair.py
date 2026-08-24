"""Apply the P25 session-split trial->C3D repair as BLOCK shifts, and verify against OMC.

check_trial_pairing + a global assignment showed three contiguous spans each uniformly off by +1,
which is what a session split at the wrong boundary produces:

    P251  R_unaffected  trial >= 12       -> C3D trial+1
    P251  L_affected    trial >= 23       -> C3D trial+1
    P252  R_unaffected  trial in {43,44}  -> C3D trial+1

Applied as blocks rather than per-trial thresholding: the shift is systematic, so shifting only the
trials that individually clear a gain threshold would leave a mixture (13 shifted, 16 not, inside one
continuous span) that is worse than either choice. It also picks up trial_27_L (own_r 0.186), which a
per-trial rule skipped only because its match landed at 0.816.

Verification is the point: re-score wrist-speed correlation with the repaired pairing.

    python scripts/apply_block_repair.py     -> out/scoring/block_repair.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H      # noqa: E402
import gnn_train as T                   # noqa: E402
import gnn_refiner as G                 # noqa: E402
import gnn_build_dataset as B           # noqa: E402  -- SINGLE source of the block table

J = G.JOINTS
BLOCKS = B._C3D_BLOCKS                  # verifier and builder must never drift apart
PARTS = sorted({b[0] for b in BLOCKS})


def repaired_path(part, trial):
    """The C3D this trial should be paired with, per the shared block table."""
    return B.c3d_path(part, trial)


def main() -> None:
    H.use_good_cams()
    rows = []
    for t in T.load_clean(need_reproj=True):
        part, trial, side = t["part"], t["trial"], t["side"]
        if part not in PARTS:
            continue
        tgt = repaired_path(part, trial)
        new = tgt.stem
        n = t["mmc"].shape[0]
        wi = J.index(f"{side}_wrist")
        mm = H._speed(t["mmc"][:, wi])
        def corr(path):
            try:
                o = H._load_omc(path.parent.parent.name, path.stem, n, repair=False)[f"{side}_wrist"]
                lg, c = H._corr_at_lag(mm, H._speed(o), 40)
                return c, lg
            except Exception:
                return np.nan, np.nan
        own = ROOT / "cache/delta" / part / "c3d" / f"{trial}.c3d"
        c_old, l_old = corr(own)
        exists = tgt.exists()
        c_new, l_new = corr(tgt) if (new != trial and exists) else (c_old, l_old)
        rows.append(dict(part=part, trial=trial, repaired=new, changed=new != trial,
                         c3d_dir=tgt.parent.parent.name, c3d_exists=exists,
                         r_old=c_old, r_new=c_new if exists else np.nan,
                         lag_old=l_old, lag_new=l_new if exists else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "out/scoring/block_repair.csv", index=False)
    ch = d[d.changed]
    print(f"PROCESSING CHECK: {len(d)} trials, {int(d.changed.sum())} re-paired, "
          f"{int((~d.c3d_exists & d.changed).sum())} with NO target C3D, "
          f"non-finite r_new {int(ch.r_new.isna().sum())}\n")
    print(f"{'part':6s} {'block':14s} {'n':>4s} {'r before':>9s} {'r after':>8s} {'|lag| after':>12s}")
    for (p, blk), g in ch.assign(blk=ch.trial.str.extract(r"trial_\d+_(.+)$")[0]).groupby(["part", "blk"]):
        v = g.dropna(subset=["r_new"])
        print(f"{p:6s} {blk:14s} {len(g):4d} {g.r_old.median():9.3f} {v.r_new.median():8.3f} "
              f"{v.lag_new.abs().median():12.0f}")
    un = d[~d.changed]
    print(f"\nunchanged trials ({len(un)}): r {un.r_old.median():.3f}  (sanity: should stay high)")
    print(f"\nimproved: {int((ch.r_new > ch.r_old).sum())}/{len(ch.dropna(subset=['r_new']))}")
    worse = ch[ch.r_new < ch.r_old]
    if len(worse):
        print(f"WORSE after repair ({len(worse)}):")
        print(worse[["part", "trial", "repaired", "r_old", "r_new"]].round(3).to_string(index=False))
    miss = ch[~ch.c3d_exists]
    if len(miss):
        print(f"\nNO TARGET C3D -- drop these ({len(miss)}):")
        print(miss[["part", "trial", "repaired"]].to_string(index=False))


if __name__ == "__main__":
    main()
