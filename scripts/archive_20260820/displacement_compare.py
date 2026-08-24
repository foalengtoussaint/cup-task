"""Alignment-free comparison of DLT vs BA against OMC, using displacement magnitudes.

score_trial fits a rigid Kabsch (omc->mmc) per trial, which absorbs any coherent shift --
so it measures shape, not position, and a large uniform BA move is invisible to it. These
quantities need NO alignment because they are invariant to rotation and translation:

  step   : |X(t) - X(t-1)|          per-frame displacement magnitude (speed x dt)
  path   : sum of step over trial   total distance travelled
  excur  : |X(t) - median(X)|       displacement from the trial's own rest position
  bone   : |joint_a - joint_b|      segment length (pure geometry, no motion)

For each we report how well MMC tracks OMC. Reads the cached BA trajectories -- no re-solve.

    python scripts/displacement_compare.py     -> out/scoring/displacement_compare.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                     # noqa: E402
import gnn_refiner as G                   # noqa: E402

CACHE = ROOT / "cache" / "ba_variants" / "fix.npz"
SEG = [("shoulder", "elbow"), ("elbow", "wrist")]


def _mask(P, valid):
    Q = P.copy(); Q[~valid] = np.nan
    return Q


def metrics(P, valid, wi):
    """Alignment-free descriptors for one joint's trajectory."""
    w = _mask(P[:, wi], valid[:, wi])
    step = np.linalg.norm(np.diff(w, axis=0), axis=1)          # (T-1,)
    rest = np.nanmedian(w, axis=0)
    excur = np.linalg.norm(w - rest, axis=1)                   # (T,)
    return step, excur


def main() -> None:
    if not CACHE.exists():
        sys.exit(f"missing {CACHE} -- run scripts/cache_ba_variants.py --cfg fix first")
    z = np.load(CACHE, allow_pickle=True)
    ba = dict(zip(z["ids"], z["traj"]))
    rows = []
    trials = T.load_clean(need_reproj=True)
    for i, t in enumerate(trials):
        tid = f"{t['part']}/{t['trial']}"
        if tid not in ba:
            continue
        mmc, omc, val, side = t["mmc"], t["omc"], t["valid"], t["side"]
        Xb = ba[tid]
        wi = G.JOINTS.index(f"{side}_wrist")
        si = G.JOINTS.index(f"{side}_shoulder"); ei = G.JOINTS.index(f"{side}_elbow")
        rec = {"id": tid, "part": t["part"]}
        s_o, e_o = metrics(omc, val, wi)
        for tag, P in (("dlt", mmc), ("ba", Xb)):
            s_m, e_m = metrics(P, val, wi)
            m = np.isfinite(s_m) & np.isfinite(s_o)
            rec[f"{tag}_step_err"] = float(np.median(np.abs(s_m - s_o)[m])) if m.any() else np.nan
            rec[f"{tag}_path"] = float(np.nansum(s_m))
            m2 = np.isfinite(e_m) & np.isfinite(e_o)
            rec[f"{tag}_excur_err"] = float(np.median(np.abs(e_m - e_o)[m2])) if m2.any() else np.nan
            rec[f"{tag}_excur_max"] = float(np.nanmax(e_m)) if np.isfinite(e_m).any() else np.nan
            # segment lengths (pure geometry)
            for a_, b_, nm in ((si, ei, "upper"), (ei, wi, "fore")):
                v = val[:, a_] & val[:, b_]
                L = np.linalg.norm(P[:, a_] - P[:, b_], axis=1); L[~v] = np.nan
                Lo = np.linalg.norm(omc[:, a_] - omc[:, b_], axis=1); Lo[~v] = np.nan
                rec[f"{tag}_{nm}_len"] = float(np.nanmedian(L))
                rec[f"{tag}_{nm}_err"] = float(np.abs(np.nanmedian(L) - np.nanmedian(Lo)))
                rec[f"{tag}_{nm}_sd"] = float(np.nanstd(L))
        rec["omc_path"] = float(np.nansum(s_o))
        rec["omc_excur_max"] = float(np.nanmax(e_o)) if np.isfinite(e_o).any() else np.nan
        rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(trials)}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "out/scoring/displacement_compare.csv", index=False)
    print(f"\nPROCESSING CHECK: {len(df)} trials, non-finite step_err "
          f"{int(df.ba_step_err.isna().sum())}\n")
    print("WRIST, alignment-free. err = |MMC - OMC|, mm. LOWER BETTER.\n")
    print(f"{'part':6s} {'n':>4s} | {'step err DLT':>12s} {'BA':>7s} | {'excur err DLT':>13s} {'BA':>7s} "
          f"| {'path DLT':>9s} {'BA':>8s} {'OMC':>8s}")
    for p, g in df.groupby("part"):
        print(f"{p:6s} {len(g):4d} | {g.dlt_step_err.median():12.2f} {g.ba_step_err.median():7.2f} "
              f"| {g.dlt_excur_err.median():13.2f} {g.ba_excur_err.median():7.2f} "
              f"| {g.dlt_path.median():9.0f} {g.ba_path.median():8.0f} {g.omc_path.median():8.0f}")
    print("\nSEGMENT LENGTHS (geometry only, no motion): |median MMC len - median OMC len| mm")
    print(f"{'part':6s} | {'upper DLT':>10s} {'BA':>7s} | {'fore DLT':>9s} {'BA':>7s} "
          f"| {'upper SD DLT':>13s} {'BA':>7s}")
    for p, g in df.groupby("part"):
        print(f"{p:6s} | {g.dlt_upper_err.median():10.2f} {g.ba_upper_err.median():7.2f} "
              f"| {g.dlt_fore_err.median():9.2f} {g.ba_fore_err.median():7.2f} "
              f"| {g.dlt_upper_sd.median():13.2f} {g.ba_upper_sd.median():7.2f}")


if __name__ == "__main__":
    main()
