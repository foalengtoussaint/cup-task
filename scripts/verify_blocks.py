"""Re-derive the trial->C3D block offsets against the COMPLETE trial list, with the stacked lag.

The first pass read the candidate trial list off cache/delta/gnn_pairs, which only held trials that
had already been built -- so spans looked contiguous when they were not (P10 appeared to jump 19->23;
trials 20-22 simply had no pair yet). It also scored with the wrist-speed argmax, the estimator that
fails on exactly the trials in question. Both are fixed here.

For every trial of the named participants, score its OMC candidates at offsets -2..+2 using
H.find_lag_best (all six signals, Fisher-z stacked) and report the best offset per trial, so a block
boundary is read off complete evidence.

    python scripts/verify_blocks.py --parts P10 P251 P252
    -> out/scoring/verify_blocks.csv
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H      # noqa: E402
import gnn_refiner as G                 # noqa: E402
import gnn_build_dataset as B           # noqa: E402

J = list(G.JOINTS)
PAT = re.compile(r"trial_(\d+)_(.+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--offsets", nargs="+", type=int, default=[-2, -1, 0, 1, 2])
    a = ap.parse_args()
    H.use_good_cams()
    rows, t0 = [], time.time()
    for part in a.parts:
        trials = sorted(p.stem for p in (B.CACHE / part).glob("*.npz")
                        if not p.name.endswith(".reproj.npz"))
        print(f"\n### {part}: {len(trials)} trials x {len(a.offsets)} offsets", flush=True)
        for i, tr in enumerate(trials):
            m = PAT.search(tr)
            if not m:
                continue
            n_, suf = int(m.group(1)), m.group(2)
            side = "right" if "_R_" in tr else "left"
            z = np.load(str(B.CACHE / part / f"{tr}.npz"))
            mmc = {j: z["mmc"][:, k] for k, j in enumerate(J)}
            n = z["mmc"].shape[0]
            best = {}
            for off in a.offsets:
                cand = f"trial_{n_+off}_{suf}"
                for d in (part, B._SIBLING.get(part)):
                    if d is None:
                        continue
                    p = H.DELTA / d / "c3d" / f"{cand}.c3d"
                    if p.exists():
                        try:
                            omc, _, _ = B._load_omc_defensive.__wrapped__(d, cand, n) \
                                if hasattr(B._load_omc_defensive, "__wrapped__") else (None, None, None)
                        except Exception:
                            omc = None
                        if omc is None:
                            try:
                                omc = H._load_omc(d, cand, n, repair=False)
                            except Exception:
                                continue
                        try:
                            lag, r, info = H.find_lag_best(mmc, omc, side)
                        except Exception:
                            continue
                        best[off] = (float(r), int(lag), f"{d}/{cand}")
                        break
            if not best:
                continue
            bo = max(best, key=lambda k: best[k][0])
            rows.append(dict(part=part, trial=tr, suffix=suf, n=n_,
                             best_off=bo, best_r=best[bo][0], best_lag=best[bo][1],
                             best_c3d=best[bo][2],
                             r_at_0=best.get(0, (np.nan,))[0],
                             r_at_m1=best.get(-1, (np.nan,))[0],
                             r_at_p1=best.get(1, (np.nan,))[0]))
            if (i + 1) % 10 == 0 or (i + 1) == len(trials):
                el = time.time() - t0
                print(f"  {part} [{i+1}/{len(trials)}] {el:5.0f}s ({el/len(rows):.2f}s/trial)  "
                      f"best_off!=0 so far: {sum(1 for r in rows if r['best_off'] != 0)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "out/scoring/verify_blocks.csv", index=False)
    np.savez(ROOT / "out/scoring/verify_blocks.npz", **{c: d[c].values for c in d.columns})
    print(f"\nPROCESSING CHECK: {len(d)} trials scored, non-finite best_r {int(d.best_r.isna().sum())}\n")
    for (p, s), g in d.groupby(["part", "suffix"]):
        g = g.sort_values("n")
        print(f"=== {p} {s} ===")
        print(f"{'trial':>6s} {'best_off':>9s} {'best_r':>7s} {'lag':>5s} | "
              f"{'r@-1':>6s} {'r@0':>6s} {'r@+1':>6s}  {'target':>28s}")
        for _, r in g.iterrows():
            mark = "" if r.best_off == 0 else "  <<<"
            print(f"{r.n:6d} {r.best_off:+9d} {r.best_r:7.3f} {r.best_lag:5d} | "
                  f"{r.r_at_m1:6.3f} {r.r_at_0:6.3f} {r.r_at_p1:6.3f}  {r.best_c3d:>28s}{mark}")
    print("DONE_VERIFY", flush=True)


if __name__ == "__main__":
    main()
