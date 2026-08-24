"""Re-pair MMC clips to C3Ds where the evidence is UNAMBIGUOUS, leave near-ties alone.

check_trial_pairing showed P251 is systematically off by +1 from trial_12 (own_r 0.69 -> 0.90
against the next trial's C3D at ~2 frames lag) and P252 slips on trials 43-44. But several other
flagged trials are near-ties -- a greedy argmax always finds SOME better match when a participant's
repetitions are near-identical (P252 trial_69: 0.871 vs 0.886). Re-pairing those would be fitting
noise.

So: build the full trial x c3d correlation matrix, solve a GLOBAL assignment (each C3D used once,
which a greedy argmax cannot guarantee), and accept a move only when it clears both
  - MIN_GAIN  : assigned_r - own_r          (the move must be a big improvement)
  - MIN_R     : assigned_r                  (and land on a genuinely good match)
Everything else keeps its original pairing.

max_lag is small on purpose: a correctly paired trial aligns within a few frames, and a WRONG
pairing cannot be rescued by any lag, so a wide search only adds cost and false matches.

    python scripts/repair_trial_pairing.py --parts P251 P252
    -> out/scoring/trial_repair_map.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H      # noqa: E402
import gnn_train as T                   # noqa: E402
import gnn_refiner as G                 # noqa: E402

J = G.JOINTS
MIN_GAIN, MIN_R = 0.15, 0.85


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P251", "P252"])
    ap.add_argument("--max-lag", type=int, default=40)
    a = ap.parse_args()
    H.use_good_cams()
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in a.parts]
    by = {}
    for t in trials:
        by.setdefault(t["part"], []).append(t)
    out = []
    for part, ts in sorted(by.items()):
        c3ds = [Path(f).stem for f in sorted(glob.glob(str(ROOT / "cache/delta" / part / "c3d/*.c3d")))]
        # side must match: an L trial can only pair with an L c3d
        M = np.full((len(ts), len(c3ds)), -1.0)
        for i, t in enumerate(ts):
            side = t["side"]; wi = J.index(f"{side}_wrist"); n = t["mmc"].shape[0]
            mm = H._speed(t["mmc"][:, wi])
            for k, c3 in enumerate(c3ds):
                if ("_L_" in t["trial"]) != ("_L_" in c3):
                    continue
                try:
                    o = H._load_omc(part, c3, n, repair=False)[f"{side}_wrist"]
                    _, c = H._corr_at_lag(mm, H._speed(o), a.max_lag)
                except Exception:
                    continue
                if np.isfinite(c):
                    M[i, k] = c
            if (i + 1) % 10 == 0:
                print(f"   {part} [{i+1}/{len(ts)}]", flush=True)
        r, cidx = linear_sum_assignment(-M)                 # global: each c3d used once
        for i, k in zip(r, cidx):
            t = ts[i]
            own = c3ds.index(t["trial"]) if t["trial"] in c3ds else -1
            own_r = M[i, own] if own >= 0 else np.nan
            new_r = M[i, k]
            move = (c3ds[k] != t["trial"]) and (new_r - own_r > MIN_GAIN) and (new_r > MIN_R)
            out.append(dict(part=part, trial=t["trial"], own_r=own_r,
                            assigned=c3ds[k], assigned_r=new_r,
                            gain=new_r - own_r, repair=bool(move),
                            use=c3ds[k] if move else t["trial"]))
    d = pd.DataFrame(out)
    d.to_csv(ROOT / "out/scoring/trial_repair_map.csv", index=False)
    print(f"\nPROCESSING CHECK: {len(d)} trials, non-finite own_r {int(d.own_r.isna().sum())}\n")
    for p, g in d.groupby("part"):
        rep = g[g.repair]
        print(f"{p}: {len(rep)}/{len(g)} re-paired   "
              f"own_r {g.own_r.median():.3f} -> effective {np.where(g.repair, g.assigned_r, g.own_r).mean():.3f}")
    rep = d[d.repair]
    if len(rep):
        print(f"\nRE-PAIRED ({len(rep)}):")
        print(rep[["part", "trial", "own_r", "assigned", "assigned_r", "gain"]].round(3).to_string(index=False))
    near = d[(~d.repair) & (d.assigned != d.trial)]
    print(f"\nLEFT ALONE despite a different assignment ({len(near)}) -- gain below {MIN_GAIN}:")
    if len(near):
        print(near[["part", "trial", "own_r", "assigned", "assigned_r", "gain"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
