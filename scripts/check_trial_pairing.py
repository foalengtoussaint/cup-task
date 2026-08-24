"""Is each MMC clip paired with the RIGHT C3D? Cross-correlate against every C3D of that participant.

P251's bad trials matched the NEXT trial number's C3D far better than their own (25->26 r 0.43->0.87,
28->29 0.44->0.90, 13->14 0.50->0.94, all at ~zero lag), which is an off-by-one between the video
trial numbering and the C3D numbering -- P25 was recorded as one session and split into P251/P252.

This is a data-integrity check, not a sync check: a wrong pairing compares two different repetitions
of the same task, so NO lag can align them and the multi-signal gate can still admit the trial on the
cup signal (the cup does much the same thing every rep).

    python scripts/check_trial_pairing.py --parts P251 P252 P07
    -> out/scoring/trial_pairing.csv
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H      # noqa: E402
import gnn_train as T                   # noqa: E402
import gnn_refiner as G                 # noqa: E402

J = G.JOINTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--max-lag", type=int, default=180)
    a = ap.parse_args()
    H.use_good_cams()
    trials = T.load_clean(need_reproj=True)
    if a.parts:
        trials = [t for t in trials if t["part"] in a.parts]
    by_part = {}
    for t in trials:
        by_part.setdefault(t["part"], []).append(t)
    rows = []
    for part, ts in sorted(by_part.items()):
        c3ds = [Path(f).stem for f in sorted(glob.glob(str(ROOT / "cache/delta" / part / "c3d/*.c3d")))]
        print(f"{part}: {len(ts)} trials vs {len(c3ds)} c3d", flush=True)
        t0 = time.time()
        for i, t in enumerate(ts):
            side = t["side"]; wi = J.index(f"{side}_wrist"); n = t["mmc"].shape[0]
            mm = H._speed(t["mmc"][:, wi])
            scores = []
            for c3 in c3ds:
                try:
                    o = H._load_omc(part, c3, n, repair=False)[f"{side}_wrist"]
                    lg, c = H._corr_at_lag(mm, H._speed(o), a.max_lag)
                except Exception:
                    continue
                if np.isfinite(c):
                    scores.append((c3, lg, c))
            if not scores:
                continue
            scores.sort(key=lambda x: -x[2])
            own = next((s for s in scores if s[0] == t["trial"]), None)
            rank = next((k for k, s in enumerate(scores) if s[0] == t["trial"]), -1)
            rows.append(dict(part=part, trial=t["trial"], own_r=own[2] if own else np.nan,
                             own_lag=own[1] if own else np.nan, own_rank=rank + 1,
                             n_c3d=len(scores), best=scores[0][0], best_r=scores[0][2],
                             best_lag=scores[0][1]))
            if (i + 1) % 5 == 0 or (i + 1) == len(ts):
                el = time.time() - t0
                nmis = sum(1 for r in rows if r["part"] == part and r["own_rank"] > 1)
                print(f"   {part} [{i+1}/{len(ts)}] {el:5.0f}s  "
                      f"({el/(i+1):.1f}s/trial)  rank>1 so far: {nmis}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "out/scoring/trial_pairing.csv", index=False)
    d["mismatch"] = d.own_rank > 1
    print(f"\nPROCESSING CHECK: {len(d)} trials scored, non-finite own_r "
          f"{int(d.own_r.isna().sum())}\n")
    print(f"{'part':6s} {'n':>4s} {'own r':>7s} {'best r':>7s} {'rank>1':>7s} "
          f"{'rank>3':>7s} {'own r<0.7':>10s}")
    for p, g in d.groupby("part"):
        print(f"{p:6s} {len(g):4d} {g.own_r.median():7.3f} {g.best_r.median():7.3f} "
              f"{int((g.own_rank > 1).sum()):7d} {int((g.own_rank > 3).sum()):7d} "
              f"{int((g.own_r < 0.7).sum()):10d}")
    bad = d[d.own_rank > 3]
    if len(bad):
        print(f"\nWRONG-PAIRING CANDIDATES (own C3D ranks >3):")
        print(bad[["part", "trial", "own_r", "own_rank", "best", "best_r", "best_lag"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
