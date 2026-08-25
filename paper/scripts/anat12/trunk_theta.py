"""Fit the sternum offset per participant x arm and emit it as a theta table the scorer can apply.

`trunk_offset_fit.py` established that a body-frame offset on the OPTICAL sternum brings trunk
displacement into agreement with the markerless shoulder midpoint (per-frame r 0.969 -> 0.976, RMSE
4.28 -> 3.45 mm, reported scalar 0.936 -> 0.950 out of sample). It was written as a probe and never
wired into the scorer, so Table III's trunk row still compares an unfitted sternum. This produces the
per-group offsets in the same form as `anat12_lopo.py`'s theta CSV, so `score_own_phases.py` can
apply them alongside the twelve arm and shoulder parameters.

WHY THE TRUNK NEEDS ITS OWN CODE PATH. The twelve arm parameters are applied in a body frame FROZEN
per trial. A frozen frame makes the offset a constant world vector, and trunk displacement is a
distance from the trial's own rest position, so a constant offset cancels EXACTLY and a frozen-frame
sternum offset would do nothing at all. The offset only bites in a frame rebuilt per frame, which
rotates with the torso and so changes the arc the point sweeps during a forward lean --- which is
precisely the sternum-versus-shoulder-midpoint discrepancy. Fitted and applied per frame, therefore.

    python paper/scripts/anat12/trunk_theta.py     # -> out/scoring/trunk_theta.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper" / "scripts" / "anat12"))

from trunk_offset_fit import build, fit_d, excursion, apply_d, agree   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "out/scoring/trunk_theta.csv"))
    a = ap.parse_args(argv)

    recs = build()
    groups: dict = {}
    for r in recs:
        groups.setdefault((r["part"], r["arm"]), []).append(r)
    print(f"\n{len(recs)} trials, {len(groups)} participant x arm groups\n", flush=True)

    rows = []
    print(f"{'group':22}{'n':>5}{'|d| mm':>9}{'r raw':>9}{'r fit':>9}{'RMSE raw':>10}{'RMSE fit':>10}")
    for (part, arm), rs in sorted(groups.items()):
        d = fit_d(rs)                       # in-sample, exactly as anat12's theta is fitted
        rr, rf, er, ef = [], [], [], []
        for r in rs:
            a_m = excursion(r["sh_m"])
            b0 = excursion(r["st"])
            b1 = excursion(apply_d(r["st"], r["B_o"], d))
            for src, rl, el in ((b0, rr, er), (b1, rf, ef)):
                g = agree(a_m, src)
                if g:
                    el.append(g[0]); rl.append(g[1])
        rows.append(dict(part=part, arm=arm, d0=d[0], d1=d[1], d2=d[2], n=len(rs)))
        print(f"{part+'/'+arm:22}{len(rs):>5}{np.linalg.norm(d):>9.1f}"
              f"{np.median(rr):>9.3f}{np.median(rf):>9.3f}"
              f"{np.median(er):>10.2f}{np.median(ef):>10.2f}", flush=True)

    D = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(D)} groups)")
    print(f"PROCESSING CHECK: |d| median {np.linalg.norm(D[['d0','d1','d2']].values, axis=1).mean():.1f} mm")
    print("DONE_TRUNK_THETA", flush=True)


if __name__ == "__main__":
    main()
