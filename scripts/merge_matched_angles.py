"""Swap the three ANGLE rows' ground truth from AutoMQ's world-axis scalars to our
matched-definition OMC values, leaving every other measure untouched.

This step was previously an ad-hoc shell heredoc and existed only in a session
transcript, which is why Table III could not be reproduced from the documented
commands (recovered 2026-08-19). It is the third stage of the Table III chain:

    python scripts/score_vs_automq.py --out out/scoring/score_vs_automq.csv --parts <11>
    python scripts/omc_matched_angles.py          -> out/scoring/omc_matched_angles.csv
    python scripts/merge_matched_angles.py        -> out/scoring/score_vs_automq_matchedangles.csv
    SCORE_CSV=out/scoring/score_vs_automq_matchedangles.csv python paper/scripts/paper_table3.py
    SCORE_CSV=out/scoring/score_vs_automq_matchedangles.csv python paper/scripts/fig4_correlation.py

Only flexion, abduction and interjoint are swapped: those are the three whose AutoMQ
definition uses world reference axes rather than a body frame. Angle rows with no
matched OMC value become NaN and drop out of the table -- that is the 695 -> 660 fall
in Table III's n, not a filtering choice made here.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWAP = {"max_shoulder_flexion", "max_shoulder_abduction", "interjoint_coordination"}
SRC = ROOT / "out/scoring/score_vs_automq.csv"
MATCHED = ROOT / "out/scoring/omc_matched_angles.csv"
OUT = ROOT / "out/scoring/score_vs_automq_matchedangles.csv"


def main() -> None:
    sc = pd.read_csv(SRC, keep_default_na=False, na_values=[""])
    mt = pd.read_csv(MATCHED, usecols=["part", "trial", "measure", "omc_matched"]).drop_duplicates()

    j = sc.merge(mt, on=["part", "trial", "measure"], how="left")
    sw = j.measure.isin(SWAP)
    before = int(j.loc[sw, "automq"].notna().sum())
    j.loc[sw, "automq"] = j.loc[sw, "omc_matched"]
    j = j.drop(columns=["omc_matched"])
    after = int(j.loc[sw, "automq"].notna().sum())

    j.to_csv(OUT, index=False)
    print(f"PROCESSING CHECK: rows {len(j)}, angle rows {int(sw.sum())}, "
          f"with matched OMC {after} (dropped {before - after} that had none)", flush=True)
    print(f"untouched measures keep AutoMQ: {sorted(set(j.measure.unique()) - SWAP)}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
