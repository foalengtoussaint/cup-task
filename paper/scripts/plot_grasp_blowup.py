"""Show HOW anchor-min's grasp blows up on MMC: the wrist->cup distance, its runs, and both rules.

Worst offenders are trials where anchor-min puts grasp ~400 frames later than sequential on MMC while
agreeing exactly on OMC (e.g. P07 trial_77: seq 105 / min 518, AutoMQ 108).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cache_seg_inputs as CSI
from pipeline.segment import _butter_lp, _interp_nan_xyz, _median_smooth, _runs, FPS, GRASP_FLAT_MMPS
from seg_sequential import LEAVE_REST, HOLD, _sustained

TRIALS = [("P07", "trial_77_R_affected"), ("P14", "trial_36_R_unaffected"),
          ("P15", "trial_67_L_affected")]
OUT = ROOT / "paper" / "grasp_blowup.png"


def signals(cup, hand, fps=FPS):
    cup = _butter_lp(_interp_nan_xyz(np.asarray(cup, float)), fps)
    hand = _butter_lp(_interp_nan_xyz(np.asarray(hand, float)), fps)
    T = min(len(cup), len(hand)); cup, hand = cup[:T], hand[:T]
    d = np.linalg.norm(hand - cup, axis=1)
    v = np.r_[0.0, np.diff(_median_smooth(d, 11))] * fps
    rest = np.median(hand[:max(int(0.5 * fps), 10)], axis=0)
    d_rest = np.linalg.norm(hand - rest, axis=1)
    r0 = _sustained(d_rest > LEAVE_REST, 0, HOLD) or 0
    big = 0.3 * float(d.max() - d.min())
    closing = [(s, e) for s, e in _runs(v < -GRASP_FLAT_MMPS)
               if e > r0 and (d[s] - d[e - 1]) >= big]
    return d, v, r0, big, closing


if __name__ == "__main__":
    recs = {(r["part"], r["trial"]): r for r in CSI.load_all()}
    fig, axes = plt.subplots(len(TRIALS), 2, figsize=(13, 3.1 * len(TRIALS)),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.16})
    for row, key in enumerate(TRIALS):
        r = recs[key]
        for col, src in enumerate(("omc", "mmc")):
            ax = axes[row, col]
            d, v, r0, big, closing = signals(r[f"cup_{src}"], r[f"wrist_{src}"])
            ax.plot(d, color="0.25", lw=1.4, label="wrist→cup distance")
            for i, (s, e) in enumerate(closing):
                ax.axvspan(s, e, color="#8ab4f8", alpha=0.35,
                           label="qualifying closing run" if i == 0 else None)
            a = int(np.nanargmin(np.where(np.arange(len(d)) > r0, d, np.inf)))
            ax.axvline(a, color="#d17c00", lw=1.6, ls=":", label="argmin anchor")
            if closing:
                ax.axvline(closing[0][1], color="#2e7d32", lw=1.8, label="SEQUENTIAL grasp (1st run)")
                before = [c for c in closing if c[1] - 1 <= a]
                pick = max(before, key=lambda c: c[1]) if before else min(closing, key=lambda c: c[0])
                ax.axvline(pick[1], color="#c62828", lw=1.8, ls="--", label="ANCHOR-MIN grasp")
            ax.set_title(f"{key[0]} {key[1]}  —  {src.upper()}", fontsize=9)
            ax.set_xlabel("frame"); ax.set_ylabel("mm", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("Why anchor-min's grasp blows up on MMC: the wrist→cup MINIMUM is not at the grasp",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.07)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
