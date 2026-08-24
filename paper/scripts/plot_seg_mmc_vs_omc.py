"""Two trials where the MMC segmentation disagrees most with the OMC segmentation (new 26x cup).

Same rule both sides; only the input cup differs. Top row of each pair: the two distances the rule
reads, from the MMC cup, with MMC boundaries (solid) and OMC boundaries (dashed). Bottom: the wrist
speed the position measures are computed from, so the effect on movement units / time-to-PV is
visible rather than inferred.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import os
os.environ.setdefault("OT_SEG_INPUTS_DIR", "seg_inputs_26x")
import cache_seg_inputs as CSI
from pipeline.segment import _butter_lp, _interp_nan_xyz, FPS
from pipeline.score import _smoothed_xyz, _hand_speed_mmps, DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER
from seg_sequential import segment_sequential

CASES = [("P12", "trial_24_L_unaffected"), ("P10", "trial_8_R_unaffected")]
COL = {"grasp": "#2e7d32", "drink on": "#c62828", "drink off": "#6a1b9a", "release": "#00695c"}
OUT = ROOT / "paper" / "seg_mmc_vs_omc.png"


def bounds(seg):
    d = {nm: (s, e) for nm, s, e in seg}
    return {"grasp": d.get("reaching", (np.nan,) * 2)[1],
            "drink on": d.get("drinking", (np.nan,) * 2)[0],
            "drink off": d.get("drinking", (np.nan,) * 2)[1],
            "release": d.get("back_transport", (np.nan,) * 2)[1]}


if __name__ == "__main__":
    recs = {(r["part"], r["trial"]): r for r in CSI.load_all()}
    fig, axes = plt.subplots(2, len(CASES), figsize=(7.2 * len(CASES), 7.0),
                             gridspec_kw={"hspace": 0.33, "height_ratios": [1.25, 1]})
    for col, key in enumerate(CASES):
        r = recs[key]
        f_ = lambda x: _butter_lp(_interp_nan_xyz(np.asarray(x, float)), FPS)
        cup_m, cup_o = f_(r["cup_mmc"]), f_(r["cup_omc"])
        wr_m, no_m = f_(r["wrist_mmc"]), f_(r["nose_mmc"])
        wr_o, no_o = f_(r["wrist_omc"]), f_(r["nose_omc"])
        T = min(map(len, (cup_m, cup_o, wr_m, no_m)))
        b_m = bounds(segment_sequential(r["cup_mmc"], r["wrist_mmc"], r["nose_mmc"]))
        b_o = bounds(segment_sequential(r["cup_omc"], r["wrist_omc"], r["nose_omc"]))

        ax = axes[0, col]
        ax.plot(np.linalg.norm(wr_m - cup_m, axis=1)[:T], color="0.25", lw=1.4, label="wrist→cup (MMC)")
        ax.plot(np.linalg.norm(cup_m - no_m, axis=1)[:T], color="#1565c0", lw=1.3, label="cup→mouth (MMC)")
        ax.plot(np.linalg.norm(cup_o - no_o, axis=1)[:T], color="#1565c0", lw=1.0, ls=":", alpha=.7,
                label="cup→mouth (OMC)")
        for k in COL:
            if np.isfinite(b_m[k]):
                ax.axvline(b_m[k], color=COL[k], lw=1.8, label=f"{k} MMC" if col == 0 else None)
            if np.isfinite(b_o[k]):
                ax.axvline(b_o[k], color=COL[k], lw=1.5, ls="--", alpha=.85,
                           label=f"{k} OMC" if col == 0 else None)
        ax.set_title(f"{key[0]} {key[1]}", fontsize=10)
        ax.set_ylabel("distance [mm]", fontsize=9); ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=6.5, ncol=2, loc="upper center", framealpha=.9)

        ax = axes[1, col]
        ok = np.isfinite(wr_m[:T]).all(1)
        sp = _hand_speed_mmps(_smoothed_xyz(wr_m[:T][ok], FPS, DEFAULT_LOWPASS_HZ,
                                            DEFAULT_BUTTER_ORDER), FPS)
        ax.plot(sp, color="#e65100", lw=1.3, label="wrist speed (MMC)")
        for k in COL:
            if np.isfinite(b_m[k]):
                ax.axvline(b_m[k], color=COL[k], lw=1.8)
            if np.isfinite(b_o[k]):
                ax.axvline(b_o[k], color=COL[k], lw=1.5, ls="--", alpha=.85)
        ax.set_xlabel("frame", fontsize=9); ax.set_ylabel("speed [mm/s]", fontsize=9)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("MMC segmentation (solid) vs OMC segmentation (dashed) — same rule, different cup",
                 fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.08)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
