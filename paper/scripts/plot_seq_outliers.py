"""The OUTLIERS of the SHIPPED sequential segmenter: where MMC and OMC input disagree most.

One row per trial, OMC left / MMC right, both signals the rule actually reads (wrist->cup and
cup->mouth distance) with the qualifying runs shaded and every boundary marked, so the failure is
visible rather than inferred. Trials chosen as the worst |MMC - OMC| per boundary.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cache_seg_inputs as CSI
from pipeline.segment import _butter_lp, _interp_nan_xyz, _median_smooth, _runs, FPS, GRASP_FLAT_MMPS
from seg_sequential import segment_sequential

CASES = [("P13", "trial_34_L_unaffected", "grasp +250"),
         ("P13", "trial_53_R_affected",   "drink onset +223"),
         ("P10", "trial_59_L_affected",   "drink offset +221"),
         ("P19", "trial_90_R_affected",   "release -726")]
COL = {"reaching": "#2e7d32", "drinking": "#c62828", "back_transport": "#6a1b9a",
       "returning": "#00695c"}
OUT = ROOT / "paper" / "seq_outliers.png"


def prep(cup, hand, nose, fps=FPS):
    c = _butter_lp(_interp_nan_xyz(np.asarray(cup, float)), fps)
    h = _butter_lp(_interp_nan_xyz(np.asarray(hand, float)), fps)
    m = _butter_lp(_interp_nan_xyz(np.asarray(nose, float)), fps)
    T = min(len(c), len(h), len(m)); c, h, m = c[:T], h[:T], m[:T]
    d_wc = np.linalg.norm(h - c, axis=1); d_cm = np.linalg.norm(c - m, axis=1)
    v_wc = np.r_[0.0, np.diff(_median_smooth(d_wc, 11))] * fps
    v_cm = np.r_[0.0, np.diff(_median_smooth(d_cm, 11))] * fps
    big_wc = 0.3 * float(d_wc.max() - d_wc.min()); big_cm = 0.3 * float(d_cm.max() - d_cm.min())
    cl_wc = [(s, e) for s, e in _runs(v_wc < -GRASP_FLAT_MMPS) if (d_wc[s] - d_wc[e - 1]) >= big_wc]
    op_wc = [(s, e) for s, e in _runs(v_wc > GRASP_FLAT_MMPS) if (d_wc[e - 1] - d_wc[s]) >= big_wc]
    cl_cm = [(s, e) for s, e in _runs(v_cm < -GRASP_FLAT_MMPS) if (d_cm[s] - d_cm[e - 1]) >= big_cm]
    op_cm = [(s, e) for s, e in _runs(v_cm > GRASP_FLAT_MMPS) if (d_cm[e - 1] - d_cm[s]) >= big_cm]
    return d_wc, d_cm, cl_wc + op_wc, cl_cm + op_cm


if __name__ == "__main__":
    recs = {(r["part"], r["trial"]): r for r in CSI.load_all()}
    fig, axes = plt.subplots(len(CASES), 2, figsize=(13.5, 3.2 * len(CASES)),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.14})
    for row, (part, trial, why) in enumerate(CASES):
        r = recs[(part, trial)]
        for col, src in enumerate(("omc", "mmc")):
            ax = axes[row, col]
            cup, hand, nose = r[f"cup_{src}"], r[f"wrist_{src}"], r[f"nose_{src}"]
            d_wc, d_cm, runs_wc, runs_cm = prep(cup, hand, nose)
            ax.plot(d_wc, color="0.25", lw=1.4, label="wrist→cup")
            ax.plot(d_cm, color="#1565c0", lw=1.2, alpha=0.8, label="cup→mouth")
            for i, (s, e) in enumerate(runs_wc):
                ax.axvspan(s, e, color="0.6", alpha=0.16,
                           label="qualifying run (wrist→cup)" if i == 0 else None)
            for i, (s, e) in enumerate(runs_cm):
                ax.axvspan(s, e, color="#1565c0", alpha=0.12,
                           label="qualifying run (cup→mouth)" if i == 0 else None)
            seg = {nm: (s, e) for nm, s, e in segment_sequential(cup, hand, nose)}
            for nm, i, lab in (("reaching", 1, "grasp"), ("drinking", 0, "drink on"),
                               ("drinking", 1, "drink off"), ("back_transport", 1, "release")):
                if nm in seg:
                    x = seg[nm][i]
                    ax.axvline(x, color=COL[nm], lw=1.7,
                               ls="-" if i == 0 else "--", label=lab if row == 0 else None)
            for nm, i in (("reaching", 1), ("drinking", 0), ("drinking", 1), ("back_transport", 1)):
                w = r[f"amq_{nm}"]
                if w[0] >= 0:
                    ax.axvline(w[i], color="0.45", lw=1.0, ls=":",
                               label="AutoMQ" if (row == 0 and nm == "reaching" and i == 1) else None)
            # NaN coverage of the raw cup, the thing that differs between the two columns
            nanmask = ~np.isfinite(np.asarray(cup, float)).all(1)
            if nanmask.any():
                ax.plot(np.flatnonzero(nanmask),
                        np.full(nanmask.sum(), ax.get_ylim()[0]), "|", color="#e53935",
                        ms=5, label="cup NaN" if row == 0 and col == 1 else None)
            ax.set_title(f"{part} {trial} — {src.upper()}" + (f"   [{why}]" if col else ""),
                         fontsize=9)
            ax.set_xlabel("frame", fontsize=8); ax.set_ylabel("mm", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6.5, loc="upper right", ncol=2, framealpha=0.9)
    fig.suptitle("Sequential segmenter outliers: same rule, OMC input (left) vs MMC input (right)",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.94, bottom=0.06)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
