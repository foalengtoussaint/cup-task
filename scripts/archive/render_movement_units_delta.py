"""Dedicated movement-unit view: WHY MMC counts 2 and OMC counts 1 (P14 sip 1).

The full-trial speed plot is too crowded to see the extra unit. Here each non-drinking phase
gets its own panel, MMC and OMC overlaid, with every counted min->max oscillation marked using
the SCORER's exact rule (score._count_movement_units). The extra MMC unit -- a jitter
oscillation in the return-home that the smooth mocap lacks -- is then plainly visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import segment
from cup_task.score import (_smoothed_xyz, _hand_speed_mmps, DEFAULT_LOWPASS_HZ,
                            DEFAULT_BUTTER_ORDER, DEFAULT_MU_AMPLITUDE_MMPS, DEFAULT_MU_TIME_GAP_S)
from scripts.score_omc_delta import _omc_cup, _shift
from scripts.compare_pose_omc_delta import _load_mmc, _load_omc, _find_lag, _lp, VIDEO_FPS

PART, TRIAL = "P14", "trial_1_R_unaffected"


def _fill(xyz):
    xyz = np.asarray(xyz, float).copy()
    for ax in range(3):
        v = np.isfinite(xyz[:, ax])
        if v.sum() >= 2:
            xyz[:, ax] = np.interp(np.arange(len(xyz)), np.flatnonzero(v), xyz[v, ax])
    return xyz


def _mu_marks(seg):
    """The scorer's exact rule (score._count_movement_units): for each local min, its NEXT
    local max counts if the max's ABSOLUTE speed > amplitude threshold and it is >= time_gap
    frames later. Returns the (min_idx, max_idx) pairs that count."""
    if len(seg) < 3:
        return []
    min_idx, _ = find_peaks(-seg)
    max_idx, _ = find_peaks(seg)
    gap = max(int(DEFAULT_MU_TIME_GAP_S * VIDEO_FPS), 1)
    out = []
    for mn in min_idx:
        for mx in max_idx:
            if mx <= mn:
                continue
            if seg[mx] > DEFAULT_MU_AMPLITUDE_MMPS and (mx - mn) >= gap:
                out.append((mn, mx))
            break
    return out


def main():
    mmc, n = _load_mmc(PART, TRIAL)
    omc = _load_omc(PART, TRIAL, n)
    lag, _ = _find_lag(mmc["right_wrist"], omc["right_wrist"])
    omc = {j: _shift(v, lag) for j, v in omc.items()}
    cup = _shift(_omc_cup(PART, TRIAL, n), lag)
    cr = _lp(np.linalg.norm(cup - np.nanmedian(cup[:30], 0), axis=1))
    wr = _lp(np.linalg.norm(omc["right_wrist"] - np.nanmedian(omc["right_wrist"][:30], 0), axis=1))
    lift = np.flatnonzero(cr > 150)[0]
    both = np.flatnonzero((np.arange(n) > lift) & (cr < 60) & (wr < 60))
    cut = min(both[0] + 20, n) if len(both) else n
    cup, mmc, omc = cup[:cut], {k: v[:cut] for k, v in mmc.items()}, {k: v[:cut] for k, v in omc.items()}

    seg = segment.segment_cup_only(cup, fps=VIDEO_FPS)
    seg = segment.refine_grasp_with_pose(seg, cup, omc["right_wrist"], omc["nose"], fps=VIDEO_FPS)
    phases = segment.to_murphy_phases(seg, omc["right_wrist"], cup, fps=VIDEO_FPS)

    def speed(P):
        return _hand_speed_mmps(_smoothed_xyz(_fill(P["right_wrist"]), VIDEO_FPS,
                                              DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER), VIDEO_FPS)
    sm, so = speed(mmc), speed(omc)

    # the phases movement units are counted over (everything except drinking)
    mu_phases = [(nm, s, e) for nm, s, e in phases
                 if nm in ("reaching", "forward_transport", "back_transport", "returning")]

    fig, axes = plt.subplots(1, len(mu_phases), figsize=(3.4 * len(mu_phases), 4.2), sharey=True)
    if len(mu_phases) == 1:
        axes = [axes]
    mmc_tot = omc_tot = 0
    for ax, (nm, s, e) in zip(axes, mu_phases):
        t = np.arange(s, e) / VIDEO_FPS
        for lab, sp, col, tot_add in [("MMC", sm, "#0077b6", "m"), ("OMC", so, "#e63946", "o")]:
            segv = sp[s:e]
            ax.plot(t, segv, color=col, lw=1.6, label=lab)
            marks = _mu_marks(segv)
            for (mn, mx) in marks:
                ax.annotate("", xy=(t[mx], segv[mx]), xytext=(t[mn], segv[mn]),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
                ax.plot(t[mx], segv[mx], "*", color=col, ms=16, mec="k", mew=0.6, zorder=5)
            if lab == "MMC":
                mmc_tot += len(marks)
            else:
                omc_tot += len(marks)
            mcount = len(marks)
            ax.text(0.03, 0.97 if lab == "MMC" else 0.88, f"{lab}: {mcount} unit(s)",
                    transform=ax.transAxes, color=col, fontsize=9, va="top", fontweight="bold")
        ax.axhline(DEFAULT_MU_AMPLITUDE_MMPS, color="gray", ls=":", lw=1)
        ax.text(t[0], DEFAULT_MU_AMPLITUDE_MMPS + 6, f"{DEFAULT_MU_AMPLITUDE_MMPS:.0f} mm/s thr",
                fontsize=7, color="gray")
        ax.set_title(f"{nm}\n({s/VIDEO_FPS:.1f}-{e/VIDEO_FPS:.1f}s)", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("hand speed mm/s")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Movement units, MMC vs OMC  (total: MMC {mmc_tot}, OMC {omc_tot}) — "
                 f"★ = a counted min→max oscillation", fontsize=11)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "out" / f"movement_units_{PART}_{TRIAL}.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  (MMC {mmc_tot} / OMC {omc_tot})", flush=True)


if __name__ == "__main__":
    main()
