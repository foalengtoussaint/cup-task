"""Show WHERE each Murphy measure comes from on the signal, MMC vs OMC, so a difference in the
scalar is visible on the curve. One complete cycle (P14 sip 1).

Panels:
  1  hand SPEED (the signal peak_velocity / time-to-peak / movement-units all read), with:
       * phase bands shaded (reaching / forward_transport / drinking / back_transport / returning)
       * the peak_velocity point marked on each side (scoped to REACHING)
       * movement-unit oscillations marked (counted over everything except drinking)
  2  trunk Y (max_trunk_displacement = max |Y - Y[0]|), excursion marked each side.

The point: read off the graph why MMC's peak is lower / why it finds an extra movement unit.
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
from cup_task.score import (_smoothed_xyz, _hand_speed_mmps, _butter_lowpass,
                            DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER,
                            DEFAULT_MU_AMPLITUDE_MMPS, DEFAULT_MU_TIME_GAP_S)
from scripts.score_omc_delta import _omc_cup, _shift
from scripts.compare_pose_omc_delta import (_load_mmc, _load_omc, _find_lag, _lp, VIDEO_FPS)

PART, TRIAL = "P14", "trial_1_R_unaffected"
PHASE_COL = {"reaching": "#8ecae6", "forward_transport": "#ffb703", "drinking": "#fb8500",
             "back_transport": "#ffb703", "returning": "#8ecae6"}


def _fill(xyz):
    xyz = np.asarray(xyz, float).copy()
    for ax in range(3):
        v = np.isfinite(xyz[:, ax])
        if v.sum() >= 2:
            xyz[:, ax] = np.interp(np.arange(len(xyz)), np.flatnonzero(v), xyz[v, ax])
    return xyz


def _mu_marks(speed, s, e):
    """Movement-unit min->max pairs in [s,e) using the SCORER's EXACT rule
    (score._count_movement_units): for each local min, the NEXT local max counts if its
    ABSOLUTE speed exceeds the amplitude threshold and it is >= time_gap later. Must match the
    scorer's logic exactly, or the picture disagrees with the number (it did: an earlier version
    thresholded the min->max RISE instead of the max's absolute value, and mis-placed the marks)."""
    seg = speed[s:e]
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
                out.append((s + mn, s + mx))
            break
    return out


def main():
    mmc, n = _load_mmc(PART, TRIAL)
    omc = _load_omc(PART, TRIAL, n)
    lag, _ = _find_lag(mmc["right_wrist"], omc["right_wrist"])
    omc = {j: _shift(v, lag) for j, v in omc.items()}
    cup = _shift(_omc_cup(PART, TRIAL, n), lag)

    # cut to first complete cycle (both cup+wrist at rest), as the scorer does
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
    t = np.arange(cut) / VIDEO_FPS

    def reach():
        for nm, s, e in phases:
            if nm == "reaching":
                return s, e
        return None
    rs, re = reach()

    from scripts.compare_pose_omc_delta import _murphy_signals

    def elbow_angle(P):
        u, v = P["right_shoulder"] - P["right_elbow"], P["right_wrist"] - P["right_elbow"]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        return _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1.3]})
    # phase bands
    for nm, s, e in phases:
        if nm in PHASE_COL:
            ax[0].axvspan(s / VIDEO_FPS, e / VIDEO_FPS, color=PHASE_COL[nm], alpha=0.25)
            ax[0].text((s + e) / 2 / VIDEO_FPS, ax[0].get_ylim()[1], nm, fontsize=6,
                       ha="center", va="top", rotation=90, alpha=0.7)
    for lab, sp, col in [("MMC (cameras)", sm, "#0077b6"), ("OMC (mocap)", so, "#e63946")]:
        ax[0].plot(t, sp, color=col, lw=1.4, label=lab)
        # peak in reaching
        seg_r = sp[rs:re]
        if len(seg_r):
            pk = rs + int(np.argmax(seg_r))
            ax[0].plot(t[pk], sp[pk], "o", color=col, ms=9, mfc="none", mew=2)
            ax[0].annotate(f"peak {sp[pk]:.0f}", (t[pk], sp[pk]), color=col, fontsize=8,
                           xytext=(0, 8), textcoords="offset points", ha="center")
        # movement units over everything except drinking -- draw each min->max rise as a bold
        # arrow so the COUNT is legible (this is why MMC reads 2 vs OMC 1)
        mu_n = 0
        for nm, s, e in phases:
            if nm == "drinking":
                continue
            for (a, b) in _mu_marks(sp, s, e):
                mu_n += 1
                ax[0].annotate("", xy=(t[b], sp[b]), xytext=(t[a], sp[a]),
                               arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, alpha=0.9))
                ax[0].plot(t[b], sp[b], "*", color=col, ms=14, mec="k", mew=0.5, zorder=5)
        ax[0].plot([], [], "*", color=col, ms=12, label=f"{lab.split()[0]} movement units: {mu_n}")
    ax[0].axvspan(rs / VIDEO_FPS, re / VIDEO_FPS, ymin=0.97, ymax=1.0, color="k", alpha=0.4)
    ax[0].set_ylabel("hand speed mm/s")
    ax[0].set_title("Murphy measures on the signal: peak_velocity (○, in reaching), "
                    "movement units (△, min→max outside drinking)", fontsize=10)
    ax[0].legend(loc="upper right", fontsize=8)

    # trunk
    for lab, P, col in [("MMC", mmc, "#0077b6"), ("OMC", omc, "#e63946")]:
        ty = _butter_lowpass(_fill((P["right_shoulder"] + P["left_shoulder"]) / 2)[:, 1],
                             VIDEO_FPS, DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER)
        d = ty - ty[0]
        ax[1].plot(t, d, color=col, lw=1.4, label=lab)
        k = int(np.argmax(np.abs(d)))
        ax[1].plot(t[k], d[k], "o", color=col, ms=9, mfc="none", mew=2)
        ax[1].annotate(f"max |disp| {abs(d[k]):.0f}mm", (t[k], d[k]), color=col, fontsize=8,
                       xytext=(0, 6), textcoords="offset points")
    ax[1].set_ylabel("trunk Y - Y[0] mm")
    ax[1].legend(fontsize=8)

    # angle panel: elbow angle + shoulder flexion, with the phase-scoped scalar points marked
    def rband(name):
        for nm, s, e in phases:
            if nm == name:
                return s, e
        return None
    rr, dd = rband("reaching"), rband("drinking")
    for lab, P, col in [("MMC", mmc, "#0077b6"), ("OMC", omc, "#e63946")]:
        elb = elbow_angle(P)
        flex = _lp(_murphy_signals(P)["shoulder_flexion"])
        ax[2].plot(t, elb, color=col, lw=1.4, label=f"{lab} elbow")
        ax[2].plot(t, flex, color=col, lw=1.1, ls="--", alpha=0.7, label=f"{lab} sh.flex")
        if rr:  # elbow_extension_reaching = MIN elbow in reaching
            k = rr[0] + int(np.argmin(elb[rr[0]:rr[1]]))
            ax[2].plot(t[k], elb[k], "v", color=col, ms=8)
        if dd:  # shoulder_flexion_drinking = MAX flex in drinking
            k = dd[0] + int(np.argmax(flex[dd[0]:dd[1]]))
            ax[2].plot(t[k], flex[k], "^", color=col, ms=8)
    ax[2].set_ylabel("angle deg"); ax[2].set_xlabel("time (s)")
    ax[2].set_title("ANGLE measures (raw-point): elbow ▽=extension_reaching(min), "
                    "sh.flex △=flexion_drinking(max)", fontsize=9)
    ax[2].legend(fontsize=7, ncol=2)

    fig.suptitle(f"DELTA {PART} {TRIAL} (first complete cycle): why the Murphy scalars differ",
                 fontsize=11)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "out" / f"measures_{PART}_{TRIAL}.png"
    fig.savefig(out, dpi=115)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
