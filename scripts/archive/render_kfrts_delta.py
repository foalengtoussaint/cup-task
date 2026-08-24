"""Show what KF+RTS smoothing does to the MMC measures vs baseline vs OMC (P14 sip 1).

Three traces: MMC baseline (raw triangulation), MMC + consensus KF+RTS, OMC (mocap).
Panels: wrist SPEED (peak_velocity ○ in reaching, movement units ★) and elbow ANGULAR VELOCITY
(peak_elbow_ang_vel). The point is to SEE why KF+RTS fixes peak LOCATION but flattens peak
MAGNITUDE and erases movement units.
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
from pipeline import segment
from pipeline.triangulate import kf_rts_smooth
from scripts.score_omc_delta import _kf_only
from pipeline.score import (_smoothed_xyz, _hand_speed_mmps, DEFAULT_LOWPASS_HZ,
                            DEFAULT_BUTTER_ORDER, DEFAULT_MU_AMPLITUDE_MMPS, DEFAULT_MU_TIME_GAP_S)
from scripts.score_omc_delta import _omc_cup, _shift
from scripts.compare_pose_omc_delta import _load_mmc, _load_omc, _find_lag, _lp, VIDEO_FPS

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


def _mu(seg):
    if len(seg) < 3:
        return []
    mn_i, _ = find_peaks(-seg); mx_i, _ = find_peaks(seg)
    gap = max(int(DEFAULT_MU_TIME_GAP_S * VIDEO_FPS), 1)
    out = []
    for mn in mn_i:
        for mx in mx_i:
            if mx <= mn:
                continue
            if seg[mx] > DEFAULT_MU_AMPLITUDE_MMPS and (mx - mn) >= gap:
                out.append(mx)
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
    cup = cup[:cut]
    mmc = {k: v[:cut] for k, v in mmc.items()}
    omc = {k: v[:cut] for k, v in omc.items()}
    mmc_k = {k: kf_rts_smooth(v, fps=VIDEO_FPS) for k, v in mmc.items()}
    mmc_f = {k: _kf_only(v, fps=VIDEO_FPS) for k, v in mmc.items()}

    seg = segment.segment_cup_only(cup, fps=VIDEO_FPS)
    seg = segment.refine_grasp_with_pose(seg, cup, omc["right_wrist"], omc["nose"], fps=VIDEO_FPS)
    phases = segment.to_murphy_phases(seg, omc["right_wrist"], cup, fps=VIDEO_FPS)
    reach = next(((s, e) for nm, s, e in phases if nm == "reaching"), None)
    t = np.arange(cut) / VIDEO_FPS

    def speed(P):
        return _hand_speed_mmps(_smoothed_xyz(_fill(P["right_wrist"]), VIDEO_FPS,
                                              DEFAULT_LOWPASS_HZ, DEFAULT_BUTTER_ORDER), VIDEO_FPS)

    def elb_angvel(P):
        u, v = P["right_shoulder"] - P["right_elbow"], P["right_wrist"] - P["right_elbow"]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        ang = _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))
        return np.abs(np.gradient(ang)) * VIDEO_FPS

    series = [("MMC baseline", mmc, "#0077b6", "-"),
              ("MMC + KF-only", mmc_f, "#52b788", "-"),
              ("MMC + KF+RTS", mmc_k, "#00b4d8", "-"),
              ("OMC (mocap)", omc, "#e63946", "-")]

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for nm, s, e in phases:
        if nm in PHASE_COL:
            ax[0].axvspan(s / VIDEO_FPS, e / VIDEO_FPS, color=PHASE_COL[nm], alpha=0.18)
            ax[0].text((s + e) / 2 / VIDEO_FPS, ax[0].get_ylim()[1], nm, fontsize=6,
                       ha="center", va="top", rotation=90, alpha=0.6)
    for lab, P, col, ls in series:
        sp = speed(P)
        ax[0].plot(t, sp, col, ls=ls, lw=1.6, label=lab)
        if reach:
            pk = reach[0] + int(np.argmax(sp[reach[0]:reach[1]]))
            ax[0].plot(t[pk], sp[pk], "o", color=col, ms=9, mfc="none", mew=2)
            ax[0].annotate(f"{sp[pk]:.0f}", (t[pk], sp[pk]), color=col, fontsize=8,
                           xytext=(2, 6), textcoords="offset points")
        nmu = 0
        for pn, s, e in phases:
            if pn == "drinking":
                continue
            for mx in _mu(sp[s:e]):
                nmu += 1
                ax[0].plot(t[s + mx], sp[s + mx], "*", color=col, ms=13, mec="k", mew=0.4, zorder=5)
        ax[0].plot([], [], col, label=f"   → {lab.split()[0] if 'MMC' in lab else 'OMC'} "
                   f"peak {sp[reach[0] + int(np.argmax(sp[reach[0]:reach[1]]))]:.0f}, MU {nmu}")
    ax[0].set_ylabel("wrist speed mm/s")
    ax[0].set_title("peak_velocity (○, in reaching) + movement units (★): KF+RTS flattens the "
                    "peak & erases the MU", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper right", ncol=2)

    for lab, P, col, ls in series:
        av = elb_angvel(P)
        ax[1].plot(t, av, col, ls=ls, lw=1.5, label=f"{lab}  peak {np.nanmax(av):.0f}°/s")
    ax[1].set_ylabel("elbow angular vel °/s"); ax[1].set_xlabel("time (s)")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_title("peak_elbow_angular_velocity: KF+RTS pulls the jittery MMC peak down",
                    fontsize=10)

    fig.suptitle(f"KF+RTS vs baseline vs OMC — {PART} {TRIAL} (sip 1)", fontsize=11)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "out" / f"kfrts_{PART}_{TRIAL}.png"
    fig.savefig(out, dpi=118)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
