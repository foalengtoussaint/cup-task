"""Body-plane / down-axis: MMC hips vs OMC hips, and what it does to shoulder flexion.

Flexion = angle of the upper arm relative to a trunk DOWN axis (shoulder_mid -> hip_mid). We freeze
that axis to a per-trial CONSTANT (median over frames) because the seated participants' hips are
OCCLUDED by the table, so a per-frame MMC hip is pure jitter. This probe shows, per participant:

  * how VISIBLE the hips even are (MMC finite % vs OMC 100%)
  * the ANGLE between the MMC down axis and the OMC down axis (deg) -- i.e. how much our body plane is
    tilted vs the true one. This tilt is the between-participant flexion OFFSET source.
  * the flexion each axis yields (max over reach->drink), and how much of the flexion error the axis
    tilt explains.

OMC hips are READ from the ground-truth markers for this DIAGNOSTIC only -- they never enter scoring
(the scorer uses OUR hips). Output: paper/downaxis_omc_vs_mmc.png + a per-participant table.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, FPS)
GRID = R._GRID_JOINTS


def _down_axis(sh_mid, hip_mid):
    """per-trial CONSTANT down axis = median of (hip_mid - sh_mid) over finite frames, normalized."""
    d = hip_mid - sh_mid
    fin = np.isfinite(d).all(1)
    if fin.sum() < 5:
        return None
    v = np.nanmedian(d[fin], 0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _flex_max(pose, side, down, win):
    sh, el = pose[f"{side}_shoulder"], pose[f"{side}_elbow"]
    arm = el - sh
    arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    fx = np.degrees(np.arccos(np.clip((arm * down[None, :]).sum(1), -1, 1)))
    if not (win and win[1] > win[0]):
        return np.nan
    seg = fx[win[0]:win[1]]; seg = seg[np.isfinite(seg)]
    return float(np.max(seg)) if len(seg) else np.nan


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    rows = []   # per-trial: part, hip_vis_mmc, tilt_deg, flex_mmc_axis, flex_omc_axis, flex_gt
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        rec = amq.get((automq_part(t["part"]), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        gt = rec.get("max_shoulder_flexion")
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        omc = H._load_omc(t["part"], t["trial"], n)      # READ-ONLY diagnostic (hips), never scored
        # FRAME MATCH: our pose is in the camera-calibration frame, OMC in the mocap-lab frame. To
        # compare down-axes we must first bring OMC into the MMC frame with ONE rigid transform, fit on
        # the ARM + SHOULDER joints (hips EXCLUDED -- they are the disputed landmark). Without this the
        # "tilt" is just the ~90 deg lab->camera rotation, not a body-plane difference. (the earlier
        # unaligned version reported ~96 deg tilt and made OMC hips look 10x WORSE -- a frame artifact.)
        FIT = [f"{side}_shoulder", f"{other}_shoulder", f"{side}_elbow", f"{side}_wrist", "nose"]
        A = np.vstack([omc[j] for j in FIT]); B = np.vstack([t["mmc"][:, GRID.index(j)] for j in FIT])
        fin = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if fin.sum() < 30:
            continue
        Rm, tm, _ = H._kabsch(A[fin], B[fin])
        def to_mmc(x):
            return x @ Rm.T + tm
        sh_mid = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2.0
        hip_mmc = (pose["right_hip"] + pose["left_hip"]) / 2.0
        hip_omc = to_mmc((omc["right_hip"] + omc["left_hip"]) / 2.0)          # OMC hips -> MMC frame
        sh_mid_omc = to_mmc((omc[f"{side}_shoulder"] + omc[f"{other}_shoulder"]) / 2.0)
        d_mmc = _down_axis(sh_mid, hip_mmc)
        d_omc = _down_axis(sh_mid_omc, hip_omc)
        if d_mmc is None or d_omc is None:
            continue
        tilt = float(np.degrees(np.arccos(np.clip(d_mmc @ d_omc, -1, 1))))
        hip_vis = float(np.isfinite(hip_mmc).all(1).mean())
        ph = automq_phases_to_video(rec["phases"], 0, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        rows.append({"part": t["part"], "hip_vis": hip_vis, "tilt": tilt,
                     "flex_mmc": _flex_max(pose, side, d_mmc, rd),
                     "flex_omc": _flex_max(pose, side, d_omc, rd),
                     "gt": gt if (gt is not None and np.isfinite(gt)) else np.nan})

    import pandas as pd
    df = pd.DataFrame(rows)
    parts = sorted(df["part"].unique())
    print(f"\n{'part':6}{'n':>4}{'hip_vis%':>9}{'tilt deg (MMC vs OMC axis)':>28}{'flex_err(MMC axis)':>20}{'flex_err(OMC axis)':>20}")
    tab = []
    for p in parts:
        g = df[df.part == p]
        tilt_med = g["tilt"].median()
        e_mmc = (g["flex_mmc"] - g["gt"]).abs().median()
        e_omc = (g["flex_omc"] - g["gt"]).abs().median()
        tab.append((p, len(g), g["hip_vis"].median()*100, tilt_med, e_mmc, e_omc))
        print(f"{p:6}{len(g):>4}{g['hip_vis'].median()*100:>8.0f}%{tilt_med:>28.1f}{e_mmc:>20.1f}{e_omc:>20.1f}")
    print(f"\nPOOLED  tilt median {df['tilt'].median():.1f} deg   "
          f"|flex_err| MMC-axis {(df['flex_mmc']-df['gt']).abs().median():.1f}  "
          f"OMC-axis {(df['flex_omc']-df['gt']).abs().median():.1f}", flush=True)

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cmap = plt.get_cmap("tab20")
    pcol = {p: cmap(i % 20) for i, p in enumerate(parts)}
    # (1) hip visibility vs axis tilt
    for p in parts:
        g = df[df.part == p]
        ax[0].scatter(g["hip_vis"]*100, g["tilt"], s=14, alpha=0.5, color=pcol[p], label=p)
    ax[0].set_xlabel("MMC hip visibility (% frames finite)"); ax[0].set_ylabel("down-axis tilt: MMC vs OMC (deg)")
    ax[0].set_title("Occluded hips → tilted body plane")
    # (2) tilt vs flexion error (MMC axis)
    for p in parts:
        g = df[df.part == p]
        ax[1].scatter(g["tilt"], (g["flex_mmc"]-g["gt"]), s=14, alpha=0.5, color=pcol[p])
    ax[1].axhline(0, color="0.6", lw=0.8); ax[1].set_xlabel("down-axis tilt (deg)")
    ax[1].set_ylabel("flexion error, MMC axis (deg)"); ax[1].set_title("Axis tilt drives the flexion offset")
    # (3) per-participant |flex err|: MMC axis vs OMC axis (bar)
    x = np.arange(len(parts)); w = 0.38
    em = [(df[df.part==p]["flex_mmc"] - df[df.part==p]["gt"]).abs().median() for p in parts]
    eo = [(df[df.part==p]["flex_omc"] - df[df.part==p]["gt"]).abs().median() for p in parts]
    ax[2].bar(x-w/2, em, w, label="MMC hips (ours)", color="#c1440e")
    ax[2].bar(x+w/2, eo, w, label="OMC hips (truth axis)", color="#e0a200")
    ax[2].set_xticks(x); ax[2].set_xticklabels(parts, rotation=45, fontsize=7)
    ax[2].set_ylabel("median |flexion error| (deg)")
    ax[2].set_title("If we had the true down axis"); ax[2].legend(fontsize=8, frameon=False)
    ax[0].legend(fontsize=6, ncol=2, frameon=False, loc="upper right")
    fig.suptitle("Down-axis (body plane): MMC hips vs OMC hips, and the flexion offset it causes", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PAPER / "downaxis_omc_vs_mmc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    df.to_csv(PAPER / "downaxis_omc_vs_mmc.csv", index=False)
    print(f"\nwrote {out}\nwrote {PAPER/'downaxis_omc_vs_mmc.csv'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
