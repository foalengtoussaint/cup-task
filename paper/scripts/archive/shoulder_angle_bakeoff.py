"""Does a RAW shoulder angle agree with OMC like the elbow angle does?

KEY FINDING from AutoMQ's notebook: `pure_shoulder_angle` = angle(upper_arm_vector, GLOBAL Z-axis)
-- NO hips. `shoulder_flexion` is that same vector decomposed in the sagittal plane. So AutoMQ's
shoulder reference is LAB VERTICAL, not the shoulder->hip axis we use. That's a DEFINITION mismatch,
not just pose error. This bakes off shoulder-angle definitions vs AutoMQ's stored scalars:

  hip_down    : angle(arm, shoulder->hip axis)                  [our current flexion recipe]
  vert_omcZ   : angle(arm, OMC lab-Z mapped into our frame)     [AutoMQ's ACTUAL definition]
  three_pt    : angle at shoulder in elbow-shoulder-hip         [a pure 3-point angle, no vertical]

vs AutoMQ `max_shoulder_flexion` (reach->drink max), and `pure_shoulder_angle` series where useful.
The elbow angle (shoulder-elbow-wrist, no reference axis) is the benchmark it should try to match.
OMC used READ-ONLY: hips + lab-Z direction for the diagnostic; scoring still uses our pose only.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, _elbow_series, FPS)
GRID = R._GRID_JOINTS


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _maxseg(a, win):
    if not (win and win[1] > win[0]):
        return np.nan
    s = a[win[0]:win[1]]; s = s[np.isfinite(s)]
    return float(np.max(s)) if len(s) else np.nan


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    keys = ["hip_down", "vert_omcZ", "vert_rigZ", "vert_partZ", "three_pt", "elbow_bench"]
    # OMC-FREE candidate: a per-PARTICIPANT vertical learned WITHOUT OMC, as the median torso-up axis
    # (shoulder_mid - hip_mid) over ALL that participant's trials/frames in OUR OWN frame. This tests
    # whether a self-derived vertical recovers the 0.89 that the borrowed OMC lab-Z gives.
    part_up = {}
    for t in GT.load_clean(need_reproj=False):
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        smid = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2.0
        hmid = (pose["right_hip"] + pose["left_hip"]) / 2.0
        up = smid - hmid                                  # torso UP (shoulder above hip)
        f = np.isfinite(up).all(1)
        if f.any():
            part_up.setdefault(t["part"], []).append(np.nanmedian(up[f], 0))
    part_up = {p: (np.median(np.stack(v), 0) / (np.linalg.norm(np.median(np.stack(v), 0)) + 1e-9))
               for p, v in part_up.items()}
    gt = {k: [] for k in keys}; ours = {k: [] for k in keys}; parts = {k: [] for k in keys}
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        rec = amq.get((automq_part(t["part"]), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        gflex = rec.get("max_shoulder_flexion"); gelb = rec.get("max_elbow_angle")
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        omc = H._load_omc(t["part"], t["trial"], n)
        # frame map OMC->MMC on arm+shoulders (hips excluded), to import lab-Z and OMC hips fairly
        FIT = [f"{side}_shoulder", f"{other}_shoulder", f"{side}_elbow", f"{side}_wrist", "nose"]
        A = np.vstack([omc[j] for j in FIT]); B = np.vstack([t["mmc"][:, GRID.index(j)] for j in FIT])
        fin = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if fin.sum() < 30:
            continue
        Rm, tm, _ = H._kabsch(A[fin], B[fin])
        zc = Rm @ np.array([0.0, 0.0, 1.0])          # OMC lab vertical, rotated into our frame
        sh, el = pose[f"{side}_shoulder"], pose[f"{side}_elbow"]
        arm = el - sh                                 # shoulder->elbow
        hip_mid = (pose["right_hip"] + pose["left_hip"]) / 2.0
        sh_mid = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2.0
        down = hip_mid - sh_mid
        dfin = np.isfinite(down).all(1)
        down_c = np.nanmedian(down[dfin], 0) if dfin.any() else np.array([0, 0, -1.0])
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        # three definitions of the shoulder angle. NB AutoMQ's upper_arm_vector = shoulder - elbow
        # (points el->sh); using el-sh flips it and adds a ~105 deg bias. Verified: (sh-el) vs OMC +Z
        # reproduces pure_shoulder_angle (42.9 vs 42.0 for P07). Lab-Z (index 2,+) IS vertical.
        upper = sh - el                                                      # AutoMQ direction (el->sh)
        a_hip = H._lp(_ang(upper, -down_c[None, :]))                          # vs hip-UP (matches el->sh)
        a_z = H._lp(_ang(upper, zc[None, :]))                                 # vs OMC lab-Z (AutoMQ def)
        a_rig = H._lp(_ang(upper, np.array([0.0, 0.0, 1.0])[None, :]))        # vs RIG world-Z (OMC-free)
        pu = part_up.get(t["part"])                                           # per-participant torso-up (OMC-free)
        a_pz = H._lp(_ang(upper, pu[None, :])) if pu is not None else np.full(len(upper), np.nan)
        a_3p = H._lp(_ang(el - sh, hip_mid - sh))                             # 3-pt elbow-shoulder-hip
        elb = _elbow_series(pose, side)                                       # benchmark 3-pt elbow angle
        vals = {"hip_down": (_maxseg(a_hip, rd), gflex),
                "vert_omcZ": (_maxseg(a_z, rd), gflex),
                "vert_rigZ": (_maxseg(a_rig, rd), gflex),
                "vert_partZ": (_maxseg(a_pz, rd), gflex),
                "three_pt": (_maxseg(a_3p, rd), gflex),
                "elbow_bench": (_maxseg(elb, (0, len(elb))), gelb)}
        for k, (o, g) in vals.items():
            if np.isfinite(o) and g is not None and np.isfinite(g):
                ours[k].append(o); gt[k].append(g); parts[k].append(t["part"])

    note = {"hip_down": "<- current flexion (hips)", "vert_omcZ": "<- AutoMQ def, borrows OMC lab-Z",
            "vert_rigZ": "<- rig world-Z (OMC-FREE)", "vert_partZ": "<- per-participant torso-up (OMC-FREE)",
            "three_pt": "<- pure 3-point elbow-sh-hip", "elbow_bench": "<- elbow benchmark"}
    print(f"{'definition':13}{'n':>5}{'rs':>8}{'bias':>9}{'med|err|':>10}   (vs AutoMQ)")
    for k in keys:
        a = np.array(gt[k]); o = np.array(ours[k])
        if len(a) < 5:
            print(f"{k:13}{len(a):>5}   too few"); continue
        rs = spearmanr(a, o).correlation
        print(f"{k:13}{len(a):>5}{rs:>+8.2f}{np.median(o-a):>+9.1f}{np.median(np.abs(o-a)):>10.1f}   {note.get(k,'')}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
