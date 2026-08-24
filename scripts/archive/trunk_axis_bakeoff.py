"""Trunk-displacement axis bake-off: fix the P07 frame bug (forward normal from occluded seated hips
mis-projects the real lean) without hurting the others. Variants (all from OUR pose, no OMC):

  perframe   : max|disp . (side x down_perframe)|   [current -> P07 collapses, cohort rs 0.65]
  constdown  : max|disp . (side x down_const)|      [median down axis, like the flexion fix]
  mag3d      : max||disp||                          [raw 3D shoulder-mid displacement magnitude]
  pca        : max|disp . pca1(disp)|               [project on the dominant motion direction]

disp = shoulder-mid(t) - shoulder-mid(first). Reports cohort rs + P07 rs vs AutoMQ trunk_displacement.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from compare_pose_omc_delta import _murphy_signals
from score_vs_automq import load_automq, automq_part


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    acc = {v: {"gt": [], "ours": [], "part": []} for v in
           ["perframe", "constdown", "mag3d", "pca", "nonact_sh", "hipmid"]}
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m: continue
        rec = amq.get((automq_part(t["part"]), int(m.group(1)), m.group(2)))
        if rec is None: continue
        gt = rec.get("max_trunk_displacement")
        if gt is None or not np.isfinite(gt): continue
        from score_vs_automq import _pose_variant_cached
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None: continue
        side = t["side"]; other = "right" if side == "left" else "left"
        sh, shL = pose[f"{side}_shoulder"], pose[f"{other}_shoulder"]
        sh_mid = (sh + shL) / 2.0
        hip_mid = (pose["right_hip"] + pose["left_hip"]) / 2.0
        fin = np.isfinite(sh_mid).all(1) & np.isfinite(hip_mid).all(1)
        if fin.sum() < 10: continue
        disp = sh_mid - sh_mid[fin][0]
        down = hip_mid - sh_mid
        sidev = sh - shL
        def n(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
        vals = {}
        # perframe forward normal
        fwd = np.cross(n(sidev), n(down))
        vals["perframe"] = np.nanmax(np.abs((disp * fwd).sum(1)[fin]))
        # constant (median) down axis
        dc = np.nanmedian(n(down)[fin], 0)
        fwd_c = np.cross(np.nanmedian(n(sidev)[fin], 0), dc); fwd_c = fwd_c / (np.linalg.norm(fwd_c) + 1e-9)
        vals["constdown"] = np.nanmax(np.abs((disp[fin] * fwd_c).sum(1)))
        # 3D magnitude
        vals["mag3d"] = np.nanmax(np.linalg.norm(disp[fin], axis=1))
        # PCA principal axis of displacement
        D = disp[fin] - disp[fin].mean(0)
        try:
            pc1 = np.linalg.svd(D, full_matrices=False)[2][0]
            vals["pca"] = np.nanmax(np.abs((disp[fin] * pc1).sum(1)))
        except Exception:
            vals["pca"] = np.nan
        # NON-ACTING shoulder (doesn't move with the reach) forward-projected on constant fwd
        shO = shL   # 'other' = non-acting side
        dispO = shO - shO[fin][0]
        vals["nonact_sh"] = np.nanmax(np.abs((dispO[fin] * fwd_c).sum(1)))
        # HIP midpoint displacement magnitude (torso base, no reach contamination)
        dh = hip_mid - hip_mid[fin][0]
        vals["hipmid"] = np.nanmax(np.linalg.norm(dh[fin], axis=1))
        for v, val in vals.items():
            if np.isfinite(val):
                acc[v]["gt"].append(gt); acc[v]["ours"].append(val); acc[v]["part"].append(t["part"])

    print(f"{'variant':11}{'n':>5}{'cohort rs':>11}{'bias':>8}{'med|err|':>10}{'P07 rs':>9}{'P07 med|err|':>14}")
    for v in ["perframe", "constdown", "mag3d", "pca", "nonact_sh", "hipmid"]:
        a = np.array(acc[v]["gt"]); o = np.array(acc[v]["ours"]); p = np.array(acc[v]["part"])
        rs = spearmanr(a, o).correlation
        m7 = p == "P07"
        rs7 = spearmanr(a[m7], o[m7]).correlation if m7.sum() > 5 else np.nan
        print(f"{v:11}{len(a):>5}{rs:>+11.2f}{np.median(o-a):>+8.1f}{np.median(np.abs(o-a)):>10.1f}"
              f"{rs7:>+9.2f}{np.median(np.abs(o[m7]-a[m7])):>14.1f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
