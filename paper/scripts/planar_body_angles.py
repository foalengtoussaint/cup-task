"""Paper-faithful shoulder angles: orthonormal BODY frame from 4 torso points (2 shoulders + 2 hips),
flexion = arm angle in the SAGITTAL plane, abduction = arm angle in the FRONTAL plane, both from the
shoulder->hip down axis (Alt Murphy PMC5933268, Table 2).

body frame (per frame; MMC freezes down+forward per-trial for occluded hips):
  down    = normalize(hip_mid - sh_mid)
  forward = normalize(shoulder_line x down)          # frontal-plane normal
  lateral = normalize(down x forward)                # cleaned medial-lateral
  flexion   = angle( arm - (arm.lateral)lateral , down )     # sagittal-plane angle
  abduction = angle( arm - (arm.forward)forward , down )     # frontal-plane angle

Runs on OMC markers and MMC (BA+SmoothNet). Reports flexion + abduction scalars (max reach->drink)
and interjoint (corr flexion,elbow reaching) vs AutoMQ stored, and MMC-vs-OMC(same method).
    python paper/scripts/planar_body_angles.py -> prints + paper/planar_body_angles.csv
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from compare_pose_omc_delta import _lp
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, _elbow_series, AUTOMQ)
GRID = R._GRID_JOINTS


def _n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _angvec(a, ref):
    a = _n(a); ref = _n(ref)
    return np.degrees(np.arccos(np.clip((a * ref).sum(-1), -1, 1)))


def planar_angles(sh, el, shO, hR, hL, freeze):
    """returns (flexion_series, abduction_series) in the body frame."""
    arm = el - sh
    sh_mid = (sh + shO) / 2.0
    hip_mid = (hR + hL) / 2.0
    down = _n(hip_mid - sh_mid)
    sline = _n(sh - shO)                          # acting shoulder outward
    forward = _n(np.cross(sline, down))           # frontal-plane normal (anterior)
    lateral = _n(np.cross(down, forward))         # cleaned medial-lateral
    if freeze:                                    # MMC: occluded hips -> per-trial constant frame
        f = np.isfinite(down).all(1) & np.isfinite(forward).all(1)
        dc = _n(np.nanmedian(down[f], 0)); fc = _n(np.nanmedian(forward[f], 0)); lc = _n(np.nanmedian(lateral[f], 0))
        down = np.broadcast_to(dc, arm.shape); forward = np.broadcast_to(fc, arm.shape); lateral = np.broadcast_to(lc, arm.shape)
    arm_sag = arm - (arm * lateral).sum(1, keepdims=True) * lateral    # onto sagittal plane
    arm_fro = arm - (arm * forward).sum(1, keepdims=True) * forward    # onto frontal plane
    flex = _lp(_angvec(arm_sag, down))
    abd = _lp(_angvec(arm_fro, down))
    return flex, abd


def _maxseg(a, w):
    if not (w and w[1] > w[0]):
        return np.nan
    s = a[w[0]:w[1]]; s = s[np.isfinite(s)]
    return float(np.max(s)) if len(s) else np.nan


def _corr(a, b, w):
    if not (w and w[1] - w[0] >= 12):
        return np.nan
    mg = int(0.1 * (w[1] - w[0])); x, y = a[w[0]+mg:w[1]-mg], b[w[0]+mg:w[1]-mg]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.std(x[m]) < 1e-6 or np.std(y[m]) < 1e-6:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _marker(mk, n):
    try:
        return mk.xs(n, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    mk_cache = {}
    rows = []
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        p = t["part"]; tn = int(m.group(1)); sd = m.group(2)
        rec = amq.get((automq_part(p), tn, sd))
        if rec is None or rec.get("phases") is None:
            continue
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"; n = t["mmc"].shape[0]
        omc = H._load_omc(p, t["trial"], n)
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        try:
            flexM, abdM = planar_angles(pose[f"{side}_shoulder"], pose[f"{side}_elbow"],
                                        pose[f"{other}_shoulder"], pose["right_hip"], pose["left_hip"], True)
        except (IndexError, ValueError):
            continue
        elbM = _elbow_series(pose, side)
        # OMC
        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        flexO = abdO = elbO = None; rr = rec["phases"].get("Reaching"); dd = rec["phases"].get("Drinking")
        rd100 = (int(rr[0]), int(dd[1])) if (rr is not None and dd is not None) else None
        reach100 = (int(rr[0]), int(rr[1])) if rr is not None else None
        if key is not None:
            mk = cdf.loc[key, "markers"]
            sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}"); wr = _marker(mk, f"hand_{sd}")
            shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
            hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
            if all(x is not None for x in (sh, el, wr, shO, hR, hL)):
                T = min(map(len, (sh, el, wr, shO, hR, hL)))
                sh, el, wr, shO, hR, hL = (a[:T] for a in (sh, el, wr, shO, hR, hL))
                flexO, abdO = planar_angles(sh, el, shO, hR, hL, False)
                u, v = sh - el, wr - el
                elbO = _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))
        rows.append({
            "part": p, "trial": tn, "side": sd,
            "flexM": _maxseg(flexM, rd), "abdM": _maxseg(abdM, rd),
            "automq_flex": rec.get("max_shoulder_flexion"), "automq_abd": rec.get("max_shoulder_abduction"),
            "ijM": _corr(flexM, elbM, reach), "automq_ij": rec.get("interjoint_coordination"),
            "flexO": _maxseg(flexO, rd100) if flexO is not None else np.nan,
            "abdO": _maxseg(abdO, rd100) if abdO is not None else np.nan,
            "ijO": _corr(flexO, elbO, reach100) if (flexO is not None and elbO is not None) else np.nan,
        })
    d = pd.DataFrame(rows)
    d.to_csv(PAPER / "planar_body_angles.csv", index=False)

    def rs(a, b):
        m = np.isfinite(d[a]) & np.isfinite(d[b])
        if m.sum() < 5:
            return (np.nan, np.nan, 0)
        return (spearmanr(d[a][m], d[b][m]).correlation, float(np.median(d[b][m] - d[a][m])), int(m.sum()))

    print(f"n={len(d)}   PLANAR body-frame angles (4-point frame, sagittal flexion + frontal abduction)\n")
    print("=== SCALARS (max reach->drink) vs AutoMQ stored ===")
    for a, b, lab in [("automq_flex", "flexM", "flexion  MMC"), ("automq_flex", "flexO", "flexion  OMC"),
                      ("automq_abd", "abdM", "abduction MMC"), ("automq_abd", "abdO", "abduction OMC")]:
        r, bias, nn = rs(a, b); print(f"  {lab:16} rs={r:+.2f}  bias={bias:+.1f}  n={nn}")
    print("\n=== INTERJOINT (corr flexion,elbow reaching) ===")
    for a, b, lab in [("automq_ij", "ijM", "MMC  vs AutoMQ stored"), ("automq_ij", "ijO", "OMC  vs AutoMQ stored"),
                      ("ijO", "ijM", "MMC  vs OMC (same method)")]:
        r, bias, nn = rs(a, b); print(f"  {lab:26} rs={r:+.2f}  bias={bias:+.2f}  n={nn}")
    print(f"\n  ij medians: MMC {d.ijM.median():.2f} (frac<0.8 {(d.ijM<0.8).mean():.2f}) | "
          f"OMC {d.ijO.median():.2f} (frac<0.8 {(d.ijO<0.8).mean():.2f}) | AutoMQ {d.automq_ij.median():.2f}")
    print(f"  P19 ij: MMC {d[d.part=='P19'].ijM.median():.2f}  OMC {d[d.part=='P19'].ijO.median():.2f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
