"""Paper-definition flexion: SAGITTAL-plane angle between (shoulder->elbow) and the shoulder->hip axis
(Alt Murphy et al., PMC5933268, Table 2). Compute on MMC (BA+SmoothNet) AND OMC markers, report the
flexion scalar (max over reach->drink) and interjoint (corr flexion,elbow over reaching) vs AutoMQ.

sagittal flexion recipe (both MMC and OMC):
  down = hip_mid - shoulder_mid        (shoulder->hip; MMC: per-trial CONSTANT median [occluded hips],
                                        OMC: per-frame)
  side = acting_shoulder - other_shoulder   (medial-lateral)
  arm_sag = (elbow-shoulder) with the side-component removed  (project onto sagittal plane)
  flexion = angle(arm_sag, down)
Compare three flexion variants for context: total(current) vs body-sagittal(paper) on both MMC and OMC.
    python paper/scripts/sagittal_flexion_paper.py -> prints + paper/sagittal_flexion_paper.csv
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


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def sagittal_flex(sh, el, shO, hR, hL, const_axis):
    """paper flexion: arm projected onto sagittal plane (perp shoulder-line), angle from shoulder->hip."""
    arm = el - sh
    down = (hR + hL) / 2.0 - (sh + shO) / 2.0
    side = sh - shO
    side_u = side / (np.linalg.norm(side, axis=1, keepdims=True) + 1e-9)
    arm_sag = arm - (arm * side_u).sum(1, keepdims=True) * side_u   # remove medial-lateral component
    if const_axis:                                                  # MMC: freeze the down axis (occluded hips)
        dn = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
        f = np.isfinite(dn).all(1)
        dc = np.nanmedian(dn[f], 0) if f.any() else dn[0]
        return _lp(_ang(arm_sag, np.broadcast_to(dc, arm_sag.shape)))
    return _lp(_ang(arm_sag, down))


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
        # MMC sagittal flexion (paper def), constant axis
        try:
            flex_mmc = sagittal_flex(pose[f"{side}_shoulder"], pose[f"{side}_elbow"],
                                     pose[f"{other}_shoulder"], pose["right_hip"], pose["left_hip"], True)
        except (IndexError, ValueError):
            continue
        elb_mmc = _elbow_series(pose, side)
        # OMC sagittal flexion (per-frame) from markers
        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        flex_omc_sag = elb_omc = None; rr = rec["phases"].get("Reaching"); dd = rec["phases"].get("Drinking")
        w100 = (int(rr[0]), int(dd[1])) if (rr is not None and dd is not None) else None
        reach100 = (int(rr[0]), int(rr[1])) if rr is not None else None
        if key is not None:
            mk = cdf.loc[key, "markers"]
            sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}"); wr = _marker(mk, f"hand_{sd}")
            shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
            hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
            if all(x is not None for x in (sh, el, wr, shO, hR, hL)):
                T = min(map(len, (sh, el, wr, shO, hR, hL)))
                sh, el, wr, shO, hR, hL = (a[:T] for a in (sh, el, wr, shO, hR, hL))
                flex_omc_sag = sagittal_flex(sh, el, shO, hR, hL, False)
                u, v = sh - el, wr - el
                elb_omc = _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))
        rows.append({
            "part": p, "trial": tn, "side": sd,
            "flex_mmc_max": _maxseg(flex_mmc, rd),
            "automq_flex": rec.get("max_shoulder_flexion"),
            "ij_mmc": _corr(flex_mmc, elb_mmc, reach),
            "automq_ij": rec.get("interjoint_coordination"),
            "flex_omc_sag_max": _maxseg(flex_omc_sag, (0, len(flex_omc_sag))) if flex_omc_sag is not None and w100 else np.nan,
            "ij_omc_sag": _corr(flex_omc_sag, elb_omc, reach100) if (flex_omc_sag is not None and elb_omc is not None) else np.nan,
        })
    d = pd.DataFrame(rows)
    d.to_csv(PAPER / "sagittal_flexion_paper.csv", index=False)

    def rs(a, b):
        m = np.isfinite(d[a]) & np.isfinite(d[b])
        if m.sum() < 5:
            return (np.nan, np.nan, 0)
        return (spearmanr(d[a][m], d[b][m]).correlation, float(np.median(d[b][m] - d[a][m])), int(m.sum()))

    print(f"n={len(d)}   PAPER-definition sagittal flexion (shoulder->hip ref, sagittal plane)\n")
    print("=== FLEXION scalar (max, reach->drink) ===")
    r, bias, nn = rs("automq_flex", "flex_mmc_max")
    print(f"  MMC sagittal  vs AutoMQ stored flexion:  rs={r:+.2f}  bias={bias:+.1f}  n={nn}")
    print("=== INTERJOINT (corr flexion,elbow over reaching) ===")
    for a, b, lab in [("automq_ij", "ij_mmc", "MMC sagittal   vs AutoMQ stored ij"),
                      ("ij_omc_sag", "ij_mmc", "MMC sagittal   vs OMC sagittal ij"),
                      ("automq_ij", "ij_omc_sag", "OMC sagittal   vs AutoMQ stored ij")]:
        r, bias, nn = rs(a, b)
        print(f"  {lab:36} rs={r:+.2f}  bias={bias:+.2f}  n={nn}")
    print(f"\n  MMC ij: median {d.ij_mmc.median():.2f} frac<0.8 {(d.ij_mmc<0.8).mean():.2f} | "
          f"OMC-sag ij: median {d.ij_omc_sag.median():.2f} frac<0.8 {(d.ij_omc_sag<0.8).mean():.2f} | "
          f"AutoMQ ij: median {d.automq_ij.median():.2f} frac<0.8 {(d.automq_ij<0.8).mean():.2f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
