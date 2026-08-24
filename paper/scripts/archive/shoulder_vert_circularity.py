"""Is the arm-vs-vert_omcZ agreement REAL, or circular (Kabsch re-reading OMC's arm back to itself)?

vert_omcZ imports OMC's vertical via a rigid Kabsch fit. If that fit includes the ARM joints, a
systematic arm-orientation error in our pose could be absorbed into R and flatter the score. Three
checks, strongest first:

  A. ARM-EXCLUDED alignment: fit Kabsch on SHOULDERS + NOSE only (no elbow/wrist), import THAT vertical,
     then score our arm-vs-vertical. The arm never touches the transform -> agreement can't be circular.
  B. per-FRAME trajectory corr: our (arm vs OMC-vert) SERIES vs OMC's OWN stored shoulder_flexion SERIES
     (computed by AutoMQ in its native frame, never touched by our Kabsch). Median per-trial Pearson r.
  C. control: shuffle the vertical across trials (wrong participant's axis) -> should COLLAPSE if the
     agreement is real (i.e. the specific vertical matters, not just 'any downward-ish axis').

vs AutoMQ max_shoulder_flexion (scalar) for A/C; vs AutoMQ shoulder_flexion series for B. OMC read-only.
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
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, AUTOMQ, C3D_RATE, FPS)
GRID = R._GRID_JOINTS


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _maxseg(a, w):
    if not (w and w[1] > w[0]):
        return np.nan
    s = a[w[0]:w[1]]; s = s[np.isfinite(s)]
    return float(np.max(s)) if len(s) else np.nan


def _resample_1d(y, n):
    y = np.asarray(y, float)
    if len(y) < 2 or n < 2:
        return np.full(n, np.nan)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y)


def _kin_lookup(part):
    cdf = pd.read_pickle(AUTOMQ / automq_part(part) / "combined_data_with_kinematics.pkl")
    out = {}
    for fk in cdf.index:
        try:
            out[(int(fk[1]), fk[2])] = cdf.loc[fk, "kinematics"]
        except Exception:
            pass
    return out


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    A_gt, A_full, A_armex, A_shuf = [], [], [], []      # check A + control
    B_framecorr = []                                    # check B per-trial series r
    z_by_part = {}                                      # cache one vertical per participant (for shuffle)
    kin_cache = {}
    trials = list(GT.load_clean(need_reproj=False))
    for t in trials:
        m = pat.search(t["trial"])
        if not m:
            continue
        rec = amq.get((automq_part(t["part"]), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        g = rec.get("max_shoulder_flexion")
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None or g is None or not np.isfinite(g):
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        omc = H._load_omc(t["part"], t["trial"], n)
        sh, el = pose[f"{side}_shoulder"], pose[f"{side}_elbow"]
        upper = sh - el                                  # shoulder angle uses ONLY shoulder & elbow
        # FULL fit (arm included) -- the original vert_omcZ (has elbow+wrist)
        FIT_full = [f"{side}_shoulder", f"{other}_shoulder", f"{side}_elbow", f"{side}_wrist", "nose"]
        # ELBOW-EXCLUDED fit -- the acting elbow (in the arm vector) is removed. Keep the far shoulder,
        # nose, and BOTH hips: a torso/head constellation that does NOT contain the scored elbow. (The
        # acting shoulder is unavoidable -- it's the vertex -- but on its own it can't fix a rotation.)
        FIT_armex = [f"{other}_shoulder", "nose", "right_hip", "left_hip"]
        def fit_z(joints):
            Aa = np.vstack([omc[j] for j in joints]); Bb = np.vstack([t["mmc"][:, GRID.index(j)] for j in joints])
            fin = np.isfinite(Aa).all(1) & np.isfinite(Bb).all(1)
            if fin.sum() < 30:
                return None
            Rm, _, _ = H._kabsch(Aa[fin], Bb[fin])
            return Rm @ np.array([0.0, 0.0, 1.0])
        z_full = fit_z(FIT_full); z_armex = fit_z(FIT_armex)
        if z_full is None or z_armex is None:
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        A_gt.append(g)
        A_full.append(_maxseg(H._lp(_ang(upper, z_full[None, :])), rd))
        A_armex.append(_maxseg(H._lp(_ang(upper, z_armex[None, :])), rd))
        z_by_part.setdefault(t["part"], []).append(z_armex)
        A_shuf.append((t["part"], upper, rd))            # resolve shuffle after we have all verticals

        # check B: our arm-vs-vert SERIES vs OMC's OWN shoulder_flexion series
        if t["part"] not in kin_cache:
            kin_cache[t["part"]] = _kin_lookup(t["part"])
        kin = kin_cache[t["part"]].get((int(m.group(1)), m.group(2)))
        if kin is not None and "shoulder_flexion" in kin.columns and rd and rd[1] > rd[0]:
            ours_series = H._lp(_ang(upper, z_armex[None, :]))
            omc_series = _resample_1d(kin["shoulder_flexion"].to_numpy(), n)
            a = ours_series[rd[0]:rd[1]]; b = omc_series[rd[0]:rd[1]]
            mm = np.isfinite(a) & np.isfinite(b)
            if mm.sum() >= 20 and np.std(a[mm]) > 1e-6 and np.std(b[mm]) > 1e-6:
                B_framecorr.append(np.corrcoef(a[mm], b[mm])[0, 1])

    # resolve the SHUFFLE control: give each trial a DIFFERENT participant's vertical
    part_z = {p: np.median(np.stack(v), 0) for p, v in z_by_part.items()}
    parts = list(part_z)
    shuf_vals = []
    rng = np.random.default_rng(0)
    for (p, upper, rd) in A_shuf:
        wrongp = rng.choice([q for q in parts if q != p]) if len(parts) > 1 else p
        shuf_vals.append(_maxseg(H._lp(_ang(upper, part_z[wrongp][None, :])), rd))

    A_gt = np.array(A_gt)
    def rs(x):
        x = np.array(x); m = np.isfinite(A_gt) & np.isfinite(x)
        return spearmanr(A_gt[m], x[m]).correlation, np.median(np.abs(x[m] - A_gt[m]))

    print("\n=== CIRCULARITY CHECKS (vs AutoMQ max_shoulder_flexion) ===")
    for lab, x in [("A. full fit (arm included)", A_full),
                   ("A. ELBOW-EXCLUDED fit (far-sh+nose+hips)", A_armex),
                   ("C. SHUFFLED vertical (wrong participant)", shuf_vals)]:
        r, e = rs(x)
        print(f"  {lab:44} rs={r:+.2f}  med|err|={e:.1f}", flush=True)
    bc = np.array(B_framecorr)
    print(f"\n  B. per-frame series corr (our arm-vs-vert vs OMC's OWN flexion series):")
    print(f"     median per-trial Pearson r = {np.median(bc):+.2f}   (n={len(bc)} trials, "
          f"IQR [{np.percentile(bc,25):+.2f}, {np.percentile(bc,75):+.2f}])")
    print("\nINTERPRETATION: if ARM-EXCLUDED stays high AND shuffle COLLAPSES AND series-corr is high,")
    print("the agreement is REAL (our arm genuinely swings against gravity like OMC's), not circular.")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
