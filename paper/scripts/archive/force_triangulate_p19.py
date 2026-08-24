"""Force P19 triangulation despite the reprojection gate (which nan's right_shoulder/nose: 3 cams
reproject at 43/28/46px > REPROJ_PX=30, so <MIN_CAMS=3 survive). Plain DLT on all good cams, NO gate,
then compute flexion/elbow/interjoint and compare to AutoMQ stored -- is the forced pose usable or junk?

Per P19 trial: triangulate every joint with triangulate_dlt on the good cams (no reproj rejection),
despike, then our shipped flexion (arm vs per-trial-const shoulder->hip) + elbow + interjoint over the
reaching window (AutoMQ phases -> video via wrist-speed lag). Report vs AutoMQ stored interjoint/flexion.
    python paper/scripts/force_triangulate_p19.py
"""
from __future__ import annotations
import sys, re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import compare_pose_omc_delta as H
import results_v3_delta as R
from compare_pose_omc_delta import _lp, _despike, JOINTS
from pipeline import triangulate as TR
from score_vs_automq import load_automq, automq_phases_to_video, automq_part, _win
JLIST = list(JOINTS)


def _force_pose(part, trial):
    """triangulate every joint on good cams with NO reproj gate (min 2 cams)."""
    d = H.DELTA / part
    cams = H._load_calib_mm(part)
    per_cam = {}
    for pj in sorted(glob.glob(str(d / H.DETS_SUBDIR / f"*{trial}*.pose.json"))):
        cam = Path(pj).name.split(".")[1]; per_cam[f"cam_{cam}"] = json.loads(Path(pj).read_text())["frames"]
    if H.GOOD_CAMS and part in H.GOOD_CAMS:
        keep = H.GOOD_CAMS[part]
        per_cam = {c: v for c, v in per_cam.items() if c in keep}; cams = {c: v for c, v in cams.items() if c in keep}
    n = max(len(v) for v in per_cam.values())
    out = {}
    for joint in JLIST:
        pf = H._kp_point(joint)
        X = np.full((n, 3), np.nan)
        for f in range(n):
            uc, up = [], []
            for ck, frames in per_cam.items():
                if f >= len(frames):
                    continue
                p = pf(frames[f])
                if p is not None:
                    uc.append(cams[ck]); up.append(p)
            if len(uc) >= 2:                                  # FORCE: plain DLT, no reproj gate
                X[f] = TR.triangulate_dlt(uc, up)
        out[joint] = _despike(X)
    return out, n


def _flex(pose, side, other):
    sh, el = pose[f"{side}_shoulder"], pose[f"{side}_elbow"]
    arm = el - sh
    down = (pose["right_hip"] + pose["left_hip"]) / 2 - (sh + pose[f"{other}_shoulder"]) / 2
    dn = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    f = np.isfinite(dn).all(1); dc = np.nanmedian(dn[f], 0) if f.any() else np.array([0, 0, -1.0])
    dc = dc / (np.linalg.norm(dc) + 1e-9)
    an = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    return _lp(np.degrees(np.arccos(np.clip((an * dc[None, :]).sum(1), -1, 1))))


def _elbow(pose, side):
    sh, el, wr = pose[f"{side}_shoulder"], pose[f"{side}_elbow"], pose[f"{side}_wrist"]
    u, v = sh - el, wr - el
    return _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))


def _corr(a, b, w):
    if not (w and w[1] - w[0] >= 12):
        return np.nan
    mg = int(0.1 * (w[1] - w[0])); x, y = a[w[0]+mg:w[1]-mg], b[w[0]+mg:w[1]-mg]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.std(x[m]) < 1e-6 or np.std(y[m]) < 1e-6:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def main():
    H.use_good_cams(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    # P19 trials that have an AutoMQ interjoint
    import gnn_train as GT
    trials = [t for t in GT.load_clean(parts=["P19"]) if amq.get((automq_part("P19"), *(lambda m: (int(m.group(1)), m.group(2)))(pat.search(t["trial"])))) is not None] \
        if False else [t for t in GT.load_clean(parts=["P19"])]
    rows = []
    finite_frac = {j: [] for j in JLIST}
    for t in trials:
        m = pat.search(t["trial"])
        if not m:
            continue
        rec = amq.get((automq_part("P19"), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        pose, n = _force_pose("P19", t["trial"])
        for j in JLIST:
            finite_frac[j].append(np.isfinite(pose[j]).all(1).mean())
        omc = H._load_omc("P19", t["trial"], n)
        lag, _ = H._find_lag(pose[f"{side}_wrist"], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach = _win(ph, "reaching")
        flex = _flex(pose, side, other); elb = _elbow(pose, side)
        rows.append({"trial": t["trial"], "side": side,
                     "ij_forced": _corr(flex, elb, reach),
                     "automq_ij": rec.get("interjoint_coordination"),
                     "flex_max": np.nanmax(flex[reach[0]:reach[1]]) if reach else np.nan,
                     "automq_flex": rec.get("max_shoulder_flexion")})
    d = pd.DataFrame(rows)
    d.to_csv(PAPER / "force_triangulate_p19.csv", index=False)
    print("FORCED P19 pose (no reproj gate) -- joint finite fraction now:")
    for j in ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "nose", "right_hip"]:
        print(f"  {j:15} {np.mean(finite_frac[j]):.2f}")
    dd = d.dropna(subset=["ij_forced", "automq_ij"])
    print(f"\nP19 interjoint  n={len(dd)}  (was 0 with the gate)")
    print(f"  forced ij:  median {dd.ij_forced.median():.2f}  min {dd.ij_forced.min():.2f}  frac<0.8 {(dd.ij_forced<0.8).mean():.2f}")
    print(f"  AutoMQ ij:  median {dd.automq_ij.median():.2f}  min {dd.automq_ij.min():.2f}  frac<0.8 {(dd.automq_ij<0.8).mean():.2f}")
    if len(dd) >= 5:
        print(f"  rs(forced, AutoMQ) = {spearmanr(dd.ij_forced, dd.automq_ij).correlation:+.2f}")
        print(f"  median |err| = {np.median(np.abs(dd.ij_forced - dd.automq_ij)):.2f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
