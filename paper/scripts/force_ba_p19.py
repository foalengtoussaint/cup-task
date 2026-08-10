"""P19 gate-relaxed: force triangulation (no reproj gate), optionally BA-refine, compute interjoint,
AND flag per trial which arm-joint-frames would NOT have passed the normal gate (n_inliers<MIN_CAMS at
REPROJ_PX=30) -- so the utility of the gate is visible.

For each P19 trial:
  - forced pose: plain DLT on good cams, no reproj rejection (min 2 cams)
  - BA pose:     refine_trial_ba on the forced init (Huber reproj + bone prior) with the trial guard
  - gate-fail:   per arm joint (shoulder/elbow/wrist), fraction of frames with <3 cams within 30px
  - interjoint (forced) + interjoint (BA) vs AutoMQ stored
    python paper/scripts/force_ba_p19.py
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
import gnn_train as GT
from compare_pose_omc_delta import _lp, _despike, JOINTS
from cup_task import triangulate as TR
from score_vs_automq import load_automq, automq_phases_to_video, automq_part, _win
import ba_refine
JLIST = list(JOINTS)
ARM = lambda side: [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]


def _percam(part, trial):
    d = H.DELTA / part; cams = H._load_calib_mm(part); per_cam = {}
    for pj in sorted(glob.glob(str(d / H.DETS_SUBDIR / f"*{trial}*.pose.json"))):
        cam = Path(pj).name.split(".")[1]; per_cam[f"cam_{cam}"] = json.loads(Path(pj).read_text())["frames"]
    if H.GOOD_CAMS and part in H.GOOD_CAMS:
        keep = H.GOOD_CAMS[part]
        per_cam = {c: v for c, v in per_cam.items() if c in keep}; cams = {c: v for c, v in cams.items() if c in keep}
    return per_cam, cams


def _force_pose(per_cam, cams, n):
    """plain DLT, no gate; also return per-joint frac of frames that would FAIL the normal gate."""
    out = {}; gatefail = {}
    for joint in JLIST:
        pf = H._kp_point(joint); X = np.full((n, 3), np.nan); fail = 0; tot = 0
        for f in range(n):
            uc, up = [], []
            for ck, frames in per_cam.items():
                if f < len(frames):
                    p = pf(frames[f])
                    if p is not None:
                        uc.append(cams[ck]); up.append(p)
            if len(uc) >= 2:
                X[f] = TR.triangulate_dlt(uc, up)
                tot += 1
                # would the NORMAL gate keep this frame? (>=MIN_CAMS within REPROJ_PX)
                errs = [np.linalg.norm(TR.project(c, X[f])[0] - p) for c, p in zip(uc, up)]
                if sum(e <= TR.REPROJ_PX for e in errs) < TR.MIN_CAMS:
                    fail += 1
        out[joint] = _despike(X)
        gatefail[joint] = fail / tot if tot else np.nan
    return out, gatefail


def _flex(pose, side, other):
    sh, el = pose[f"{side}_shoulder"], pose[f"{side}_elbow"]; arm = el - sh
    down = (pose["right_hip"] + pose["left_hip"]) / 2 - (sh + pose[f"{other}_shoulder"]) / 2
    dn = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9); f = np.isfinite(dn).all(1)
    dc = np.nanmedian(dn[f], 0) if f.any() else np.array([0, 0, -1.0]); dc = dc / (np.linalg.norm(dc) + 1e-9)
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
    reproj = {t["trial"]: t for t in GT.load_clean(parts=["P19"], need_reproj=True)}
    rows = []
    for t in GT.load_clean(parts=["P19"]):
        m = pat.search(t["trial"])
        if not m:
            continue
        rec = amq.get((automq_part("P19"), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        per_cam, cams = _percam("P19", t["trial"]); n = max(len(v) for v in per_cam.values())
        pose, gf = _force_pose(per_cam, cams, n)
        # BA on forced init (if reproj sidecar available)
        ba_pose = None
        rt = reproj.get(t["trial"])
        if rt is not None:
            tt = dict(rt); tt["mmc"] = np.stack([pose[j] for j in JLIST], 1).astype(np.float32)
            tt["valid"] = np.isfinite(tt["mmc"]).all(2)
            try:
                Xba, _ = ba_refine.refine_trial_ba(tt, lam_bone=0.0, huber_px=20.0, iters=40, trial_guard_mm=150.0)
                ba_pose = {j: Xba[:, i] for i, j in enumerate(JLIST)}
            except Exception as e:
                ba_pose = None
        omc = H._load_omc("P19", t["trial"], n)
        lag, _ = H._find_lag(pose[f"{side}_wrist"], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach = _win(ph, "reaching")
        ij_f = _corr(_flex(pose, side, other), _elbow(pose, side), reach)
        ij_b = _corr(_flex(ba_pose, side, other), _elbow(ba_pose, side), reach) if ba_pose else np.nan
        arm_gatefail = float(np.mean([gf[j] for j in ARM(side)]))
        rows.append({"trial": t["trial"], "side": side, "arm_gatefail_frac": arm_gatefail,
                     "ij_forced": ij_f, "ij_ba": ij_b, "automq_ij": rec.get("interjoint_coordination")})
    d = pd.DataFrame(rows)
    d.to_csv(PAPER / "force_ba_p19.csv", index=False)
    print(f"P19 forced+BA interjoint  n={len(d)}\n")
    print("per-trial (sorted by AutoMQ ij; * = would FAIL normal gate on the acting arm):")
    print(f"{'trial':22}{'arm_gatefail':>13}{'forced_ij':>11}{'ba_ij':>9}{'automq_ij':>11}")
    for _, r in d.sort_values("automq_ij").iterrows():
        star = " *" if r.arm_gatefail_frac > 0.5 else ""
        print(f"  {r.trial:20}{r.arm_gatefail_frac:>13.2f}{r.ij_forced:>11.2f}{r.ij_ba:>9.2f}{r.automq_ij:>11.2f}{star}")
    dd = d.dropna(subset=["automq_ij"])
    for col, lab in [("ij_forced", "forced"), ("ij_ba", "BA")]:
        s = dd.dropna(subset=[col])
        if len(s) >= 5:
            print(f"\n  {lab:7} vs AutoMQ:  rs={spearmanr(s[col], s.automq_ij).correlation:+.2f}  "
                  f"median|err|={np.median(np.abs(s[col]-s.automq_ij)):.2f}  n={len(s)}")
    print(f"\n  trials that would FAIL the normal gate on the acting arm (>50% frames): "
          f"{(d.arm_gatefail_frac>0.5).sum()}/{len(d)}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
