"""Relax the reprojection gate for EVERYONE and see if it changes the results.

For every cohort trial: compare measures computed on the GATED pose (t['mmc'], the shipped robust-consensus
DLT with its reproj gate) vs a FORCED pose (plain DLT on good cams, NO gate, min 2 cams). Same processing
level (raw DLT + the same _lp smoothing + same measure operators) so the ONLY difference is the gate.

Reports, per measure (flexion/abduction/elbow/peak_velocity/interjoint): rs vs AutoMQ on the COMMON
trials (both give a value) AND on ALL-forced trials (incl. newly-added), + how many trials the relaxation
ADDS (gated gave NaN -> forced gives a value), and which participants they come from.
    python paper/scripts/relax_gate_all.py    (BACKGROUND -- ~690 trials re-triangulated)
"""
from __future__ import annotations
import sys, re, glob, json, time
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
from compare_pose_omc_delta import _lp, _despike, JOINTS
from pipeline import triangulate as TR
from score_vs_automq import load_automq, automq_phases_to_video, automq_part, _win
from planar_body_angles import planar_angles          # NEW DEFAULT: 4-point body-frame sagittal/frontal
JLIST = list(JOINTS)


def _forced(part, trial):
    d = H.DELTA / part; cams = H._load_calib_mm(part); pc = {}
    for pj in sorted(glob.glob(str(d / H.DETS_SUBDIR / f"*{trial}*.pose.json"))):
        cam = Path(pj).name.split(".")[1]; pc[f"cam_{cam}"] = json.loads(Path(pj).read_text())["frames"]
    if H.GOOD_CAMS and part in H.GOOD_CAMS:
        keep = H.GOOD_CAMS[part]; pc = {c: v for c, v in pc.items() if c in keep}; cams = {c: v for c, v in cams.items() if c in keep}
    if not pc:
        return None
    n = max(len(v) for v in pc.values())
    out = {}
    for j in JLIST:
        pf = H._kp_point(j); X = np.full((n, 3), np.nan)
        for f in range(n):
            uc, up = [], []
            for ck, frames in pc.items():
                if f < len(frames):
                    p = pf(frames[f])
                    if p is not None:
                        uc.append(cams[ck]); up.append(p)
            if len(uc) >= 2:
                X[f] = TR.triangulate_dlt(uc, up)
        out[j] = _despike(X)
    return out, n


def _posed(pose_arr):
    """dict-view of a (T,J,3) array by joint name."""
    return {j: pose_arr[:, i] for i, j in enumerate(JLIST)}


FPS = 60.0


def _measures(pose, side, other, ph):
    from scipy.signal import find_peaks
    reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
    ret = _win(ph, "returning"); fwd = _win(ph, "forward_transport"); bwd = _win(ph, "back_transport")
    rd = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
    sh, el, wr = pose[f"{side}_shoulder"], pose[f"{side}_elbow"], pose[f"{side}_wrist"]
    # NEW DEFAULT: planar 4-point body-frame flexion (sagittal) + abduction (frontal); freeze the
    # frame per-trial for the (possibly occluded) hips, as the MMC pipeline requires. interjoint below
    # then uses THIS flexion. (Old total/lateral definitions retired.)
    try:
        flex, abd = planar_angles(sh, el, pose[f"{other}_shoulder"], pose["right_hip"], pose["left_hip"], True)
    except (IndexError, ValueError):
        flex = abd = np.full(len(sh), np.nan)
    u, v = sh - el, wr - el
    elb = _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))
    eav = np.abs(np.gradient(elb)) * FPS                                   # elbow angular velocity
    sp = _lp(H._speed(wr))
    sh_mid = (sh + pose[f"{other}_shoulder"]) / 2.0                        # trunk displacement ref

    def mx(a, w):
        if not (w and w[1] > w[0]):
            return np.nan
        s = a[w[0]:w[1]]; s = s[np.isfinite(s)]
        return float(np.max(s)) if len(s) else np.nan

    def ij(w):
        if not (w and w[1] - w[0] >= 12):
            return np.nan
        mg = int(0.1 * (w[1] - w[0])); x, y = flex[w[0]+mg:w[1]-mg], elb[w[0]+mg:w[1]-mg]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10 or np.std(x[m]) < 1e-6 or np.std(y[m]) < 1e-6:
            return np.nan
        return float(np.corrcoef(x[m], y[m])[0, 1])

    # timing (from frame 0, AutoMQ convention)
    ttp = ttfp = np.nan
    if reach and reach[1] > reach[0]:
        seg = sp[reach[0]:reach[1]]
        if np.isfinite(seg).any():
            pk = int(np.nanargmax(seg)); ttp = (reach[0] + pk) / FPS
            pv = float(np.nanmax(seg))
            fp, _ = find_peaks(np.nan_to_num(seg), prominence=max(50.0, 0.15 * pv))
            idx = int(fp[0]) if len(fp) else pk
            ttfp = (reach[0] + idx) / FPS
    # total movement time (phase-derived -> gate-independent)
    tmt = ((ret[1] - reach[0]) / FPS) if (reach and ret) else np.nan
    # movement units over reaching+transports+returning
    munits = np.nan
    span = [w for w in (reach, fwd, bwd, ret) if w]
    if span:
        s0 = min(w[0] for w in span); s1 = max(w[1] for w in span)
        vseg = sp[s0:s1]; vseg = vseg[np.isfinite(vseg)]
        if len(vseg) > 5:
            mins, _ = find_peaks(-vseg); maxs, _ = find_peaks(vseg); c = 0
            for mn in mins:
                later = maxs[maxs > mn]
                if len(later) and (vseg[later[0]] - vseg[mn]) > 60.0 and (later[0] - mn) >= 9:
                    c += 1
            munits = float(c)
    # trunk displacement: p98 of shoulder-mid excursion from rest
    trunk = np.nan
    finm = np.isfinite(sh_mid).all(1)
    if finm.sum() >= 15:
        smf = sh_mid[finm]; trunk = float(np.percentile(np.linalg.norm(smf - np.median(smf[:15], 0), axis=1), 98))
    return {"flex": mx(flex, rd), "abd": mx(abd, rd), "elbow": mx(elb, (0, len(elb))),
            "peakvel": mx(sp, reach), "ij": ij(reach), "eav": mx(eav, reach),
            "ttp": ttp, "ttfp": ttfp, "munits": munits, "tmt": tmt, "trunk": trunk}


def main():
    H.use_good_cams(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([RL])_")
    trials = GT.load_clean(need_reproj=False)
    print(f"{len(trials)} trials; re-triangulating forced (no gate)...", flush=True)
    rows = []; t0 = time.time()
    for i, t in enumerate(trials):
        m = pat.search(t["trial"])
        if not m:
            continue
        p = t["part"]; rec = amq.get((automq_part(p), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"; n = t["mmc"].shape[0]
        omc = H._load_omc(p, t["trial"], n)
        # phase window uses the GATED wrist lag (stable); reused for both
        lag, _ = H._find_lag(t["mmc"][:, JLIST.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        gm = _measures(_posed(t["mmc"]), side, other, ph)
        fr = _forced(p, t["trial"])
        fm = _measures(_posed(np.stack([fr[0][j] for j in JLIST], 1)), side, other, ph) if fr else {k: np.nan for k in gm}
        rows.append({"part": p, "trial": t["trial"],
                     **{f"g_{k}": v for k, v in gm.items()}, **{f"f_{k}": v for k, v in fm.items()},
                     "a_flex": rec.get("max_shoulder_flexion"), "a_abd": rec.get("max_shoulder_abduction"),
                     "a_elbow": rec.get("max_elbow_angle"), "a_peakvel": rec.get("peak_velocity"),
                     "a_ij": rec.get("interjoint_coordination"),
                     "a_eav": rec.get("peak_elbow_angular_velocity"),
                     "a_ttp": rec.get("time_to_peak_velocity"), "a_ttfp": rec.get("time_to_first_peak_velocity"),
                     "a_munits": rec.get("number_of_movement_units"), "a_tmt": rec.get("total_movement_time"),
                     "a_trunk": rec.get("max_trunk_displacement")})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(trials)}  elapsed {time.time()-t0:.0f}s  rows={len(rows)}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(PAPER / "relax_gate_all.csv", index=False)

    amap = {"peakvel": "a_peakvel", "eav": "a_eav", "ttp": "a_ttp", "ttfp": "a_ttfp",
            "munits": "a_munits", "tmt": "a_tmt", "ij": "a_ij", "trunk": "a_trunk",
            "flex": "a_flex", "elbow": "a_elbow", "abd": "a_abd"}
    print(f"\nPROCESSING CHECK: rows={len(d)}", flush=True)
    print(f"\n{'measure':10}{'gated rs (n)':>18}{'forced rs COMMON (n)':>24}{'forced rs ALL (n)':>22}{'+trials':>9}")
    for k, acol in amap.items():
        g, f, a = d[f"g_{k}"], d[f"f_{k}"], d[acol]
        common = np.isfinite(g) & np.isfinite(f) & np.isfinite(a)
        gmask = np.isfinite(g) & np.isfinite(a)
        fmask = np.isfinite(f) & np.isfinite(a)
        added = int((~np.isfinite(g) & np.isfinite(f) & np.isfinite(a)).sum())
        rg = spearmanr(g[gmask], a[gmask]).correlation if gmask.sum() >= 5 else np.nan
        rfc = spearmanr(f[common], a[common]).correlation if common.sum() >= 5 else np.nan
        rfa = spearmanr(f[fmask], a[fmask]).correlation if fmask.sum() >= 5 else np.nan
        print(f"{k:10}{f'{rg:+.2f} ({gmask.sum()})':>18}{f'{rfc:+.2f} ({common.sum()})':>24}{f'{rfa:+.2f} ({fmask.sum()})':>22}{added:>9}")
    # where do the added trials come from
    add_ij = d[~np.isfinite(d.g_ij) & np.isfinite(d.f_ij) & np.isfinite(d.a_ij)]
    print(f"\n  interjoint +trials by participant: {add_ij.part.value_counts().to_dict()}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
