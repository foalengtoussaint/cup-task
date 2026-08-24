"""WHO carries the flexion offset -- OMC marker placement, or our keypoints?

Two frame-invariant tests per trial, MMC vs OMC, both with the PLANAR construction:
  (1) REST flexion: over rest_pre (frames before Reaching starts) the arm hangs -> flexion should be
      ~0 in BOTH modalities. Whichever reads ~20 deg at rest carries a static geometry offset.
  (2) UPPER-ARM LENGTH |shoulder-elbow|: within a person L and R should be near-equal. An asymmetric
      OMC segment where MMC is symmetric => a displaced mocap MARKER, not our keypoint.
Parallel over trials, with progress (both of which the last probe lacked).
"""
import sys, re
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
from score_vs_automq import (_pose_variant_cached, _planar_body_angles, load_automq,
                             automq_part, automq_phases_to_video, _win)
PARTS = {"P12","P15","P17","P07","P13"}
JN = ["right_shoulder","left_shoulder","right_elbow","left_elbow","right_wrist","left_wrist",
      "right_hip","left_hip","nose"]
GRID = R._GRID_JOINTS
pat = re.compile(r"trial_(\d+)_([RL])_")
H.use_good_cams(); _ba = R._ba_traj_cache(); _amq = load_automq()

def one(t):
    p, tr, side = t["part"], t["trial"], t["side"]
    m = pat.search(tr)
    if not m: return None
    rec = _amq.get((automq_part(p), int(m.group(1)), m.group(2)))
    if rec is None or rec.get("phases") is None: return None
    n = t["mmc"].shape[0]
    pose = _pose_variant_cached(t, "BA", "smoothnet", _ba)
    if pose is None: return None
    omc = H._load_omc(p, tr, n); wr = f"{side}_wrist"
    if wr not in omc or not np.isfinite(omc[wr]).any() or not all(j in omc for j in JN): return None
    lag,_ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
    po = {j: R._shift(omc[j], lag) for j in JN}
    other = "right" if side=="left" else "left"
    try:
        fm,_ = _planar_body_angles(pose, side, other)
        fo,_ = _planar_body_angles(po, side, other)
    except Exception: return None
    ph = automq_phases_to_video(rec["phases"], lag, n)
    if not ph: return None
    reach = _win(ph, "reaching")
    if not reach or reach[0] < 10: return None
    r0, r1 = 0, reach[0]                      # rest_pre
    def med(a, s, e):
        v = a[s:e]; v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.nan
    def seglen(P, sd_):
        d = np.linalg.norm(P[f"{sd_}_elbow"] - P[f"{sd_}_shoulder"], axis=1)
        d = d[np.isfinite(d)]
        return float(np.median(d)) if len(d) else np.nan
    # WHICH PLANE of the trunk actually moves during a trial? Frozen frames are only defensible for
    # the components that DON'T move. All within-modality (frame-invariant relative to the trial's
    # own first frame), computed over reach->drink:
    #   yaw   = axial rotation of the shoulder line about the trunk's own down axis (toward the cup)
    #   lean  = sagittal tilt of the down axis
    #   frontal = lateral tilt of the down axis
    reach2, drink2 = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach2[0], drink2[1]) if (reach2 and drink2) else reach2
    if not w: return None
    s_, e_ = w
    def planes(P):
        nn = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
        shm = (P["right_shoulder"]+P["left_shoulder"])/2.0
        hipm = (P["right_hip"]+P["left_hip"])/2.0
        down = nn(hipm - shm)[s_:e_]
        sline = nn(P["right_shoulder"]-P["left_shoulder"])[s_:e_]
        ok = np.isfinite(down).all(1) & np.isfinite(sline).all(1)
        if ok.sum() < 20: return (np.nan,)*3
        d0 = down[ok][0]; s0 = sline[ok][0]
        # yaw: angle of the shoulder line about the down axis, relative to frame 0
        def proj(v, ax):
            return v - (v*ax).sum(-1, keepdims=True)*ax
        p0 = proj(s0, d0); p0 = p0/ (np.linalg.norm(p0)+1e-9)
        pv = proj(sline[ok], d0); pv = pv/(np.linalg.norm(pv,axis=1,keepdims=True)+1e-9)
        yaw = np.degrees(np.arccos(np.clip((pv*p0).sum(1), -1, 1)))
        tilt = np.degrees(np.arccos(np.clip((down[ok]*d0).sum(1), -1, 1)))   # total tilt of down axis
        rng = lambda v: float(np.percentile(v,95)-np.percentile(v,5))
        return rng(yaw), rng(tilt), float(np.nanmax(yaw))
    ym, tm, ymx = planes(pose); yo, to_, yomx = planes(po)
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                side=side, yaw_range_mmc=ym, yaw_range_omc=yo,
                tilt_range_mmc=tm, tilt_range_omc=to_, yaw_max_omc=yomx)

if __name__ == "__main__":
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in PARTS]
    print(f"{len(trials)} trials over {sorted(PARTS)}", flush=True)
    out=[]
    with Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(one, trials, chunksize=4)):
            if r: out.append(r)
            if (i+1) % 40 == 0: print(f"  [{i+1}/{len(trials)}] kept {len(out)}", flush=True)
    df = pd.DataFrame(out)
    print(f"\nPROCESSING CHECK: trials {len(trials)}, kept {len(df)}\n")
    g = df.groupby("part")[["yaw_range_omc","tilt_range_omc","yaw_range_mmc","tilt_range_mmc",
                            "yaw_max_omc"]].median().round(1)
    print("How much the TRUNK moves within a reach (deg, 5-95pct range), per participant.")
    print("yaw = axial rotation about the trunk's own down axis (transverse plane, toward the cup)")
    print("tilt = total tilt of the down axis (sagittal lean + frontal combined)")
    print("A FROZEN body frame assumes both are ~0.\n")
    print(g.to_string())
    print(f"\nCOHORT MEDIAN (OMC):  yaw range {df.yaw_range_omc.median():.1f} deg   "
          f"tilt range {df.tilt_range_omc.median():.1f} deg   max yaw {df.yaw_max_omc.median():.1f} deg")

    df.to_csv(ROOT/"out/scoring/trunk_plane_motion.csv", index=False)
    print("\nwrote out/scoring/trunk_plane_motion.csv")
