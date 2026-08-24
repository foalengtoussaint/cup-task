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
    # FRAME-INVARIANT trunk geometry: everything below is computed INSIDE one modality (a length,
    # or an angle between two of its own vectors), so MMC and OMC are comparable without alignment.
    def trunk(P):
        shmid = (P["right_shoulder"] + P["left_shoulder"]) / 2.0
        hipmid = (P["right_hip"] + P["left_hip"]) / 2.0
        d = np.linalg.norm(shmid - hipmid, axis=1)                 # trunk LENGTH (invariant)
        down = hipmid - shmid
        sline = P["right_shoulder"] - P["left_shoulder"]
        nd = down/(np.linalg.norm(down,axis=1,keepdims=True)+1e-9)
        ns = sline/(np.linalg.norm(sline,axis=1,keepdims=True)+1e-9)
        perp = np.degrees(np.arccos(np.clip((nd*ns).sum(1),-1,1)))  # down vs shoulder-line: ~90 deg
        sw = np.linalg.norm(P["right_shoulder"]-P["left_shoulder"], axis=1)   # shoulder WIDTH
        f = lambda v: float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan
        g = lambda v: float(np.std(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan
        return f(d), g(d), f(perp), f(sw)
    tm, tms, pm, swm = trunk(pose)
    to_, tos, po_, swo = trunk(po)
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                side=side, trunk_mmc=tm, trunk_omc=to_, trunk_sd_mmc=tms, trunk_sd_omc=tos,
                perp_mmc=pm, perp_omc=po_, shw_mmc=swm, shw_omc=swo)

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
    g = df.groupby("part").agg(n=("trunk_mmc","size"),
        trunkLEN_MMC=("trunk_mmc","median"), trunkLEN_OMC=("trunk_omc","median"),
        trunkSD_MMC=("trunk_sd_mmc","median"), trunkSD_OMC=("trunk_sd_omc","median"),
        downVSshoulder_MMC=("perp_mmc","median"), downVSshoulder_OMC=("perp_omc","median"),
        shoulderW_MMC=("shw_mmc","median"), shoulderW_OMC=("shw_omc","median")).round(1)
    g["trunkLEN_DIFF"] = (g.trunkLEN_MMC - g.trunkLEN_OMC).round(1)
    g["perp_DIFF"] = (g.downVSshoulder_MMC - g.downVSshoulder_OMC).round(1)
    print("ALL FRAME-INVARIANT (lengths + within-modality angles). down-vs-shoulderline should be ~90.\n")
    print(g.to_string())

    df.to_csv(ROOT/"out/scoring/trunk_geometry.csv", index=False)
    print("\nwrote out/scoring/trunk_geometry.csv")
