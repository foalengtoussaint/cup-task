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
    # Flexion = arm vs the BODY FRAME. Elbow angle uses the same arm vector but NO body frame, and
    # it is accurate for these participants -> suspect the FRAME. Measure the angle between the MMC
    # and OMC frame axes directly (frozen per-trial medians, as _planar_body_angles does).
    def axes(P, sd_, ot_):
        n_ = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
        sh = P[f"{sd_}_shoulder"]; shO = P[f"{ot_}_shoulder"]
        down = n_((P["right_hip"]+P["left_hip"])/2.0 - (sh+shO)/2.0)
        sline = n_(sh - shO)
        f = np.isfinite(down).all(1) & np.isfinite(sline).all(1)
        if not f.any(): return None, None
        return n_(np.nanmedian(down[f],0)), n_(np.nanmedian(sline[f],0))
    dm, sm = axes(pose, side, other); do_, so_ = axes(po, side, other)
    if dm is None or do_ is None: return None
    ang = lambda a,b: float(np.degrees(np.arccos(np.clip(np.dot(a,b), -1, 1))))
    # arm vector agreement (median over the trial) as the control
    am = pose[f"{side}_elbow"]-pose[f"{side}_shoulder"]; ao = po[f"{side}_elbow"]-po[f"{side}_shoulder"]
    am = am/(np.linalg.norm(am,axis=1,keepdims=True)+1e-9); ao = ao/(np.linalg.norm(ao,axis=1,keepdims=True)+1e-9)
    k = np.isfinite(am).all(1)&np.isfinite(ao).all(1)
    arm_ang = float(np.median(np.degrees(np.arccos(np.clip((am[k]*ao[k]).sum(1),-1,1))))) if k.any() else np.nan
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                side=side, down_axis_deg=ang(dm,do_), shoulderline_deg=ang(sm,so_), arm_vec_deg=arm_ang)

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
    g = df.groupby(["part","arm"]).agg(n=("down_axis_deg","size"),
        DOWNaxis_deg=("down_axis_deg","median"), SHOULDERline_deg=("shoulderline_deg","median"),
        ARMvector_deg=("arm_vec_deg","median")).round(1)
    print("MMC-vs-OMC disagreement of each geometric ingredient (deg):")
    print("  DOWNaxis/SHOULDERline = the BODY FRAME (used by flexion, NOT by elbow angle)")
    print("  ARMvector = shared by both -> the CONTROL\n")
    print(g.to_string())

    df.to_csv(ROOT/"out/scoring/frame_axes.csv", index=False)
    print("\nwrote out/scoring/frame_axes.csv")
