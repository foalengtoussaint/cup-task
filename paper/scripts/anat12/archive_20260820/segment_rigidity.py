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
    # RIGIDITY of every segment the flexion angle depends on. All FRAME-INVARIANT (lengths within
    # one modality). A rigid body keeps these constant: cv ~ 0 and no correlation with the angle.
    # TRUNK (shoulder_mid->hip_mid) is the one flexion's down-axis rests on and elbow angle ignores.
    reach2, drink2 = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach2[0], drink2[1]) if (reach2 and drink2) else reach2
    if not w: return None
    s_, e_ = w
    def L(P, a, b):
        return np.linalg.norm(P[a]-P[b], axis=1)[s_:e_]
    segs = {
        "trunk":    lambda P: np.linalg.norm((P["right_shoulder"]+P["left_shoulder"])/2.0
                                             - (P["right_hip"]+P["left_hip"])/2.0, axis=1)[s_:e_],
        "shoulderW":lambda P: L(P, "right_shoulder", "left_shoulder"),
        "hipW":     lambda P: L(P, "right_hip", "left_hip"),
        "upperarm": lambda P: L(P, f"{side}_shoulder", f"{side}_elbow"),
        "forearm":  lambda P: L(P, f"{side}_elbow", f"{side}_wrist"),
    }
    out = dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"), side=side)
    A_m, A_o = fm[s_:e_], fo[s_:e_]
    def cv(v):
        v = v[np.isfinite(v)]
        return float(np.std(v)/np.mean(v)*100) if len(v)>10 and np.mean(v)>1 else np.nan
    def cr(v, a):
        k = np.isfinite(v)&np.isfinite(a)
        if k.sum()<20 or np.std(v[k])<1e-9 or np.std(a[k])<1e-9: return np.nan
        return float(np.corrcoef(v[k],a[k])[0,1])
    for nm, fn in segs.items():
        try:
            vm, vo = fn(pose), fn(po)
        except KeyError:
            continue
        out[f"cv_{nm}_mmc"] = cv(vm); out[f"cv_{nm}_omc"] = cv(vo)
        out[f"r_{nm}_mmc"] = cr(vm, A_m); out[f"r_{nm}_omc"] = cr(vo, A_o)
    return out

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
    SEGS = ["trunk","shoulderW","hipW","upperarm","forearm"]
    print("cv% = how much the segment LENGTH moves (0 = rigid). r = corr(length, flexion angle).")
    print("All frame-invariant. TRUNK/shoulderW/hipW build the down-axis that flexion uses and elbow ignores.\n")
    for nm in SEGS:
        cols = {f"cv_{nm}_mmc":"cv_MMC", f"cv_{nm}_omc":"cv_OMC", f"r_{nm}_mmc":"r_MMC", f"r_{nm}_omc":"r_OMC"}
        have = [c for c in cols if c in df.columns]
        if not have: continue
        g = df.groupby(["part","arm"])[have].median().round(2).rename(columns=cols)
        print(f"--- {nm.upper()}"); print(g.to_string()); print()

    df.to_csv(ROOT/"out/scoring/segment_rigidity.csv", index=False)
    print("\nwrote out/scoring/segment_rigidity.csv")
