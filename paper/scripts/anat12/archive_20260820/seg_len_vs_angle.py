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
    # RIGID-SEGMENT test: |shoulder-elbow| must be CONSTANT. If it varies WITH the flexion angle,
    # that modality's shoulder endpoint is moving relative to the bone (soft-tissue artefact on the
    # OMC side, or keypoint degradation on ours). Restricted to the reach->drink window.
    reach2, drink2 = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach2[0], drink2[1]) if (reach2 and drink2) else reach2
    if not w: return None
    def lenser(P, sd_):
        return np.linalg.norm(P[f"{sd_}_elbow"] - P[f"{sd_}_shoulder"], axis=1)
    Lm, Lo = lenser(pose, side)[w[0]:w[1]], lenser(po, side)[w[0]:w[1]]
    A_m, A_o = fm[w[0]:w[1]], fo[w[0]:w[1]]
    def cv(v):
        v = v[np.isfinite(v)]
        return float(np.std(v)/np.mean(v)*100) if len(v)>10 and np.mean(v)>1 else np.nan
    def corr(a, b):
        k = np.isfinite(a)&np.isfinite(b)
        if k.sum()<20 or np.std(a[k])<1e-9 or np.std(b[k])<1e-9: return np.nan
        return float(np.corrcoef(a[k],b[k])[0,1])
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                side=side, rest_mmc=med(fm, r0, r1), rest_omc=med(fo, r0, r1),
                ua_mmc=seglen(pose, side), ua_omc=seglen(po, side),
                cv_mmc=cv(Lm), cv_omc=cv(Lo),
                len_vs_angle_mmc=corr(Lm, A_m), len_vs_angle_omc=corr(Lo, A_o))

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
    g = df.groupby(["part","arm"]).agg(n=("rest_mmc","size"),
        cv_MMC=("cv_mmc","median"), cv_OMC=("cv_omc","median"),
        lenVang_MMC=("len_vs_angle_mmc","median"), lenVang_OMC=("len_vs_angle_omc","median")).round(2)
    print("cv_* = % variation of |shoulder-elbow| (0 = perfectly rigid)")
    print("lenVang_* = corr(segment length, flexion angle): |r| high => the endpoint slides WITH elevation\n")
    print(g.to_string())
    print("\n(skipped L-R asymmetry here; see rest_and_segments.csv)")
    if False:
        print("")
    for p, h in df.groupby("part"):
        mm = h.groupby("side").ua_mmc.median(); oo = h.groupby("side").ua_omc.median()
        if len(mm)==2 and len(oo)==2:
            print(f"  {p:6} MMC L-R {abs(mm.get('left',np.nan)-mm.get('right',np.nan)):5.1f} mm   "
                  f"OMC L-R {abs(oo.get('left',np.nan)-oo.get('right',np.nan)):5.1f} mm")
    df.to_csv(ROOT/"out/scoring/seg_len_vs_angle.csv", index=False)
    print("\nwrote out/scoring/seg_len_vs_angle.csv")
