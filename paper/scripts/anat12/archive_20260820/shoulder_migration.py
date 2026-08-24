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
    # Does the ACTING SHOULDER migrate during elevation? Frame-invariant: distance from the acting
    # shoulder to (a) the contralateral shoulder and (b) hip_mid, compared REST vs PEAK-FLEXION.
    # The frozen body frame cannot cause a movement-dependent bias, so the live ARM VECTOR must --
    # which means the shoulder or elbow endpoint moving relative to the rest of the body.
    reach2, drink2 = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach2[0], drink2[1]) if (reach2 and drink2) else reach2
    if not w: return None
    def dists(P):
        sh = P[f"{side}_shoulder"]
        d_contra = np.linalg.norm(sh - P[f"{other}_shoulder"], axis=1)
        d_hip = np.linalg.norm(sh - (P["right_hip"]+P["left_hip"])/2.0, axis=1)
        return d_contra, d_hip
    def at(v, s, e):
        x = v[s:e]; x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else np.nan
    outd = {}
    for tag, P, A in (("mmc", pose, fm), ("omc", po, fo)):
        dc, dh = dists(P)
        seg = A[w[0]:w[1]]
        pk = int(np.nanargmax(seg)) + w[0] if np.isfinite(seg).any() else None
        if pk is None: return None
        lo, hi = max(pk-5, 0), pk+5
        outd[f"contra_rest_{tag}"] = at(dc, r0, r1); outd[f"contra_peak_{tag}"] = at(dc, lo, hi)
        outd[f"hip_rest_{tag}"]   = at(dh, r0, r1); outd[f"hip_peak_{tag}"]   = at(dh, lo, hi)
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                side=side, **outd)

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
    for tag in ("mmc","omc"):
        df[f"d_contra_{tag}"] = df[f"contra_peak_{tag}"] - df[f"contra_rest_{tag}"]
        df[f"d_hip_{tag}"]    = df[f"hip_peak_{tag}"] - df[f"hip_rest_{tag}"]
    g = df.groupby(["part","arm"])[["d_contra_mmc","d_contra_omc","d_hip_mmc","d_hip_omc"]].median().round(1)
    g["contra_EXCESS"] = (g.d_contra_mmc - g.d_contra_omc).round(1)
    g["hip_EXCESS"]    = (g.d_hip_mmc - g.d_hip_omc).round(1)
    print("Acting-shoulder MIGRATION rest->peak-flexion (mm). d_* = how far it moved in that modality;")
    print("EXCESS = ours minus mocap's. Large EXCESS => OUR shoulder keypoint slides during elevation.\n")
    print(g.to_string())

    df.to_csv(ROOT/"out/scoring/shoulder_migration.csv", index=False)
    print("\nwrote out/scoring/shoulder_migration.csv")
