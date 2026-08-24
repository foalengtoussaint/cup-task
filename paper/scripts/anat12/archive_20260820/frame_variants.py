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
    # THREE body-frame variants, computed the SAME way on both sides (matched):
    #   frozen   = shipped _planar_body_angles: down/fwd/lat all collapsed to per-trial medians
    #   hipfroz  = ONLY hip_mid frozen; the SHOULDERS stay live, so trunk rotation/lean/scapular
    #              motion moves the frame as it physically does
    #   perframe = nothing frozen
    # The freeze exists because our per-frame HIPS are jitter -- but it also freezes the SHOULDERS,
    # which genuinely move. If the freeze is the problem, hipfroz/perframe should shrink the bias.
    def planar(P, variant):
        nn = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
        sh, el = P[f"{side}_shoulder"], P[f"{side}_elbow"]
        shO = P[f"{other}_shoulder"]
        arm = el - sh
        hip = (P["right_hip"] + P["left_hip"]) / 2.0
        if variant == "hipfroz":
            hf = np.isfinite(hip).all(1)
            hip = np.broadcast_to(np.nanmedian(hip[hf], 0) if hf.any() else hip[0], hip.shape)
        down = nn(hip - (sh + shO)/2.0)
        sline = nn(sh - shO)
        fwd = nn(np.cross(sline, down)); lat = nn(np.cross(down, fwd))
        if variant == "frozen":
            f_ = np.isfinite(down).all(1) & np.isfinite(fwd).all(1)
            if f_.any():
                dc, fc, lc = nn(np.nanmedian(down[f_],0)), nn(np.nanmedian(fwd[f_],0)), nn(np.nanmedian(lat[f_],0))
                down = np.broadcast_to(dc, arm.shape); fwd = np.broadcast_to(fc, arm.shape); lat = np.broadcast_to(lc, arm.shape)
        arm_sag = arm - (arm*lat).sum(1, keepdims=True)*lat
        return H._lp(np.degrees(np.arccos(np.clip((nn(arm_sag)*down).sum(1), -1, 1))))
    reach2, drink2 = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach2[0], drink2[1]) if (reach2 and drink2) else reach2
    if not w: return None
    out = dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"), side=side)
    for v in ("frozen","hipfroz","perframe"):
        try:
            a, b = planar(pose, v)[w[0]:w[1]], planar(po, v)[w[0]:w[1]]
        except Exception:
            continue
        k = min(len(a), len(b)); a, b = a[:k], b[:k]
        ok = np.isfinite(a)&np.isfinite(b)
        if ok.sum()<20 or np.std(b[ok])<1e-6: continue
        out[f"bias_{v}"] = float(np.median(a[ok]-b[ok]))
        out[f"corr_{v}"] = float(np.corrcoef(a[ok],b[ok])[0,1])
        out[f"slope_{v}"] = float(np.polyfit(b[ok], a[ok], 1)[0])
        out[f"max_mmc_{v}"] = float(np.nanmax(a[ok])); out[f"max_omc_{v}"] = float(np.nanmax(b[ok]))
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
    cols = [c for v in ("frozen","hipfroz","perframe") for c in (f"bias_{v}", f"slope_{v}", f"corr_{v}")]
    g = df.groupby(["part","arm"])[cols].median().round(2)
    print("Per-frame BIAS / SLOPE / CORR under three body-frame variants (matched both sides).")
    print("frozen = shipped (down/fwd/lat all per-trial medians); hipfroz = only hips frozen; perframe = none.\n")
    print(g.to_string())
    from scipy.stats import spearmanr as _sp
    print("\nPOOLED r_s of the max-flexion measure, per variant:")
    for v in ("frozen","hipfroz","perframe"):
        h = df.dropna(subset=[f"max_mmc_{v}", f"max_omc_{v}"])
        print(f"  {v:9} r_s {_sp(h[f'max_omc_{v}'], h[f'max_mmc_{v}']).correlation:.3f}   "
              f"|bias| median {h[f'bias_{v}'].abs().median():.1f} deg   n={len(h)}")

    df.to_csv(ROOT/"out/scoring/frame_variants.csv", index=False)
    print("\nwrote out/scoring/frame_variants.csv")
