"""Is the P12/P15/P17 flexion offset a CONSTANT BIAS in the per-frame series, or a gain/shape error?

Per trial, over the reach->drink window, with PLANAR angles on BOTH sides:
  bias  = median(MMC - OMC)            constant component
  sd    = sd(MMC - OMC)                how constant it is (small => pure offset)
  corr  = Pearson(MMC, OMC)            does the shape track at all
  slope = OLS slope of MMC on OMC      1.0 => pure offset; <1 => we compress the range
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
from score_vs_automq import (_pose_variant_cached, _planar_body_angles, load_automq,
                             automq_part, automq_phases_to_video, _win)
import re
H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
GRID = R._GRID_JOINTS
JN = ["right_shoulder","left_shoulder","right_elbow","left_elbow","right_wrist","left_wrist",
      "right_hip","left_hip","nose"]
pat = re.compile(r"trial_(\d+)_([RL])_")
rows=[]
PARTS = {"P12","P15","P17","P07","P13"}   # the 3 suspects + 2 clean controls (was: all 11 = 8min of I/O)
for t in GT.load_clean(need_reproj=False):
    if t["part"] not in PARTS: continue
    p, tr, side = t["part"], t["trial"], t["side"]
    m = pat.search(tr)
    if not m: continue
    rec = amq.get((automq_part(p), int(m.group(1)), m.group(2)))
    if rec is None or rec.get("phases") is None: continue
    n = t["mmc"].shape[0]
    pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
    if pose is None: continue
    omc = H._load_omc(p, tr, n); wr=f"{side}_wrist"
    if wr not in omc or not np.isfinite(omc[wr]).any() or not all(j in omc for j in JN): continue
    lag,_ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
    pose_omc = {j: R._shift(omc[j], lag) for j in JN}
    other = "right" if side=="left" else "left"
    try:
        f_mmc,_ = _planar_body_angles(pose, side, other)
        f_omc,_ = _planar_body_angles(pose_omc, side, other)
    except Exception: continue
    ph = automq_phases_to_video(rec["phases"], lag, n)
    if not ph: continue
    reach, drink = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach[0], drink[1]) if (reach and drink) else reach
    if not w: continue
    a, b = f_mmc[w[0]:w[1]], f_omc[w[0]:w[1]]
    k = min(len(a), len(b)); a, b = a[:k], b[:k]
    ok = np.isfinite(a)&np.isfinite(b)
    if ok.sum()<20 or np.std(b[ok])<1e-6: continue
    d = a[ok]-b[ok]
    slope = np.polyfit(b[ok], a[ok], 1)[0]
    rows.append(dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                     side=side, bias=float(np.median(d)), sd=float(np.std(d)),
                     corr=float(np.corrcoef(a[ok],b[ok])[0,1]), slope=float(slope)))
df = pd.DataFrame(rows)
g = df.groupby(["part","arm"]).agg(n=("bias","size"), bias=("bias","median"), sd=("sd","median"),
                                   corr=("corr","median"), slope=("slope","median")).round(2)
print(g.to_string())
df.to_csv(ROOT/"out/scoring/flex_series_bias.csv", index=False)
print("\nwrote out/scoring/flex_series_bias.csv")
