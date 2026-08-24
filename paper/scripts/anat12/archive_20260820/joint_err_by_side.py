"""Per-participant per-SIDE 3D error of shoulder/elbow/wrist (MMC vs OMC markers).
If flexion is ~20deg low on one arm, the joints defining it should be displaced on that side."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
from score_vs_automq import _pose_variant_cached
H.use_good_cams(); ba = R._ba_traj_cache()
GRID = R._GRID_JOINTS
rows=[]
for t in GT.load_clean(need_reproj=False):
    p, tr, side = t["part"], t["trial"], t["side"]
    n = t["mmc"].shape[0]
    pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
    if pose is None: continue
    omc = H._load_omc(p, tr, n)
    wr = f"{side}_wrist"
    if wr not in omc or not np.isfinite(omc[wr]).any(): continue
    lag,_ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
    for j in (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"):
        if j not in omc or j not in pose: continue
        o = R._shift(omc[j], lag); m = pose[j]
        k = min(len(o), len(m)); o, m = o[:k], m[:k]
        d = np.linalg.norm(o-m, axis=1)
        d = d[np.isfinite(d)]
        if len(d) < 20: continue
        rows.append(dict(part=p, side=side, joint=j.split("_")[1], med_mm=float(np.median(d))))
df = pd.DataFrame(rows)
piv = df.pivot_table(index="part", columns=["side","joint"], values="med_mm", aggfunc="median").round(1)
print(piv.to_string())
df.to_csv("/home/imove/Documents/cup-task/out/scoring/joint_err_by_side.csv", index=False)
print("\nwrote out/scoring/joint_err_by_side.csv")
