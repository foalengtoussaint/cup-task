"""Does camera COVERAGE (angular spread) matter, holding camera COUNT fixed?

Reuses the camera-drop subsets (paper/augment_camdrop.csv): at a FIXED k, different k-camera
subsets of the same participant have different coverage. Correlate subset coverage vs its pose
error -- WITHIN participant, WITHIN k -- so count is held constant and only geometry varies.

coverage(subset) = mean pairwise angle (deg) between the kept cameras' view directions to the
subject centroid (camera centres from the reproj sidecar; centroid = median OMC over the trial set).
Data only.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
PAIRS = REPO / "cache/delta/gnn_pairs"
cam = pd.read_csv(REPO / "paper/augment_camdrop.csv")
cam["subset"] = cam["subset"].astype(str)


def centres_centroid(part):
    fs = sorted(f for f in glob.glob(str(PAIRS / part / "*.npz")) if not f.endswith(".reproj.npz"))
    rj = np.load(fs[0].replace(".npz", ".reproj.npz"), allow_pickle=True)
    R, t = rj["R"], rj["t"]
    C = np.stack([-R[c].T @ t[c] for c in range(len(R))])
    ctrs = []
    for f in fs[:20]:
        d = np.load(f, allow_pickle=True)
        omc, val = d["omc"], d["valid"]
        ctrs.append(np.nanmedian(np.where(val[..., None], omc, np.nan).reshape(-1, 3), axis=0))
    return C, np.nanmedian(np.stack(ctrs), axis=0)


def cov(C, ctr, idx):
    dirs = ctr[None] - C[idx]
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    ang = [np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1)))
           for i in range(len(idx)) for j in range(i + 1, len(idx))]
    return float(np.mean(ang)) if ang else np.nan


rows = []
for part, g in cam.groupby("part"):
    C, ctr = centres_centroid(part)
    for _, r in g.iterrows():
        idx = list(map(int, r["subset"]))
        rows.append((part, r["k"], r["subset"], cov(C, ctr, idx), r["err_wr"], r["err_el"], r["err_sh"]))
d = pd.DataFrame(rows, columns=["part", "k", "subset", "cov", "err_wr", "err_el", "err_sh"])

print("== coverage vs wrist error, WITHIN participant & FIXED k (Spearman r_s) ==")
print("positive r_s = wider coverage -> MORE error; negative = wider coverage -> LESS error\n")
print(f"{'part':<6}" + "".join(f"{f'k={k}':>12}" for k in range(2, 6)))
for part, gp in d.groupby("part"):
    cells = []
    for k in range(2, 6):
        gg = gp[gp.k == k].dropna(subset=["cov", "err_wr"])
        # collapse to per-subset medians so each subset weighs once
        m = gg.groupby("subset").agg(cov=("cov", "first"), err=("err_wr", "median")).reset_index()
        if len(m) > 3 and m["cov"].std() > 1e-6:
            r = spearmanr(m["cov"], m["err"]).correlation
            cells.append(f"{r:>7.2f}(n{len(m)})")
        else:
            cells.append(f"{'--':>12}")
    print(f"{part:<6}" + "".join(f"{c:>12}" for c in cells))

# pooled per k, mean-centred within (part,k) to remove participant offsets
print("\n== pooled (residualised within part,k) coverage vs error ==")
for k in range(2, 6):
    gg = d[d.k == k].dropna(subset=["cov", "err_wr"]).copy()
    if len(gg) < 10:
        continue
    gg["covz"] = gg.groupby("part")["cov"].transform(lambda x: x - x.mean())
    gg["errz"] = gg.groupby("part")["err_wr"].transform(lambda x: x - x.mean())
    r = spearmanr(gg["covz"], gg["errz"]).correlation
    print(f"k={k}: r_s(cov, wrist_err) = {r:+.2f}  (n={len(gg)} subsets, {gg.part.nunique()} parts)")

d.to_csv(REPO / "paper/seg_coverage.csv", index=False)
print(f"\nwrote {REPO/'paper/seg_coverage.csv'}\nDONE")
