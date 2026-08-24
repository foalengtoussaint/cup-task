"""Audit the DLT 30px gate PER TRIAL: is the relaxed path a per-trial fault or spread evenly?

Mask A = joint-frames where <MIN_CAMS cameras reproject within REPROJ_PX, so
robust_triangulate abandons filtering and uses every camera.

Per-trial fault (mis-cut clip, bumped camera) => within a participant the rate is
BIMODAL: some trials bad, most clean, and the SAME camera carries it on the bad ones.
Pose/occlusion-driven => rate is similar across all trials of that participant.

Also reports, per trial, each camera's OUTLIER RATE (fraction of joint-frames where that
camera alone reprojects >30px), which is what identifies a specific trial's bad camera.

    python scripts/dlt_gate_audit.py   -> out/scoring/dlt_gate_audit.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                    # noqa: E402
import gnn_refiner as G                  # noqa: E402

REPROJ_PX, MIN_CAMS = 30.0, 3


def gate_state(t):
    """THE gate computation. (inl, resid, fin) for one trial.

    inl   (T,C,J) bool  -- camera c is an inlier for joint j at frame t
    resid (T,C,J) float -- that camera's reprojection residual in px
    fin   (T,J)   bool  -- the DLT produced a point at all

    Both dlt_gate_audit and render_gate_trial call this, so the video and the
    numbers can never diverge (feedback_shared_code_metric_and_render).
    """
    mmc = t["mmc"]; fin = np.isfinite(mmc).all(-1)
    X = torch.from_numpy(np.nan_to_num(mmc).astype(np.float32))[None]
    inl, res = [], []
    for c in range(t["uv"].shape[1]):
        uvp, _ = G.project_torch(X, *[torch.from_numpy(t[k][c].astype(np.float32))
                                      for k in ("K", "dist", "R", "t")])
        r = np.linalg.norm(uvp[0].numpy() - t["uv"][:, c], axis=-1)
        res.append(r)
        inl.append(t["uv_valid"][:, c] & np.isfinite(r) & (r <= REPROJ_PX))
    return np.stack(inl, 1), np.stack(res, 1), fin


def relaxed_mask(t):
    """(T,J) bool -- joint-frames where <MIN_CAMS inliers, so the gate falls back to all cams."""
    inl, _, fin = gate_state(t)
    return (inl.sum(1) < MIN_CAMS) & fin, inl, fin


def main() -> None:
    trials = T.load_clean(need_reproj=True)
    rows = []
    for t in tqdm(trials, mininterval=3, ncols=90, file=sys.stdout):
        inl, _, fin = gate_state(t)
        mmc = t["mmc"]; C = inl.shape[1]
        n = inl.sum(1)                                           # (T,J)
        relaxed = ((n < MIN_CAMS) & fin)
        d = {"part": t["part"], "trial": t["trial"], "n_cams": C,
             "relaxed_pct": 100 * relaxed.sum() / max(fin.sum(), 1)}
        # per-camera outlier rate: camera seen the joint but failed the gate
        for c in range(C):
            seen = t["uv_valid"][:, c] & fin
            d[f"cam{c+1}_out_pct"] = 100 * ((~inl[:, c]) & seen).sum() / max(seen.sum(), 1)
        rows.append(d)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "out/scoring/dlt_gate_audit.csv", index=False)
    print(f"\nPROCESSING CHECK: {len(df)} trials, non-finite {int(df.relaxed_pct.isna().sum())}\n")

    print("WITHIN-PARTICIPANT SPREAD of relaxed_pct across that participant's trials")
    print("(per-trial fault => wide spread / high max; pose-driven => tight)\n")
    print(f"{'part':6s} {'cams':>4s} {'trials':>6s} {'median':>7s} {'IQR':>7s} {'min':>6s} "
          f"{'max':>7s} {'>50%':>5s}")
    for p, g in df.groupby("part"):
        q1, q3 = np.percentile(g.relaxed_pct, [25, 75])
        print(f"{p:6s} {g.n_cams.iloc[0]:4d} {len(g):6d} {g.relaxed_pct.median():7.2f} "
              f"{q3-q1:7.2f} {g.relaxed_pct.min():6.2f} {g.relaxed_pct.max():7.2f} "
              f"{int((g.relaxed_pct > 50).sum()):5d}")

    print("\nPER-CAMERA OUTLIER RATE — median across trials [min, max]")
    for p, g in df.groupby("part"):
        cams = [c for c in g.columns if c.endswith("_out_pct") and g[c].notna().any()]
        s = "  ".join(f"{c[:4]} {g[c].median():5.1f} [{g[c].min():5.1f},{g[c].max():5.1f}]"
                      for c in cams)
        print(f"  {p:6s} {s}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
