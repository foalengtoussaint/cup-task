"""What drives pose accuracy in the ways the Murphy measures care about?

Per-trial OBSERVATIONAL table (no augmentation yet) joining, for every cohort trial:

  PREDICTORS (rig / calibration)
    n_cam        cameras actually used for this trial (from reproj sidecar)   [participant-level]
    cov_maxang   max pairwise angle between camera->subject view directions, deg  [trial-level]
    cov_meanang  mean pairwise angle, deg                                          [trial-level]
    reproj_med   in-task DLT reprojection residual, median px over frames x upper-limb joints [trial-level]

  NB the study's Calibration_errors.csv is DROPPED: its error was computed over the FULL camera set
  (cam_used, 7-10 cams) but we triangulate from a pruned subset (the uncalibrated cams were removed),
  so it describes a camera set we don't use. reproj_med is the in-task calibration-quality signal,
  computed only over the cameras actually used.

  OUTCOMES
    pose error vs OMC (mm): shoulder / elbow / wrist mean over valid frames (npz mmc vs omc)
    measure error: |mmc - automq| per Murphy measure (out/scoring/score_vs_automq.csv, BA+smoothnet)

Saves per-trial arrays (seg_factors.npz) + long CSV, prints a processing check + correlation tables.
Data only -- no interpretation.
"""
from __future__ import annotations
import glob, os, ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
PAIRS = REPO / "cache/delta/gnn_pairs"
CALIB = REPO / "cache/delta/study_docs/2024_10_23_1138_Calibration_errors.csv"
SCORE = Path(os.environ.get("SCORE_CSV", REPO / "out/scoring/score_vs_automq.csv"))
OUT = REPO / "paper"
PARTS = ["P07", "P08", "P10", "P12", "P13", "P14", "P15", "P17", "P19", "P251", "P252"]
# joint index order in the npz (from the trial json "joints")
JIDX = {"right_wrist": 0, "right_elbow": 1, "right_shoulder": 2,
        "left_wrist": 3, "left_elbow": 4, "left_shoulder": 5,
        "right_hip": 6, "left_hip": 7, "nose": 8}
UPPER = [0, 1, 2, 3, 4, 5]  # wrists/elbows/shoulders -- the measure-relevant joints


def _calib_map():
    d = pd.read_csv(CALIB, sep=";")
    return dict(zip(d["id_p"], pd.to_numeric(d["error"], errors="coerce")))


def _dlt(Ps, uvs):
    """DLT triangulate one point from >=2 (P, uv) pairs. Ps:(m,3,4) uvs:(m,2)."""
    A = []
    for P, uv in zip(Ps, uvs):
        A.append(uv[0] * P[2] - P[0])
        A.append(uv[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    h = Vt[-1]
    return h[:3] / h[3]


def _reproj_residual(rj):
    """Median in-task DLT reprojection residual (px) over frames x upper-limb joints."""
    K, R, t = rj["K"], rj["R"], rj["t"]
    uv, val = rj["uv"], rj["uv_valid"]            # (T,C,9,2),(T,C,9)
    P = np.stack([K[c] @ np.hstack([R[c], t[c].reshape(3, 1)]) for c in range(len(K))])
    T = uv.shape[0]
    res = []
    for f in range(T):
        for j in UPPER:
            cams = np.where(val[f, :, j])[0]
            if len(cams) < 2:
                continue
            X = _dlt(P[cams], uv[f, cams, j])
            Xh = np.append(X, 1.0)
            for c in cams:
                p = P[c] @ Xh
                if abs(p[2]) < 1e-6:
                    continue
                res.append(np.linalg.norm(p[:2] / p[2] - uv[f, c, j]))
    return float(np.median(res)) if res else np.nan


def _coverage(rj, npz):
    """Max/mean pairwise angle (deg) between camera->subject view directions."""
    R, t = rj["R"], rj["t"]
    C = np.stack([-R[c].T @ t[c] for c in range(len(R))])      # camera centres
    omc, val = npz["omc"], npz["valid"]
    ctr = np.nanmedian(np.where(val[..., None], omc, np.nan).reshape(-1, 3), axis=0)
    dirs = ctr[None] - C
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    n = len(dirs)
    ang = []
    for i in range(n):
        for k in range(i + 1, n):
            ang.append(np.degrees(np.arccos(np.clip(dirs[i] @ dirs[k], -1, 1))))
    return (float(np.max(ang)), float(np.mean(ang))) if ang else (np.nan, np.nan)


def _kabsch(A, B):
    """Rigid R,t mapping A->B (no scale). A,B: (N,3)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def _pose_err(npz, side):
    """Mean 3D error (mm) vs OMC for shoulder/elbow/wrist, after per-trial rigid (Kabsch) align.

    The stored mmc/omc are in different frames; align mmc->omc on ALL valid upper-limb points
    (pooled over frames) first, then residual per joint. Alignment-free measures (angles/vels)
    come from the score CSV -- this is only for the 3D-error diagnostic."""
    mmc, omc, val = npz["mmc"], npz["omc"], npz["valid"]
    A, B = [], []
    for j in UPPER:
        m = val[:, j]
        A.append(mmc[m, j]); B.append(omc[m, j])
    A = np.concatenate(A); B = np.concatenate(B)
    if len(A) < 3:
        return {n: np.nan for n in ("shoulder", "elbow", "wrist")}
    R, t = _kabsch(A, B)
    pre = "left" if side == "left" else "right"
    out = {}
    for name in ("shoulder", "elbow", "wrist"):
        j = JIDX[f"{pre}_{name}"]
        m = val[:, j]
        if m.any():
            pred = (R @ mmc[m, j].T).T + t
            out[name] = float(np.nanmean(np.linalg.norm(pred - omc[m, j], axis=1)))
        else:
            out[name] = np.nan
    return out


def main():
    rows = []
    n_trials = n_noreproj = 0
    for part in PARTS:
        for npzf in sorted(glob.glob(str(PAIRS / part / "*.npz"))):
            if npzf.endswith(".reproj.npz"):
                continue
            stem = Path(npzf).stem
            rjf = PAIRS / part / f"{stem}.reproj.npz"
            if not rjf.exists():
                continue
            n_trials += 1
            npz = np.load(npzf, allow_pickle=True)
            rj = np.load(rjf, allow_pickle=True)
            side = "left" if "_L_" in stem else "right"
            cov_max, cov_mean = _coverage(rj, npz)
            rp = _reproj_residual(rj)
            if not np.isfinite(rp):
                n_noreproj += 1
            pe = _pose_err(npz, side)
            rows.append(dict(part=part, trial=stem, side=side,
                             n_cam=int(len(rj["cams"])),
                             cov_maxang=cov_max, cov_meanang=cov_mean, reproj_med=rp,
                             pose_sh=pe["shoulder"], pose_el=pe["elbow"], pose_wr=pe["wrist"]))
    df = pd.DataFrame(rows)
    print(f"\nPROCESSING CHECK: trials {n_trials}, table rows {len(df)}, "
          f"no-reproj {n_noreproj}, non-finite reproj {int((~np.isfinite(df['reproj_med'])).sum())}",
          flush=True)

    # join per-measure |mmc-automq|
    sc = pd.read_csv(SCORE, keep_default_na=False, na_values=[""])
    sc = sc[(sc["variant"] == "BA+smoothnet") & (sc["peak_metric"].isin(["n/a", "max"]))].copy()
    sc["automq"] = pd.to_numeric(sc["automq"], errors="coerce")
    sc["mmc"] = pd.to_numeric(sc["mmc"], errors="coerce")
    sc["abserr"] = (sc["mmc"] - sc["automq"]).abs()
    merr = sc.pivot_table(index=["part", "trial"], columns="measure", values="abserr", aggfunc="first")
    merr.columns = [f"err_{c}" for c in merr.columns]
    full = df.merge(merr.reset_index(), on=["part", "trial"], how="left")

    OUT.mkdir(exist_ok=True)
    full.to_csv(OUT / "seg_factors.csv", index=False)
    np.savez(OUT / "seg_factors.npz", **{c: full[c].to_numpy() for c in full.columns
                                          if full[c].dtype != object},
             part=full["part"].to_numpy(), trial=full["trial"].to_numpy())

    # ---- correlation tables (data only) ----
    pred = ["n_cam", "cov_maxang", "cov_meanang", "reproj_med"]
    print("\n== per-PARTICIPANT means (n_cam is participant-level) ==")
    pm = df.groupby("part").agg(n_cam=("n_cam", "first"),
                                cov_maxang=("cov_maxang", "mean"), reproj_med=("reproj_med", "mean"),
                                pose_sh=("pose_sh", "mean"), pose_el=("pose_el", "mean"),
                                pose_wr=("pose_wr", "mean"))
    print(pm.round(2).to_string())

    print("\n== Spearman r_s : PREDICTOR vs POSE ERROR (trial-level, all trials) ==")
    print(f"{'predictor':<12}" + "".join(f"{o:>10}" for o in ("pose_sh", "pose_el", "pose_wr")))
    for p in pred:
        cells = []
        for o in ("pose_sh", "pose_el", "pose_wr"):
            g = df[[p, o]].dropna()
            r = spearmanr(g[p], g[o]).correlation if len(g) > 3 else np.nan
            cells.append(f"{r:>10.2f}")
        print(f"{p:<12}" + "".join(cells))

    errcols = [c for c in full.columns if c.startswith("err_")]
    print("\n== Spearman r_s : PREDICTOR vs MEASURE |error| (trial-level, all trials) ==")
    print(f"{'measure':<34}" + "".join(f"{p:>12}" for p in pred))
    for ec in sorted(errcols):
        cells = []
        for p in pred:
            g = full[[p, ec]].dropna()
            r = spearmanr(g[p], g[ec]).correlation if len(g) > 3 else np.nan
            cells.append(f"{r:>12.2f}")
        print(f"{ec:<34}" + "".join(cells))

    print(f"\nwrote {OUT/'seg_factors.csv'}\nwrote {OUT/'seg_factors.npz'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
