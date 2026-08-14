"""Augmentation: vary rig factors WITHIN each trial (person held fixed) and re-triangulate,
to get a confound-free causal read of what the observational seg_factors table couldn't separate.

  A) CAMERA DROP    -- for every trial, re-triangulate from every camera subset of size 2..n.
                       pose error vs OMC (Kabsch-aligned) as a function of #cameras kept.
  B) MISCALIBRATION -- perturb ONE camera in place (rotate about its optical centre by theta),
                       re-triangulate with the full set, sweep theta. Severity reported as the
                       INDUCED median reprojection px so it maps onto the real observed range
                       (clean ~8 px .. P19 ~34 px). 3 random axes per (cam, theta), averaged.

Batched DLT (stacked SVD over frames). Frames subsampled by FRAME_STRIDE for speed.
Saves per-item CSVs + npz, prints LIVE per-trial progress. Data only.
"""
from __future__ import annotations
import glob, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAIRS = REPO / "cache/delta/gnn_pairs"
OUT = REPO / "paper"
PARTS = ["P07", "P08", "P10", "P12", "P13", "P14", "P15", "P17", "P19", "P251", "P252"]
JIDX = {"right_wrist": 0, "right_elbow": 1, "right_shoulder": 2,
        "left_wrist": 3, "left_elbow": 4, "left_shoulder": 5}
UPPER = [0, 1, 2, 3, 4, 5]
FRAME_STRIDE = 4
ROT_DEG = [0.5, 1.0, 2.0, 4.0, 8.0]   # per-camera in-place rotation sweep
N_AXES = 3                             # random rotation axes per (cam, theta)
RNG = np.random.default_rng(0)


def _P(K, R, t):
    return np.stack([K[c] @ np.hstack([R[c], t[c].reshape(3, 1)]) for c in range(len(K))])


def tri_batch(P_sub, uv, val):
    """Batched DLT. P_sub:(m,3,4) uv:(F,m,2) val:(F,m)bool -> X:(F,3) (NaN if <2 valid)."""
    F, m, _ = uv.shape
    P0, P1, P2 = P_sub[:, 0], P_sub[:, 1], P_sub[:, 2]        # (m,4)
    A = np.zeros((F, 2 * m, 4))
    u, v = uv[..., 0], uv[..., 1]
    A[:, 0::2, :] = u[..., None] * P2[None] - P0[None]
    A[:, 1::2, :] = v[..., None] * P2[None] - P1[None]
    w = val.astype(float)[..., None]
    A[:, 0::2, :] *= w; A[:, 1::2, :] *= w
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)   # invalid uv is NaN -> zeroed rows
    _, _, Vt = np.linalg.svd(A)
    h = Vt[:, -1, :]
    X = h[:, :3] / h[:, 3:4]
    X[val.sum(1) < 2] = np.nan
    return X


def _kabsch_err(Xj, omc, val, side):
    """Xj: dict joint->(F,3). Kabsch-align all valid upper points to OMC, per-joint mean mm err."""
    A, B = [], []
    for j in UPPER:
        m = val[:, j] & np.isfinite(Xj[j]).all(1)
        A.append(Xj[j][m]); B.append(omc[m, j])
    A = np.concatenate(A); B = np.concatenate(B)
    if len(A) < 3:
        return {}
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rk = Vt.T @ np.diag([1, 1, d]) @ U.T
    tk = cb - Rk @ ca
    pre = "left" if side == "left" else "right"
    out = {}
    for name in ("shoulder", "elbow", "wrist"):
        j = JIDX[f"{pre}_{name}"]
        m = val[:, j] & np.isfinite(Xj[j]).all(1)
        if m.any():
            pred = (Rk @ Xj[j][m].T).T + tk
            out[name] = float(np.nanmean(np.linalg.norm(pred - omc[m, j], axis=1)))
    return out


def _reproj_med(P, Xj, uv, val):
    """Median reprojection px over frames x upper joints for the given P + triangulated Xj."""
    res = []
    for j in UPPER:
        X = Xj[j]                                    # (F,3)
        Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
        for c in range(len(P)):
            p = (P[c] @ Xh.T).T                       # (F,3)
            good = (np.abs(p[:, 2]) > 1e-6) & val[:, c, j] & np.isfinite(X).all(1)
            if good.any():
                proj = p[good, :2] / p[good, 2:3]
                res.append(np.linalg.norm(proj - uv[good, c, j], axis=1))
    return float(np.median(np.concatenate(res))) if res else np.nan


def _rodrigues(axis, deg):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    th = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main():
    OUT.mkdir(exist_ok=True)
    cam_rows, mis_rows = [], []
    t0 = time.time()
    n = 0
    trials = [(p, f) for p in PARTS
              for f in sorted(glob.glob(str(PAIRS / p / "*.npz"))) if not f.endswith(".reproj.npz")]
    print(f"{len(trials)} trials; stride {FRAME_STRIDE}; rot sweep {ROT_DEG}deg x{N_AXES} axes",
          flush=True)
    for part, npzf in trials:
        stem = Path(npzf).stem
        rjf = PAIRS / part / f"{stem}.reproj.npz"
        if not rjf.exists():
            continue
        npz = np.load(npzf, allow_pickle=True)
        rj = np.load(rjf, allow_pickle=True)
        side = "left" if "_L_" in stem else "right"
        omc, valj = npz["omc"], npz["valid"]                 # (T,9,3),(T,9)
        uv, uvv = rj["uv"], rj["uv_valid"]                   # (T,C,9,2),(T,C,9)
        K, Rr, tt = rj["K"], rj["R"], rj["t"]
        sl = slice(None, None, FRAME_STRIDE)
        omc, valj, uv, uvv = omc[sl], valj[sl], uv[sl], uvv[sl]
        ncam = len(K)
        Pfull = _P(K, Rr, tt)

        # ---- A) camera drop: every subset size 2..ncam ----
        for k in range(2, ncam + 1):
            for sub in itertools.combinations(range(ncam), k):
                sub = list(sub)
                Ps = Pfull[sub]
                Xj = {j: tri_batch(Ps, uv[:, sub, j], uvv[:, sub, j]) for j in UPPER}
                e = _kabsch_err(Xj, omc, valj, side)
                if e:
                    cam_rows.append(dict(part=part, trial=stem, k=k, subset="".join(map(str, sub)),
                                         err_sh=e.get("shoulder", np.nan), err_el=e.get("elbow", np.nan),
                                         err_wr=e.get("wrist", np.nan)))

        # ---- B) miscalibration: perturb one cam in place, sweep theta ----
        # baseline (full set, no perturb)
        Xb = {j: tri_batch(Pfull, uv[:, :, j], uvv[:, :, j]) for j in UPPER}
        eb = _kabsch_err(Xb, omc, valj, side)
        rb = _reproj_med(Pfull, Xb, uv, uvv)
        for c in range(ncam):
            for deg in ROT_DEG:
                for a in range(N_AXES):
                    dR = _rodrigues(RNG.standard_normal(3), deg)
                    Rp, tp = Rr.copy(), tt.copy()
                    Rp[c] = dR @ Rr[c]                       # rotate cam c about its optical centre
                    tp[c] = dR @ tt[c]
                    Pp = _P(K, Rp, tp)
                    Xj = {j: tri_batch(Pp, uv[:, :, j], uvv[:, :, j]) for j in UPPER}
                    e = _kabsch_err(Xj, omc, valj, side)
                    rpx = _reproj_med(Pp, Xj, uv, uvv)
                    if e:
                        mis_rows.append(dict(part=part, trial=stem, cam=c, rot_deg=deg, axis=a,
                                             induced_reproj=rpx, base_reproj=rb,
                                             err_sh=e.get("shoulder", np.nan), err_el=e.get("elbow", np.nan),
                                             err_wr=e.get("wrist", np.nan),
                                             base_err_wr=eb.get("wrist", np.nan),
                                             base_err_el=eb.get("elbow", np.nan),
                                             base_err_sh=eb.get("shoulder", np.nan)))
        n += 1
        if n % 40 == 0:
            cd = pd.DataFrame(cam_rows)
            m2 = cd[cd.k == 2]["err_wr"].median() if len(cd) else np.nan
            mf = cd[cd.k == cd.k.max()]["err_wr"].median() if len(cd) else np.nan
            print(f"[{n}/{len(trials)}] {time.time()-t0:5.0f}s  cam_rows={len(cam_rows)} "
                  f"mis_rows={len(mis_rows)}  wrist_err 2cam~{m2:.1f} full~{mf:.1f}mm", flush=True)

    cam = pd.DataFrame(cam_rows); mis = pd.DataFrame(mis_rows)
    cam.to_csv(OUT / "augment_camdrop.csv", index=False)
    mis.to_csv(OUT / "augment_miscal.csv", index=False)
    np.savez(OUT / "augment.npz",
             **{f"cam_{c}": cam[c].to_numpy() for c in cam.columns if cam[c].dtype != object},
             **{f"mis_{c}": mis[c].to_numpy() for c in mis.columns if mis[c].dtype != object})
    print(f"\nPROCESSING CHECK: trials {n}, camdrop rows {len(cam)}, miscal rows {len(mis)}, "
          f"nonfinite camdrop_wr {int(cam['err_wr'].isna().sum())}", flush=True)

    # ---- A) within-participant error vs #cameras ----
    print("\n== CAMERA DROP: median wrist / elbow / shoulder error (mm) vs #cameras kept ==")
    print(f"{'part':<6}" + "".join(f"{f'k={k}':>18}" for k in range(2, 6)))
    for part, g in cam.groupby("part"):
        cells = []
        for k in range(2, 6):
            gg = g[g.k == k]
            cells.append(f"{gg['err_wr'].median():>5.0f}/{gg['err_el'].median():>3.0f}/{gg['err_sh'].median():>3.0f}"
                         if len(gg) else f"{'':>18}")
        print(f"{part:<6}" + "".join(f"{c:>18}" for c in cells))
    print("pooled median wrist err by k:",
          {int(k): round(float(g["err_wr"].median()), 1) for k, g in cam.groupby("k")})

    # ---- B) miscalibration: pose error vs induced reprojection px (binned) ----
    print("\n== MISCALIBRATION: wrist error (mm) vs INDUCED reprojection (px), pooled ==")
    mis = mis.dropna(subset=["induced_reproj", "err_wr"])
    bins = [0, 10, 15, 20, 30, 45, 70, 1e9]
    lab = ["<10", "10-15", "15-20", "20-30", "30-45", "45-70", ">70"]
    mis["bin"] = pd.cut(mis["induced_reproj"], bins=bins, labels=lab)
    base_wr = mis["base_err_wr"].median()
    print(f"baseline (unperturbed full set) wrist err = {base_wr:.1f} mm")
    print(f"{'induced px':>10}{'n':>8}{'wrist mm':>10}{'elbow mm':>10}{'shoulder mm':>12}")
    for b in lab:
        gg = mis[mis["bin"] == b]
        if len(gg):
            print(f"{b:>10}{len(gg):>8}{gg['err_wr'].median():>10.1f}"
                  f"{gg['err_el'].median():>10.1f}{gg['err_sh'].median():>12.1f}")
    print(f"\nwrote {OUT/'augment_camdrop.csv'}\nwrote {OUT/'augment_miscal.csv'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
