"""Stronger miscalibration sweep -- find the calibration level at which reconstruction degrades.

Model = MOVED CAMERA: rotate a camera about its optical centre by `deg` AND translate its centre by
`mm` (a real bump/move, not a pure spin). Perturbation is CONSTANT over the whole trial (one fixed
pose used for every frame). Applied to ONE camera and to camera PAIRS. Severity swept wide, well past
the real observed range (real in-task reproj: median ~9px, P19 ~31px), and reported against the
INDUCED reprojection px so the x-axis is grounded. Per-camera / per-pair records kept (NOT averaged)
so we can report the WORST camera, not a diluted mean.

Outcomes per variant: wrist/elbow/shoulder 3D error (mm, Kabsch vs OMC) + peak_velocity + interjoint
(scorer's own functions vs AutoMQ). Saves per-variant CSV, prints LIVE progress + degradation curve.
Data only.
"""
from __future__ import annotations
import sys, glob, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO))
import score_vs_automq as S
import compare_pose_omc_delta as H
import gnn_train as GT

PAIRS = REPO / "cache/delta/gnn_pairs"; OUT = REPO / "paper"; FPS = S.FPS
JIDX = {"right_wrist": 0, "right_elbow": 1, "right_shoulder": 2,
        "left_wrist": 3, "left_elbow": 4, "left_shoulder": 5, "right_hip": 6, "left_hip": 7}
NEED = list(JIDX); UPPER = [0, 1, 2, 3, 4, 5]
UPPER_NAMES = [nm for nm, i in JIDX.items() if i in UPPER]   # wrists/elbows/shoulders
# (rotation deg, translation mm) -- moved-camera severity, wide
LEVELS = [(1, 10), (2, 20), (4, 40), (8, 80), (16, 150), (30, 300)]
AXIS = np.array([1., 1., 1.]); TDIR = np.array([1., -1., 1.])   # one fixed move direction


def _P(K, R, t):
    return np.stack([K[c] @ np.hstack([R[c], t[c].reshape(3, 1)]) for c in range(len(K))])


def tri(P, uv, val):
    F, m, _ = uv.shape
    P0, P1, P2 = P[:, 0], P[:, 1], P[:, 2]
    A = np.zeros((F, 2 * m, 4)); u, v = uv[..., 0], uv[..., 1]
    A[:, 0::2, :] = u[..., None] * P2[None] - P0[None]
    A[:, 1::2, :] = v[..., None] * P2[None] - P1[None]
    w = val.astype(float)[..., None]; A[:, 0::2] *= w; A[:, 1::2] *= w
    A = np.nan_to_num(A)
    _, _, Vt = np.linalg.svd(A); h = Vt[:, -1, :]; X = h[:, :3] / h[:, 3:4]
    X[val.sum(1) < 2] = np.nan; return X


def _rod(axis, deg):
    a = axis / (np.linalg.norm(axis) + 1e-12); th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def move(K, R, t, idxs, deg, mm):
    """Return perturbed P with cameras `idxs` rotated `deg` about their centre + centre moved `mm`."""
    R2, t2 = R.copy(), t.copy()
    dR = _rod(AXIS, deg); dt = mm * TDIR / (np.linalg.norm(TDIR) + 1e-12)
    for c in idxs:
        C = -R[c].T @ t[c]
        R2[c] = dR @ R[c]
        t2[c] = -R2[c] @ (C + dt)
    return _P(K, R2, t2)


def pose_err(Xj, omc, val, side):
    A, B = [], []
    for nm in UPPER_NAMES:
        j = JIDX[nm]; m = val[:, j] & np.isfinite(Xj[nm]).all(1)
        A.append(Xj[nm][m]); B.append(omc[m, j])
    A = np.concatenate(A); B = np.concatenate(B)
    if len(A) < 3:
        return {}
    ca, cb = A.mean(0), B.mean(0); Hm = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rk = Vt.T @ np.diag([1, 1, d]) @ U.T; tk = cb - Rk @ ca
    pre = "left" if side == "left" else "right"; out = {}
    for nm in ("shoulder", "elbow", "wrist"):
        key = f"{pre}_{nm}"; j = JIDX[key]; m = val[:, j] & np.isfinite(Xj[key]).all(1)
        if m.any():
            out[nm] = float(np.nanmean(np.linalg.norm((Rk @ Xj[key][m].T).T + tk - omc[m, j], axis=1)))
    return out


def reproj_px(P, Xj, uv, val):
    res = []
    for nm in UPPER_NAMES:
        j = JIDX[nm]; X = Xj[nm]; Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
        for c in range(len(P)):
            p = (P[c] @ Xh.T).T
            g = (np.abs(p[:, 2]) > 1e-6) & val[:, c, j] & np.isfinite(X).all(1)
            if g.any():
                res.append(np.linalg.norm(p[g, :2] / p[g, 2:3] - uv[g, c, j], axis=1))
    return float(np.median(np.concatenate(res))) if res else np.nan


def main():
    OUT.mkdir(exist_ok=True); S.H.use_good_cams()
    amq = S.load_automq()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in S.COHORT_PARTS]
    import re; pat = re.compile(r"trial_(\d+)_([LR])_")
    print(f"{len(trials)} trials; levels {LEVELS}", flush=True)
    rows = []; t0 = time.time(); n = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((S.automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        rjf = PAIRS / part / f"{trial}.reproj.npz"
        npf = PAIRS / part / f"{trial}.npz"
        if not rjf.exists():
            continue
        nfr = t["mmc"].shape[0]
        omc_s = H._load_omc(part, trial, nfr)
        lag, _ = H._find_lag(t["mmc"][:, S.GRID_JOINTS.index(f"{side}_wrist")], omc_s[f"{side}_wrist"])
        ph = S.automq_phases_to_video(rec["phases"], lag, nfr)
        if ph is None:
            continue
        d = np.load(npf, allow_pickle=True); rj = np.load(rjf, allow_pickle=True)
        omc, val = d["omc"], d["valid"]
        uv, uvv, K, R, tt = rj["uv"], rj["uv_valid"], rj["K"], rj["R"], rj["t"]
        T = min(len(uv), nfr, len(omc)); uv, uvv, omc, val = uv[:T], uvv[:T], omc[:T], val[:T]
        ncam = len(K)
        gtpv = rec.get("peak_velocity_wrist"); gtij = rec.get("interjoint_coordination")

        def eval_P(P, nsel, level_deg, level_mm, which):
            Xj = {nm: tri(P, uv[:, :, JIDX[nm]], uvv[:, :, JIDX[nm]]) for nm in NEED}
            pe = pose_err(Xj, omc, val, side)
            rp = reproj_px(P, Xj, uv, uvv)
            pose = Xj
            pv = S.peak_velocity_reduce(pose, ph, side, "max")
            ang = S.angle_measures_automq(pose, ph, side, "max")
            rows.append(dict(part=part, trial=trial, which=which, nsel=nsel, deg=level_deg, mm=level_mm,
                             induced=rp, err_wr=pe.get("wrist", np.nan), err_el=pe.get("elbow", np.nan),
                             err_sh=pe.get("shoulder", np.nan),
                             pv=pv, gt_pv=gtpv if gtpv else np.nan,
                             ij=ang.get("interjoint_coordination", np.nan),
                             gt_ij=gtij if gtij is not None else np.nan))

        # baseline (unperturbed)
        eval_P(_P(K, R, tt), 0, 0, 0, "base")
        for deg, mm in LEVELS:
            for c in range(ncam):                                   # single camera
                eval_P(move(K, R, tt, [c], deg, mm), 1, deg, mm, "single")
            for pr in itertools.combinations(range(ncam), 2):       # camera pair
                eval_P(move(K, R, tt, list(pr), deg, mm), 2, deg, mm, "pair")
        n += 1
        if n % 40 == 0:
            print(f"[{n}] {time.time()-t0:5.0f}s rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "augment_miscal2.csv", index=False)
    base_wr = df[df.which == "base"]["err_wr"].median()
    print(f"\nPROCESSING CHECK: rows {len(df)}, trials {n}, baseline wrist {base_wr:.1f}mm", flush=True)

    bins = [0, 10, 15, 20, 30, 45, 70, 120, 1e9]
    lab = ["<10", "10-15", "15-20", "20-30", "30-45", "45-70", "70-120", ">120"]
    for which in ("single", "pair"):
        g = df[df.which == which].dropna(subset=["induced"]).copy()
        g["bin"] = pd.cut(g["induced"], bins, labels=lab)
        print(f"\n== {which.upper()}-camera miscalibration: degradation vs INDUCED reproj px ==")
        print(f"{'induced px':>10}{'n':>7}{'wrist mm':>10}{'elbow mm':>10}{'|pv err|':>10}{'ij r_s':>9}")
        for b in lab:
            gg = g[g.bin == b]
            if len(gg) < 5:
                continue
            pverr = (gg["pv"] - gg["gt_pv"]).abs().median()
            gj = gg.dropna(subset=["ij", "gt_ij"])
            ij = spearmanr(gj["gt_ij"], gj["ij"]).correlation if len(gj) > 5 else np.nan
            print(f"{b:>10}{len(gg):>7}{gg['err_wr'].median():>10.1f}{gg['err_el'].median():>10.1f}"
                  f"{pverr:>10.0f}{ij:>9.2f}")
    print(f"\nreal observed in-task reproj for reference: median ~9px, P19 ~31px, cohort max ~93px")
    print(f"\nwrote {OUT/'augment_miscal2.csv'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
