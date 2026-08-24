"""Does the 2D flow difference survive into 3D SPEED -- and is the LOO consensus the right fuser?

Two questions in one pass, because both need the same expensive RAFT recompute:

  (1) TAILS. Per-camera-frame pixel error for PyrLK vs RAFT, full distribution. A method that is
      slightly worse typically but never fails badly may still be preferable: one 55px outlier at
      the peak corrupts peak_velocity for a whole trial, while 0.25px of typical error averages out.

  (2) SURVIVAL. Does any of it reach the 3D speed? The fuser already discards outliers across 5
      cameras, so it may absorb PyrLK's tail entirely -- in which case the pixel comparison is moot.

FUSERS COMPARED (the LOO consensus is a hand-tuned heuristic -- drop a camera when leave-one-out
disagreement exceeds 20mm/s -- and it is only ~60% right per frame, so it is worth checking against
alternatives that do not hard-drop anything):

    plain      least squares over all cameras (breakdown point 0: one bad vector moves the answer)
    loo        the current leave-one-out consensus
    huber      IRLS with a Huber loss -- DOWN-WEIGHTS outliers smoothly instead of discarding them
    l1         iteratively reweighted least squares toward an L1 (median-like) solution
    trimmed    drop the single worst-residual camera, no threshold, no iteration

    python scripts/flow_3d_survival.py --trials 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

CROP = 192
MOVING = 50.0


def jacobian(cal, X):
    """d(pixel)/d(world position) at X for this camera -- the 2x3 projection Jacobian."""
    R_ = np.asarray(cal.R); t = np.asarray(cal.t).ravel(); K = np.asarray(cal.K)
    Xc = R_ @ np.asarray(X) + t
    x, y, z = Xc
    if z <= 1:
        return None
    fx, fy = K[0, 0], K[1, 1]
    return np.array([[fx / z, 0, -fx * x / z ** 2],
                     [0, fy / z, -fy * y / z ** 2]]) @ R_


def solve_velocity(rows, mode="plain", fps=60.0, huber_k=1.5, iters=8):
    """rows = [(J_2x3, flow_2), ...] -> 3D velocity (mm/s).

    Solves u_dot = J v directly, which is the correct model: each camera constrains only the
    component of motion it can actually see. `mode` selects how outliers are handled.
    """
    if len(rows) < 2:
        return None
    A = np.vstack([r[0] for r in rows])
    b = np.concatenate([r[1] for r in rows])
    if mode == "plain":
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        return v * fps
    if mode == "trimmed" and len(rows) >= 3:
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (A @ v - b).reshape(-1, 2)
        keep = [i for i in range(len(rows)) if i != int(np.argmax(np.linalg.norm(res, axis=1)))]
        A = np.vstack([rows[i][0] for i in keep])
        b = np.concatenate([rows[i][1] for i in keep])
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        return v * fps
    # IRLS: huber or l1
    v, *_ = np.linalg.lstsq(A, b, rcond=None)
    for _ in range(iters):
        r = (A @ v - b).reshape(-1, 2)
        rn = np.linalg.norm(r, axis=1)
        s = 1.4826 * np.median(rn) + 1e-6
        if mode == "huber":
            w = np.where(rn <= huber_k * s, 1.0, huber_k * s / np.maximum(rn, 1e-9))
        else:                                   # l1
            w = 1.0 / np.maximum(rn, 1e-3)
        W = np.repeat(w, 2)[:, None]
        v, *_ = np.linalg.lstsq(A * W, b * W[:, 0], rcond=None)
    return v * fps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args(argv)

    import cv2
    import torch
    import torch.nn.functional as Fn
    from scipy.signal import find_peaks
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import flow_velocity_probe as F
    from pipeline import flow_speed
    from pipeline.kalman_3d import triangulate_dlt

    H.use_good_cams()
    dev = "cuda"
    print("loading raft_large ...", flush=True)
    rl = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()

    def raft_at(ga, gb, p2):
        h, w = ga.shape
        x, y = int(round(p2[0])), int(round(p2[1]))
        c = CROP // 2
        x0, y0 = max(0, x - c), max(0, y - c)
        x1, y1 = min(w, x0 + CROP), min(h, y0 + CROP)
        x0, y0 = max(0, x1 - CROP), max(0, y1 - CROP)
        ca, cb = ga[y0:y1, x0:x1], gb[y0:y1, x0:x1]
        if ca.shape[0] < 64 or ca.shape[1] < 64:
            return None

        def t(g):
            z = torch.from_numpy(np.ascontiguousarray(g)).float()[None, None].to(dev) / 255.0
            z = z.repeat(1, 3, 1, 1) * 2 - 1
            H_, W_ = z.shape[-2:]
            return Fn.pad(z, (0, (8 - W_ % 8) % 8, 0, (8 - H_ % 8) % 8))

        with torch.no_grad():
            fl = rl(t(ca), t(cb))[-1][0].cpu().numpy().transpose(1, 2, 0)
        lx, ly = x - x0, y - y0
        return fl[ly, lx] if (0 <= ly < fl.shape[0] and 0 <= lx < fl.shape[1]) else None

    FUSERS = ["plain", "loo", "huber", "l1", "trimmed"]
    px_err = {"pyrlk": [], "raft": []}
    spd = {f"{m}_{k}": {"mv": [], "peak": []} for m in ("pyrlk", "raft") for k in FUSERS}
    t0 = time.time()

    for part, (trials, side) in R.TRIALS.items():
        calib = R._calib(part)
        for trial in trials[:a.trials]:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            omcs = {j: R._shift(v, lag) for j, v in omc.items()}
            fitj = [j for j in H.JOINTS if "hip" not in j]
            A_ = np.vstack([omcs[j] for j in fitj]); B_ = np.vstack([mmc[j] for j in fitj])
            if (np.isfinite(A_).all(1) & np.isfinite(B_).all(1)).sum() < 50:
                continue
            Rm, _, _ = H._kabsch(A_, B_)
            ow = omcs[f"{side}_wrist"]
            so = H._lp(H._speed(ow))

            px = F.load_wrist_px(part, trial, f"{side}_wrist")
            flp = {}
            for c in px:
                p = ROOT / "cache" / "flow_vel" / f"delta_{part}_{trial}.{c.split('_')[1]}__pyrlk.npy"
                if p.exists() and c in calib:
                    flp[c] = np.load(p)
            caps = {}
            for cam in flp:
                v = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{cam.split('_')[1]}.mp4"
                if v.exists():
                    caps[cam] = cv2.VideoCapture(str(v))
            if not caps:
                continue

            flr = {c: np.full((n, 2), np.nan) for c in flp}
            prev = {c: None for c in caps}
            for f in range(n):
                gr = {}
                for cam, cap in caps.items():
                    ok, im = cap.read()
                    gr[cam] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if ok else None
                for cam in flp:
                    ga, gb = prev.get(cam), gr.get(cam)
                    if ga is None or gb is None or f - 1 < 0 or f - 1 >= len(px[cam]):
                        continue
                    p2 = px[cam][f - 1]
                    if not np.isfinite(p2).all():
                        continue
                    r = raft_at(ga, gb, p2)
                    if r is not None:
                        flr[cam][f - 1] = r
                prev = dict(gr)
            for c in caps.values():
                c.release()

            # ---- (1) per-camera PIXEL error vs the true projected flow ----
            for f in range(1, n - 1):
                if not (np.isfinite(so[f]) and so[f] > 200
                        and np.isfinite(ow[f]).all() and np.isfinite(ow[f + 1]).all()):
                    continue
                op = {c: px[c][f] for c in flp
                      if f < len(px[c]) and np.isfinite(px[c][f]).all()}
                if len(op) < 3:
                    continue
                X = triangulate_dlt([calib[c] for c in op], [np.asarray(op[c]) for c in op])
                if X is None:
                    continue
                vt = Rm @ (np.asarray(ow[f + 1]) - np.asarray(ow[f]))
                for cam in op:
                    cal = calib[cam]

                    def proj(P3):
                        Y = np.asarray(cal.R) @ np.asarray(P3) + np.asarray(cal.t).ravel()
                        if Y[2] <= 1:
                            return None
                        u = np.asarray(cal.K) @ Y
                        return u[:2] / u[2]

                    aa, bb = proj(X), proj(np.asarray(X) + vt)
                    if aa is None or bb is None:
                        continue
                    true = bb - aa
                    if f < len(flp[cam]) and np.isfinite(flp[cam][f]).all():
                        px_err["pyrlk"].append(float(np.linalg.norm(flp[cam][f] - true)))
                    if f < len(flr[cam]) and np.isfinite(flr[cam][f]).all():
                        px_err["raft"].append(float(np.linalg.norm(flr[cam][f] - true)))

            # ---- (2) 3D SPEED under each fuser ----
            for mname, fld in (("pyrlk", flp), ("raft", flr)):
                tracks = {k: np.full(n, np.nan) for k in FUSERS}
                prev3d = None
                for f in range(n):
                    obs = {c: px[c][f] for c in fld
                           if f < len(px[c]) and f < len(fld[c])
                           and np.isfinite(px[c][f]).all() and np.isfinite(fld[c][f]).all()}
                    if len(obs) < 2:
                        continue
                    X = triangulate_dlt([calib[c] for c in obs],
                                        [np.asarray(obs[c]) for c in obs])
                    if X is None:
                        continue
                    rows, cams = [], []
                    for c in obs:
                        J = jacobian(calib[c], X)
                        if J is None:
                            continue
                        rows.append((J, np.asarray(fld[c][f]))); cams.append(c)
                    if len(rows) < 2:
                        continue
                    for k in FUSERS:
                        if k == "loo":
                            P = {c: obs[c] for c in cams}
                            PV = {c: obs[c] + fld[c][f] for c in cams}
                            keep = (flow_speed.flow_consensus_cams(P, PV, calib)
                                    if len(P) >= 3 else list(P))
                            sub = [rows[i] for i, c in enumerate(cams) if c in keep]
                            v = solve_velocity(sub if len(sub) >= 2 else rows, "plain")
                        else:
                            v = solve_velocity(rows, k)
                        if v is not None:
                            tracks[k][f] = float(np.linalg.norm(v))
                for k in FUSERS:
                    s = H._lp(tracks[k])
                    m = np.isfinite(s) & np.isfinite(so) & (so > MOVING)
                    if m.sum() > 20:
                        spd[f"{mname}_{k}"]["mv"].append(float(np.median(np.abs(s[m] - so[m]))))
                    for p_ in find_peaks(so, height=300, distance=30, prominence=150)[0]:
                        pk, _ = find_peaks(s, height=200, distance=25, prominence=100)
                        if len(pk):
                            j = pk[np.argmin(np.abs(pk - p_))]
                            if abs(j - p_) <= 20:
                                spd[f"{mname}_{k}"]["peak"].append(abs(s[j] - so[p_]))
            print(f"  {part}_{trial.split('_')[1]:>10}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez(ROOT / "out" / "figures" / "flow_3d_survival.npz",
             **{f"px_{k}": np.array(v) for k, v in px_err.items()})

    print(f"\n=== (1) PER-CAMERA PIXEL ERROR -- full distribution ===")
    print(f"{'method':8} {'p25':>6} {'MED':>6} {'p75':>6} {'p90':>6} {'p95':>6} {'p99':>7} "
          f"{'max':>8} {'>2px':>7} {'>5px':>7}")
    for k in ("pyrlk", "raft"):
        m = np.array(px_err[k])
        if not len(m):
            continue
        q = np.percentile(m, [25, 50, 75, 90, 95, 99])
        print(f"{k:8} {q[0]:6.2f} {q[1]:6.2f} {q[2]:6.2f} {q[3]:6.2f} {q[4]:6.2f} {q[5]:7.2f} "
              f"{m.max():8.2f} {(m>2).mean()*100:6.1f}% {(m>5).mean()*100:6.1f}%")

    print(f"\n=== (2) 3D SPEED -- does it survive, and which FUSER? (mm/s) ===")
    print(f"{'fuser':10} {'PyrLK moving':>13} {'RAFT moving':>12} | {'PyrLK peak':>11} {'RAFT peak':>10}")
    print("-" * 64)
    for k in FUSERS:
        a1 = spd[f"pyrlk_{k}"]; a2 = spd[f"raft_{k}"]
        f_ = lambda v: np.median(v) if v else float("nan")
        print(f"{k:10} {f_(a1['mv']):12.2f} {f_(a2['mv']):11.2f} | "
              f"{f_(a1['peak']):10.2f} {f_(a2['peak']):9.2f}")
    print("\n  'plain' has breakdown point 0; huber/l1/trimmed down-weight or drop smoothly and need")
    print("  no hand-set threshold, unlike 'loo'. If they match loo, prefer them -- fewer knobs.")


if __name__ == "__main__":
    main()
