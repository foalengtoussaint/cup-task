"""Bundle-adjust the CUP after greedy consensus, and bake it off against consensus alone.

The shipped cup 3D is consensus.consensus3: a LINEAR DLT over the best-agreeing camera subset, with
a hard inlier cut and a velocity gate. Two things BA can add that the linear solve cannot:
  * lens DISTORTION -- consensus triangulates from raw pixels; project_torch models k1,k2,k3,p1,p2.
    The cup spends the transport phase near frame edges, where distortion is largest.
  * SOFT rejection -- Huber down-weights a slightly-off camera instead of discarding it, which
    matters because the >=3-camera floor exists precisely because discarding leaves too few.

There is no bone prior here (one point, no neighbours), so this is purely robust nonlinear
triangulation -- BA's usual depth-ray constraint via a neighbouring joint does not apply.

Variants (all initialised from the consensus point, or from its interpolation where it is missing):
  consensus : shipped, unchanged
  ba_kept   : Huber + distortion, restricted to the cameras consensus kept
  ba_all    : Huber + distortion over EVERY camera with a tracker point -- Huber does the rejecting.
              This is the one that can recover frames consensus dropped, and also the one that can
              re-admit a static distractor the velocity gate was there to reject. Both show up in
              the coverage/error split below.

Truth = the lag-aligned OMC cup already cached in seg_inputs (cup_omc), so no re-alignment here.

    python scripts/cup_ba.py --limit 0        -> cache/cup_ba/<variant>.npz + out/scoring/cup_ba.csv
    tail -f out/scoring/cup_ba.log
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H       # noqa: E402
import results_v3_delta as R             # noqa: E402
import ba_refine as BA                   # noqa: E402
from pipeline import consensus           # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEG = ROOT / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs_final")
TRACKS = ROOT / "cache" / (__import__("os").environ.get("OT_TRACKS_DIR") or "tracks_uetrack_26x")
OUT = ROOT / "cache" / "cup_ba"


def _project_np(X, cam):
    """Pinhole + full Brown-Conrady, numpy. Same model as gnn_refiner.project_torch."""
    Xc = (np.asarray(cam.R, float) @ X.T).T + np.asarray(cam.t, float).ravel()
    z = np.clip(Xc[:, 2], 1e-3, None)
    xn, yn = Xc[:, 0] / z, Xc[:, 1] / z
    d = np.asarray(cam.dist, float).ravel()
    k1, k2, p1, p2, k3 = (list(d) + [0.0] * 5)[:5]
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    K = np.asarray(cam.K, float)
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)


def _interp_gaps(X):
    """Linear-interpolate NaN rows so BA has a starting point on frames consensus dropped."""
    X = np.asarray(X, float).copy()
    good = np.isfinite(X).all(1)
    if good.sum() < 2:
        return X, good
    idx = np.arange(len(X))
    for c in range(3):
        X[:, c] = np.interp(idx, idx[good], X[good, c])
    return X, good


def solve(uv, valid, cams, calib, X0, huber_px=20.0, iters=60):
    """Per-frame robust nonlinear triangulation of ONE point. uv (T,C,2), valid (T,C)."""
    T, C = valid.shape
    S = dict(
        uv=torch.from_numpy(np.nan_to_num(uv)[:, :, None, :].astype(np.float32)).to(DEV),
        uvc=torch.ones((T, C, 1), dtype=torch.float32, device=DEV),
        uvv=torch.from_numpy(valid[:, :, None]).to(DEV),
        K=torch.from_numpy(np.stack([calib[c].K for c in cams]).astype(np.float32)).to(DEV),
        dist=torch.from_numpy(np.stack([np.asarray(calib[c].dist).ravel()[:5]
                                        for c in cams]).astype(np.float32)).to(DEV),
        R=torch.from_numpy(np.stack([calib[c].R for c in cams]).astype(np.float32)).to(DEV),
        tt=torch.from_numpy(np.stack([np.asarray(calib[c].t).ravel()
                                      for c in cams]).astype(np.float32)).to(DEV),
    )
    X = torch.from_numpy(X0[None, :, None, :].astype(np.float32)).to(DEV).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([X], lr=1.0, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e, w = BA._reproj_residual(X, S, huber_px)
        loss = e / w.clamp(min=1.0)
        loss.backward()
        return loss

    opt.step(closure)
    return X.detach()[0, :, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--huber-px", type=float, default=20.0)
    a = ap.parse_args()
    H.use_good_cams()
    files = sorted(SEG.glob("*.npz"))
    if a.limit:
        files = files[::max(1, len(files) // a.limit)][:a.limit]
    print(f"cup BA bake-off: {len(files)} trials  (seg cache {SEG.name}, tracks {TRACKS.name})",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, store = [], {v: {} for v in ("consensus", "ba_kept", "ba_all")}
    t0 = time.time()
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        part, trial = str(z["part"]), str(z["trial"])
        cup_omc = np.asarray(z["cup_omc"], float)
        n = cup_omc.shape[0]
        cf = TRACKS / f"{part}__{trial}__uetrack__fs1.json"
        if not cf.exists():
            continue
        calib = R._calib(part)
        rec = json.loads(cf.read_text())
        cams = sorted(calib)
        uv = np.full((n, len(cams), 2), np.nan)
        for fr in range(n):
            row = rec.get(str(fr)) or {}
            for ci, c in enumerate(cams):
                v = row.get(c)
                if v and v.get("trk") is not None:
                    uv[fr, ci] = v["trk"]
        have = np.isfinite(uv).all(-1)                       # (T,C) tracker point present
        # shipped consensus + which cameras it kept
        tr = R.cup_track.track_cup_3d_from_cache(cf, calib) if hasattr(R, "cup_track") else None
        if tr is None:
            from pipeline import cup_track as CT
            tr = CT.track_cup_3d_from_cache(cf, calib)
        Xc = np.full((n, 3), np.nan); kept = np.zeros((n, len(cams)), bool)
        for t_ in tr:
            fr = t_["frame"]
            if fr >= n:
                continue
            if t_["X"] is not None:
                Xc[fr] = t_["X"]
            for c in t_.get("kept", []):
                if c in cams:
                    kept[fr, cams.index(c)] = True
        X0, _ = _interp_gaps(Xc)
        if not np.isfinite(X0).all():
            continue
        Xk = solve(uv, kept & have, cams, calib, X0, a.huber_px)
        Xa = solve(uv, have, cams, calib, X0, a.huber_px)
        # a frame is only CLAIMED where there was evidence: consensus keeps its own mask; ba_kept
        # keeps consensus's frames; ba_all claims any frame with >=2 tracker points.
        okc = np.isfinite(Xc).all(1)
        Xk_ = np.where(okc[:, None], Xk, np.nan)
        Xa_ = np.where((have.sum(1) >= 2)[:, None], Xa, np.nan)
        for nm, X in (("consensus", Xc), ("ba_kept", Xk_), ("ba_all", Xa_)):
            store[nm][f"{part}/{trial}"] = X.astype(np.float32)
            # NOTE: MMC and OMC live in DIFFERENT WORLD FRAMES, so a direct |X - cup_omc| is
            # meaningless (it reads ~1.6 m, which is the frame offset, not error). The segmenter only
            # ever uses frame-INVARIANT quantities, so score those:
            #   speed_r  = correlation of the cup SPEED profile with the OMC cup -- accuracy
            #   reproj   = median per-camera reprojection residual (px) -- fit quality, but this is
            #              BA's own objective, so it is diagnostic, not evidence of accuracy
            sp_m, sp_o = H._speed(X), H._speed(cup_omc)
            mm_ = np.isfinite(sp_m) & np.isfinite(sp_o)
            speed_r = float(np.corrcoef(H._lp(sp_m)[mm_], H._lp(sp_o)[mm_])[0, 1]) \
                if mm_.sum() > 40 else np.nan
            rp = []
            fin = np.isfinite(X).all(1)
            for ci, c_ in enumerate(cams):
                sel = fin & have[:, ci]
                if not sel.any():
                    continue
                uvp = _project_np(X[sel], calib[c_])
                rp.append(np.linalg.norm(uvp - uv[sel, ci], axis=1))
            reproj = float(np.median(np.concatenate(rp))) if rp else np.nan
            rows.append(dict(part=part, trial=trial, variant=nm,
                             cover=float(fin.mean()), n_scored=int(mm_.sum()),
                             speed_r=speed_r, reproj=reproj))
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            d = pd.DataFrame(rows); el = time.time() - t0
            g = d.groupby("variant")
            msg = "  ".join(f"{k} r {v.speed_r.median():.3f} reproj {v.reproj.median():.1f}px"
                            for k, v in g)
            print(f"  [{i+1}/{len(files)}] {el:5.0f}s ({el/(i+1):.2f}s/trial)  {msg}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "out/scoring/cup_ba.csv", index=False)
    for nm, dd in store.items():
        np.savez(OUT / f"{nm}.npz", ids=np.array(list(dd)),
                 traj=np.array(list(dd.values()), dtype=object))
    print(f"\nPROCESSING CHECK: {d.trial.nunique()} trials, {len(d)} rows, "
          f"non-finite speed_r {int(d.speed_r.isna().sum())}, reproj {int(d.reproj.isna().sum())}",
          flush=True)
    print(f"\n{'variant':11s} {'n':>5s} {'cover%':>7s} {'speed_r':>9s} {'reproj px':>10s}")
    for k, g in d.groupby("variant"):
        print(f"{k:11s} {len(g):5d} {100*g.cover.mean():7.2f} {g.speed_r.median():9.3f} "
              f"{g.reproj.median():10.2f}")
    print("\nPAIRED vs consensus (same trials):")
    for col, unit, better in (("speed_r", "", "higher"), ("reproj", " px", "lower")):
        p = d.pivot_table(index=["part", "trial"], columns="variant", values=col)
        print(f"  -- {col} ({better} is better)")
        for v in ("ba_kept", "ba_all"):
            q = p[["consensus", v]].dropna()
            win = (q[v] > q.consensus) if better == "higher" else (q[v] < q.consensus)
            print(f"     {v:9s} n={len(q):4d}  better {int(win.sum()):4d}  worse "
                  f"{int((~win & (q[v] != q.consensus)).sum()):4d}  "
                  f"median delta {float((q[v] - q.consensus).median()):+.4f}{unit}")
    print("DONE_CUPBA", flush=True)


if __name__ == "__main__":
    main()
