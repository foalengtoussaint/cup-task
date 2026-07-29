"""LOPO train + evaluate the ST-GCN 3D->3D refiner (gnn_refiner.py) on the DELTA gnn_pairs cache.

Leave-one-participant-out: for each held-out participant, train on the others' clean trials, then
refine the held-out trials and score refined-vs-OMC against raw-MMC-vs-OMC on the metrics that
actually matter downstream:
    PA-MPJPE (mm)      per-frame Procrustes shape error, arm+head    -- the training objective
    wrist PA err (mm)  affected-wrist error after that alignment      -- the clinical joint
    jitter (mm/s^2)    mean |accel| of the affected wrist             -- smoothness (SmoothNet metric)
    peak-vel err (%)   |peak wrist speed refined - OMC| / OMC, low-passed, signed (neg=undershoot)

CLEAN filter: sync_corr >= 0.7 AND affected-wrist valid_frac >= 0.5 (measured: 171/279 trials).
Windows: centre-frame prediction over a (win) temporal window, stride 1; batches drawn across trials.
NO hip root (PA loss is translation/rotation invariant). Prints per-epoch val PA-MPJPE (flush) so a
long run is never a silent hang. Caches the best per-fold checkpoint under out/gnn/.

    python scripts/gnn_train.py --epochs 40 --win 31 --hidden 64 --blocks 3
    python scripts/gnn_train.py --folds P15 --epochs 40          # single fold, fast iterate
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import gnn_refiner as G          # noqa: E402
import compare_pose_omc_delta as H   # noqa: E402  (reuse _lp/_speed for the speed metrics)

CACHE = ROOT / "cache" / "delta" / "gnn_pairs"
OUT = ROOT / "out" / "gnn"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
JOINTS = G.JOINTS
ARM = [j for j in JOINTS if "hip" not in j]
ARM_I = [JOINTS.index(j) for j in ARM]


# --------------------------------------------------------------------------- data
def load_clean(parts=None, sync_thr=0.7, wr_thr=0.5, need_reproj=False):
    """Return list of trial dicts: {part, side, mmc(T,J,3), omc(T,J,3), valid(T,J)} (mm).

    need_reproj: also attach the reprojection sidecar (per-cam 2D + calib) for --loss reproj:
        uv (T,C,J,2) uv_conf (T,C,J) uv_valid (T,C,J) K (C,3,3) dist (C,5) R (C,3,3) t (C,3).
    A trial without a sidecar is SKIPPED when need_reproj (can't compute the loss for it)."""
    trials = []
    for mj in sorted(glob.glob(str(CACHE / "*" / "*.json"))):
        m = json.loads(Path(mj).read_text())
        if parts and m["part"] not in parts:
            continue
        if m["sync_corr"] < sync_thr or m["wrist_valid_frac"] < wr_thr:
            continue
        d = np.load(str(Path(mj).with_suffix(".npz")))
        rec = {"part": m["part"], "trial": m["trial"], "side": m["side"],
               "mmc": d["mmc"].astype(np.float32), "omc": d["omc"].astype(np.float32),
               "valid": d["valid"]}
        if need_reproj:
            rp = Path(mj).with_suffix(".reproj.npz")
            if not rp.exists():
                continue
            r = np.load(str(rp))
            rec["uv"] = r["uv"].astype(np.float32)
            rec["uv_conf"] = r["uv_conf"].astype(np.float32)
            rec["uv_valid"] = r["uv_valid"]
            rec["K"] = r["K"].astype(np.float32)
            rec["dist"] = r["dist"].astype(np.float32)
            rec["R"] = r["R"].astype(np.float32)
            rec["t"] = r["t"].astype(np.float32)
        trials.append(rec)
    return trials


class WindowSet(Dataset):
    """Sliding windows across trials. Item = (mmc_win, omc_win, valid_win) each (win,J,3)/(win,J).
    A window is kept only if its CENTRE frame has the arm valid (so the loss has signal).

    reproj=True: also returns per-window (uv, uv_conf, uv_valid, K, dist, R, t) padded to `max_c`
    cameras (padded cams have uv_valid=False -> 0 loss weight) so a batch mixing participants with
    different camera counts collates cleanly."""
    def __init__(self, trials, win, reproj=False, max_c=None):
        self.win = win
        self.index = []      # (trial_idx, start)
        self.trials = trials
        self.reproj = reproj
        self.max_c = max_c or (max((t["uv"].shape[1] for t in trials), default=0) if reproj else 0)
        h = win // 2
        for ti, t in enumerate(trials):
            T = t["mmc"].shape[0]
            cvalid = t["valid"][:, ARM_I].sum(1) >= 4
            for c in range(h, T - h):
                if cvalid[c]:
                    self.index.append((ti, c - h))

    def __len__(self):
        return len(self.index)

    def _pad_cams(self, arr, C, cam_axis):
        """Pad the camera axis up to self.max_c with zeros. cam_axis is EXPLICIT (the caller knows
        it) -- do NOT guess from shape: a (C,3,3) calib matrix with C==3 is ambiguous by shape and
        was the [3,5,3]-vs-[5,3,3] collate crash when a 3-cam participant (P19) joined the batch."""
        # trim if this trial already has >= max_c cams
        if C >= self.max_c:
            sl = [slice(None)] * arr.ndim
            sl[cam_axis] = slice(0, self.max_c)
            return arr[tuple(sl)]
        pad_c = self.max_c - C
        pad_shape = list(arr.shape)
        pad_shape[cam_axis] = pad_c
        return np.concatenate([arr, np.zeros(pad_shape, arr.dtype)], axis=cam_axis)

    def __getitem__(self, k):
        ti, s = self.index[k]
        t = self.trials[ti]
        e = s + self.win
        mmc = t["mmc"][s:e].copy()
        omc = t["omc"][s:e].copy()
        val = t["valid"][s:e].copy()
        # replace NaN with 0 (masked out in the loss); keeps conv finite
        mmc[~np.isfinite(mmc)] = 0.0
        omc[~np.isfinite(omc)] = 0.0
        if not self.reproj:
            return (torch.from_numpy(mmc), torch.from_numpy(omc),
                    torch.from_numpy(val.astype(np.bool_)))
        C = t["uv"].shape[1]
        # per-frame arrays: camera axis = 1 (win, C, ...).  calib arrays: camera axis = 0 (C, ...).
        uv = self._pad_cams(t["uv"][s:e].copy(), C, 1)          # (win, maxc, J, 2)
        uvc = self._pad_cams(t["uv_conf"][s:e].copy(), C, 1)    # (win, maxc, J)
        uvv = self._pad_cams(t["uv_valid"][s:e].astype(np.bool_), C, 1)
        uv[~np.isfinite(uv)] = 0.0
        K = self._pad_cams(t["K"].copy(), C, 0)                 # (maxc,3,3)
        dist = self._pad_cams(t["dist"].copy(), C, 0)           # (maxc,5)
        R = self._pad_cams(t["R"].copy(), C, 0)                 # (maxc,3,3)
        tv = self._pad_cams(t["t"].copy(), C, 0)                # (maxc,3)
        return (torch.from_numpy(mmc), torch.from_numpy(omc),
                torch.from_numpy(val.astype(np.bool_)),
                torch.from_numpy(uv), torch.from_numpy(uvc),
                torch.from_numpy(uvv), torch.from_numpy(K),
                torch.from_numpy(dist), torch.from_numpy(R), torch.from_numpy(tv))


# --------------------------------------------------------------------------- metrics (numpy, on full trials)
def _pa_align_np(pred, targ):
    """Procrustes-align pred->targ. Robust to degenerate constellations: np.linalg.svd can raise
    'SVD did not converge' on a near-rank-deficient covariance (e.g. a frame where the 6 arm joints
    are momentarily near-collinear). Regularise Hm and fall back to translation-only if SVD still
    fails, so one bad frame can't kill a whole trial's eval."""
    cP, cT = pred.mean(0), targ.mean(0)
    P0, T0 = pred - cP, targ - cT
    Hm = P0.T @ T0
    Hm = Hm + np.eye(3) * 1e-8 * (np.trace(Hm @ Hm.T) ** 0.5 + 1e-9)
    try:
        U, _, Vt = np.linalg.svd(Hm)
    except np.linalg.LinAlgError:
        return P0 + cT   # translation-only fallback (no rotation) for a pathological frame
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return P0 @ R.T + cT


def score_trial(mmc, omc, valid, side):
    """PER-TRIAL-aligned arm error, affected-wrist err, wrist jitter, peak-vel, elbow angle.

    ALIGNMENT: ONE rigid Kabsch (omc->mmc) fit over the WHOLE trial's arm constellation, applied to
    every frame. This is correct for a FIXED, calibrated rig -- the camera->lab transform is a single
    session constant, not per-frame. (An earlier version used PER-FRAME Procrustes, which re-absorbs
    placement error every frame and inflated the wrist gain ~57% vs the honest ~40%; discarded.)"""
    wi = JOINTS.index(f"{side}_wrist")
    # fit ONE transform on all valid arm joint-frames
    A, B = [], []
    for f in range(mmc.shape[0]):
        vv = valid[f, ARM_I]
        for k in range(len(ARM_I)):
            if vv[k]:
                A.append(omc[f, ARM_I[k]]); B.append(mmc[f, ARM_I[k]])
    pa, wr = [], []
    if A:
        R, t, _ = H._kabsch(np.array(A), np.array(B))      # omc -> mmc, whole trial
        omc_al = omc @ R.T + t
        for f in range(mmc.shape[0]):
            vv = valid[f, ARM_I]
            if vv.sum() < 4:
                continue
            idx = [ARM_I[k] for k in range(len(ARM_I)) if vv[k]]
            e = np.linalg.norm(mmc[f, idx] - omc_al[f, idx], axis=1)
            pa.append(np.median(e))
            if wi in idx:
                wr.append(e[idx.index(wi)])
    # jitter + peak-vel on the affected wrist (masked, NaN where invalid)
    w_mmc = mmc[:, wi].copy(); w_mmc[~valid[:, wi]] = np.nan
    w_omc = omc[:, wi].copy(); w_omc[~valid[:, wi]] = np.nan
    a = w_mmc[2:] - 2 * w_mmc[1:-1] + w_mmc[:-2]
    mm = np.isfinite(a).all(1)
    jit = float(np.mean(np.linalg.norm(a[mm], axis=1)) * H.VIDEO_FPS**2) if mm.any() else np.nan
    sm, so = H._lp(H._speed(w_mmc)), H._lp(H._speed(w_omc))
    pm, po = np.nanmax(sm), np.nanmax(so)
    pve = float((pm - po) / po * 100) if np.isfinite(po) and po > 0 else np.nan
    # ELBOW ANGLE err (deg) -- INTRINSIC (no alignment), the metric the GNN actually improves and the
    # one that feeds the Murphy elbow-extension measure. shoulder-elbow-wrist on the affected side.
    si, ei = JOINTS.index(f"{side}_shoulder"), JOINTS.index(f"{side}_elbow")
    def elbow_ang(P):
        u = P[:, si] - P[:, ei]; w = P[:, wi] - P[:, ei]
        c = (u * w).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    va = valid[:, si] & valid[:, ei] & valid[:, wi]
    am, ao = elbow_ang(mmc), elbow_ang(omc)
    ang_err = float(np.nanmedian(np.abs(am - ao)[va])) if va.any() else np.nan
    # SHOULDER flexion + abduction (Murphy shoulder_flexion/abduction measures). Reuse the harness
    # raw-point proxies (H._murphy_signals: side-aware, uses shoulders+hips for the trunk axis).
    def as_dict(P):
        return {j: P[:, k] for k, j in enumerate(JOINTS)}
    try:
        sm_mmc = H._murphy_signals(as_dict(mmc), side=side)
        sm_omc = H._murphy_signals(as_dict(omc), side=side)
        def sig_err(name):
            a, o = H._lp(sm_mmc[name]), H._lp(sm_omc[name])
            mk = np.isfinite(a) & np.isfinite(o)
            # offset-removed |Δ| (angle proxies have a convention offset; the SHAPE is what matters)
            if mk.sum() < 20:
                return np.nan
            off = np.median(a[mk] - o[mk])
            return float(np.median(np.abs((a[mk] - off) - o[mk])))
        flex_err = sig_err("shoulder_flexion")
        abd_err = sig_err("shoulder_abduction")
    except Exception:
        flex_err = abd_err = np.nan
    return {"pa": np.median(pa) if pa else np.nan,
            "wr": np.median(wr) if wr else np.nan,
            "jit": jit, "pve": pve, "elb": ang_err,
            "sflex": flex_err, "sabd": abd_err}


def smooth_baseline(mmc, valid, kind="butter"):
    """Per-joint, per-axis low-pass baseline (what a GNN must BEAT to be worth training).
    butter = the harness 6Hz Butterworth (H._lp); savgol = window-21 order-3. Position-domain,
    NaN-gap-safe, restores per-joint NaN. These CANNOT reduce PA-MPJPE shape error -- only jitter/
    peak -- which is exactly the point: if the GNN only matches them on speed it added nothing."""
    from scipy.signal import savgol_filter
    out = mmc.copy()
    T, J, _ = mmc.shape
    for j in range(J):
        col = mmc[:, j, :]
        v = np.isfinite(col).all(1)
        if v.sum() < 8:
            continue
        idx = np.flatnonzero(v)
        for ax in range(3):
            xi = np.interp(np.arange(T), idx, col[idx, ax])
            if kind == "savgol":
                out[:, j, ax] = savgol_filter(xi, 21, 3, mode="interp")
            else:
                out[:, j, ax] = H._lp(xi)
        out[~v, j, :] = np.nan
    return out


def refine_trial(model, mmc, valid, win, chunk=512):
    """Slide the model over a full trial, average overlaps -> refined (T,J,3).

    BATCHED: all (T-win+1) windows are stacked and run through the model in chunks of `chunk`, not
    one forward per window. On a ~400-frame trial that's 1 (or 2) GPU calls instead of ~370, which is
    where the eval wall-clock went (the model is 12k params -- launch-bound, not compute-bound)."""
    T = mmc.shape[0]
    x = mmc.copy(); x[~np.isfinite(x)] = 0.0
    acc = np.zeros_like(x); cnt = np.zeros((T, 1, 1))
    model.eval()
    nwin = T - win + 1
    if nwin <= 0:
        out = mmc.copy(); out[~valid] = np.nan; return out
    starts = np.arange(nwin)
    # (nwin, win, J, 3) view via stride tricks -> contiguous tensor
    windows = np.stack([x[s:s + win] for s in starts])   # cheap: J*3 small
    with torch.no_grad():
        for c0 in range(0, nwin, chunk):
            wb = torch.from_numpy(windows[c0:c0 + chunk]).to(DEV)   # (b,win,J,3)
            yb = model(wb).cpu().numpy()                            # (b,win,J,3)
            for k, s in enumerate(starts[c0:c0 + chunk]):
                acc[s:s + win] += yb[k]
                cnt[s:s + win, 0, 0] += 1
    out = acc / np.maximum(cnt, 1)
    out[cnt[:, 0, 0] == 0] = mmc[cnt[:, 0, 0] == 0]     # untouched edges = raw
    out[~valid] = np.nan
    return out


# --------------------------------------------------------------------------- train one fold
def train_fold(held, all_trials, args):
    train = [t for t in all_trials if t["part"] != held]
    test = [t for t in all_trials if t["part"] == held]
    if not test:
        return None
    use_reproj = (args.loss == "reproj")
    # pad to the GLOBAL max cam count so a fold trained on 5-cam parts + tested on 3-cam is consistent
    max_c = max((t["uv"].shape[1] for t in all_trials), default=0) if use_reproj else 0
    ds = WindowSet(train, args.win, reproj=use_reproj, max_c=max_c)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=(DEV == "cuda"))
    model = G.GNNRefiner(hidden=args.hidden, blocks=args.blocks,
                         t_kernel=args.t_kernel, dropout=args.dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    jw = torch.ones(G.NJ)
    if args.jw_scheme == "angle":
        # Focus on the joints that feed the in-scope Murphy ANGLE measures (elbow_extension,
        # shoulder_flexion/abduction). Elbow+shoulder appear in every angle measure and were the
        # biggest raw errors; wrist is KEPT (it defines the elbow angle shoulder-elbow-wrist, so a
        # bad wrist corrupts it); nose feeds NOTHING (and degraded); hips only the coarse trunk axis.
        for j in ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]:
            jw[G.JOINTS.index(j)] = 3.0
        for j in ["left_wrist", "right_wrist"]:
            jw[G.JOINTS.index(j)] = 1.5
        # HIPS are load-bearing, NOT junk: shoulder-mid -> hip-mid is the TRUNK VERTICAL AXIS that
        # shoulder_flexion/abduction are measured RELATIVE TO (the "Z" reference), and it drives
        # trunk_y_displacement. A wrong hip => wrong shoulder angles even with a perfect arm. Keep them.
        for j in ["left_hip", "right_hip"]:
            jw[G.JOINTS.index(j)] = 2.0
        jw[G.JOINTS.index("nose")] = 0.25   # nose feeds NO in-scope measure (and it degraded)
    else:
        # legacy: up-weight both wrists
        jw[G.wrist_idx("right")] = args.wrist_w
        jw[G.wrist_idx("left")] = args.wrist_w

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = nb = 0.0
        for batch in dl:
            if use_reproj:
                mmc, omc, val, uv, uvc, uvv, Ks, dists, Rs, ts = [b.to(DEV) for b in batch]
            else:
                mmc, omc, val = [b.to(DEV) for b in batch]
            pred = model(mmc)
            B, Tw, J, _ = pred.shape
            if args.loss == "reproj":
                # SELF-SUPERVISED: pull refined 3D toward the per-cam 2D dets (NO omc, NO alignment).
                # Huber-robust so a disagreeing cam can't yank it; smooth_w buys a cleaner trajectory
                # (the jitter fix). Where the good cams agree the raw is already on the rays -> ~0 loss
                # -> good frames left alone (the self-gating the OMC losses lacked).
                loss = G.reprojection_loss(pred, uv, uvc, uvv, Ks, dists, Rs, ts,
                                           smooth_w=args.smooth_w, mask=val, joint_weights=jw,
                                           huber_px=args.huber_px)
            elif args.loss == "relational":
                # impairment-agnostic 'arm as coupled linkage': supervise RELATIVE bone vectors +
                # bone-length only, NO absolute pose -> can't learn a healthy prior (user's idea).
                loss = G.relational_loss(pred, omc, val, bone_w=args.bone_w)
            elif args.loss == "window":
                # ONE alignment per window -> offset-invariant BUT temporally coherent (user's point:
                # per-frame PA has no temporal notion; a shared-window rotation penalises jitter).
                loss = G.window_aligned_loss(pred, omc, val, joint_weights=jw)
            elif args.loss == "velocity":
                # offset-invariant VELOCITY loss (see gnn_refiner.velocity_loss). Works on the whole
                # window so the temporal conv sees the trajectory. Peak term targets Murphy peak-vel.
                wi = G.wrist_idx("right")   # both wrists up-weighted via jw; peak term on the R idx
                loss = G.velocity_loss(pred, omc, val, joint_weights=jw,
                                       peak_w=args.peak_w, wrist_i=wi, smooth_w=args.smooth_w)
            else:
                # PA-MPJPE (position shape). Trains on all frames of the window for more signal.
                loss = G.pa_mpjpe_loss(pred.reshape(B * Tw, J, 3),
                                       omc.reshape(B * Tw, J, 3),
                                       val.reshape(B * Tw, J), joint_weights=jw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        do_val = ep % args.log_every == 0 or ep == args.epochs - 1
        if not do_val:
            # lightweight per-epoch line so the log is never silent for minutes (no expensive val eval)
            print(f"  [{held}] ep{ep:3d} train_loss {tot/max(nb,1):6.2f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        else:
            # quick val PA on held-out (raw vs refined), median over trials
            vr = [score_trial(t["mmc"], t["omc"], t["valid"], t["side"]) for t in test]
            rr = [score_trial(refine_trial(model, t["mmc"], t["valid"], args.win),
                              t["omc"], t["valid"], t["side"]) for t in test]
            def med(rows, k):
                v=[r[k] for r in rows if np.isfinite(r[k])]; return np.median(v) if v else float('nan')
            print(f"  [{held}] ep{ep:3d} train_loss {tot/max(nb,1):6.2f}  "
                  f"val PA {med(vr,'pa'):.1f}->{med(rr,'pa'):.1f}  "
                  f"wr {med(vr,'wr'):.1f}->{med(rr,'wr'):.1f}  "
                  f"elb {med(vr,'elb'):.1f}->{med(rr,'elb'):.1f}  "
                  f"jit {med(vr,'jit'):.0f}->{med(rr,'jit'):.0f}  "
                  f"pve {med(vr,'pve'):+.0f}->{med(rr,'pve'):+.0f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # final eval: raw, butter, savgol, GNN, GNN+savgol -- all vs OMC, same metrics.
    # GNN+savgol tests the complementarity claim: GNN fixes SHAPE (PA-MPJPE), which smoothers
    # cannot; savgol then removes the JITTER the GNN leaves untouched. Best-of-both if it holds.
    rows = []
    for t in test:
        raw = score_trial(t["mmc"], t["omc"], t["valid"], t["side"])
        but = score_trial(smooth_baseline(t["mmc"], t["valid"], "butter"),
                          t["omc"], t["valid"], t["side"])
        sav = score_trial(smooth_baseline(t["mmc"], t["valid"], "savgol"),
                          t["omc"], t["valid"], t["side"])
        gnn = refine_trial(model, t["mmc"], t["valid"], args.win)
        ref = score_trial(gnn, t["omc"], t["valid"], t["side"])
        gns = score_trial(smooth_baseline(gnn, t["valid"], "savgol"),
                          t["omc"], t["valid"], t["side"])
        rows.append((t["trial"], raw, ref, but, sav, gns))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / f"gnn_{held}.pt")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", nargs="+", default=None, help="held-out parts (default: all present)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--win", type=int, default=31)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--t-kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wrist-w", type=float, default=3.0)
    ap.add_argument("--jw-scheme", choices=["wrist", "angle"], default="wrist",
                    help="wrist = up-weight wrists (legacy); angle = focus elbow+shoulder (the joints "
                         "feeding the in-scope Murphy angle measures), keep wrist, downweight nose/hips")
    ap.add_argument("--bone-w", type=float, default=0.3, help="bone-length term weight (relational loss)")
    ap.add_argument("--loss", choices=["pampjpe", "velocity", "window", "relational", "reproj"],
                    default="pampjpe",
                    help="pampjpe = position shape (overfits rig geometry); "
                         "velocity = offset-invariant speed-domain loss; "
                         "reproj = SELF-SUPERVISED reprojection to per-cam 2D (no OMC, self-gating) "
                         "+ smooth_w jerk penalty -- the less-biased signal")
    ap.add_argument("--huber-px", type=float, default=20.0,
                    help="reproj loss Huber threshold (px): a cam past this is capped so a "
                         "disagreeing/other-object 2D can't yank the 3D (soft consensus gate)")
    ap.add_argument("--peak-w", type=float, default=0.5,
                    help="weight of the peak-wrist-speed term (velocity loss only)")
    ap.add_argument("--smooth-w", type=float, default=1.0,
                    help="weight of the acceleration/smoothness prior (velocity loss only) -- the "
                         "DENOISING incentive; 0 = the plain loss that parked at identity")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--sync-thr", type=float, default=0.7)
    ap.add_argument("--wr-thr", type=float, default=0.5)
    ap.add_argument("--only-parts", nargs="+", default=None,
                    help="restrict BOTH training and folds to these participants (e.g. the true "
                         "5-cam consistent rigs P07 P08 P15) -- tests transfer within one geometry")
    a = ap.parse_args(argv)

    print(f"device={DEV}  loading clean pairs...", flush=True)
    trials = load_clean(sync_thr=a.sync_thr, wr_thr=a.wr_thr, need_reproj=(a.loss == "reproj"))
    if a.only_parts:
        trials = [t for t in trials if t["part"] in a.only_parts]
        print(f"restricted to {a.only_parts}", flush=True)
    parts = sorted({t["part"] for t in trials})
    from collections import Counter
    print(f"clean trials: {len(trials)}  by part: {dict(Counter(t['part'] for t in trials))}", flush=True)
    folds = a.folds or parts

    all_rows = []
    for held in folds:
        print(f"\n=== FOLD hold-out {held} (train on {[p for p in parts if p!=held]}) ===", flush=True)
        rows = train_fold(held, trials, a)
        if rows:
            all_rows.extend([(held,) + r for r in rows])

    # overall summary. tuple layout: (held, trial, raw, ref, but, sav, gns)
    #   slot map: raw=2  GNN=3  butter=4  savgol=5  GNN+savgol=6
    def col(slot, k, absol=False):
        v = [(abs(x[slot][k]) if absol else x[slot][k]) for x in all_rows
             if np.isfinite(x[slot][k])]
        return np.median(v) if v else float("nan")
    print(f"\n{'='*82}\nOVERALL LOPO ({len(all_rows)} trials)  medians\n{'='*82}", flush=True)
    print(f"  {'metric':12}{'raw':>9}{'butter':>9}{'savgol':>9}{'GNN':>9}{'GNN+sav':>10}", flush=True)
    for lab, k, ab in [("PA-MPJPE mm", "pa", False), ("wrist PA mm", "wr", False),
                       ("elbow ang deg", "elb", False),
                       ("sh flex deg", "sflex", False), ("sh abd deg", "sabd", False),
                       ("jitter", "jit", False), ("|peakvel| %", "pve", True)]:
        one_dp = k in ("pa", "wr", "elb", "sflex", "sabd")
        fmt = "9.1f" if one_dp else "9.0f"
        print(f"  {lab:12}"
              f"{col(2,k,ab):{fmt}}{col(4,k,ab):{fmt}}{col(5,k,ab):{fmt}}{col(3,k,ab):{fmt}}"
              f"{col(6,k,ab):{'10.1f' if one_dp else '10.0f'}}", flush=True)
    # per-fold PA-MPJPE + wrist (raw -> GNN)
    print("\n  per-fold PA-MPJPE / wrist PA (raw -> GNN):", flush=True)
    for held in folds:
        fr = [(x[2], x[3]) for x in all_rows if x[0] == held]   # (raw, gnn)
        if not fr: continue
        def m(rows, sel, k):
            v=[r[sel][k] for r in rows if np.isfinite(r[sel][k])]; return np.median(v) if v else float('nan')
        print(f"    {held}: PA {m(fr,0,'pa'):.1f}->{m(fr,1,'pa'):.1f}   "
              f"wrist {m(fr,0,'wr'):.1f}->{m(fr,1,'wr'):.1f}  (n={len(fr)})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lopo_results.json").write_text(json.dumps(
        [{"held": x[0], "trial": x[1], "raw": x[2], "refined": x[3],
          "butter": x[4], "savgol": x[5], "gnn_savgol": x[6]} for x in all_rows], indent=1))
    print(f"\nwrote {OUT/'lopo_results.json'}", flush=True)


if __name__ == "__main__":
    main()
