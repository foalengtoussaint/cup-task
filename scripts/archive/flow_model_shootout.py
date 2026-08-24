"""Deep dense flow (RAFT) vs PyrLK at the wrist, scored against OMC.

Every method is scored the SAME way: compute the flow at the wrist pixel in each camera, and compare
it to the TRUE flow -- triangulate the wrist in 3D, add the OMC velocity (rotation-only Kabsch, so
the translation part of the alignment cancels in a velocity), project both into that camera, take the
pixel difference. Error is then |measured - true| in px.

Reports two things, because they answer different questions:
  * MAGNITUDE error  -- how far off, regardless of direction. The usual number.
  * SIGNED BIAS      -- the error projected ONTO the true motion direction, so positive means the
                        method OVER-reads the displacement. PyrLK over-reads ~17% (motion blur
                        smears the patch, so the match runs long). A method with similar scatter but
                        LESS bias would still be better at the peak, which is what feeds
                        peak_velocity. Reported vs displacement, since bias that GROWS with
                        displacement is what actually hurts the fast frames.

    python scripts/flow_model_shootout.py --trials 1          # smoke test, one trial
    python scripts/flow_model_shootout.py --trials 2          # the usual 8-trial run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

CROP = 192          # bigger than the old 256-crop attempt that OOM'd; still memory-safe
MOVING = 200.0      # mm/s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=2, help="trials per participant")
    ap.add_argument("--models", default="large,small")
    a = ap.parse_args(argv)

    import cv2
    import torch
    import torch.nn.functional as Fn
    from torchvision.models.optical_flow import (raft_large, Raft_Large_Weights,
                                                 raft_small, Raft_Small_Weights)
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import flow_velocity_probe as F
    from pipeline.kalman_3d import triangulate_dlt

    H.use_good_cams()
    dev = "cuda"
    want = [m.strip() for m in a.models.split(",") if m.strip()]
    models = {}
    if "large" in want:
        print("loading raft_large ...", flush=True)
        models["raft_large"] = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()
    if "small" in want:
        print("loading raft_small ...", flush=True)
        models["raft_small"] = raft_small(weights=Raft_Small_Weights.DEFAULT).to(dev).eval()
    print(f"models ready: {list(models)}", flush=True)

    def raft_at(model, ga, gb, p2):
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
            fl = model(t(ca), t(cb))[-1][0].cpu().numpy().transpose(1, 2, 0)
        lx, ly = x - x0, y - y0
        if 0 <= ly < fl.shape[0] and 0 <= lx < fl.shape[1]:
            return fl[ly, lx]
        return None

    names = ["pyrlk"] + list(models)
    rec = {k: {"mag": [], "bias": [], "disp": []} for k in names}
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
            ow = omcs[f"{side}_wrist"]; so = H._speed(ow)

            px = F.load_wrist_px(part, trial, f"{side}_wrist")
            fl = {}
            for c in px:
                p = ROOT / "cache" / "flow_vel" / f"delta_{part}_{trial}.{c.split('_')[1]}__pyrlk.npy"
                if p.exists() and c in calib:
                    fl[c] = np.load(p)
            caps = {}
            for cam in fl:
                v = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{cam.split('_')[1]}.mp4"
                if v.exists():
                    caps[cam] = cv2.VideoCapture(str(v))
            if not caps:
                continue

            prev = {c: None for c in caps}
            nf = 0
            for f in range(1, n - 1):
                gr = {}
                for cam, cap in caps.items():
                    ok, im = cap.read()
                    gr[cam] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if ok else None
                if not (np.isfinite(so[f]) and so[f] > MOVING
                        and np.isfinite(ow[f]).all() and np.isfinite(ow[f + 1]).all()):
                    prev = dict(gr); continue
                op = {}
                for cam in fl:
                    if f >= len(px[cam]) or f >= len(fl[cam]):
                        continue
                    c2, v2 = px[cam][f], fl[cam][f]
                    if np.isfinite(c2).all() and np.isfinite(v2).all():
                        op[cam] = (c2, v2)
                if len(op) < 3:
                    prev = dict(gr); continue
                X = triangulate_dlt([calib[c] for c in op], [np.asarray(op[c][0]) for c in op])
                if X is None:
                    prev = dict(gr); continue
                vt = Rm @ (np.asarray(ow[f + 1]) - np.asarray(ow[f]))

                for cam, (p2, v2) in op.items():
                    ga, gb = prev.get(cam), gr.get(cam)
                    if ga is None or gb is None:
                        continue
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
                    pr = float(np.linalg.norm(true))
                    if pr < 0.5:
                        continue
                    u = true / pr
                    got = {"pyrlk": v2}
                    bad = False
                    for mname, mdl in models.items():
                        r = raft_at(mdl, ga, gb, p2)
                        if r is None:
                            bad = True; break
                        got[mname] = r
                    if bad:
                        continue
                    for k, v in got.items():
                        d = np.asarray(v) - true
                        rec[k]["mag"].append(float(np.linalg.norm(d)))
                        rec[k]["bias"].append(float(np.dot(d, u)))   # + = OVER-reads
                        rec[k]["disp"].append(pr)
                    nf += 1
                prev = dict(gr)
            for c in caps.values():
                c.release()
            print(f"  {part}_{trial.split('_')[1]:>10}  n={nf:5d}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    # SAVE the per-frame errors. The run costs minutes of GPU; printing only summary statistics
    # and discarding the raw arrays means any new question (tails, thresholds, a different split)
    # forces a full recompute.
    outnpz = ROOT / "out" / "figures" / "flow_model_shootout.npz"
    outnpz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outnpz, **{f"{k}_{fld}": np.array(rec[k][fld])
                        for k in names for fld in ("mag", "bias", "disp")})
    print(f"\nsaved per-frame errors -> {outnpz}")
    print(f"n={len(rec['pyrlk']['mag'])} camera-frames, {time.time()-t0:.0f}s total")
    # FULL distribution, not just two points: a median win can hide a worse tail, and the tail is
    # what breaks a speed estimate (one bad frame moves a peak).
    print(f"\n{'method':12} {'p25':>6} {'MED':>6} {'p75':>6} {'p90':>6} {'p95':>6} {'p99':>7} "
          f"{'max':>7} {'mean':>7}")
    print("-" * 72)
    for k in names:
        m = np.array(rec[k]["mag"])
        if not len(m):
            continue
        q = np.percentile(m, [25, 50, 75, 90, 95, 99])
        print(f"{k:12} {q[0]:6.2f} {q[1]:6.2f} {q[2]:6.2f} {q[3]:6.2f} {q[4]:6.2f} "
              f"{q[5]:7.2f} {m.max():7.2f} {m.mean():7.2f}")

    print(f"\n{'method':12} {'BIAS med':>9} {'bias mean':>10}   fraction of frames with |err| over:")
    print(f"{'':12} {'':9} {'':10}   {'1px':>7} {'2px':>7} {'5px':>7}")
    print("-" * 72)
    for k in names:
        m = np.array(rec[k]["mag"]); b = np.array(rec[k]["bias"])
        if not len(m):
            continue
        print(f"{k:12} {np.median(b):+8.2f} {np.mean(b):+9.2f}   "
              f"{(m>1).mean()*100:6.1f}% {(m>2).mean()*100:6.1f}% {(m>5).mean()*100:6.1f}%")

    # --- how does the error DEPEND on displacement? bias and magnitude, both ways ---
    from scipy.stats import spearmanr
    d = np.array(rec[names[0]]["disp"])
    print(f"\nDEPENDENCE ON DISPLACEMENT  (Spearman as well as Pearson: the relationship is")
    print(f"thresholded, not linear, so Pearson alone understates it)")
    print(f"{'method':12} {'r(disp,|err|)':>16} {'rho(disp,|err|)':>17} "
          f"{'r(disp,bias)':>14} {'rho(disp,bias)':>16}")
    print("-" * 78)
    for k in names:
        m = np.array(rec[k]["mag"]); b = np.array(rec[k]["bias"])
        if len(m) < 20:
            continue
        print(f"{k:12} {np.corrcoef(d,m)[0,1]:+15.3f} {spearmanr(d,m).statistic:+16.3f} "
              f"{np.corrcoef(d,b)[0,1]:+13.3f} {spearmanr(d,b).statistic:+15.3f}")
    print("  r(disp,|err|) > 0  = the method degrades as the target moves faster")
    print("  r(disp,bias)  < 0  = it increasingly UNDER-reads fast motion (blur runs the match long")
    print("                       the other way, so PyrLK is expected POSITIVE here)")

    print(f"\nSIGNED BIAS by displacement (+ = over-reads). Relative bias in brackets = bias/disp,")
    print(f"which is what actually distorts a speed estimate.")
    hdr = f"{'disp px':>12}" + "".join(f"{k:>20}" for k in names)
    print(hdr); print("-" * len(hdr))
    for lo, hi in [(0.5, 1.5), (1.5, 3), (3, 6), (6, 12), (12, 40)]:
        sel = (d >= lo) & (d < hi)
        if sel.sum() < 25:
            continue
        row = f"{lo:5.1f}-{hi:<6.1f}"
        for k in names:
            b = np.array(rec[k]["bias"])[sel]
            rel = np.median(b / d[sel]) * 100
            row += f"{np.median(b):+9.2f} ({rel:+4.0f}%)"
        print(row + f"  (n={sel.sum()})")


if __name__ == "__main__":
    main()
