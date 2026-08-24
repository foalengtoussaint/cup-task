"""Two questions in one run (3 BA solves, tqdm, backgrounded):

1. PER-PARTICIPANT wrist + jitter: pipeline vs BA+backup(savgol). Is BA ONLY a high-error rescue
   (like peak-vel, concentrated in P17/P19) or a broad small position gain everywhere?

2. THE BA SMOOTHER: in-solve smoothness (smooth_w inside the L-BFGS) vs post-hoc savgol. In-solve
   smoothness competes against the reprojection DATA term -> smooths only where cameras are weak,
   PRESERVES the real peak where they agree (the property a blind savgol lacks). All with DLT-backup
   (fallback_mm=inf) for stability. Compare peak-vel (per-cycle matched) + jitter:
      pipeline | BA+savgol | BA+insolve(sw=.003) | BA+insolve(sw=.01)
"""
import sys, time
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T, gnn_refiner as G, ba_refine as BA
import compare_pose_omc_delta as H
JOINTS = G.JOINTS; FPS = H.VIDEO_FPS
med = lambda a: float(np.nanmedian([z for z in a if np.isfinite(z)]))


def pv_matched(sm, so, win=15):
    if np.isfinite(so).sum() < 20:
        return []
    pk, _ = find_peaks(np.nan_to_num(so, nan=0.0), prominence=np.nanmax(so) * 0.25, distance=win)
    out = []
    for p in pk:
        if not np.isfinite(so[p]) or so[p] <= 0:
            continue
        seg = sm[max(0, p-win):min(len(sm), p+win+1)]
        pm = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
        out.append((pm - so[p]) / so[p] * 100.0 if np.isfinite(pm) else np.nan)
    return out


def main():
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in ("P07","P08","P15","P17","P19")]
    print(f"trials: {len(trials)}", flush=True)
    parts = sorted(set(t["part"] for t in trials))

    # per-trial storage
    rows = {p: {} for p in parts}     # wrist/jit by participant, per variant
    pv = {}; jt = {}                  # overall peak-vel list + jitter list, per variant
    VARS = ["pipeline", "BA+savgol", "BA+insolve.003", "BA+insolve.01"]
    for v in VARS:
        pv[v] = []; jt[v] = []
        for p in parts:
            rows[p][v] = {"wr": [], "jit": []}

    for t in tqdm(trials, mininterval=3, ncols=90, file=sys.stdout):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist"); pp = t["part"]
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        # 3 solves (reuse solve1 for both pipeline-comparison and BA+savgol)
        ba_plain, _ = BA.refine_trial_ba(t, 0.0, iters=60, fallback_mm=float("inf"))
        ba_s3, _ = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=0.003, fallback_mm=float("inf"))
        ba_s1, _ = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=0.01, fallback_mm=float("inf"))
        variants = {
            "pipeline":        T.smooth_baseline(t["mmc"], t["valid"], kind="savgol"),
            "BA+savgol":       T.smooth_baseline(ba_plain, t["valid"], kind="savgol"),
            "BA+insolve.003":  ba_s3,      # in-solve smoothed -> NO post savgol
            "BA+insolve.01":   ba_s1,
        }
        for v, P in variants.items():
            sc = T.score_trial(P, t["omc"], t["valid"], side)
            rows[pp][v]["wr"].append(sc["wr"]); rows[pp][v]["jit"].append(sc["jit"])
            jt[v].append(sc["jit"])
            sm = H._lp(H._speed(P[:, wi]))
            pv[v] += [x for x in pv_matched(sm, so) if np.isfinite(x)]

    print("\n=== Q1: PER-PARTICIPANT wrist(mm) / jitter — pipeline vs BA+savgol ===", flush=True)
    print(f"  {'part':6s} {'wrist pipe':>10s} {'wrist BA':>9s}   {'jit pipe':>9s} {'jit BA':>9s}", flush=True)
    for p in parts:
        wp = med(rows[p]["pipeline"]["wr"]); wb = med(rows[p]["BA+savgol"]["wr"])
        jp = med(rows[p]["pipeline"]["jit"]); jb = med(rows[p]["BA+savgol"]["jit"])
        print(f"  {p:6s} {wp:10.1f} {wb:9.1f}   {jp:9.0f} {jb:9.0f}"
              f"   [{'wr+' if wb<wp else 'wr='}{'jit+' if jb<jp else 'jit='}]", flush=True)

    print("\n=== Q2: THE BA SMOOTHER — peak-vel(matched) + jitter, all DLT-backup ===", flush=True)
    print(f"  {'variant':16s} {'peak-vel':>9s} {'jitter':>9s}", flush=True)
    for v in VARS:
        print(f"  {v:16s} {med(pv[v]):+7.2f}% {med(jt[v]):9.0f}", flush=True)
    print("\nBA+insolve wins if peak-vel <= BA+savgol AND jitter comparable (peak-preserving smoother).",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
