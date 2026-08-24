"""Push the in-solve BA smoother HIGHER (user: jitter 12332->5580 from sw .003->.01, go higher).
Sweep smooth_w = 0.03, 0.1, 0.3 (+ .01 reference), all DLT-backup (fallback_mm=inf), NO post-savgol.
Compare peak-vel(matched) + jitter to the BA+savgol reference (+4.80% / 1323). The in-solve smoother
WINS if some sw gives peak-vel <= savgol AND jitter ~ savgol -- i.e. it preserves the peak better than
a blind low-pass. tqdm progress; watch with:  tail -f out/gnn/ba_smoother_hi.log
"""
import sys
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T, gnn_refiner as G, ba_refine as BA
import compare_pose_omc_delta as H
JOINTS = G.JOINTS
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
    SWS = [0.03, 0.1, 0.3]   # ONLY the new higher weights (refs below are already known, not recomputed)
    variants = [f"BA insolve sw={s}" for s in SWS]
    pv = {v: [] for v in variants}; jt = {v: [] for v in variants}

    for t in tqdm(trials, mininterval=3, ncols=90, file=sys.stdout):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        for s in SWS:
            P = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=s, fallback_mm=float("inf"))[0]
            v = f"BA insolve sw={s}"
            jt[v].append(T.score_trial(P, t["omc"], t["valid"], side)["jit"])
            pv[v] += [x for x in pv_matched(H._lp(H._speed(P[:, wi])), so) if np.isfinite(x)]

    print(f"\n=== in-solve smoother HIGH sweep (peak-vel matched + jitter) ===", flush=True)
    print(f"  {'variant':20s} {'peak-vel':>9s} {'jitter':>9s}", flush=True)
    print(f"  (ref) pipeline+savgol    +6.60%      1435", flush=True)
    print(f"  (ref) BA+savgol          +4.80%      1323   <- target", flush=True)
    print(f"  (ref) insolve sw=.01    +24.79%      5580", flush=True)
    for v in variants:
        print(f"  {v:20s} {med(pv[v]):+7.2f}% {med(jt[v]):9.0f}", flush=True)
    print("\nWIN = an sw with peak-vel <= BA+savgol AND jitter ~ BA+savgol (peak-preserving).", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
