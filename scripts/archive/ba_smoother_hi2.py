"""In-solve BA smoother at EVEN HIGHER weights (1.0, 3.0, 10.0). Also: (a) verify EVERY trial is
processed (count processed / zero-cycle / non-finite jitter), and (b) SAVE per-trial data to npz
(not just the final median) so it's kept and re-analyzable.

Watch:  tail -f out/gnn/ba_smoother_hi2.log
Data:   out/gnn/ba_smoother_hi2.npz  (per-trial jitter arrays + per-trial cycle-count + flat cycle pv)
Target to beat: BA+savgol +4.80% / jitter 1323.
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
    N = len(trials)
    print(f"trials: {N}", flush=True)
    SWS = [1.0, 3.0, 10.0]

    # per-trial storage (kept): jitter[sw] (N,), n_cycles[sw] (N,), and flat cycle pv lists
    jit = {s: np.full(N, np.nan) for s in SWS}
    ncyc = {s: np.zeros(N, int) for s in SWS}
    pv_flat = {s: [] for s in SWS}
    n_bad_jit = {s: 0 for s in SWS}      # trials whose jitter came back non-finite
    ids = []

    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        ids.append(f"{t['part']}/{t['trial']}")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        for s in SWS:
            P = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=s, fallback_mm=float("inf"))[0]
            j = T.score_trial(P, t["omc"], t["valid"], side)["jit"]
            jit[s][i] = j
            if not np.isfinite(j):
                n_bad_jit[s] += 1
            cy = [x for x in pv_matched(H._lp(H._speed(P[:, wi])), so) if np.isfinite(x)]
            ncyc[s][i] = len(cy); pv_flat[s] += cy

    # SAVE the per-trial data
    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_smoother_hi2.npz",
             ids=np.array(ids), sws=np.array(SWS),
             **{f"jit_{s}": jit[s] for s in SWS},
             **{f"ncyc_{s}": ncyc[s] for s in SWS},
             **{f"pv_{s}": np.array(pv_flat[s]) for s in SWS})

    print(f"\n=== PROCESSING CHECK (N={N} trials) ===", flush=True)
    for s in SWS:
        print(f"  sw={s:<5g}  jitter finite on {np.isfinite(jit[s]).sum()}/{N} trials "
              f"({n_bad_jit[s]} non-finite)   trials with >=1 cycle: {(ncyc[s]>0).sum()}/{N}   "
              f"total cycles: {ncyc[s].sum()}", flush=True)

    print(f"\n=== HIGH-HIGH sweep (peak-vel matched + jitter) ===", flush=True)
    print(f"  (ref) BA+savgol          +4.80%      1323   <- target", flush=True)
    for s in SWS:
        print(f"  BA insolve sw={s:<5g}   {med(pv_flat[s]):+7.2f}% {med(jit[s]):9.0f}", flush=True)
    print("\nsaved out/gnn/ba_smoother_hi2.npz (per-trial jitter + cycle counts + all cycle pv)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
