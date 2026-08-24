"""Quick focused test (user): BA in-solve smoother PLUS post-hoc savgol. Does doing BOTH beat plain
BA+savgol (+4.80% / 1323)?  Controls: BA(sw=0)+savgol [= the reference]. All fallback_mm=inf, iters=60.
Saves per-trial data + coverage. Watch: tail -f out/gnn/ba_smooth_plus.log
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
    SWS = [0, 10, 100]      # sw=0 -> plain BA+savgol (the reference control)
    labels = [f"BA(sw={s})+savgol" for s in SWS]
    jit = {l: np.full(N, np.nan) for l in labels}
    ncyc = {l: np.zeros(N, int) for l in labels}
    pv_flat = {l: [] for l in labels}
    ids = []
    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        ids.append(f"{t['part']}/{t['trial']}")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        for s in SWS:
            P = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=s, fallback_mm=float("inf"))[0]
            P = T.smooth_baseline(P, t["valid"], kind="savgol")     # + post savgol
            lab = f"BA(sw={s})+savgol"
            jit[lab][i] = T.score_trial(P, t["omc"], t["valid"], side)["jit"]
            cy = [x for x in pv_matched(H._lp(H._speed(P[:, wi])), so) if np.isfinite(x)]
            ncyc[lab][i] = len(cy); pv_flat[lab] += cy
    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_smooth_plus.npz",
             ids=np.array(ids), **{f"jit_{l}": jit[l] for l in labels},
             **{f"ncyc_{l}": ncyc[l] for l in labels},
             **{f"pv_{l}": np.array(pv_flat[l]) for l in labels})
    print(f"\n=== PROCESSING CHECK (N={N}) ===", flush=True)
    for l in labels:
        print(f"  {l:20s} jitter finite {np.isfinite(jit[l]).sum()}/{N}   "
              f">=1 cycle {(ncyc[l]>0).sum()}/{N}   cycles {ncyc[l].sum()}", flush=True)
    print(f"\n=== BA in-solve + savgol (peak-vel matched + jitter) ===", flush=True)
    for l in labels:
        print(f"  {l:20s} {med(pv_flat[l]):+7.2f}% {med(jit[l]):9.0f}", flush=True)
    print("  (sw=0 row IS the plain BA+savgol reference ~ +4.80%/1323)", flush=True)
    print("saved out/gnn/ba_smooth_plus.npz", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
