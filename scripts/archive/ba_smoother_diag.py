"""DIAGNOSE why the in-solve smoother floors at jitter ~3000 despite the smoothness term already
dominating the objective at sw>=10. Separate WEIGHT from CONVERGENCE from FALLBACK:

  sw=100  iters=60           baseline-ish bigger weight
  sw=100  iters=300          CONVERGENCE test (5x iters, same weight) -> if jitter craters, it was
                             under-convergence, NOT weight
  sw=1000 iters=60           EXTREME weight, few iters -> if this doesn't help, weight isn't it
  sw=100  iters=300 NO-fb    isolate the DLT-backup: does reverting NaN->pipeline re-inject jitter?
                             (jitter measured only on trials that stayed fully finite)

Saves per-trial data (out/gnn/ba_smoother_diag.npz) + PROCESSING CHECK. Watch:
  tail -f out/gnn/ba_smoother_diag.log
Target: BA+savgol +4.80% / 1323.
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
    # solves (iters=150 = 2.5x baseline, enough to test convergence; it300 x3 was ~40min):
    #   A sw100 it60 fb ; B sw100 it150 fb (reused for +SAVGOL) ; C sw1000 it60 fb ; D sw100 it150 NO-fb
    LABELS = ["sw100_it60", "sw100_it150", "sw100_it150_SAVGOL", "sw1000_it60", "sw100_it150_nofb"]
    jit = {l: np.full(N, np.nan) for l in LABELS}
    ncyc = {l: np.zeros(N, int) for l in LABELS}
    pv_flat = {l: [] for l in LABELS}
    ids = []

    def rec(lab, P, t, so, wi):
        jit[lab][i] = T.score_trial(P, t["omc"], t["valid"], t["side"])["jit"]
        cy = [x for x in pv_matched(H._lp(H._speed(P[:, wi])), so) if np.isfinite(x)]
        ncyc[lab][i] = len(cy); pv_flat[lab] += cy

    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        ids.append(f"{t['part']}/{t['trial']}")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        A = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=100, fallback_mm=float("inf"))[0]
        B = BA.refine_trial_ba(t, 0.0, iters=150, smooth_w=100, fallback_mm=float("inf"))[0]
        C = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=1000, fallback_mm=float("inf"))[0]
        D = BA.refine_trial_ba(t, 0.0, iters=150, smooth_w=100, fallback_mm=None)[0]
        rec("sw100_it60", A, t, so, wi)
        rec("sw100_it150", B, t, so, wi)
        rec("sw100_it150_SAVGOL", T.smooth_baseline(B, t["valid"], kind="savgol"), t, so, wi)
        rec("sw1000_it60", C, t, so, wi)
        rec("sw100_it150_nofb", D, t, so, wi)
    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_smoother_diag.npz",
             ids=np.array(ids), **{f"jit_{l}": jit[l] for l in LABELS},
             **{f"ncyc_{l}": ncyc[l] for l in LABELS},
             **{f"pv_{l}": np.array(pv_flat[l]) for l in LABELS})

    print(f"\n=== PROCESSING CHECK (N={N}) ===", flush=True)
    for lab in LABELS:
        print(f"  {lab:20s} jitter finite {np.isfinite(jit[lab]).sum()}/{N}   "
              f">=1 cycle {(ncyc[lab]>0).sum()}/{N}   cycles {ncyc[lab].sum()}", flush=True)
    print(f"\n=== DIAGNOSTIC (peak-vel matched + jitter) ===", flush=True)
    print(f"  {'(ref) BA+savgol':20s}   +4.80%      1323   <- target", flush=True)
    for lab in LABELS:
        print(f"  {lab:20s} {med(pv_flat[lab]):+7.2f}% {med(jit[lab]):9.0f}", flush=True)
    print("\nREAD: it60->it300 drops jitter => UNDER-CONVERGENCE. sw100->sw1000 helps => weight.", flush=True)
    print("      nofb << fb on jitter => the DLT-backup re-injects jitter.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
