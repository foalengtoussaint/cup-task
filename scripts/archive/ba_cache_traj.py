"""Build a TRAJECTORY CACHE for BA configs (solve ONCE, save refined 3D; all future metrics free).

For each trial, solve 3 configs with a ROBUST fallback (fallback_mm=500: reverts NaN AND finite-huge
run-offs to the pipeline point, but leaves legit <500mm corrections -> can't crash Kabsch, unlike the
NaN-only fallback that let a 1e10 value through):
    sw=0   plain BA (the recipe)          sw=10, sw=100  in-solve smoother
Saves refined (T,9,3) per trial per config to  cache/ba_traj/traj_sw{X}.npz  (the cache).
Then computes peak-vel(matched) + jitter for each config, RAW and +savgol, from the just-solved
trajectories -> answers "in-solve + savgol" too. Coverage printed. Watch:
  tail -f out/gnn/ba_cache_traj.log
Re-analysis later reads cache/ba_traj/*.npz with NO GPU/solve.
"""
import sys, os
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T, gnn_refiner as G, ba_refine as BA
import compare_pose_omc_delta as H
JOINTS = G.JOINTS
CACHE = "/home/imove/Documents/cup-task/cache/ba_traj"
os.makedirs(CACHE, exist_ok=True)
# fallback threshold: legit corrections are <=106mm (p99.9); blowups jump to 324-498mm. 150mm sits in
# the clean gap -> catches the ~0.05% blowup frames (incl P19 trial_63's 498mm) WITHOUT touching legit.
FALLBACK_MM = 150.0
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


def metrics_from(P, t, so, wi):
    # isfinite-filtered valid mask -> never feed non-finite into Kabsch/SVD (the crash cause)
    vfin = t["valid"] & np.isfinite(P).all(-1)
    jit = T.score_trial(P, t["omc"], vfin, t["side"])["jit"]
    cy = [x for x in pv_matched(H._lp(H._speed(P[:, wi])), so) if np.isfinite(x)]
    return jit, cy


def main():
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in ("P07","P08","P15","P17","P19")]
    N = len(trials)
    print(f"trials: {N}   cache -> {CACHE}", flush=True)
    SWS = [0, 10, 100]
    traj = {s: [] for s in SWS}                 # cached refined trajectories (object arrays)
    ids = []
    # metrics: [raw | +savgol] x [jit | pv] per config
    jit = {(s, sav): np.full(N, np.nan) for s in SWS for sav in (0, 1)}
    pv = {(s, sav): [] for s in SWS for sav in (0, 1)}
    ncyc = {(s, sav): np.zeros(N, int) for s in SWS for sav in (0, 1)}

    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        ids.append(f"{t['part']}/{t['trial']}")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        for s in SWS:
            try:
                P = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=s, fallback_mm=FALLBACK_MM)[0]
            except Exception as e:
                print(f"  [solve fail] {ids[-1]} sw={s}: {type(e).__name__}", flush=True)
                P = np.full_like(t["mmc"], np.nan)
            traj[s].append(P.astype(np.float32))            # -> cache (even if all-NaN)
            for sav in (0, 1):
                try:
                    Q = T.smooth_baseline(P, t["valid"], kind="savgol") if sav else P
                    j, cy = metrics_from(Q, t, so, wi)
                    jit[(s, sav)][i] = j; ncyc[(s, sav)][i] = len(cy); pv[(s, sav)] += cy
                except Exception as e:
                    print(f"  [metric fail] {ids[-1]} sw={s} sav={sav}: {type(e).__name__}", flush=True)

    # SAVE the trajectory cache -- fb-tagged filename so the fb=500 cache is KEPT (keep-all-data)
    fbtag = f"_fb{int(FALLBACK_MM)}"
    for s in SWS:
        np.savez(f"{CACHE}/traj_sw{s}{fbtag}.npz",
                 ids=np.array(ids), traj=np.array(traj[s], dtype=object))
    print(f"\ncached trajectories: {CACHE}/traj_sw{{0,10,100}}{fbtag}.npz", flush=True)

    print(f"\n=== PROCESSING CHECK (N={N}, fallback_mm={FALLBACK_MM}) ===", flush=True)
    for s in SWS:
        f = np.isfinite(jit[(s, 0)]).sum()
        print(f"  sw={s:<4} jitter finite {f}/{N}   cycles(raw) {ncyc[(s,0)].sum()}", flush=True)
    print(f"\n=== peak-vel(matched) / jitter — RAW vs +SAVGOL ===", flush=True)
    print(f"  {'config':16s} {'raw pv':>8s} {'raw jit':>8s}   {'+savgol pv':>10s} {'+savgol jit':>11s}",
          flush=True)
    for s in SWS:
        tag = "BA plain" if s == 0 else f"BA insolve{s}"
        print(f"  {tag:16s} {med(pv[(s,0)]):+7.2f}% {med(jit[(s,0)]):8.0f}   "
              f"{med(pv[(s,1)]):+9.2f}% {med(jit[(s,1)]):11.0f}", flush=True)
    print("\nsw=0 +savgol = the recipe (~+4.80%/1323). in-solve+savgol wins only if it beats that.",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
