"""Rebuild the BA trajectory cache for ALL 11 cohort participants (sw=0 = the recipe only).

The committed cache/ba_traj/traj_sw0_fb150.npz only covers the original 5 parts (P07/08/15/17/19,
328 trials). The 6 added parts (P10/P12/P13/P14/P251/P252) have no BA trajectories, so
`_ba_traj_cache()` returns None for them and BA+SmoothNet can't be scored there. This rebuilds the
SAME file over the full 826-trial cohort so BA is available for the AutoMQ scoring on everyone.

Only sw=0 is solved (fb150) -- that is the recipe and the only config the scorer loads. Drops the
sw=10/100 in-solve-smoother variants and the OMC metric computation from ba_cache_traj.py (~3x less
work). Solve-once: refined (T,9,3) per trial saved; all downstream metrics read the cache with no GPU.

Watch:  tail -f out/gnn/ba_cache_traj_all.log
"""
import sys, os, time
import numpy as np
from tqdm import tqdm
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T, gnn_refiner as G, ba_refine as BA
JOINTS = G.JOINTS
CACHE = "/home/imove/Documents/cup-task/cache/ba_traj"
os.makedirs(CACHE, exist_ok=True)
FALLBACK_MM = 150.0        # same threshold as ba_cache_traj.py: catches ~0.05% blowups, keeps legit


def main():
    trials = T.load_clean(need_reproj=True)     # all 11 parts, 826 trials
    N = len(trials)
    from collections import Counter
    per = dict(Counter(t["part"] for t in trials))
    print(f"trials: {N}   cache -> {CACHE}/traj_sw0_fb{int(FALLBACK_MM)}.npz", flush=True)
    print(f"per part: {per}", flush=True)

    traj, ids = [], []
    n_fail = 0
    t0 = time.time()
    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        ids.append(f"{t['part']}/{t['trial']}")
        try:
            P = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=0, fallback_mm=FALLBACK_MM)[0]
        except Exception as e:
            print(f"  [solve fail] {ids[-1]}: {type(e).__name__}: {e}", flush=True)
            P = np.full_like(t["mmc"], np.nan)
            n_fail += 1
        traj.append(P.astype(np.float32))                        # -> cache (even if all-NaN)
        # LIVE per-40 progress: count + elapsed + running finite-fraction
        if (i + 1) % 40 == 0:
            fin = sum(np.isfinite(p).any() for p in traj)
            print(f"    [{i+1}/{N}] {time.time()-t0:5.0f}s  finite-traj {fin}/{i+1}  fail {n_fail}",
                  flush=True)

    out = f"{CACHE}/traj_sw0_fb{int(FALLBACK_MM)}.npz"
    np.savez(out, ids=np.array(ids), traj=np.array(traj, dtype=object))
    fin = sum(np.isfinite(p).any() for p in traj)
    print(f"\nPROCESSING CHECK: {N} trials, {fin} finite-traj, {n_fail} solve-fail", flush=True)
    print(f"wrote {out}  ({time.time()-t0:.0f}s)", flush=True)
    # per-part coverage in the cache
    pc = Counter(i.split("/")[0] for i in ids)
    print(f"cache per-part: {dict(pc)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
