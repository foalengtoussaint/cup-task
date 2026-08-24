"""Measure the BA divergence-guard rate over the full 826-trial cohort.

Answers the Methods TODO: the only rate ever recorded (~0.05% of frames, 1 trial in 328)
predates the cohort expansion and the miscalibrated participants that actually cause
divergence. `ba_cache_traj_all.py` computes these counters on every trial and then
discards them -- it keeps only `refine_trial_ba(...)[0]` -- so the number was never
persisted, not genuinely unobtainable. Same solve, same recipe (fb150, sw=0), counters kept.

NOTE the shipped recipe passes fallback_mm only, never trial_guard_mm, so
n_guarded_joints is 0 by construction; it is recorded anyway so the claim is checkable.

    python scripts/ba_guard_rate.py     -> out/scoring/ba_guard_rate.{npz,csv} + log
"""
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                                  # noqa: E402
import ba_refine as BA                                 # noqa: E402

FALLBACK_MM = 150.0        # the recipe's threshold, as in ba_cache_traj_all.py
OUT = ROOT / "out" / "automq"


def main() -> None:
    trials = T.load_clean(need_reproj=True)
    N = len(trials)
    print(f"trials: {N}   fallback_mm={FALLBACK_MM}", flush=True)

    ids, nfb, ngd, nframes, njoints, failed = [], [], [], [], [], []
    t0 = time.time()
    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        tid = f"{t['part']}/{t['trial']}"
        try:
            P, info = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=0, fallback_mm=FALLBACK_MM)
            ids.append(tid)
            nfb.append(int(info.get("n_fallback", 0)))
            ngd.append(int(info.get("n_guarded_joints", 0)))
            nframes.append(int(P.shape[0])); njoints.append(int(P.shape[1]))
        except Exception as e:
            print(f"  [solve fail] {tid}: {type(e).__name__}: {e}", flush=True)
            failed.append(tid)
            ids.append(tid); nfb.append(-1); ngd.append(-1); nframes.append(0); njoints.append(0)
        if (i + 1) % 40 == 0:
            a = np.array(nfb); f = np.array(nframes) * np.array(njoints)
            ok = a >= 0
            rate = 100.0 * a[ok].sum() / max(f[ok].sum(), 1)
            print(f"    [{i+1}/{N}] {time.time()-t0:5.0f}s  trials-with-fallback "
                  f"{int((a[ok] > 0).sum())}  running frame-rate {rate:.4f}%", flush=True)

    ids = np.array(ids); nfb = np.array(nfb); ngd = np.array(ngd)
    nframes = np.array(nframes); njoints = np.array(njoints)
    ok = nfb >= 0
    jf = nframes * njoints                                    # joint-frames per trial

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "ba_guard_rate.npz", ids=ids, n_fallback=nfb, n_guarded=ngd,
             n_frames=nframes, n_joints=njoints)
    import pandas as pd
    pd.DataFrame({"id": ids, "n_fallback": nfb, "n_guarded": ngd,
                  "n_frames": nframes, "n_joints": njoints}).to_csv(
        OUT / "ba_guard_rate.csv", index=False)

    tw = int((nfb[ok] > 0).sum())
    print(f"\nPROCESSING CHECK: processed {ok.sum()}/{N}, solve-fail {len(failed)}, "
          f"non-finite 0", flush=True)
    print(f"\nBA DIVERGENCE GUARD over {ok.sum()} trials:", flush=True)
    print(f"  joint-frames reverted : {nfb[ok].sum():,} / {jf[ok].sum():,} "
          f"= {100.0*nfb[ok].sum()/jf[ok].sum():.4f}% of joint-frames", flush=True)
    print(f"  trials with >=1 revert: {tw} / {ok.sum()} = {100.0*tw/ok.sum():.2f}% of trials",
          flush=True)
    print(f"  n_guarded_joints total: {ngd[ok].sum()} (0 expected: trial_guard_mm unused)",
          flush=True)
    if tw:
        aff = nfb[ok]; sel = aff > 0
        print(f"  among affected trials : median {np.median(aff[sel]):.0f}, "
              f"p90 {np.percentile(aff[sel],90):.0f}, max {aff[sel].max()} joint-frames", flush=True)
        part = np.array([i.split("/")[0] for i in ids])[ok][sel]
        u, c = np.unique(part, return_counts=True)
        print("  affected trials by participant: "
              + ", ".join(f"{a}:{b}" for a, b in sorted(zip(u, c), key=lambda x: -x[1])), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
