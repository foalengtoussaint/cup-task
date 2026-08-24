"""Temporal structure of the two failure masks: isolated frames, or long runs?

Mask A: <3 cameras pass the 30px gate  -> the DLT relaxed-gate path.
Mask B: BA joint-frames the guard reverts (non-finite OR >150mm from DLT).

Both are (T,J) boolean. For each joint independently we take runs of consecutive
flagged frames and report how the flagged cells distribute over run length. 60 fps,
so 1 frame = 16.7 ms.

    python scripts/ba_revert_runlengths.py  -> out/scoring/revert_runlengths.{npz,csv}
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                       # noqa: E402
import gnn_refiner as G                     # noqa: E402
import ba_refine as BA                      # noqa: E402

REPROJ_PX, MIN_CAMS, FALLBACK_MM, FPS = 30.0, 3, 150.0, 60.0
BINS = [(1, 1), (2, 5), (6, 30), (31, 100), (101, 10**9)]
LBL = ["1 frame (isolated)", "2-5", "6-30", "31-100", ">100 frames"]


def runs_of(mask1d):
    """Yield lengths of consecutive-True runs."""
    if not mask1d.any():
        return []
    d = np.diff(np.concatenate(([0], mask1d.view(np.int8), [0])))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()


def main() -> None:
    trials = T.load_clean(need_reproj=True)
    accA = defaultdict(lambda: np.zeros(len(BINS)))   # part -> cells per bin
    accB = defaultdict(lambda: np.zeros(len(BINS)))
    runsA, runsB, ids = [], [], []

    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        p, tid = t["part"], f"{t['part']}/{t['trial']}"
        mmc = t["mmc"]; fin = np.isfinite(mmc).all(-1)
        # --- mask A: inlier count from the DLT point
        X = torch.from_numpy(np.nan_to_num(mmc).astype(np.float32))[None]
        ok = []
        for c in range(t["uv"].shape[1]):
            uvp, _ = G.project_torch(X, *[torch.from_numpy(t[k][c].astype(np.float32))
                                          for k in ("K", "dist", "R", "t")])
            r = np.linalg.norm(uvp[0].numpy() - t["uv"][:, c], axis=-1)
            ok.append(t["uv_valid"][:, c] & np.isfinite(r) & (r <= REPROJ_PX))
        A = (np.stack(ok, 1).sum(1) < MIN_CAMS) & fin
        # --- mask B: what the guard reverts
        try:
            Xr, _ = BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=0, fallback_mm=None)
            moved = np.linalg.norm(Xr - mmc, axis=-1)
            B = (~np.isfinite(Xr).all(-1) | (np.isfinite(moved) & (moved > FALLBACK_MM))) & fin
        except Exception:
            B = np.zeros_like(A)

        for M, acc, store in ((A, accA, runsA), (B, accB, runsB)):
            for j in range(M.shape[1]):
                for L in runs_of(M[:, j]):
                    store.append(L)
                    for b, (lo, hi) in enumerate(BINS):
                        if lo <= L <= hi:
                            acc[p][b] += L; break
        ids.append(tid)
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(trials)}] runsA={len(runsA):,} runsB={len(runsB):,}", flush=True)

    OUT = ROOT / "out" / "automq"
    np.savez(OUT / "revert_runlengths.npz", runsA=np.array(runsA), runsB=np.array(runsB),
             ids=np.array(ids))

    def show(name, acc, runs):
        tot = sum(acc[p].sum() for p in acc)
        print(f"\n=== {name} — {tot:,.0f} flagged joint-frames in {len(runs):,} runs ===")
        print(f"{'run length':22s} {'cells':>12s} {'% of flagged':>13s} {'ms':>9s}")
        agg = np.sum([acc[p] for p in acc], axis=0)
        for b, L in enumerate(LBL):
            lo, hi = BINS[b]
            ms = f"{lo/FPS*1e3:.0f}-{min(hi,999)/FPS*1e3:.0f}" if lo != hi else f"{lo/FPS*1e3:.0f}"
            print(f"{L:22s} {agg[b]:12,.0f} {100*agg[b]/max(tot,1):12.1f}% {ms:>9s}")
        r = np.array(runs)
        if len(r):
            print(f"  median run {np.median(r):.0f} fr, p90 {np.percentile(r,90):.0f} fr, "
                  f"max {r.max():,} fr ({r.max()/FPS:.1f} s)")
        print("  P19 vs rest (% of that group's flagged cells in runs >100 fr):")
        for grp, sel in (("P19", ["P19"]), ("rest", [k for k in acc if k != "P19"])):
            g = np.sum([acc[k] for k in sel], axis=0); gt = g.sum()
            print(f"    {grp:5s} {100*g[-1]/max(gt,1):5.1f}%   ({gt:,.0f} cells)")

    show("MASK A: <3 inliers (relaxed-gate path)", accA, runsA)
    show("MASK B: BA guard reverts", accB, runsB)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
