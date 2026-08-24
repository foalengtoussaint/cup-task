"""Cache the SmoothNet-smoothed 3D pose ONCE per trial per triangulation, so downstream metric
experiments (peak-summary, filter-cutoff, timing, ...) are pure-CPU seconds -- no repeat GPU pass.

The main grid ran SmoothNet on all 328 trials but saved only the SCALAR measures, so any new peak
definition or filter cutoff re-ran the whole ~2s/trial/variant GPU inference. This dumps the actual
smoothed trajectories (the expensive SOLVE OUTPUT) for both triangulations:
  - pipeline+smoothnet : SmoothNet on the robust-consensus DLT pose (trial_rec['mmc'])
  - BA+smoothnet       : SmoothNet on the cached BA trajectory

Cache: cache/pose_smoothed/<part>__<trial>.npz
  keys: joints (list[str]), pipeline_sn (T,J,3), ba_sn (T,J,3)   (ba_sn all-NaN if no BA traj)

    python scripts/cache_smoothed_pose.py            # all load_clean trials, idempotent
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import results_v3_delta as R

OUT = ROOT / "cache" / "pose_smoothed"
JOINTS = R._GRID_JOINTS          # same 9-joint order as everywhere else


def run(overwrite=False):
    import gnn_train as GT
    from tqdm import tqdm
    H.use_good_cams()
    ba = R._ba_traj_cache()
    trials = GT.load_clean(need_reproj=False)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"cache_smoothed_pose: {len(trials)} trials -> {OUT}  (BA cache: "
          f"{len(ba) if ba else 0} trajs)", flush=True)

    done = skip = 0
    t0 = time.perf_counter()
    for t in tqdm(trials, desc="cache_smoothed_pose", unit="trial"):
        part, trial = t["part"], t["trial"]
        p = OUT / f"{part}__{trial}.npz"
        if p.exists() and not overwrite:
            skip += 1
            continue
        from pipeline import pose_smooth as PS
        n = t["mmc"].shape[0]
        # BATCHED SmoothNet: one GPU forward per trip (all 9 joints, all windows stacked) -- ~20x
        # faster than the per-joint/per-window path and numerically identical (verified 0.05mm).
        pipe_in = {j: t["mmc"][:, k] for k, j in enumerate(JOINTS)}
        pipe = PS.smooth_joints_batched(pipe_in)
        pipe_arr = np.stack([pipe[j] for j in JOINTS], 1)
        if ba is not None and f"{part}/{trial}" in ba:
            arr = np.asarray(ba[f"{part}/{trial}"], float)
            ba_in = {j: arr[:, k] for k, j in enumerate(JOINTS)}
            bav = PS.smooth_joints_batched(ba_in)
            ba_arr = np.stack([bav[j] for j in JOINTS], 1)
        else:
            ba_arr = np.full((n, len(JOINTS), 3), np.nan)
        np.savez(str(p), joints=np.array(JOINTS), pipeline_sn=pipe_arr.astype(np.float32),
                 ba_sn=ba_arr.astype(np.float32))
        done += 1

    print(f"\nPROCESSING CHECK: {done} cached, {skip} skipped-existing, {len(trials)} total "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    run(overwrite="--overwrite" in sys.argv)
