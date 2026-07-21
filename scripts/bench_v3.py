"""v3 pipeline benchmark: ONLINE fps + OFFLINE post-processing wall time.

The v3 design splits the pipeline at the point where a stage stops needing raw PIXELS:

  ONLINE  (needs the frame while it is in hand): decode, YOLO-pose, cup detect/UETrack, PyrLK flow.
  OFFLINE (needs the whole trial, works on numbers): triangulation, consensus, SmoothNet, blend,
          segmentation, Murphy scoring.

This script measures both halves and, crucially, the cost of the ONE stage that could go either way
-- PyrLK flow. Flow needs the raw frame PAIR, so running it offline forces a SECOND full decode of
every camera's video. We measure that decode explicitly, because it is the entire justification for
putting flow online.

    python scripts/bench_v3.py --cams 1 5 10 --frames 200          # everything
    python scripts/bench_v3.py --what online                       # fps only
    python scripts/bench_v3.py --what offline                      # post-processing only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

FPS = 60.0
_UETB = {}


def _fill(xyz):
    x = np.asarray(xyz, dtype=float).copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    return x


def _batched_uetrack(ncam):
    from uetrack_wrap import UETrackBatch
    if ncam not in _UETB:
        _UETB[ncam] = UETrackBatch(ncam)
    return _UETB[ncam]


def _sample_frames(n_frames):
    """Decode n real frames from a DELTA clip. Real pixels so flow/tracking do real work."""
    import cv2, glob
    import compare_pose_omc_delta as H
    vids = sorted(glob.glob(str(H.DELTA / "P07" / "staged" / "*.mp4")))
    if not vids:
        return [np.random.randint(0, 255, (1080, 1920, 3), np.uint8) for _ in range(n_frames)], True, None
    cap = cv2.VideoCapture(vids[0]); frames = []
    while len(frames) < n_frames:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    return frames, False, vids[0]


def bench_online(cam_counts, n_frames, device="0"):
    """Per-rig-frame cost of every ONLINE stage, then the combined live loop."""
    import cv2
    import torch
    from ultralytics import YOLO
    from cup_task import flow_speed

    frames, synth, vid = _sample_frames(n_frames)
    n_frames = len(frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    # Pre-convert BGR->RGB ONCE. A per-frame `[:, :, ::-1].copy()` on 1080p costs 6.96ms -- MORE
    # than UETrack's own 4.04ms step -- and charging it to the tracker made it read 91fps instead
    # of 247. In a real rig the colour conversion happens once per frame anyway (pose needs it too),
    # so it is capture-loop overhead, not tracker cost.
    rgbs = [f[:, :, ::-1].copy() for f in frames]
    print(f"ONLINE loop: {n_frames} real 1080p frames{' (SYNTH)' if synth else ''} from "
          f"{Path(vid).name if vid else 'synthetic'}, cuda:{device}\n", flush=True)

    pose_model = YOLO(str(ROOT / "models" / "yolo26n-pose.pt"))
    pose_model.to(f"cuda:{device}")
    _ = pose_model.predict(frames[0], verbose=False, device=f"cuda:{device}")

    # ---- the flow stage alone (CPU, per wrist per camera) ----
    wpx = np.array([960.0, 540.0])
    t0 = time.perf_counter()
    reps = 200
    for i in range(reps):
        flow_speed.flow_at(grays[i % n_frames], grays[(i + 1) % n_frames], wpx)
    flow_ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  PyrLK flow          {flow_ms:6.2f} ms / wrist / camera   (CPU)", flush=True)

    # ---- decode alone (the cost flow would DOUBLE if run offline) ----
    if vid:
        cap = cv2.VideoCapture(vid)
        t0 = time.perf_counter(); k = 0
        while k < 200:
            ok, im = cap.read()
            if not ok:
                break
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); k += 1
        cap.release()
        dec_ms = (time.perf_counter() - t0) / max(k, 1) * 1000
        print(f"  decode+gray         {dec_ms:6.2f} ms / frame / camera   (CPU, the offline-flow tax)",
              flush=True)
    else:
        dec_ms = float("nan")

    print(f"\n{'cams':>5} {'pose fps':>9} {'cup fps':>9} {'flow +ms':>9} {'BOTH+flow fps':>14} "
          f"{'realtime?':>10}", flush=True)
    print("  " + "-" * 62, flush=True)
    rows = []
    for ncam in cam_counts:
        batch = [frames[i % n_frames] for i in range(ncam)]

        # pose: batched across cameras, one forward
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = 40
        for i in range(r):
            b = [frames[(i + k) % n_frames] for k in range(ncam)]
            _ = pose_model.predict(b, verbose=False, device=f"cuda:{device}")
        torch.cuda.synchronize()
        pose_fps = 1.0 / ((time.perf_counter() - t0) / r)

        # cup: batched UETrack, one forward across cameras
        btrk = _batched_uetrack(ncam)
        rgb0 = rgbs[0]
        for c in range(ncam):
            btrk.init(c, rgb0, (900, 700, 60, 60))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(r):
            btrk.update([rgbs[i % n_frames]] * ncam)
        torch.cuda.synchronize()
        cup_fps = 1.0 / ((time.perf_counter() - t0) / r)

        # flow for this rig-frame = ncam wrists (one per camera) on a CPU thread pool. PyrLK
        # releases the GIL, so threads give real parallelism (~2.4x at 5-10 cams). Reported as the
        # MARGINAL cost measured below -- what flow actually adds on top of the GPU pass, which is
        # the only number that matters for the loop. (ncam * per-wrist would be the SERIAL cost and
        # overstates it by ~30x at 5 cams, because the GPU pass hides nearly all of it.)
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=max(2, ncam)) as _ex:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i in range(r):
                pose_model.predict(batch, verbose=False, device=f"cuda:{device}")
                torch.cuda.synchronize()
            gpu_only = (time.perf_counter() - t0) / r
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i in range(r):
                _f = [_ex.submit(flow_speed.flow_at, grays[i % n_frames],
                                 grays[(i + 1) % n_frames], wpx) for _ in range(ncam)]
                pose_model.predict(batch, verbose=False, device=f"cuda:{device}")
                torch.cuda.synchronize()
                for x in _f:
                    x.result()
            gpu_flow = (time.perf_counter() - t0) / r
        flow_tot = max(0.0, (gpu_flow - gpu_only) * 1000)      # MARGINAL ms, overlapped

        # combined: pose + cup back-to-back, flow on a CPU THREAD POOL.
        #
        # The two GPU nets are deliberately NOT threaded against each other. Measured: threading
        # them is WORSE than running them serially (1/5/10cam = 0.81x/0.87x/0.91x), and serial
        # already equals the sum of the parts (10.1 vs 10.9ms at 1cam). Both are compute-bound, so
        # they serialize on the device whatever the CUDA streams / host threads say; extra threads
        # only add contention. More throughput needs a 2nd GPU or a lighter backbone, not better
        # scheduling. FLOW is different -- pure CPU, genuinely overlaps -- so it IS threaded.
        from concurrent.futures import ThreadPoolExecutor
        sp, sc = torch.cuda.Stream(), torch.cuda.Stream()
        with ThreadPoolExecutor(max_workers=max(2, ncam)) as ex:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i in range(r):
                futs = [ex.submit(flow_speed.flow_at, grays[i % n_frames],
                                  grays[(i + 1) % n_frames], wpx) for _ in range(ncam)]
                with torch.cuda.stream(sp):
                    pose_model.predict(batch, verbose=False, device=f"cuda:{device}")
                with torch.cuda.stream(sc):
                    btrk.update([rgbs[i % n_frames]] * ncam)
                torch.cuda.synchronize()
                for f_ in futs:
                    f_.result()
            all_fps = 1.0 / ((time.perf_counter() - t0) / r)

        print(f"{ncam:5d} {pose_fps:9.1f} {cup_fps:9.1f} {flow_tot:9.2f} {all_fps:14.1f} "
              f"{'YES' if all_fps >= 60 else 'no':>10}", flush=True)
        rows.append((ncam, pose_fps, cup_fps, flow_tot, all_fps))
        torch.cuda.empty_cache()

    print(f"\n  fps = RIG-frames/s (all cameras advance one frame). 60fps capture => need >=60.", flush=True)
    print(f"  'flow +ms' = MARGINAL cost of threaded flow on top of the GPU pass (measured, not")
    print(f"  ncam*per-wrist): PyrLK releases the GIL and the GPU pass hides most of the rest.")
    print(f"  BOTH+flow runs pose and cup on separate CUDA streams with flow on a CPU thread pool.")
    print(f"  It lands BELOW min(pose,cup): on one GPU the two nets contend for the same SMs rather")
    print(f"  than overlapping, so combining them costs more than the slower stage alone.", flush=True)
    return rows, dec_ms, flow_ms


def bench_offline(dec_ms=None, device="0"):
    """Per-trial wall time of every OFFLINE stage, on a real cached trial."""
    import compare_pose_omc_delta as H
    import flow_velocity_probe as F
    from cup_task import cup_track, pose_smooth, segment, flow_speed, speed_blend
    from cup_task.score import compute_position_measures
    H.use_good_cams()
    part, trial, side = "P07", "trial_10_L_unaffected", "left"
    joint = f"{side}_wrist"
    calib = H._load_calib_mm(part)
    if part in H.GOOD_CAMS:
        calib = {c: v for c, v in calib.items() if c in H.GOOD_CAMS[part]}
    cf = ROOT / "cache" / "tracks_uetrack" / f"{part}__{trial}__uetrack__fs1.json"
    mmc, n = H._load_mmc(part, trial)
    ncam = len(calib)

    print(f"\nOFFLINE post-processing, per trial ({part} {trial}: {n} frames = "
          f"{n/FPS:.1f}s @60fps, {ncam} cams)\n", flush=True)
    times = {}

    # cup 3D: consensus over the tracker points
    t0 = time.perf_counter()
    cup = cup_track.track_cup_3d_from_cache(cf, calib)
    times["cup 3D (consensus)"] = time.perf_counter() - t0

    # pose SmoothNet: 3 joints (2 wrists + mouth)
    tr = [{"frame": f, "X": (None if not np.isfinite(p).all() else [float(v) for v in p])}
          for f, p in enumerate(mmc[joint])]
    pose_smooth.smooth_track(tr)                       # warm (model load excluded, ~950ms once)
    t0 = time.perf_counter()
    for _ in range(3):
        sm = pose_smooth.smooth_track(tr)
    times["SmoothNet (3 joints)"] = time.perf_counter() - t0

    # flow speed from cached per-camera flow (the online path hands these over directly)
    px = F.load_wrist_px(part, trial, joint)
    fl = {}
    for c in px:
        p = ROOT / "cache" / "flow_vel" / f"delta_{part}_{trial}.{c.split('_')[1]}__pyrlk.npy"
        if p.exists() and c in calib:
            fl[c] = np.load(p)
    # 2D flow -> 3D speed: per frame triangulate BOTH the wrist pixel and (wrist pixel + flow), and
    # difference the two 3D points. Not a position derivative -- a velocity measurement.
    t0 = time.perf_counter()
    sp_fl = flow_speed.speed_from_cached_flow(px, fl, calib, n)
    times["flow 2D->3D speed"] = time.perf_counter() - t0

    # blend
    sn_xyz = np.array([t["X"] if t["X"] else [np.nan] * 3 for t in sm])
    sp_sn = H._speed(sn_xyz)
    t0 = time.perf_counter()
    _ = speed_blend.blend(H._lp(sp_fl), H._lp(sp_sn))
    times["speed blend"] = time.perf_counter() - t0

    # segmentation
    cupX = np.array([t["X"] if t["X"] else [np.nan] * 3 for t in cup])
    x = cupX.copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    t0 = time.perf_counter()
    seg = segment.segment_cup_only(x, fps=FPS)
    times["segmentation"] = time.perf_counter() - t0

    # Murphy scoring (needs the 7-phase names, so derive them first -- that derivation is part of
    # the scoring cost and is timed with it)
    trunk = (mmc[f"{side}_shoulder"] + mmc[f"{'right' if side=='left' else 'left'}_shoulder"]) / 2
    t0 = time.perf_counter()
    try:
        ph = segment.to_murphy_phases(seg, _fill(sn_xyz), x, fps=FPS)
        compute_position_measures(sn_xyz, trunk, ph, side, fps=FPS)
    except Exception as e:
        print(f"  (scoring FAILED: {type(e).__name__}: {e})", flush=True)
    times["Murphy scoring"] = time.perf_counter() - t0

    tot = sum(times.values())
    for k, v in times.items():
        print(f"  {k:24s} {v*1000:8.1f} ms", flush=True)
    print(f"  {'-'*24} {'-'*8}", flush=True)
    print(f"  {'TOTAL offline':24s} {tot*1000:8.1f} ms/trial   "
          f"({tot/(n/FPS)*100:.2f}% of realtime)", flush=True)

    if dec_ms and np.isfinite(dec_ms):
        tax = dec_ms * n * ncam / 1000.0
        print(f"\n  If flow ran OFFLINE it would need a second decode of all {ncam} cameras:", flush=True)
        print(f"    re-decode tax        {tax*1000:8.1f} ms/trial  = {tax/tot:.0f}x the whole "
              f"offline budget above", flush=True)
        print(f"  -> flow belongs ONLINE (frames already in hand); the lift above is all that is left.",
              flush=True)
    return times


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--device", default="0")
    ap.add_argument("--what", choices=["online", "offline", "both"], default="both")
    a = ap.parse_args(argv)
    dec = None
    if a.what in ("online", "both"):
        _, dec, _ = bench_online(a.cams, a.frames, a.device)
    if a.what in ("offline", "both"):
        bench_offline(dec, a.device)


if __name__ == "__main__":
    main()
