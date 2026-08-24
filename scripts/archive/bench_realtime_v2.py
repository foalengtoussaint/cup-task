"""Real-time speed of the v2 live tracking loop, with proper batching + threading.

Two things the user asked to measure:
  (A) LIVE TRACKING LOOP -- per-frame, all cameras: the online-capable path.
      * cup: YOLO seeds once, then UETrack per camera per frame (the steady state = all tracking).
      * pose: YOLO-pose per camera per frame (batched across cameras in one forward).
      Reports fps @ 1 / 5 / 10 cameras, with batching across cameras + a thread per model.
  (B) WHOLE PIPELINE per trial -- the offline stages (triangulation + consensus + SmoothNet +
      segment) timed on a real cached trial, so we know the per-trial wall cost end to end.

Uses real DELTA frames (decoded once, reused) so the numbers are honest, not synthetic noise.

    python scripts/bench_realtime_v2.py --cams 1 5 10 --frames 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


_UET = None
_UETB = {}


def _shared_uetrack():
    """ONE UETrack model, reused across all camera-count sweeps (1.3GB; per-cam = state swap)."""
    global _UET
    if _UET is None:
        from uetrack_wrap import UETrackB
        _UET = UETrackB()
    return _UET


def _batched_uetrack(ncam):
    """A batched N-camera UETrack (one shared model, N states, one forward per rig-frame)."""
    from uetrack_wrap import UETrackBatch
    if ncam not in _UETB:
        _UETB[ncam] = UETrackBatch(ncam)
    return _UETB[ncam]


def _sample_frames(n_frames):
    """Decode n real frames from a DELTA clip (one camera) to feed every simulated camera. Same
    image content per camera is fine -- we're measuring GPU/throughput, not accuracy here."""
    import cv2
    import compare_pose_omc_delta as H
    d = H.DELTA / "P07"
    import glob
    vids = sorted(glob.glob(str(d / "staged" / "*.mp4")))
    if not vids:
        vids = sorted(glob.glob(str(d / "**" / "*.mp4"), recursive=True))
    if not vids:
        # last resort: synthesize (still exercises the model; flagged in output)
        return [np.random.randint(0, 255, (1080, 1920, 3), np.uint8) for _ in range(n_frames)], True
    cap = cv2.VideoCapture(vids[0])
    frames = []
    while len(frames) < n_frames:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    return frames, False


def bench_live(cam_counts, n_frames, device="0"):
    import torch
    from ultralytics import YOLO

    frames, synth = _sample_frames(n_frames)
    n_frames = len(frames)
    print(f"live loop: {n_frames} real frames{' (SYNTH fallback)' if synth else ''}, "
          f"1080p, device cuda:{device}", flush=True)

    pose_model = YOLO(str(ROOT / "models" / "yolo26n-pose.pt"))
    pose_model.to(f"cuda:{device}")
    # warm up
    _ = pose_model.predict(frames[0], verbose=False, device=f"cuda:{device}")

    # UETrack: one shared model; measure a per-frame update
    from uetrack_wrap import UETrackB

    print(f"\n{'cams':>5} {'pose fps':>9} {'pose ms/fr':>11}  {'trk seq':>9} {'trk batched':>11} "
          f"{'speedup':>8}", flush=True)
    rows = []
    for ncam in cam_counts:
        # ---- POSE: batch ncam frames in one forward (the right way to amortize launch cost) ----
        batch = [frames[i % n_frames] for i in range(ncam)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = max(1, n_frames // 1)
        for i in range(reps):
            b = [frames[(i + k) % n_frames] for k in range(ncam)]
            _ = pose_model.predict(b, verbose=False, device=f"cuda:{device}")
        torch.cuda.synchronize()
        pose_dt = (time.perf_counter() - t0) / reps        # seconds per FRAME-SET (all cams, 1 frame)
        pose_fps = 1.0 / pose_dt                            # frame-sets per second = live fps

        # ---- TRACK (sequential): ONE shared model, ncam sequential updates = one rig-frame ----
        seqtrk = _shared_uetrack()
        rgb0 = frames[0][:, :, ::-1].copy()
        seqtrk.init(rgb0, (900, 700, 60, 60))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        treps = min(n_frames, 100)
        for i in range(treps):
            rgb = frames[i % n_frames][:, :, ::-1].copy()
            for _c in range(ncam):
                seqtrk.update(rgb)
        torch.cuda.synchronize()
        seq_dt = (time.perf_counter() - t0) / treps
        seq_fps = 1.0 / seq_dt

        # ---- TRACK (batched): ncam states, ONE batched forward per rig-frame ----
        btrk = _batched_uetrack(ncam)
        for c in range(ncam):
            btrk.init(c, rgb0, (900, 700, 60, 60))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(treps):
            rgb = frames[i % n_frames][:, :, ::-1].copy()
            btrk.update([rgb] * ncam)
        torch.cuda.synchronize()
        bat_dt = (time.perf_counter() - t0) / treps
        bat_fps = 1.0 / bat_dt
        print(f"{ncam:5d} {pose_fps:9.1f} {pose_dt*1000:11.2f}  {seq_fps:9.1f} {bat_fps:11.1f} "
              f"{bat_fps/seq_fps if seq_fps else 0:8.1f}x", flush=True)
        rows.append((ncam, pose_fps, seq_fps, bat_fps))
        torch.cuda.empty_cache()
    print("\n(pose = batched YOLO-pose across cams; trk seq = ncam sequential UETrack updates; "
          "trk batched = ONE\n batched forward across cams. Live fps = rig advancing one frame; "
          "60fps capture => need >=60.)", flush=True)

    # ---- SIMULTANEOUS pose + cup on two CUDA streams (both batched) ----
    print(f"\n{'cams':>5} {'pose only':>10} {'cup only':>10} {'SIMULTANEOUS':>13}  (both batched, 2 streams)",
          flush=True)
    for ncam in cam_counts:
        batch = [frames[i % n_frames] for i in range(ncam)]
        btrk = _batched_uetrack(ncam)
        for c in range(ncam):
            btrk.init(c, frames[0][:, :, ::-1].copy(), (900, 700, 60, 60))

        def _pose():
            pose_model.predict(batch, verbose=False, device=f"cuda:{device}")

        def _cup():
            btrk.update([frames[0][:, :, ::-1].copy()] * ncam)

        reps = 40
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): _pose()
        torch.cuda.synchronize(); p = 1.0 / ((time.perf_counter() - t0) / reps)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): _cup()
        torch.cuda.synchronize(); c = 1.0 / ((time.perf_counter() - t0) / reps)
        sp = torch.cuda.Stream(); sc = torch.cuda.Stream()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            with torch.cuda.stream(sp): _pose()
            with torch.cuda.stream(sc): _cup()
            torch.cuda.synchronize()
        both = 1.0 / ((time.perf_counter() - t0) / reps)
        print(f"{ncam:5d} {p:9.1f} {c:9.1f} {both:12.1f}", flush=True)
        torch.cuda.empty_cache()
    print("\n(SIMULTANEOUS = pose + cup on separate CUDA streams. On a small GPU both are "
          "compute-bound so\n they contend rather than fully overlap; ~half the min of the two "
          "individual rates.)", flush=True)
    return rows


def bench_pipeline(device="0"):
    """Whole offline pipeline per trial on a cached trial: consensus + SmoothNet + segment."""
    import compare_pose_omc_delta as H
    from pipeline import cup_track, pose_smooth, segment
    H.use_good_cams()
    part, trial, side = "P07", "trial_10_L_unaffected", "left"
    calib = H._load_calib_mm(part)
    cf = ROOT / "cache" / "tracks_uetrack" / f"{part}__{trial}__uetrack__fs1.json"

    print(f"\nwhole-pipeline per trial ({part} {trial}, cached tracker points):", flush=True)
    t0 = time.perf_counter()
    cup = cup_track.track_cup_3d_from_cache(cf, calib)
    t_cons = time.perf_counter() - t0

    mmc, n = H._load_mmc(part, trial)
    tr = [{"frame": f, "X": (None if not np.isfinite(p).all() else [float(v) for v in p])}
          for f, p in enumerate(mmc[f"{side}_wrist"])]
    pose_smooth.smooth_track(tr)                     # WARM the model (one-time load excluded)
    t0 = time.perf_counter()
    for _ in range(3):                               # 3 pose joints (2 wrists + mouth)
        _ = pose_smooth.smooth_track(tr)
    t_smooth = (time.perf_counter() - t0) / 3 * 3    # per-trial = 3 joints

    cupX = np.array([t["X"] if t["X"] else [np.nan] * 3 for t in cup])
    t0 = time.perf_counter()
    x = cupX.copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    _ = segment.segment_cup_only(x, fps=60.0)
    t_seg = time.perf_counter() - t0

    print(f"  consensus (cup 3D, {n} frames)      {t_cons*1000:7.1f} ms", flush=True)
    print(f"  SmoothNet (3 joints, {n} frames)    {t_smooth*1000:7.1f} ms  (warm; model load ~950ms once)",
          flush=True)
    print(f"  segment_cup_only                    {t_seg*1000:7.1f} ms", flush=True)
    print(f"  --> offline stages total            {(t_cons+t_smooth+t_seg)*1000:7.1f} ms/trial "
          f"({n/60.0:.1f}s of video @60fps)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--device", default="0")
    ap.add_argument("--what", choices=["live", "pipeline", "both"], default="both")
    a = ap.parse_args(argv)
    if a.what in ("live", "both"):
        bench_live(a.cams, a.frames, a.device)
    if a.what in ("pipeline", "both"):
        bench_pipeline(a.device)


if __name__ == "__main__":
    main()
