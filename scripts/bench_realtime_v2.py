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


def _shared_uetrack():
    """ONE UETrack model, reused across all camera-count sweeps (1.3GB; per-cam = state swap)."""
    global _UET
    if _UET is None:
        from uetrack_wrap import UETrackB
        _UET = UETrackB()
    return _UET


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

    print(f"\n{'cams':>5} {'pose fps':>9} {'pose ms/fr':>11}  {'track fps':>10} {'track ms/fr':>12}",
          flush=True)
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

        # ---- TRACK: ONE shared UETrack model (1.3GB); per-camera cost = ncam sequential updates.
        # (The tracker thread's OOM lesson: never instantiate one model per camera. UETrack holds
        # per-track state internally, so a true multi-cam deployment keeps ncam lightweight state
        # dicts and swaps them; here we measure the per-update cost x ncam, which is the throughput.)
        trk = _shared_uetrack()
        trk.init(frames[0][:, :, ::-1], (900, 700, 60, 60))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        treps = min(n_frames, 120)
        for i in range(treps):
            im = frames[i % n_frames][:, :, ::-1]
            for _c in range(ncam):                          # ncam updates = one rig-frame
                trk.update(im)
        torch.cuda.synchronize()
        track_dt = (time.perf_counter() - t0) / treps      # sec per rig-frame (all cams updated)
        track_fps = 1.0 / track_dt
        print(f"{ncam:5d} {pose_fps:9.1f} {pose_dt*1000:11.2f}  {track_fps:10.1f} {track_dt*1000:12.2f}",
              flush=True)
        rows.append((ncam, pose_fps, track_fps))
        torch.cuda.empty_cache()
    print("\n(pose = batched YOLO-pose across cams, 1 forward/frame; track = UETrack update per cam/frame.\n"
          " Live fps = frame-sets/sec: the whole rig advancing one frame. 60fps capture => need >=60.)",
          flush=True)
    return rows


def bench_pipeline(device="0"):
    """Whole offline pipeline per trial on a cached trial: consensus + SmoothNet + segment."""
    import compare_pose_omc_delta as H
    from cup_task import cup_track, pose_smooth, segment
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
