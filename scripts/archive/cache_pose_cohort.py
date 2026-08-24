"""Batch-cache yolo26s-pose 2D keypoints for the WHOLE cohort, no GPU re-run later.

The cup is already cached (drink_study student_dets_clean3d_refill/), so the pose net owns
the whole GPU. That lets us batch to ~32 (measured saturation point on the 3060 Ti: ~293
cam-frames/s for yolo26s, VRAM only ~1GB so it is compute-bound, not memory-bound) and feed
frames via NVDEC (ffmpeg h264_cuvid) so decode overlaps inference instead of bottlenecking on
the CPU.

Output is per-rep, IDENTICAL in shape to scripts/cache_pose_models.py's <stem>/yolo26s.2d.json
(keyed cam_1..cam_N, one {"kps": {...}} dict per frame), so segment.py / the OMC comparison
read it with no changes. The 2D->3D triangulation is left to the reader (triangulate.py),
same as the rest of the pipeline -- this script only does the GPU-bound part once.

  --minutes N   stop cleanly after ~N minutes (preliminary slice; the rest resumes later
                because finished reps are skipped).

    conda activate object_tracking   # ultralytics 8.4.x -- idrink's 8.3.40 can't load Pose26
    python scripts/cache_pose_cohort.py --clips /home/imove/Documents/clips \
        --model s --batch 32 --minutes 10
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# reuse the EXACT parse the single-clip path uses, so output is bit-compatible
from pipeline.pose_keypoints import COCO17, KP_INDEX, MIN_KP_CONF, TASK_KP

# NVDEC decode (ffmpeg h264_cuvid) with cv2 fallback -- lives in object_tracking
OT_LIB = Path("/home/imove/Documents/object_tracking/experiments/drink_study/lib")
sys.path.insert(0, str(OT_LIB))
import gpu_decode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "pose_models"
IMGSZ = 640


def _parse_result(r):
    """One ultralytics Results -> {"kps": {...}} exactly as pose_keypoints does."""
    if r.boxes is None or len(r.boxes) == 0:
        return {"kps": {}, "box_conf": 0.0}
    confs = r.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    kxy = r.keypoints.xy[best].cpu().numpy()
    kconf = r.keypoints.conf[best].cpu().numpy()
    kps = {}
    for name in TASK_KP:
        j = KP_INDEX[name]
        c = float(kconf[j])
        if c >= MIN_KP_CONF:
            kps[name] = [float(kxy[j, 0]), float(kxy[j, 1]), c]
    return {"kps": kps, "box_conf": float(confs[best])}


def _rep_clips(clips_root: Path):
    """Group per-camera mp4s into reps: {stem: {cam_idx: path}}. Skips '(1)' dup dirs."""
    reps: dict[str, dict[int, Path]] = {}
    for mp4 in sorted(clips_root.glob("*/*.mp4")):
        if "(" in mp4.parent.name:            # P03 (1) etc. -- duplicate-dir cache collision
            continue
        m = re.match(r"(.+)\.(\d+)\.mp4$", mp4.name)
        if not m:
            continue
        stem, cam = m.group(1), int(m.group(2))
        reps.setdefault(stem, {})[cam] = mp4
    return reps


_SENTINEL = object()


def _decode_worker(clips, q):
    """Decode every camera's frames (NVDEC) onto a bounded queue, tagged by cam+frame idx.

    Runs on its own thread so H.264 decode overlaps GPU inference: while the card computes
    one batch, ffmpeg is already decoding the next frames. The queue bound (maxsize) caps how
    far decode may run ahead, so RAM stays flat. Result parsing on the main thread reorders
    by (cam, frame) so output is identical to the serial path regardless of interleave.
    """
    try:
        for cam, path in sorted(clips.items()):
            for fi, img in enumerate(gpu_decode.frames(path)):
                q.put((cam, fi, img))
    finally:
        q.put(_SENTINEL)


def _run_rep(model, clips: dict[int, Path], batch: int, device):
    """Overlapped decode+infer for one rep. Decode thread fills a queue; GPU drains batches."""
    q: queue.Queue = queue.Queue(maxsize=batch * 3)
    t = threading.Thread(target=_decode_worker, args=(clips, q), daemon=True)
    t.start()

    # collect results as {cam: {frame_idx: parsed}} then flatten to dense per-cam lists
    got: dict[int, dict[int, dict]] = {c: {} for c in clips}
    buf_imgs, buf_tags = [], []

    def flush():
        if not buf_imgs:
            return
        results = model.predict(buf_imgs, imgsz=IMGSZ, device=device,
                                verbose=False, batch=len(buf_imgs))
        for (cam, fi), r in zip(buf_tags, results):
            got[cam][fi] = _parse_result(r)
        buf_imgs.clear(); buf_tags.clear()

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        cam, fi, img = item
        buf_imgs.append(img); buf_tags.append((cam, fi))
        if len(buf_imgs) >= batch:
            flush()
    flush()
    t.join()

    per_cam, n_frames = {}, 0
    for cam in sorted(clips):
        frames_d = got[cam]
        dense = [frames_d[i] for i in range(len(frames_d))]
        per_cam[f"cam_{cam}"] = dense
        n_frames = max(n_frames, len(dense))
    return per_cam, n_frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--model", default="s", choices=list("nsmx"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=0)
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="stop cleanly after ~N min (0 = run all). Finished reps are skipped.")
    ap.add_argument("--limit", type=int, default=0, help="cap reps (debug)")
    ap.add_argument("--match", default="drinking",
                    help="substring stems must contain (default 'drinking' = the OMC-truth "
                         "set; '' = every rep incl. ARAT)")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model_path = ROOT / "models" / f"yolo26{a.model}-pose.pt"
    model = YOLO(str(model_path))
    model.to(f"cuda:{a.device}" if str(a.device).isdigit() else a.device)
    out_name = f"yolo26{a.model}.2d.json"

    reps = _rep_clips(a.clips)
    stems = sorted(s for s in reps if a.match in s)
    if a.limit:
        stems = stems[:a.limit]

    todo = [s for s in stems if not (CACHE / s / out_name).exists()]
    print(f"cohort: {len(stems)} reps total, {len(stems) - len(todo)} already cached, "
          f"{len(todo)} to do", flush=True)
    print(f"model=yolo26{a.model} batch={a.batch} nvdec={gpu_decode.gpu_available()} "
          f"minutes={a.minutes or 'ALL'}", flush=True)

    gpu = gpu_decode.gpu_available()
    if not gpu:
        print("  WARNING: NVDEC unavailable, falling back to CPU decode (will be slow)",
              flush=True)

    t0 = time.time()
    done_frames = 0
    for i, stem in enumerate(todo):
        if a.minutes and (time.time() - t0) / 60 >= a.minutes:
            print(f"\n== hit {a.minutes} min budget, stopping cleanly ==", flush=True)
            break
        ts = time.time()
        per_cam, nfr = _run_rep(model, reps[stem], a.batch, a.device)
        (CACHE / stem).mkdir(parents=True, exist_ok=True)
        (CACHE / stem / out_name).write_text(json.dumps(per_cam))

        done_frames += nfr * len(per_cam)
        el = time.time() - t0
        rate = done_frames / el if el else 0
        eta_min = (len(todo) - i - 1) * (el / (i + 1)) / 60
        print(f"[{i+1}/{len(todo)}] {stem}  {len(per_cam)}cam x {nfr}fr  "
              f"{time.time()-ts:4.1f}s  | {rate:.0f} cam-frame/s  "
              f"elapsed {el/60:4.1f}m  ETA-all {eta_min:5.1f}m", flush=True)

    el = time.time() - t0
    print(f"\ndone: {done_frames:,} cam-frames in {el/60:.1f} min "
          f"({done_frames/el:.0f} cam-frame/s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
