"""Pose-only, MULTI-rep batched detector: load yolo26s-pose ONCE, loop over many reps in-process.

Why: scripts/detect_rep_batched.py is per-rep (a fresh `python` per rep reimports torch + reloads
BOTH nets ~5-8s -- that dominated the 14s/rep wall time, GPU idle most of it). Here the model loads
once and every rep reuses it; and we skip the CUP net entirely (the GNN only needs pose), halving GPU
work. Output is byte-identical <clip>.pose.json (reuses pose_payload), so gnn_build_dataset.py reads it
unchanged. Resumable: skips reps whose .pose.json already exist.

    python scripts/detect_pose_multi.py --parts P07 P08          # all staged reps missing dets
"""
from __future__ import annotations

import argparse, json, queue, re, sys, threading, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task.pose_keypoints import to_payload as pose_payload
OT_LIB = Path("/home/imove/Documents/object_tracking/experiments/drink_study/lib")
sys.path.insert(0, str(OT_LIB))
import gpu_decode  # noqa: E402
# reuse the exact parse + decode-worker from the single-rep batched detector
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_rep_batched as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DELTA = ROOT / "cache" / "delta"
IMGSZ = D.IMGSZ


def _run_pose(model, clips, batch, device):
    """Pose-only version of D._run_rep: single net, single stream, decode-overlapped."""
    q: queue.Queue = queue.Queue(maxsize=batch * 3)
    t = threading.Thread(target=D._decode_worker, args=(clips, q), daemon=True)
    t.start()
    got = {c: {} for c in clips}
    buf_imgs, buf_tags = [], []

    def flush():
        if not buf_imgs:
            return
        res = model.predict(buf_imgs, imgsz=IMGSZ, device=device, verbose=False, batch=len(buf_imgs))
        for (cam, fi), pr in zip(buf_tags, res):
            got[cam][fi] = D._parse_pose(pr, fi)
        buf_imgs.clear(); buf_tags.clear()

    while True:
        item = q.get()
        if item is D._SENTINEL:
            break
        cam, fi, img = item
        buf_imgs.append(img); buf_tags.append((cam, fi))
        if len(buf_imgs) >= batch:
            flush()
    flush(); t.join()
    return {c: [got[c][i] for i in range(len(got[c]))] for c in clips}


def _rep_prefixes(stage: Path):
    pres = set()
    for mp4 in stage.glob("*.mp4"):
        m = re.match(r"(.+)\.(\d+)\.mp4$", mp4.name)
        if m:
            pres.add(m.group(1))
    return sorted(pres)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["P07", "P08"])
    ap.add_argument("--pose-model", default="models/yolo26s-pose.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=0)
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(str(ROOT / a.pose_model)); model.to(f"cuda:{a.device}")
    print(f"loaded {a.pose_model} once | batch={a.batch} nvdec={gpu_decode.gpu_available()}", flush=True)

    t0 = time.time(); done_frames = 0
    for part in a.parts:
        stage = DELTA / part / "staged"; dets = DELTA / part / "dets"; dets.mkdir(parents=True, exist_ok=True)
        pres = _rep_prefixes(stage)
        todo = [p for p in pres if not list(dets.glob(f"{p}.*.pose.json"))]
        print(f"\n### {part}: {len(pres)} staged reps, {len(pres)-len(todo)} have dets, {len(todo)} to do", flush=True)
        for i, pre in enumerate(todo):
            clips = {}
            for mp4 in sorted(stage.glob(f"{pre}.*.mp4")):
                m = re.match(rf"{re.escape(pre)}\.(\d+)\.mp4$", mp4.name)
                if m:
                    clips[int(m.group(1))] = mp4
            if not clips:
                continue
            ts = time.time()
            dense = _run_pose(model, clips, a.batch, a.device)
            nfr = 0
            for cam, clip in sorted(clips.items()):
                poses = dense[cam]
                (dets / f"{clip.stem}.pose.json").write_text(json.dumps(pose_payload(clip, poses, str(a.pose_model))))
                nfr = max(nfr, len(poses))
            done_frames += nfr * len(clips)
            el = time.time() - t0
            eta = (len(todo) - i - 1) * (el / (i + 1)) / 60
            print(f"  [{i+1:3d}/{len(todo)}] {pre[:34]:34} {len(clips)}cam x {nfr}fr  "
                  f"{time.time()-ts:4.1f}s | {done_frames/el:.0f} cam-fr/s  ETA {eta:4.1f}m", flush=True)
    el = time.time() - t0
    print(f"\ndone: {done_frames:,} cam-frames in {el/60:.1f} min ({done_frames/max(el,1):.0f} cam-fr/s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
