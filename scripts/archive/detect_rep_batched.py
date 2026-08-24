"""Batched cup + pose detection for one rep, across all camera clips at once.

Same output as scripts/detect_rep.py (<clip>.cup.json + <clip>.pose.json, ready for
triangulate.py) but the GPU-bound part is batched + NVDEC-overlapped instead of serial,
the way scripts/cache_pose_cohort.py does it for pose alone. detect_rep.py runs each clip
through model.predict(stream=True) one frame at a time (~70-90 fps/clip); here a decode
thread fills a bounded queue while the GPU drains batches of both nets, so 10 cams x ~710
frames finishes in ~30s instead of ~2min.

Both nets see the SAME decoded frame batch (decode once, infer twice), at imgsz 640 (the
training size -- 1280 is the documented recall-killing bug). Parsing reuses the exact
cup_detect / pose_keypoints logic so the JSON is byte-shape-identical to the serial path.

    python scripts/detect_rep_batched.py STAGE_DIR REP_PREFIX -o OUT_DIR
    # STAGE_DIR holds REP_PREFIX.<cam>.mp4 (same layout detect_rep.py globs)
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cup_detect import CUP_CLASS, MIN_CONF, CupDet, to_payload as cup_payload
from pipeline.pose_keypoints import (KP_INDEX, MIN_KP_CONF, TASK_KP, FramePose,
                                     to_payload as pose_payload)

OT_LIB = Path("/home/imove/Documents/object_tracking/experiments/drink_study/lib")
sys.path.insert(0, str(OT_LIB))
import gpu_decode  # noqa: E402

IMGSZ = 640
_SENTINEL = object()


def _parse_cup(result, frame_i):
    """One ultralytics Results -> CupDet, exactly as cup_detect._pick_cup does."""
    b = result.boxes
    if b is None or len(b) == 0:
        return CupDet(frame=frame_i, conf=0.0, box=None, center=None)
    cls = b.cls.cpu().numpy()
    conf = b.conf.cpu().numpy()
    idx = [i for i in range(len(conf)) if cls[i] == CUP_CLASS]
    if not idx:
        return CupDet(frame=frame_i, conf=0.0, box=None, center=None)
    best = max(idx, key=lambda i: conf[i])
    if conf[best] < MIN_CONF:
        return CupDet(frame=frame_i, conf=0.0, box=None, center=None)
    box = [float(v) for v in b.xyxy[best].cpu().numpy()]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return CupDet(frame=frame_i, conf=float(conf[best]), box=box, center=[cx, cy])


def _parse_pose(result, frame_i):
    """One ultralytics Results -> FramePose, exactly as pose_keypoints.extract_pose does."""
    if result.boxes is None or len(result.boxes) == 0:
        return FramePose(frame=frame_i, box_conf=0.0, kps={})
    confs = result.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    kxy = result.keypoints.xy[best].cpu().numpy()
    kconf = result.keypoints.conf[best].cpu().numpy()
    kps = {}
    for name in TASK_KP:
        j = KP_INDEX[name]
        c = float(kconf[j])
        if c >= MIN_KP_CONF:
            kps[name] = [float(kxy[j, 0]), float(kxy[j, 1]), c]
    return FramePose(frame=frame_i, box_conf=float(confs[best]), kps=kps)


def _decode_worker(clips, q):
    try:
        for cam, path in sorted(clips.items()):
            for fi, img in enumerate(gpu_decode.frames(path)):
                q.put((cam, fi, img))
    finally:
        q.put(_SENTINEL)


def _run_rep(cup_model, pose_model, clips, batch, device):
    q: queue.Queue = queue.Queue(maxsize=batch * 3)
    t = threading.Thread(target=_decode_worker, args=(clips, q), daemon=True)
    t.start()
    cup_got = {c: {} for c in clips}
    pose_got = {c: {} for c in clips}
    buf_imgs, buf_tags = [], []

    # The two nets run concurrently on SEPARATE CUDA streams so their kernels interleave on
    # the device (realtime.md finding #3 -- two threads on the default stream only overlap
    # CPU work; a stream each lets the GPU run both). The stream context is thread-local, so
    # it MUST be entered inside the worker thread, and we sync BOTH streams before reading
    # results (else we'd time launches, not completions). Each stream warms up separately.
    s_cup, s_pose = torch.cuda.Stream(), torch.cuda.Stream()
    ex = ThreadPoolExecutor(max_workers=2)

    def _infer(stream, model, imgs, **kw):
        with torch.cuda.stream(stream):
            return model.predict(imgs, imgsz=IMGSZ, device=device, verbose=False,
                                 batch=len(imgs), **kw)

    def flush():
        if not buf_imgs:
            return
        fc = ex.submit(_infer, s_cup, cup_model, buf_imgs, conf=MIN_CONF)
        fp = ex.submit(_infer, s_pose, pose_model, buf_imgs)
        cup_res, pose_res = fc.result(), fp.result()
        torch.cuda.synchronize()
        for (cam, fi), cr, pr in zip(buf_tags, cup_res, pose_res):
            cup_got[cam][fi] = _parse_cup(cr, fi)
            pose_got[cam][fi] = _parse_pose(pr, fi)
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
    ex.shutdown()
    t.join()

    cup_dense = {c: [cup_got[c][i] for i in range(len(cup_got[c]))] for c in clips}
    pose_dense = {c: [pose_got[c][i] for i in range(len(pose_got[c]))] for c in clips}
    return cup_dense, pose_dense


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage_dir", type=Path)
    ap.add_argument("rep_prefix")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--cup-model", default="models/cup_clean3d_refill.pt")
    ap.add_argument("--pose-model", default="models/yolo26s-pose.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=0)
    a = ap.parse_args(argv)

    clips = {}
    for mp4 in sorted(a.stage_dir.glob(f"{a.rep_prefix}.*.mp4")):
        m = re.match(rf"{re.escape(a.rep_prefix)}\.(\d+)\.mp4$", mp4.name)
        if m:
            clips[int(m.group(1))] = mp4
    if not clips:
        print(f"no clips matching {a.rep_prefix}.*.mp4 in {a.stage_dir}", flush=True)
        return 1
    print(f"rep {a.rep_prefix}: {len(clips)} cams  batch={a.batch} "
          f"nvdec={gpu_decode.gpu_available()}", flush=True)

    from ultralytics import YOLO
    cup_model = YOLO(str(a.cup_model)); cup_model.to(f"cuda:{a.device}")
    pose_model = YOLO(str(a.pose_model)); pose_model.to(f"cuda:{a.device}")

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cup_dense, pose_dense = _run_rep(cup_model, pose_model, clips, a.batch, a.device)

    total_fr = 0
    for cam, clip in sorted(clips.items()):
        cups, poses = cup_dense[cam], pose_dense[cam]
        (a.out / f"{clip.stem}.cup.json").write_text(json.dumps(cup_payload(clip, cups, str(a.cup_model))))
        (a.out / f"{clip.stem}.pose.json").write_text(json.dumps(pose_payload(clip, poses, str(a.pose_model))))
        n = len(cups)
        total_fr += n
        cup_n = sum(1 for d in cups if d.box)
        pose_n = sum(1 for f in poses if f.kps)
        print(f"  cam{cam:<2} {n}fr  cup {cup_n:3d} ({cup_n/n*100:2.0f}%)  "
              f"person {pose_n:3d} ({pose_n/n*100:3.0f}%)", flush=True)

    el = time.time() - t0
    print(f"done: {total_fr} cam-frames x2 nets in {el:.1f}s ({total_fr/el:.0f} cam-frame/s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
