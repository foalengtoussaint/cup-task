"""GPU (NVDEC) video frame decoding via ffmpeg, with a CPU fallback.

Copied from object_tracking/experiments/drink_study/lib/gpu_decode.py (the working batch-job
decoder) — stripped of the drink_study lib path shim so it's self-contained in cup-task. This
cv2 build has no CUDA, but ffmpeg ships NVDEC (`-hwaccel cuda -c:v h264_cuvid`); we decode on the
GPU's dedicated NVDEC engine and pipe raw BGR24 to numpy, offloading the CPU-bound H.264 decode
that bottlenecks the label builder.

    from gpu_decode import frames, dims
    for img in frames(path):        # BGR uint8 (h,w,3), same as cv2 cap.read()
        ...

Set OT_FORCE_CPU_DECODE=1 to force the cv2 path. gpu_available() reports NVDEC h264 usability.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

import cv2
import numpy as np


def dims(path) -> tuple[int, int, int, float]:
    c = cv2.VideoCapture(str(path))
    w, h = int(c.get(3)), int(c.get(4))
    n, fps = int(c.get(7)), c.get(5)
    c.release()
    return w, h, n, fps


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    if os.environ.get("OT_FORCE_CPU_DECODE"):
        return False
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                             capture_output=True, text=True, timeout=10).stdout
        return "h264_cuvid" in out
    except Exception:
        return False


def _gpu_frames(path, w, h):
    fsize = w * h * 3
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-hwaccel", "cuda", "-c:v", "h264_cuvid", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=fsize)
    try:
        while True:
            buf = proc.stdout.read(fsize)
            if len(buf) < fsize:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.wait()


def _cpu_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
    finally:
        cap.release()


def frames(path):
    """Yield BGR uint8 frames; GPU (NVDEC) if available, else cv2 CPU."""
    if gpu_available():
        w, h, _, _ = dims(path)
        if w > 0 and h > 0:
            yield from _gpu_frames(path, w, h)
            return
    yield from _cpu_frames(path)
