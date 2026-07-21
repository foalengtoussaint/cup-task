"""Pose-model speed comparison: YOLO-pose vs RTMPose vs BlazePose, on THIS env, honestly by device.

ENV CONSTRAINT (real): onnxruntime-gpu needs CUDA 12; this env is CUDA 11.8 (torch's), so RTMPose runs
on CPU here. BlazePose (MediaPipe) is a stateful single-person graph with NO batch entry point -- CPU,
sequential, one image at a time BY DESIGN. So this compares:
  YOLO-pose : GPU, batched across cameras (native predict([N imgs]))     -- the incumbent
  RTMPose   : CPU (onnxruntime), top-down, loops person crops            -- no GPU here
  BlazePose : CPU (MediaPipe TFLite), single-person, sequential per image
Reports fps @ 1/5/10 cams (rig-frames/sec). Device labelled per model -- NOT apples-to-apples.

    python scripts/bench_pose_models_multi.py --cams 1 5 10 --frames 60
"""
from __future__ import annotations
import argparse, glob, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SP = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
          "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad")
RTM_CK = Path.home() / ".cache/rtmlib/hub/checkpoints"


def _frames(n):
    import cv2
    vids = sorted(glob.glob(str(ROOT / "cache/delta/P07/staged/*.mp4")))
    out = []
    if vids:
        cap = cv2.VideoCapture(vids[0])
        while len(out) < n:
            ok, im = cap.read()
            if not ok: break
            out.append(im)
        cap.release()
    while len(out) < n:
        out.append(np.random.randint(0, 255, (1080, 1920, 3), np.uint8))
    return out


def bench_yolo(cams, frames, device="0"):
    from ultralytics import YOLO
    import torch
    m = YOLO(str(ROOT / "models" / "yolo26n-pose.pt")); m.to(f"cuda:{device}")
    m.predict(frames[0], verbose=False, device=f"cuda:{device}")
    res = {}
    for n in cams:
        torch.cuda.synchronize(); t0 = time.perf_counter(); reps = 40
        for i in range(reps):
            b = [frames[(i + k) % len(frames)] for k in range(n)]
            m.predict(b, verbose=False, device=f"cuda:{device}")
        torch.cuda.synchronize(); res[n] = (time.perf_counter() - t0) / reps
    return res


def bench_rtmpose(cams, frames):
    try:
        from rtmlib import RTMPose
    except Exception as e:
        print("  rtmlib import failed:", e); return None
    ck = sorted(glob.glob(str(RTM_CK / "rtmpose-*.onnx")))
    if not ck: return None
    pose = RTMPose(onnx_model=ck[0], model_input_size=(288, 384), backend="onnxruntime", device="cpu")
    bbox = [[0, 0, 1920, 1080]]
    pose(frames[0], bboxes=bbox)
    res = {}
    for n in cams:
        t0 = time.perf_counter(); reps = 6
        for i in range(reps):
            for k in range(n): pose(frames[(i + k) % len(frames)], bboxes=bbox)
        res[n] = (time.perf_counter() - t0) / reps
    return res


def bench_blazepose(cams, frames):
    try:
        import mediapipe as mp, cv2
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except Exception as e:
        print("  mediapipe import failed:", e); return None
    task = SP / "blazepose" / "pose_landmarker_full.task"
    if not task.exists(): return None
    opts = vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(task)),
                                        running_mode=vision.RunningMode.IMAGE)
    lm = vision.PoseLandmarker.create_from_options(opts)
    mps = [mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    lm.detect(mps[0])
    res = {}
    for n in cams:
        t0 = time.perf_counter(); reps = 6
        for i in range(reps):
            for k in range(n): lm.detect(mps[(i + k) % len(mps)])
        res[n] = (time.perf_counter() - t0) / reps
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    frames = _frames(a.frames)
    print(f"pose-model speed, {len(frames)} real 1080p frames, cams {a.cams}\n", flush=True)
    print("running YOLO-pose (GPU, batched)...", flush=True); yolo = bench_yolo(a.cams, frames, a.device)
    print("running RTMPose (CPU, onnxruntime)...", flush=True); rtm = bench_rtmpose(a.cams, frames)
    print("running BlazePose (CPU, MediaPipe)...", flush=True); blz = bench_blazepose(a.cams, frames)
    print(f"\n{'model':22} {'device':>7}  " + "  ".join(f"{n:>2}cam fps" for n in a.cams), flush=True)
    print("-" * (33 + 11 * len(a.cams)), flush=True)
    def row(name, dev, res):
        cells = ("  ".join(f"{'n/a':>8}" for _ in a.cams) if res is None
                 else "  ".join(f"{1.0/res[n]:8.1f}" for n in a.cams))
        print(f"{name:22} {dev:>7}  {cells}", flush=True)
    row("YOLO-pose (batched)", "GPU", yolo)
    row("RTMPose (top-down)", "CPU", rtm)
    row("BlazePose (1-person)", "CPU", blz)
    print("\nfps = rig-frames/sec (all cameras, one frame). 60fps capture => need >=60.", flush=True)
    print("NOT apples-to-apples: YOLO GPU-batched; RTMPose/BlazePose CPU (RTMPose needs CUDA-12 env for\n"
          "GPU; BlazePose = single-person stateful graph, no batch entry point).", flush=True)


if __name__ == "__main__":
    main()
