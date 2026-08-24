"""TensorRT vs PyTorch for the live nets: export engines, benchmark, and check the outputs match.

WHY TRT is the remaining lever: fp16 in PyTorch gave 1.20x on pose because ~35% of the frame is CPU
and the forward is launch-bound at small batch. TRT fuses kernels and runs a single optimised engine,
attacking both. The worklog named it the one untapped lever; it was never installed until now.

SCOPE / EXPECTATION SETTING (measured, so the result is read honestly):
  In the live loop at 5 cams, pose is 11.1ms of a ~25ms model budget and UETrack is ~12.3ms. TRT here
  covers YOLO ONLY (pose + the cup seeder). UETrack is a custom multi-modal ViT with dict inputs and
  per-camera state -- not a drop-in ONNX/TRT export -- so even a perfect pose engine caps the live
  gain at roughly 11.1ms -> ~5ms, i.e. ~40 -> ~47 rig-fps. TRT on UETrack is a separate project.

Engines are SHAPE-SPECIFIC: one per (model, batch). Built per camera count.

    python scripts/trt_bench.py --cams 1 5 10 --iters 40          # export if missing, then bench
    python scripts/trt_bench.py --export-only --cams 5
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

POSE_W = ROOT / "models/yolo26n-pose.pt"
CUP_W = ROOT / "models/cup_clean3d_refill.pt"
ENGDIR = ROOT / "models/trt"
IMGSZ = 640


def frames_1080p(n):
    import cv2
    v = sorted(glob.glob(str(ROOT / "cache/delta/P07/staged/*.mp4")))[0]
    cap = cv2.VideoCapture(v)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def engine_path(weights: Path, batch: int) -> Path:
    return ENGDIR / f"{weights.stem}__b{batch}__{IMGSZ}__fp16.engine"


def build(weights: Path, batch: int) -> Path:
    """Export a fp16 TRT engine for a fixed batch. Ultralytics writes next to the .pt; we move it."""
    from ultralytics import YOLO
    ENGDIR.mkdir(parents=True, exist_ok=True)
    dst = engine_path(weights, batch)
    if dst.exists():
        print(f"  engine exists: {dst.name}", flush=True)
        return dst
    print(f"  building {dst.name} (this takes minutes; TRT optimises per shape)...", flush=True)
    t0 = time.time()
    m = YOLO(str(weights))
    out = m.export(format="engine", imgsz=IMGSZ, half=True, batch=batch, device=0,
                   dynamic=False, verbose=False)
    src = Path(out)
    src.rename(dst)
    print(f"  built in {time.time()-t0:.0f}s -> {dst.name}", flush=True)
    return dst


def bench(model, imgs, iters, warm=5):
    import torch
    for _ in range(warm):
        model.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
    return np.array(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--nets", nargs="+", default=["pose", "cup"])
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    try:
        import tensorrt as trt
        print(f"tensorrt {trt.__version__}", flush=True)
    except Exception as e:
        raise SystemExit(f"tensorrt not importable: {e}")

    frames = frames_1080p(12)
    nets = {"pose": POSE_W, "cup": CUP_W}
    rows = []

    for name in a.nets:
        w = nets[name]
        print(f"\n===== {name} ({w.name}) =====", flush=True)
        for ncam in a.cams:
            eng = build(w, ncam)
            if a.export_only:
                continue
            imgs = [frames[i % len(frames)] for i in range(ncam)]

            # PyTorch fp16 reference (the current best config in live_v3.py)
            pt = YOLO(str(w)).to("cuda:0")
            pt.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)
            pt.predictor.model.model.half()
            pt.predictor.model.fp16 = True
            assert next(pt.predictor.model.model.parameters()).dtype == torch.float16
            ms_pt = bench(pt, imgs, a.iters)

            tr = YOLO(str(eng), task="pose" if name == "pose" else "segment")
            ms_tr = bench(tr, imgs, a.iters)

            sp = np.median(ms_pt) / np.median(ms_tr)
            print(f"  {ncam:2d} cam: pytorch-fp16 {np.median(ms_pt):6.2f} ms   "
                  f"TRT-fp16 {np.median(ms_tr):6.2f} ms   speedup {sp:4.2f}x   "
                  f"({1000/np.median(ms_tr):5.1f} fps)", flush=True)

            # output agreement, in pixels, on the same frames
            d = []
            r1 = pt.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)
            r2 = tr.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)
            lost = 0
            for A, B in zip(r1, r2):
                na = 0 if A.boxes is None else len(A.boxes)
                nb = 0 if B.boxes is None else len(B.boxes)
                if not na or not nb:
                    lost += int(na != nb)
                    continue
                ia, ib = int(A.boxes.conf.argmax()), int(B.boxes.conf.argmax())
                if name == "pose" and A.keypoints is not None and B.keypoints is not None:
                    ka = A.keypoints.xy[ia].float().cpu().numpy()
                    kb = B.keypoints.xy[ib].float().cpu().numpy()
                    m = (ka != 0).any(1) & (kb != 0).any(1)
                    if m.any():
                        d.extend(np.linalg.norm(ka[m] - kb[m], axis=1).tolist())
                else:
                    ca = A.boxes.xywh[ia, :2].float().cpu().numpy()
                    cb = B.boxes.xywh[ib, :2].float().cpu().numpy()
                    d.append(float(np.linalg.norm(ca - cb)))
            d = np.array(d)
            if len(d):
                print(f"          output |delta| px: med {np.median(d):.3f}  p95 "
                      f"{np.percentile(d,95):.3f}  max {d.max():.3f}   detections lost/gained {lost}",
                      flush=True)
            rows.append(dict(net=name, cams=ncam, pt_ms=float(np.median(ms_pt)),
                             trt_ms=float(np.median(ms_tr)), speedup=float(sp),
                             px_med=float(np.median(d)) if len(d) else np.nan,
                             px_max=float(d.max()) if len(d) else np.nan, det_diff=lost))
            del pt, tr
            torch.cuda.empty_cache()

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        outp = ROOT / "out/speed"
        outp.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp / "trt_bench.csv", index=False)
        print("\n" + df.round(3).to_string(index=False), flush=True)
        print(f"\nPROCESSING CHECK: rows {len(df)}/{len(a.nets)*len(a.cams)}, "
              f"non-finite {int(df.trt_ms.isna().sum())}", flush=True)
        print(f"wrote {outp/'trt_bench.csv'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
