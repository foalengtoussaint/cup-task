"""Why doesn't fp16 make the PIPELINE much faster, when the isolated forward gained 1.26-1.4x?

Hypothesis: on the real pipeline the input is 1080p, so per rig-frame there is a fixed CPU cost
(letterbox/resize per camera, BGR->RGB copy, per-camera GPU->CPU result transfer, UETrack's crop)
that fp16 cannot touch. This measures the SPLIT, at 5 cams, on real 1080p frames:

  POSE     ultralytics' own preprocess / inference / postprocess timers, fp32 vs REAL fp16
  UETRACK  crop+preproc (CPU) vs batched forward (GPU) vs box mapping, fp32 vs autocast-fp16

If the fp16 gain is confined to the 'inference'/'forward' rows while the CPU rows are unchanged and
large, the pipeline speedup is capped by Amdahl, not by a mistake in how fp16 was applied.

    python scripts/where_time_goes_1080p.py --cams 5 --iters 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

POSE_W = ROOT / "models/yolo26n-pose.pt"
IMGSZ = 640


def real_1080p(n):
    """n real 1080p BGR frames from a staged DELTA clip (NOT letterboxed -- that is the point)."""
    import cv2
    import glob
    vids = sorted(glob.glob(str(ROOT / "cache/delta/P07/staged/*.mp4")))
    cap = cv2.VideoCapture(vids[0])
    out = []
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    print(f"  {len(out)} frames at {out[0].shape[1]}x{out[0].shape[0]} from {Path(vids[0]).name}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    frames = real_1080p(12)
    batch = [frames[i % len(frames)] for i in range(a.cams)]
    rows = []

    print(f"\n===== POSE predict() split, {a.cams} cams, 1080p input (ms per rig-frame) =====",
          flush=True)
    for dtype in ("fp32", "fp16"):
        m = YOLO(str(POSE_W)).to("cuda:0")
        m.predict(batch, imgsz=IMGSZ, device=0, verbose=False)
        if dtype == "fp16":
            m.predictor.model.model.half()
            m.predictor.model.fp16 = True
            assert next(m.predictor.model.model.parameters()).dtype == torch.float16
        acc = {"preprocess": [], "inference": [], "postprocess": []}
        wall = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            rs = m.predict(batch, imgsz=IMGSZ, device=0, verbose=False)
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1e3)
            for k in acc:
                acc[k].append(sum(r.speed[k] for r in rs))
        med = {k: float(np.median(v)) for k, v in acc.items()}
        w = float(np.median(wall))
        print(f"  {dtype}: pre {med['preprocess']:6.2f}  infer {med['inference']:6.2f}  "
              f"post {med['postprocess']:6.2f}   wall {w:6.2f}", flush=True)
        rows.append(dict(stage="pose", dtype=dtype, wall_ms=w, **med))
        del m
        torch.cuda.empty_cache()

    print(f"\n===== UETRACK update() split, {a.cams} cams, 1080p (ms per rig-frame) =====",
          flush=True)
    from uetrack_wrap import UETrackBatch
    rgbs = [f[:, :, ::-1].copy() for f in frames]
    b = UETrackBatch(a.cams)
    for c in range(a.cams):
        b.init(c, rgbs[0], (900, 700, 60, 60))

    # time the CPU crop stage alone (what sample_target costs for N cams)
    st = b.states[0]
    t0 = time.perf_counter()
    for i in range(a.iters):
        for c in range(a.cams):
            b._sample_target(rgbs[i % len(rgbs)], b.states[c]["state"],
                             b._t.params.search_factor, output_sz=b._t.params.search_size)
    crop_ms = (time.perf_counter() - t0) / a.iters * 1e3

    # rgb copy cost per rig-frame (the BGR->RGB the loop must do anyway)
    t0 = time.perf_counter()
    for i in range(a.iters):
        for c in range(a.cams):
            _ = frames[i % len(frames)][:, :, ::-1].copy()
    rgbcopy_ms = (time.perf_counter() - t0) / a.iters * 1e3

    for dtype in ("fp32", "fp16"):
        ctx = (torch.autocast("cuda", dtype=torch.float16) if dtype == "fp16"
               else torch.autocast("cuda", enabled=False))
        with ctx:
            for _ in range(5):
                b.update([rgbs[0]] * a.cams)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctx:
            for i in range(a.iters):
                b.update([rgbs[i % len(rgbs)]] * a.cams)
        torch.cuda.synchronize()
        tot = (time.perf_counter() - t0) / a.iters * 1e3
        print(f"  {dtype}: update total {tot:6.2f}   of which CPU crop {crop_ms:6.2f} "
              f"({crop_ms/tot:.0%})   [BGR->RGB copy, separate: {rgbcopy_ms:6.2f}]", flush=True)
        rows.append(dict(stage="uetrack", dtype=dtype, wall_ms=tot, preprocess=crop_ms,
                         inference=tot - crop_ms, postprocess=float("nan")))

    import pandas as pd
    df = pd.DataFrame(rows)
    outp = ROOT / "out/speed"
    outp.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp / "where_time_goes_1080p.csv", index=False)
    print("\n" + df.round(2).to_string(index=False), flush=True)

    p = df[df.stage == "pose"].set_index("dtype")
    u = df[df.stage == "uetrack"].set_index("dtype")
    print(f"\n  POSE   : inference {p.loc['fp32','inference']:.2f} -> {p.loc['fp16','inference']:.2f}ms "
          f"({p.loc['fp32','inference']/p.loc['fp16','inference']:.2f}x)   but non-inference "
          f"{p.loc['fp32','preprocess']+p.loc['fp32','postprocess']:.2f}ms is fp16-PROOF", flush=True)
    print(f"  UETRACK: total {u.loc['fp32','wall_ms']:.2f} -> {u.loc['fp16','wall_ms']:.2f}ms "
          f"({u.loc['fp32','wall_ms']/u.loc['fp16','wall_ms']:.2f}x), CPU crop "
          f"{crop_ms:.2f}ms of it", flush=True)
    print(f"\nPROCESSING CHECK: rows {len(df)}/4, non-finite wall "
          f"{int(df.wall_ms.isna().sum())}", flush=True)
    print(f"wrote {outp/'where_time_goes_1080p.csv'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
