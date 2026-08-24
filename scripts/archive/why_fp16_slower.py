"""WHY is fp16 slower than fp32 at batch=1 (and faster at batch=10)?

Decisive test: separate GPU-BUSY time from WALL time, and count kernel launches.
  - if fp16 GPU-busy DROPS but wall does not -> launch/overhead-bound, math was never the limit
  - if fp16 launches MORE kernels than fp32 -> the .half() model is inserting casts
  - per-kernel dtype breakdown shows whether convs actually hit fp16 tensor-core kernels

Run: python scripts/why_fp16_slower.py
Saves out/speed/why_fp16_slower.csv (per-config wall/busy/launch counts) + prints the top kernels.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_speed_levers import load_frames, to_batch, POSE_W, IMGSZ  # noqa: E402


def measure(mod, x, iters=40):
    """wall ms/iter, GPU-busy ms/iter, kernel launches/iter."""
    import torch
    from torch.profiler import profile, ProfilerActivity

    with torch.inference_mode():
        for _ in range(8):
            mod(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mod(x)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3 / iters

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            for _ in range(10):
                mod(x)
            torch.cuda.synchronize()

    # KERNEL-level events ONLY. key_averages() also contains the CPU-side aten:: op with the same
    # device time attributed to it -- summing both double-counts (it produced negative idle time).
    evs = [e for e in pr.key_averages()
           if e.device_type.name == "CUDA" and not e.key.startswith("aten::")]
    busy = sum(e.self_device_time_total for e in evs) / 1e3 / 10
    launches = sum(e.count for e in evs if e.self_device_time_total > 0) / 10
    top = sorted([e for e in evs if e.self_device_time_total > 0],
                 key=lambda e: -e.self_device_time_total)[:6]
    return wall, busy, launches, [(e.key[:52], e.self_device_time_total / 1e3 / 10, e.count / 10)
                                  for e in top]


def main():
    import torch
    from ultralytics import YOLO

    dev = "cuda:0"
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    frames = load_frames(4)

    f32 = YOLO(str(POSE_W)).to(dev).model.eval()
    f16 = YOLO(str(POSE_W)).to(dev).model.eval().half()
    f16cl = YOLO(str(POSE_W)).to(dev).model.eval().half().to(memory_format=torch.channels_last)

    rows = []
    for ncam in (1, 10):
        for cfg, mod, dt, cl in (("fp32", f32, torch.float32, False),
                                 ("fp16", f16, torch.float16, False),
                                 ("fp16_cl", f16cl, torch.float16, True)):
            x = to_batch(frames, ncam, 0, dev, dt, cl)
            wall, busy, nl, top = measure(mod, x)
            idle = wall - busy
            print(f"\n--- {ncam} cam, {cfg}: wall {wall:.2f}ms  GPU-busy {busy:.2f}ms  "
                  f"IDLE {idle:.2f}ms ({idle/wall:.0%})  launches {nl:.0f}", flush=True)
            for k, ms, c in top:
                print(f"      {ms:6.3f}ms x{c:4.0f}  {k}", flush=True)
            rows.append(dict(cams=ncam, cfg=cfg, wall_ms=wall, busy_ms=busy,
                             idle_ms=idle, idle_frac=idle / wall, launches=nl))

    import pandas as pd
    df = pd.DataFrame(rows)
    outd = ROOT / "out/speed"
    outd.mkdir(parents=True, exist_ok=True)
    df.to_csv(outd / "why_fp16_slower.csv", index=False)
    print("\n" + df.round(2).to_string(index=False), flush=True)
    print(f"\nPROCESSING CHECK: configs {len(df)}/6, non-finite {int(df.wall_ms.isna().sum())}",
          flush=True)
    print(f"wrote {outd/'why_fp16_slower.csv'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
