"""Is the pipeline actually BATCHING across cameras, or silently looping per camera?

Two independent checks, because an fps number alone cannot tell you:

  (1) STRUCTURAL: hook the nets and count forward calls per rig-frame + record the batch dim.
      Correct batching = ONE call with batch=N. A per-camera loop = N calls with batch=1.
  (2) SCALING: cost at N cams vs N x cost at 1 cam. Batching shows sublinear growth; a loop
      shows ~linear. Reported as the "batch efficiency" = (N * t1) / tN  (1.0 = no benefit,
      N = perfect).

Run on real 1080p frames, for pose (ultralytics predict) and UETrackBatch, at 1/5/10 cams.

    python scripts/verify_batching.py --cams 1 5 10 --iters 30
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO
    from uetrack_wrap import UETrackBatch

    frames = frames_1080p(12)
    rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    rows = []

    # ---------------- POSE ----------------
    print("\n===== POSE (ultralytics predict) =====", flush=True)
    m = YOLO(str(POSE_W)).to("cuda:0")
    m.predict(frames[:1], imgsz=IMGSZ, device=0, verbose=False)
    calls = {"n": 0, "shapes": []}
    real_fwd = m.predictor.model.model.forward

    def counting_fwd(x, *args, **kw):
        calls["n"] += 1
        calls["shapes"].append(tuple(x.shape) if hasattr(x, "shape") else None)
        return real_fwd(x, *args, **kw)
    m.predictor.model.model.forward = counting_fwd

    t1 = None
    for ncam in a.cams:
        batch = [frames[i % len(frames)] for i in range(ncam)]
        calls["n"] = 0; calls["shapes"] = []
        m.predict(batch, imgsz=IMGSZ, device=0, verbose=False)
        nc, shp = calls["n"], calls["shapes"][0] if calls["shapes"] else None
        for _ in range(3):
            m.predict(batch, imgsz=IMGSZ, device=0, verbose=False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(a.iters):
            m.predict(batch, imgsz=IMGSZ, device=0, verbose=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / a.iters * 1e3
        if ncam == 1:
            t1 = ms
        eff = (ncam * t1) / ms if t1 else float("nan")
        print(f"  {ncam:2d} cam: forward calls/rig-frame = {nc}  first batch shape {shp}   "
              f"{ms:6.2f} ms   batch-efficiency {eff:4.2f}x of {ncam}", flush=True)
        rows.append(dict(net="pose", cams=ncam, fwd_calls=nc, batch_dim=shp[0] if shp else None,
                         ms=ms, efficiency=eff))
    m.predictor.model.model.forward = real_fwd
    del m
    torch.cuda.empty_cache()

    # ---------------- UETRACK ----------------
    print("\n===== UETRACK (UETrackBatch) =====", flush=True)
    t1 = None
    for ncam in a.cams:
        b = UETrackBatch(ncam)
        for c in range(ncam):
            b.init(c, rgbs[0], (900, 700, 60, 60))
        calls = {"n": 0, "shapes": []}
        real_enc = b.net.forward_encoder

        def counting_enc(tmpl, search, *args, **kw):
            calls["n"] += 1
            s = search[0] if isinstance(search, (list, tuple)) else search
            calls["shapes"].append(tuple(s.shape) if hasattr(s, "shape") else None)
            return real_enc(tmpl, search, *args, **kw)
        b.net.forward_encoder = counting_enc

        b.update([rgbs[0]] * ncam)
        nc, shp = calls["n"], calls["shapes"][0] if calls["shapes"] else None
        b.net.forward_encoder = real_enc
        for _ in range(3):
            b.update([rgbs[0]] * ncam)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(a.iters):
            b.update([rgbs[i % len(rgbs)]] * ncam)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / a.iters * 1e3
        if ncam == 1:
            t1 = ms
        eff = (ncam * t1) / ms if t1 else float("nan")
        print(f"  {ncam:2d} cam: encoder calls/rig-frame = {nc}  search batch shape {shp}   "
              f"{ms:6.2f} ms   batch-efficiency {eff:4.2f}x of {ncam}", flush=True)
        rows.append(dict(net="uetrack", cams=ncam, fwd_calls=nc,
                         batch_dim=shp[0] if shp else None, ms=ms, efficiency=eff))
        del b
        torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)
    outp = ROOT / "out/speed"
    outp.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp / "verify_batching.csv", index=False)
    print("\n" + df.round(2).to_string(index=False), flush=True)
    bad = df[(df.cams > 1) & (df.fwd_calls > 1)]
    print(f"\n  VERDICT: {'BATCHING BROKEN (>1 forward per rig-frame)' if len(bad) else 'batching OK: exactly 1 forward per rig-frame at every camera count'}",
          flush=True)
    print(f"\nPROCESSING CHECK: rows {len(df)}/{2*len(a.cams)}, non-finite ms "
          f"{int(df.ms.isna().sum())}", flush=True)
    print(f"wrote {outp/'verify_batching.csv'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
