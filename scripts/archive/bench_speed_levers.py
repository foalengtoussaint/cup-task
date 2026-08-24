"""Are there free speed levers left on the two live nets (pose + cup)?

Shipping baseline (docs/WORKLOG.md): 26fps @ 10 cams, batched + separate CUDA streams, imgsz=640,
fp32 .pt. The worklog names TensorRT as the one untapped lever and records that `half=True` on a
`.pt` is a NO-OP through ultralytics `predict()`. TensorRT is NOT installed here (and onnxruntime is
a CPU-only build), so this script measures the levers that need NO new dependency:

  predict_fp32        ultralytics predict() -- the shipping path (letterbox + NMS included)
  raw_fp32            raw torch module on a pre-letterboxed batched tensor (no NMS/py overhead)
  raw_fp16            .half() on the raw module          <- real fp16, not the predict() no-op
  raw_fp16_cl         + channels_last memory format
  raw_fp16_compile    + torch.compile(mode="reduce-overhead")  <- CUDA graphs, attacks the launch bound

Also reports fp16-vs-fp32 output divergence on the SAME batch (raw head tensors) as a first-pass
numerical guard. That is NOT an accuracy result -- a real verdict needs good_frame% / 3D error on the
cache; this only says whether fp16 changes the tensors at all.

Per-config timings are saved to out/speed/bench_speed_levers.npz + .csv (never just the medians).

    python scripts/bench_speed_levers.py --cams 1 5 10 --iters 60
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

STAGED = ROOT / "cache/delta/P07/staged"
POSE_W = ROOT / "models/yolo26n-pose.pt"
CUP_W = ROOT / "models/cup_clean3d_refill.pt"
IMGSZ = 640


def load_frames(n, imgsz=IMGSZ):
    """n real letterboxed frames (HWC uint8 RGB) from one DELTA clip."""
    import cv2
    vids = sorted(STAGED.glob("*.mp4"))
    if not vids:
        raise SystemExit(f"no clips in {STAGED}")
    cap = cv2.VideoCapture(str(vids[0]))
    out = []
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        s = imgsz / max(h, w)
        r = cv2.resize(fr, (int(round(w * s)), int(round(h * s))))
        pad = np.full((imgsz, imgsz, 3), 114, np.uint8)
        pad[:r.shape[0], :r.shape[1]] = r
        out.append(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise SystemExit(f"could not decode {vids[0]}")
    print(f"  frames: {len(out)} from {vids[0].name}", flush=True)
    return out


def to_batch(frames, ncam, i, device, dtype, channels_last=False):
    import torch
    idx = [(i + k) % len(frames) for k in range(ncam)]
    x = np.stack([frames[j] for j in idx]).transpose(0, 3, 1, 2)
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device).to(dtype).div_(255.0)
    if channels_last:
        t = t.contiguous(memory_format=torch.channels_last)
    return t


def first_tensor(o):
    """Raw head output can be a tensor, a tuple, or a nested tuple (seg nets: (preds, (protos,...))).
    Return the first tensor found, depth-first."""
    import torch
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (list, tuple)):
        for it in o:
            t = first_tensor(it)
            if t is not None:
                return t
    return None


def save(rows, per_iter):
    """Write CSV+npz NOW (called after every net, so a later crash can't lose earlier data)."""
    import pandas as pd
    outd = ROOT / "out/speed"
    outd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outd / "bench_speed_levers.csv", index=False)
    np.savez(outd / "bench_speed_levers.npz", **per_iter,
             keys=np.array(list(per_iter)))
    return outd


def timed(fn, iters, warmup, label):
    """Return per-iteration ms (np array). Prints live progress every 20 iters."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = np.empty(iters)
    t0 = time.time()
    for k in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ms[k] = (time.perf_counter() - t) * 1e3
        if (k + 1) % 20 == 0:
            print(f"      {label} [{k+1}/{iters}] {time.time()-t0:4.0f}s  "
                  f"running med {np.median(ms[:k+1]):.2f}ms", flush=True)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--net", choices=["pose", "cup", "both"], default="both")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    dev = "cuda:0"
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}", flush=True)
    frames = load_frames(a.frames)

    nets = {"pose": POSE_W, "cup": CUP_W}
    if a.net != "both":
        nets = {a.net: nets[a.net]}

    rows = []
    per_iter = {}

    for name, wpath in nets.items():
        print(f"\n===== {name}  ({wpath.name}) =====", flush=True)
        y = YOLO(str(wpath))
        y.to(dev)
        base = y.model.eval()

        # a reference fp32 output for the numerical guard
        with torch.inference_mode():
            xr = to_batch(frames, 1, 0, dev, torch.float32)
            ref = first_tensor(base(xr)).float().clone()

        variants = {}
        variants["raw_fp32"] = (base, torch.float32, False)
        half = YOLO(str(wpath)).to(dev).model.eval().half()
        variants["raw_fp16"] = (half, torch.float16, False)
        variants["raw_fp16_cl"] = (half.to(memory_format=torch.channels_last),
                                   torch.float16, True)

        for ncam in a.cams:
            print(f"\n  --- {name}, {ncam} cam(s) ---", flush=True)

            # (0) shipping path: ultralytics predict() on a list of ncam frames
            imgs = [frames[k % len(frames)] for k in range(ncam)]
            def _pred():
                y.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)
            ms = timed(_pred, a.iters, a.warmup, f"predict_fp32 c{ncam}")
            rows.append(dict(net=name, cfg="predict_fp32", cams=ncam,
                             med_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95))))
            per_iter[f"{name}|predict_fp32|{ncam}"] = ms

            # (1..3) raw module variants
            for cfg, (mod, dt, cl) in variants.items():
                x = to_batch(frames, ncam, 0, dev, dt, cl)
                def _raw(mod=mod, x=x):
                    with torch.inference_mode():
                        mod(x)
                ms = timed(_raw, a.iters, a.warmup, f"{cfg} c{ncam}")
                rows.append(dict(net=name, cfg=cfg, cams=ncam,
                                 med_ms=float(np.median(ms)),
                                 p95_ms=float(np.percentile(ms, 95))))
                per_iter[f"{name}|{cfg}|{ncam}"] = ms

            # (4) torch.compile reduce-overhead (CUDA graphs) on the fp16 module
            try:
                comp = torch.compile(variants["raw_fp16"][0], mode="reduce-overhead")
                x = to_batch(frames, ncam, 0, dev, torch.float16)
                def _c(comp=comp, x=x):
                    with torch.inference_mode():
                        comp(x)
                t0 = time.time()
                _c()
                print(f"      compile warm {time.time()-t0:.0f}s", flush=True)
                ms = timed(_c, a.iters, a.warmup, f"raw_fp16_compile c{ncam}")
                rows.append(dict(net=name, cfg="raw_fp16_compile", cams=ncam,
                                 med_ms=float(np.median(ms)),
                                 p95_ms=float(np.percentile(ms, 95))))
                per_iter[f"{name}|raw_fp16_compile|{ncam}"] = ms
            except Exception as e:
                print(f"      raw_fp16_compile FAILED: {type(e).__name__}: {e}", flush=True)
                rows.append(dict(net=name, cfg="raw_fp16_compile", cams=ncam,
                                 med_ms=np.nan, p95_ms=np.nan))

        # numerical guard: fp16 vs fp32 raw head output, 1 cam
        with torch.inference_mode():
            o = first_tensor(variants["raw_fp16"][0](to_batch(frames, 1, 0, dev, torch.float16)))
            d = (o.float() - ref).abs()
        print(f"\n  fp16-vs-fp32 head output: max {d.max():.4g}  median {d.median():.4g}  "
              f"(ref |x| median {ref.abs().median():.4g})", flush=True)
        rows.append(dict(net=name, cfg="fp16_maxabsdiff", cams=0,
                         med_ms=float(d.median()), p95_ms=float(d.max())))
        save(rows, per_iter)
        del y, base, half, variants
        torch.cuda.empty_cache()

    import pandas as pd
    outd = save(rows, per_iter)
    df = pd.DataFrame(rows)

    print("\n===== SUMMARY: median ms per rig-frame (fps = 1000/ms) =====", flush=True)
    t = df[df.cfg != "fp16_maxabsdiff"]
    for name in t.net.unique():
        print(f"\n{name}", flush=True)
        piv = t[t.net == name].pivot(index="cfg", columns="cams", values="med_ms")
        print(piv.round(2).to_string(), flush=True)
        print("fps:", flush=True)
        print((1000 / piv).round(1).to_string(), flush=True)

    n_exp = len(nets) * len(a.cams) * 5
    print(f"\nPROCESSING CHECK: configs {len(t)}/{n_exp}, "
          f"non-finite {int(t.med_ms.isna().sum())}", flush=True)
    print(f"wrote {outd/'bench_speed_levers.csv'} + .npz\nDONE", flush=True)


if __name__ == "__main__":
    main()
