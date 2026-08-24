"""(a) WHERE does predict()'s extra time go (preprocess / inference / postprocess)?
(b) Does fp16 change the OUTPUTS -- in pixels, not raw head units?

(a) uses ultralytics' own per-call `Results.speed` dict, so the split is its instrumentation, not mine.
(b) compares fp32 vs fp16 `predict()` on the SAME real frames:
      pose : per-keypoint pixel distance, plus box-conf delta
      cup  : detections gained/lost, box-centre pixel distance, conf delta
    Reported as median / p95 / max (the tail is the part that matters -- a median hides drift).

This is still an OUTPUT-level check on a handful of frames, NOT the good_frame%/3D verdict.

    python scripts/fp16_vs_fp32_outputs.py --frames 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_speed_levers import load_frames, POSE_W, CUP_W, IMGSZ  # noqa: E402


def speed_split(y, imgs, iters=30):
    """Median preprocess/inference/postprocess ms per rig-frame, from ultralytics' own timers."""
    acc = {"preprocess": [], "inference": [], "postprocess": []}
    for _ in range(iters):
        rs = y.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)
        # speed is per-image; sum across the batch = cost of one rig-frame
        for k in acc:
            acc[k].append(sum(r.speed[k] for r in rs))
    return {k: float(np.median(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--cams", type=int, default=10)
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    frames = load_frames(a.frames)

    def fp16_model(w, warm):
        """A genuinely-fp16 YOLO predictor.

        Two traps, both hit and fixed here:
          1. `predict(half=True)` is a NO-OP on a .pt in ultralytics 8.4.49 -- args.half is set but
             the params stay fp32 (verified). Comparing that to fp32 gives all-zero deltas, which
             LOOKS like "fp16 is safe" while testing nothing.
          2. Pre-casting with .model.half() then calling predict() crashes: predictor setup runs
             fuse_conv_and_bn, which mixes Half weights with float bn stats.
        So: build the predictor in fp32 (one warm call), THEN cast the AutoBackend module and set its
        .fp16 flag (that flag is what makes preprocess cast the input tensor). Dtype is asserted."""
        m = YOLO(str(w)).to("cuda:0")
        m.predict(warm, imgsz=IMGSZ, device=0, verbose=False)     # builds+fuses predictor in fp32
        m.predictor.model.model.half()
        m.predictor.model.fp16 = True
        assert next(m.predictor.model.model.parameters()).dtype == torch.float16, "cast failed"
        return m
    print(f"\n=== (a) predict() time split, {a.cams} cams, median ms per rig-frame ===", flush=True)
    imgs = [frames[k % len(frames)] for k in range(a.cams)]
    splits = {}
    for name, w in (("pose", POSE_W), ("cup", CUP_W)):
        y = YOLO(str(w)).to("cuda:0")
        y.predict(imgs, imgsz=IMGSZ, device=0, verbose=False)      # warm
        sp = speed_split(y, imgs)
        tot = sum(sp.values())
        splits[name] = sp
        print(f"  {name:5s} pre {sp['preprocess']:6.2f}  infer {sp['inference']:6.2f}  "
              f"post {sp['postprocess']:6.2f}   (sum {tot:6.2f})", flush=True)
        print(f"        -> non-inference = {tot-sp['inference']:.2f}ms "
              f"({(tot-sp['inference'])/tot:.0%} of the call)", flush=True)
        del y

    print(f"\n=== (b) fp16 vs fp32 OUTPUTS on {len(frames)} real frames ===", flush=True)
    rows = []

    # ---- pose: keypoint pixel distances ----
    y = YOLO(str(POSE_W)).to("cuda:0")
    y16 = fp16_model(POSE_W, frames[:1])
    print(f"  pose fp32 dtype {next(y.model.parameters()).dtype}, "
          f"fp16 dtype {next(y16.predictor.model.model.parameters()).dtype}", flush=True)
    d_kp, d_conf, n_both, n_only32, n_only16 = [], [], 0, 0, 0
    for i in range(0, len(frames), 10):
        batch = frames[i:i + 10]
        r32 = y.predict(batch, imgsz=IMGSZ, device=0, half=False, verbose=False)
        r16 = y16.predict(batch, imgsz=IMGSZ, device=0, half=True, verbose=False)
        for A, B in zip(r32, r16):
            a_ok = A.boxes is not None and len(A.boxes)
            b_ok = B.boxes is not None and len(B.boxes)
            if not a_ok and not b_ok:
                continue
            if a_ok and not b_ok:
                n_only32 += 1
                continue
            if b_ok and not a_ok:
                n_only16 += 1
                continue
            n_both += 1
            ia = int(A.boxes.conf.argmax()); ib = int(B.boxes.conf.argmax())
            ka = A.keypoints.xy[ia].cpu().numpy(); kb = B.keypoints.xy[ib].cpu().numpy()
            m = (ka != 0).any(1) & (kb != 0).any(1)
            if m.any():
                d_kp.extend(np.linalg.norm(ka[m] - kb[m], axis=1).tolist())
            d_conf.append(abs(float(A.boxes.conf[ia]) - float(B.boxes.conf[ib])))
    d_kp = np.array(d_kp); d_conf = np.array(d_conf)
    print(f"\n  pose  frames both-detect {n_both}, fp32-only {n_only32}, fp16-only {n_only16}",
          flush=True)
    if len(d_kp):
        print(f"        keypoint |delta| px : med {np.median(d_kp):.4f}  p95 "
              f"{np.percentile(d_kp,95):.4f}  max {d_kp.max():.4f}   "
              f"(>1px: {np.mean(d_kp>1):.2%}  >5px: {np.mean(d_kp>5):.2%})", flush=True)
        print(f"        box conf |delta|    : med {np.median(d_conf):.5f}  "
              f"max {d_conf.max():.5f}", flush=True)
        rows.append(dict(net="pose", n=len(d_kp), med=np.median(d_kp),
                         p95=np.percentile(d_kp, 95), mx=d_kp.max(),
                         frac_gt1px=float(np.mean(d_kp > 1)),
                         only32=n_only32, only16=n_only16))
    del y

    # ---- cup: detection presence + centre distance ----
    y = YOLO(str(CUP_W)).to("cuda:0")
    y16 = fp16_model(CUP_W, frames[:1])
    d_c, d_cf, both, o32, o16, neither = [], [], 0, 0, 0, 0
    for i in range(0, len(frames), 10):
        batch = frames[i:i + 10]
        r32 = y.predict(batch, imgsz=IMGSZ, device=0, half=False, verbose=False)
        r16 = y16.predict(batch, imgsz=IMGSZ, device=0, half=True, verbose=False)
        for A, B in zip(r32, r16):
            na = 0 if A.boxes is None else len(A.boxes)
            nb = 0 if B.boxes is None else len(B.boxes)
            if not na and not nb:
                neither += 1
            elif na and not nb:
                o32 += 1
            elif nb and not na:
                o16 += 1
            else:
                both += 1
                ia = int(A.boxes.conf.argmax()); ib = int(B.boxes.conf.argmax())
                ca = A.boxes.xywh[ia, :2].cpu().numpy(); cb = B.boxes.xywh[ib, :2].cpu().numpy()
                d_c.append(float(np.linalg.norm(ca - cb)))
                d_cf.append(abs(float(A.boxes.conf[ia]) - float(B.boxes.conf[ib])))
    d_c = np.array(d_c); d_cf = np.array(d_cf)
    print(f"\n  cup   both-detect {both}, fp32-only {o32}, fp16-only {o16}, neither {neither}",
          flush=True)
    if len(d_c):
        print(f"        box centre |delta| px: med {np.median(d_c):.4f}  p95 "
              f"{np.percentile(d_c,95):.4f}  max {d_c.max():.4f}", flush=True)
        print(f"        conf |delta|         : med {np.median(d_cf):.5f}  "
              f"max {d_cf.max():.5f}", flush=True)
        rows.append(dict(net="cup", n=len(d_c), med=np.median(d_c),
                         p95=np.percentile(d_c, 95), mx=d_c.max(),
                         frac_gt1px=float(np.mean(d_c > 1)), only32=o32, only16=o16))

    import pandas as pd
    outd = ROOT / "out/speed"
    outd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outd / "fp16_vs_fp32_outputs.csv", index=False)
    np.savez(outd / "fp16_vs_fp32_outputs.npz", pose_kp_px=d_kp, cup_centre_px=d_c,
             pose_conf=d_conf, cup_conf=d_cf,
             splits=np.array(str(splits)))
    print(f"\nPROCESSING CHECK: frames {len(frames)}, pose kp pairs {len(d_kp)}, "
          f"cup pairs {len(d_c)}, non-finite "
          f"{int(np.sum(~np.isfinite(d_kp)) + np.sum(~np.isfinite(d_c)))}", flush=True)
    print(f"wrote {outd/'fp16_vs_fp32_outputs.csv'} + .npz\nDONE", flush=True)


if __name__ == "__main__":
    main()
