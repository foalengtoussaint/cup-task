"""Whole-pipeline fp32 vs fp16 A/B on real DELTA trials: SPEED, and does it move the 3D points?

Runs the v3 online path on GPU per trial, once in fp32 and once in fp16:
    decode (NVDEC if available) -> YOLO-pose batched across cams -> detect-once UETrack cup
    -> triangulate pose joints (robust consensus) -> triangulate cup (consensus3 via cup_track)

fp16 is applied to BOTH nets:
  * YOLO-pose : build+fuse the predictor in fp32, then cast the AutoBackend module and set .fp16
                (`predict(half=True)` alone is a NO-OP on a .pt in ultralytics 8.4.49 -- verified;
                and pre-casting crashes in fuse_conv_and_bn). Dtype is ASSERTED, not assumed.
  * UETrack   : torch.autocast(fp16) around init/update -- uetrack_wrap has no fp16 path, and its
                templates/preproc are built fp32, so autocast is the non-invasive way in.

Each dtype gets its OWN decode pass (a 5-cam 1080p trial does not fit in RAM), so decode is timed
and reported SEPARATELY from model time -- otherwise decode (the dominant term) would mask the
model speedup.

Outputs per trial AND per dtype (never just a summary):
    out/speed/fp16_ab/<part>__<trial>__<dtype>.npz   3D joints (n,3) each + cup (n,3) + timings
    out/speed/fp16_ab_summary.csv                     one row per trial x dtype
    out/speed/fp16_ab_delta.csv                       per-trial fp32-vs-fp16 3D deltas (mm)

    python scripts/pipeline_fp16_ab.py --parts P07 P08 --limit 3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUTD = ROOT / "out/speed/fp16_ab"
POSE_W = ROOT / "models/yolo26n-pose.pt"
IMGSZ = 640

# Full COCO-17 keypoint ORDER -- needed to index result.keypoints. pipeline.pose_keypoints.TASK_KP
# is a 13-joint SUBSET (no ankles/knees) so it cannot be used for indexing.
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]


def pose_model(dtype, warm):
    """A YOLO-pose predictor in fp32 or genuinely-fp16 (dtype asserted)."""
    import torch
    from ultralytics import YOLO
    m = YOLO(str(POSE_W)).to("cuda:0")
    m.predict(warm, imgsz=IMGSZ, device=0, verbose=False)       # builds + fuses in fp32
    if dtype == "fp16":
        m.predictor.model.model.half()
        m.predictor.model.fp16 = True
        assert next(m.predictor.model.model.parameters()).dtype == torch.float16, "pose cast failed"
    return m


def seed_boxes(part, trial, cams):
    """{frame: {cam: [x,y,w,h]}} for the FIRST cup detection of each camera (detect-once seeds)."""
    import cache_uetrack_tracks as CU
    out = {}
    for cam in cams:
        sb = CU._seed_box_xywh(part, trial, cam)
        if sb is None:
            continue
        f, box = sb
        out.setdefault(int(f), {})[cam] = [float(v) for v in box]
    return out


def clip_paths(part, trial, cams):
    import cache_uetrack_tracks as CU
    d = CU._staged_dir(part)
    stem = CU._clip_stem(part, trial)
    got = {}
    for cam in cams:
        n = cam.split("_")[1]
        p = d / f"{stem}.{n}.mp4"
        if p.exists():
            got[cam] = str(p)
    return got


def run_trial(part, trial, cams, calib, dtype, joints, max_frames=None):
    """One full GPU pass. Returns (per-joint 3D dict, cup 3D (n,3), timings dict)."""
    import cv2
    import torch
    from pipeline import triangulate as T
    from pipeline.consensus import consensus3
    from uetrack_wrap import UETrackBatch
    import gpu_decode

    paths = clip_paths(part, trial, cams)
    cams = [c for c in cams if c in paths]
    if len(cams) < 3:
        return None
    seeds = seed_boxes(part, trial, cams)
    if not seeds:
        return None

    tm = dict(decode=0.0, pose=0.0, cup=0.0, tri=0.0)
    warm = np.zeros((64, 64, 3), np.uint8)
    pm = pose_model(dtype, [warm])
    btrk = UETrackBatch(len(cams))
    ac = (torch.autocast("cuda", dtype=torch.float16) if dtype == "fp16"
          else torch.autocast("cuda", enabled=False))

    gens = {c: gpu_decode.frames(paths[c]) for c in cams}
    per_cam = {c: [] for c in cams}          # pose frames (triangulate_target format)
    cup_obs = []                             # per frame: {cam: (u,v)}
    seeded = set()
    f = 0
    t_all = time.time()
    while True:
        t0 = time.perf_counter()
        batch = []
        for c in cams:
            fr = next(gens[c], None)
            batch.append(fr)
        tm["decode"] += time.perf_counter() - t0
        if any(b is None for b in batch):
            break
        if max_frames and f >= max_frames:
            break

        # ---- pose: one batched forward across cameras ----
        t0 = time.perf_counter()
        rs = pm.predict(batch, imgsz=IMGSZ, device=0, verbose=False)
        tm["pose"] += time.perf_counter() - t0
        for c, r in zip(cams, rs):
            kps = {}
            if r.boxes is not None and len(r.boxes):
                i = int(r.boxes.conf.argmax())
                kxy = r.keypoints.xy[i].cpu().numpy()
                kcf = r.keypoints.conf[i].cpu().numpy()
                from pipeline.pose_keypoints import MIN_KP_CONF
                for j, nm in enumerate(COCO17):
                    if nm in joints and kcf[j] >= MIN_KP_CONF:
                        kps[nm] = [float(kxy[j, 0]), float(kxy[j, 1]), float(kcf[j])]
            per_cam[c].append({"frame": f, "box_conf": 0.0, "kps": kps})

        # ---- cup: seed on this camera's first-detection frame, then batched track ----
        t0 = time.perf_counter()
        # cv2.cvtColor, NOT `b[:, :, ::-1].copy()`. MEASURED at 5 cams 1080p per rig-frame:
        # numpy strided flip 32.50ms vs cvtColor 6.32ms (1 thread) / 3.08ms (4 threads). The numpy
        # version was 25ms of the ~35ms this stage cost -- i.e. the "UETrack is the bottleneck"
        # reading was mostly a channel flip charged to the tracker (UETrack's own step is 11.3ms).
        rgbs = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in batch]
        with ac:
            if f in seeds:
                for cam, box in seeds[f].items():
                    ci = cams.index(cam)
                    btrk.init(ci, rgbs[ci], box)
                    seeded.add(ci)
            boxes = btrk.update([rgbs[i] if i in seeded else None for i in range(len(cams))])
        tm["cup"] += time.perf_counter() - t0
        obs = {}
        for i, c in enumerate(cams):
            b = boxes[i]
            if b is not None:
                obs[c] = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
        cup_obs.append(obs)

        f += 1
        if f % 60 == 0:
            print(f"      {part} {trial} [{dtype}] frame {f}  {time.time()-t_all:5.1f}s "
                  f"(pose {tm['pose']:.1f}s cup {tm['cup']:.1f}s dec {tm['decode']:.1f}s)",
                  flush=True)

    n = f
    if n < 30:
        return None

    # ---- triangulate ----
    t0 = time.perf_counter()
    out3d = {}
    for j in joints:
        # always our own point_fn: T.POINT_FN's entries are not all plain callables
        tr = T.triangulate_target(per_cam, calib, _kp_fn(j), n)
        out3d[j] = np.array([(r["X"] if r["X"] else [np.nan] * 3) for r in tr], float)
    cup = np.full((n, 3), np.nan)
    prev = None
    for i, obs in enumerate(cup_obs):
        if len(obs) >= 2:
            X, kept, _ = consensus3(obs, calib, prev=prev)
            if X is not None:
                cup[i] = X
                prev = X
    tm["tri"] = time.perf_counter() - t0
    tm["total"] = time.time() - t_all
    tm["frames"] = n
    del pm, btrk
    torch.cuda.empty_cache()
    return out3d, cup, tm


def _kp_fn(name):
    def fn(fr):
        k = fr.get("kps", {})
        v = k.get(name)
        return None if v is None else np.array(v[:2], float)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P07", "P08"])
    ap.add_argument("--limit", type=int, default=3, help="trials per participant")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()

    import compare_pose_omc_delta as H
    import results_v3_delta as R
    H.use_good_cams()
    JOINTS = ["right_wrist", "right_elbow", "right_shoulder",
              "left_wrist", "left_elbow", "left_shoulder", "nose"]
    OUTD.mkdir(parents=True, exist_ok=True)

    import cache_uetrack_tracks as CU
    rows, drows = [], []
    for part in a.parts:
        calib = R._calib(part)
        cams = sorted(calib)
        d = CU._staged_dir(part)
        trials = sorted({Path(p).name.split(".")[0].replace(f"delta_{part}_", "")
                         for p in glob.glob(str(d / f"delta_{part}_*.mp4"))})[:a.limit]
        print(f"\n=== {part}: {len(trials)} trial(s), cams {cams}", flush=True)
        for trial in trials:
            res = {}
            for dtype in ("fp32", "fp16"):
                t0 = time.time()
                out = run_trial(part, trial, cams, calib, dtype, JOINTS, a.max_frames)
                if out is None:
                    print(f"  {part} {trial} [{dtype}] SKIPPED (no seeds/clips/frames)", flush=True)
                    continue
                j3, cup, tm = out
                res[dtype] = (j3, cup)
                np.savez(OUTD / f"{part}__{trial}__{dtype}.npz", cup=cup,
                         **{f"j_{k}": v for k, v in j3.items()},
                         timings=np.array(str(tm)))
                model_s = tm["pose"] + tm["cup"]
                print(f"  {part} {trial} [{dtype}] {tm['frames']}fr  total {tm['total']:5.1f}s  "
                      f"decode {tm['decode']:5.1f}s  pose {tm['pose']:5.1f}s  cup {tm['cup']:5.1f}s "
                      f" tri {tm['tri']:4.1f}s  -> model-only {model_s:5.1f}s "
                      f"({tm['frames']/model_s:5.1f} rig-fps)", flush=True)
                rows.append(dict(part=part, trial=trial, dtype=dtype, frames=tm["frames"],
                                 total_s=tm["total"], decode_s=tm["decode"], pose_s=tm["pose"],
                                 cup_s=tm["cup"], tri_s=tm["tri"], model_s=model_s,
                                 model_fps=tm["frames"] / model_s))
            if len(res) == 2:
                (ja, ca), (jb, cb) = res["fp32"], res["fp16"]
                for j in JOINTS:
                    A, B = ja[j], jb[j]
                    m = np.isfinite(A).all(1) & np.isfinite(B).all(1)
                    dd = np.linalg.norm(A[m] - B[m], axis=1) if m.any() else np.array([])
                    drows.append(dict(part=part, trial=trial, target=j, n=int(m.sum()),
                                      med_mm=float(np.median(dd)) if len(dd) else np.nan,
                                      p95_mm=float(np.percentile(dd, 95)) if len(dd) else np.nan,
                                      max_mm=float(dd.max()) if len(dd) else np.nan,
                                      cov32=int(np.isfinite(A).all(1).sum()),
                                      cov16=int(np.isfinite(B).all(1).sum())))
                m = np.isfinite(ca).all(1) & np.isfinite(cb).all(1)
                dd = np.linalg.norm(ca[m] - cb[m], axis=1) if m.any() else np.array([])
                drows.append(dict(part=part, trial=trial, target="cup", n=int(m.sum()),
                                  med_mm=float(np.median(dd)) if len(dd) else np.nan,
                                  p95_mm=float(np.percentile(dd, 95)) if len(dd) else np.nan,
                                  max_mm=float(dd.max()) if len(dd) else np.nan,
                                  cov32=int(np.isfinite(ca).all(1).sum()),
                                  cov16=int(np.isfinite(cb).all(1).sum())))
                print(f"    -> 3D delta fp32-vs-fp16 written for {part} {trial}", flush=True)

    import pandas as pd
    outp = ROOT / "out/speed"
    df = pd.DataFrame(rows); dd = pd.DataFrame(drows)
    df.to_csv(outp / "fp16_ab_summary.csv", index=False)
    dd.to_csv(outp / "fp16_ab_delta.csv", index=False)

    print("\n===== SPEED (per trial) =====", flush=True)
    if len(df):
        g = df.groupby("dtype")[["frames", "decode_s", "pose_s", "cup_s", "tri_s",
                                 "model_s", "total_s", "model_fps"]].median()
        print(g.round(2).to_string(), flush=True)
        if {"fp32", "fp16"} <= set(g.index):
            print(f"\n  model-only speedup fp16: {g.loc['fp32','model_s']/g.loc['fp16','model_s']:.2f}x"
                  f"   (pose {g.loc['fp32','pose_s']/g.loc['fp16','pose_s']:.2f}x, "
                  f"cup {g.loc['fp32','cup_s']/g.loc['fp16','cup_s']:.2f}x)", flush=True)
            print(f"  end-to-end incl decode:  {g.loc['fp32','total_s']/g.loc['fp16','total_s']:.2f}x",
                  flush=True)

    print("\n===== 3D DELTA fp32 vs fp16 (mm) =====", flush=True)
    if len(dd):
        p = dd.groupby("target")[["n", "med_mm", "p95_mm", "max_mm", "cov32", "cov16"]].median()
        print(p.round(3).to_string(), flush=True)
        print(f"\n  worst single trial/target: "
              f"{dd.loc[dd.max_mm.idxmax(), ['part','trial','target','max_mm']].to_dict()}",
              flush=True)

    print(f"\nPROCESSING CHECK: rows {len(df)}, trials x dtype expected "
          f"{len(a.parts)*a.limit*2}, delta rows {len(dd)}, "
          f"non-finite med {int(dd.med_mm.isna().sum()) if len(dd) else 0}", flush=True)
    print(f"wrote {outp/'fp16_ab_summary.csv'}, {outp/'fp16_ab_delta.csv'}, per-trial npz in {OUTD}"
          f"\nDONE", flush=True)


if __name__ == "__main__":
    main()
