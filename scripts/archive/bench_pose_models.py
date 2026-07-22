"""yolo26 n vs s vs m -pose: is the bigger net worth it? Speed AND accuracy.

Speed alone cannot answer this. A bigger pose net only earns its cost if the KEYPOINTS get
better -- and "better" has to be measured in the terms the pipeline actually consumes, which
is 3D, not 2D confidence.

WE HAVE NO KEYPOINT GROUND TRUTH here (mocap has head/cup markers, not COCO joints). So we
score with the two signals that are available and that a wrong keypoint cannot fake:

  1. REPROJECTION RESIDUAL of the multi-view triangulation. Ten cameras each vote on where
     a joint is in 3D. If a net's 2D keypoints are noisy or biased, the rays do not meet:
     the triangulated point reprojects badly into the individual views. This is
     self-consistency across 10 independent viewpoints -- hard to fake by being confidently
     wrong in one camera, because the other nine disagree.

  2. 3D JITTER of a joint that should be SMOOTH. A wrist in a slow reach has bounded
     acceleration; a real human cannot teleport. Frame-to-frame 3D jerk is therefore mostly
     ESTIMATION NOISE, not motion. Less jitter = a steadier keypoint.

Neither is a substitute for real truth, and both are stated as such. But they are the same
signals the downstream measures live or die on: score.py differentiates the hand position
twice, so keypoint jitter is exactly what pollutes peak_velocity and the movement-unit count.

    python scripts/bench_pose_models.py --clipdir CLIPDIR --stem STEM --calib CALIB.toml
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import pose_keypoints, triangulate
from cup_task.kalman_3d import load_calibration, project

ROOT = Path(__file__).resolve().parents[1]
MODELS = {n: ROOT / "models" / f"yolo26{n}-pose.pt" for n in ("n", "s", "m")}
IMGSZ = 640
WARMUP = 10


def speed(model, frames, imgsz, batch):
    def _b(i):
        return [frames[(i + k) % len(frames)] for k in range(batch)] if batch > 1 \
            else frames[i % len(frames)]
    for i in range(WARMUP):
        model.predict(_b(i), imgsz=imgsz, device=0, verbose=False)
    torch.cuda.synchronize()
    lat = []
    for i in range(30):
        t0 = time.perf_counter()
        model.predict(_b(i), imgsz=imgsz, device=0, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lat))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clipdir", type=Path, required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 4, 10])
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    clips = sorted(a.clipdir.glob(f"{a.stem}.*.mp4"),
                   key=lambda p: int(re.search(r"\.(\d+)\.mp4$", p.name).group(1)))
    cams = load_calibration(a.calib, target_size=(1920, 1080))
    print(f"GPU: {torch.cuda.get_device_name(0)}   {len(clips)} cams   imgsz={a.imgsz}\n",
          flush=True)

    # frames of ONE camera, for the speed test
    cap = cv2.VideoCapture(str(clips[3]))
    sfr = []
    while len(sfr) < 40:
        ok, f = cap.read()
        if not ok:
            break
        sfr.append(f)
    cap.release()

    # ---------------- SPEED ----------------
    print("=== SPEED (pose net alone, batched) ===", flush=True)
    hdr = "  ".join(f"{c:>2d}cam" for c in a.cams)
    print(f"{'model':>14}  {hdr}     params", flush=True)
    speeds = {}
    for k, p in MODELS.items():
        m = YOLO(str(p))
        n_par = sum(x.numel() for x in m.model.parameters()) / 1e6
        row = [speed(m, sfr, a.imgsz, c) for c in a.cams]
        speeds[k] = row
        cells = "  ".join(f"{v:5.1f}" for v in row)
        print(f"  yolo26{k}-pose  {cells}   {n_par:5.1f}M", flush=True)
        del m
        torch.cuda.empty_cache()

    # ---------------- ACCURACY ----------------
    print("\n=== ACCURACY (all 10 cams -> 3D; no keypoint GT, see docstring) ===",
          flush=True)
    print(f"{'model':>14} {'reproj px':>18} {'wrist 3D jitter':>18} {'coverage':>10}",
          flush=True)

    for k, p in MODELS.items():
        per_cam = {}
        for clip in clips:
            ck = triangulate._cam_key_from_clip(clip.name)
            if ck not in cams:
                continue
            fr = pose_keypoints.extract_pose(clip, model_path=str(p), device=0,
                                             imgsz=a.imgsz, progress_every=0)
            per_cam[ck] = [{"kps": f.kps} for f in fr]
        n = max(len(v) for v in per_cam.values())

        res = {}
        for tgt, fn in (("right_wrist", lambda f: triangulate._wrist_point(f, "right")),
                        ("left_wrist", lambda f: triangulate._wrist_point(f, "left")),
                        ("mouth", triangulate._mouth_point)):
            res[tgt] = triangulate.triangulate_target(per_cam, cams, fn, n)

        # reprojection residual, pooled over the joints we actually use
        px = [t["reproj_px"] for tr in res.values() for t in tr
              if t["reproj_px"] is not None]
        cov = np.mean([t["X"] is not None for tr in res.values() for t in tr])

        # 3D jitter of the DOMINANT wrist: third difference (jerk) magnitude.
        def jerk(tr):
            X = np.array([t["X"] if t["X"] else [np.nan] * 3 for t in tr], float)
            X = X[np.isfinite(X).all(1)]
            if len(X) < 4:
                return np.nan
            return float(np.median(np.linalg.norm(np.diff(X, n=3, axis=0), axis=1)))

        dom = max(("right_wrist", "left_wrist"),
                  key=lambda w: np.nanstd([t["X"][0] for t in res[w] if t["X"]] or [0]))
        j = jerk(res[dom])

        print(f"  yolo26{k}-pose  median {np.median(px):5.2f}  p90 {np.percentile(px,90):5.2f}"
              f"   {j:7.2f} mm      {cov:5.0%}", flush=True)

    print("\nreproj px = do 10 independent cameras AGREE on where the joint is (lower=better)")
    print("3D jitter = median |3rd difference| of the dominant wrist -- a real arm cannot")
    print("            jerk; this is mostly estimation noise, and it is what score.py's")
    print("            double-differentiation turns into bogus peak_velocity / movement units.")


if __name__ == "__main__":
    sys.exit(main())
