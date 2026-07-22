"""Cache every pose model's keypoints (2D + fused 3D) for one rep, so the comparison can be
studied offline with no GPU.

WHAT "GROUND TRUTH" MEANS HERE -- read this before trusting any number downstream.

`yolo26x-pose` is used as a REFERENCE, and it is a weak one in a specific way: it is the
same architecture family, trained on the same COCO data, as n/s/m. So it shares their
BIASES. If the whole family systematically places the wrist 10mm up the forearm, x agrees
with them, every consistency metric looks perfect, and every one of them is wrong together.
x can rank n vs s vs m ("which is closest to the best model in the family"). It CANNOT tell
you whether the family itself is right.

The genuinely INDEPENDENT reference available in this project is **MeTRAbs multicam 3D**
(different architecture, different training) -- already used to settle the consensus-gate
question. QTM mocap is sub-mm but carries HEAD and CUP markers, not COCO joints, so it
cannot score a wrist directly.

So: x = a sharper ruler of the same make. MeTRAbs = a second opinion. Report both, and never
call x "ground truth" without the qualifier -- self-consistency metrics (reprojection
residual, 3D smoothness) all reward the SAME failure mode, which is being *consistently and
smoothly wrong*.

Writes cache/pose_models/<stem>/<model>.{2d,3d}.json -- reusable, no GPU needed after this.

    python scripts/cache_pose_models.py --clipdir DIR --stem STEM --calib CALIB.toml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import pose_keypoints, triangulate
from cup_task.kalman_3d import load_calibration

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "pose_models"
VARIANTS = ("n", "s", "m", "x")
TARGETS = ("left_wrist", "right_wrist", "mouth", "trunk")
IMGSZ = 640


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clipdir", type=Path, required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--models", nargs="+", default=list(VARIANTS))
    a = ap.parse_args(argv)

    out = CACHE / a.stem
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(a.clipdir.glob(f"{a.stem}.*.mp4"),
                   key=lambda p: int(re.search(r"\.(\d+)\.mp4$", p.name).group(1)))
    cams = load_calibration(a.calib, target_size=(1920, 1080))
    print(f"{len(clips)} clips, {len(cams)} calibrated cams, imgsz={a.imgsz}", flush=True)

    for v in a.models:
        mp = ROOT / "models" / f"yolo26{v}-pose.pt"
        f2d, f3d = out / f"yolo26{v}.2d.json", out / f"yolo26{v}.3d.json"

        if f2d.exists():
            print(f"\nyolo26{v}: 2D cached", flush=True)
            per_cam = {k: v2 for k, v2 in json.loads(f2d.read_text()).items()}
        else:
            print(f"\nyolo26{v}: detecting {len(clips)} cams", flush=True)
            per_cam = {}
            t0 = time.time()
            for clip in clips:
                ck = triangulate._cam_key_from_clip(clip.name)
                if ck not in cams:
                    continue
                fr = pose_keypoints.extract_pose(clip, model_path=str(mp), device=0,
                                                 imgsz=a.imgsz, progress_every=0)
                per_cam[ck] = [{"kps": f.kps, "box_conf": f.box_conf} for f in fr]
                print(f"    {ck}", flush=True)
            f2d.write_text(json.dumps(per_cam))
            print(f"  wrote {f2d.name} ({time.time()-t0:.0f}s)", flush=True)

        if f3d.exists():
            print(f"yolo26{v}: 3D cached", flush=True)
            continue

        n = max(len(x) for x in per_cam.values())
        res = {}
        for tgt in TARGETS:
            _, fn = triangulate.POINT_FN[tgt]
            tr = triangulate.triangulate_target(per_cam, cams, fn, n)
            got = sum(1 for t in tr if t["X"] is not None)
            px = [t["reproj_px"] for t in tr if t["reproj_px"] is not None]
            print(f"    {tgt:12s} 3D {got}/{n} ({got/n:.0%})  "
                  f"reproj {np.median(px):.2f}px", flush=True)
            res[tgt] = tr
        f3d.write_text(json.dumps({"targets": res, "imgsz": a.imgsz,
                                   "model": f"yolo26{v}-pose"}))
        print(f"  wrote {f3d.name}", flush=True)

    print(f"\ncached under {out}", flush=True)
    print("NOTE: yolo26x is a SHARPER RULER OF THE SAME MAKE, not ground truth -- it shares")
    print("the family's biases. See the module docstring.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
