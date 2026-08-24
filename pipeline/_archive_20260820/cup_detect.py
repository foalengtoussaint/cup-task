"""Cup detection per camera, using the shipped drink_study segmentation model.

The model is the `clean3d_refill` (reproject-fill) single-class cup segmenter that
wins every axis in the research pipeline — task=segment, class 0 = 'cup'. Here we
only need the box + confidence per frame (the mask is available but unused until a
metric wants cup orientation); the box centroid is what gets triangulated to 3D.

Emits one CupDet per frame, dense (None-box frames included) so indices line up
1:1 with the video and with the pose keypoints from pose_keypoints.py.

Usage:
    python -m pipeline.cup_detect CLIP.mp4 -o out.cup.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "cup_clean3d_refill.pt")
CUP_CLASS = 0
MIN_CONF = 0.25   # matches the cache's c0.25 detection threshold

# RUN AT THE RESOLUTION THE MODEL WAS TRAINED AT. This was 1280 and that was a BUG --
# not a slow-but-safe choice, a broken one. Measured on P07 cam_4 vs the research cache
# (which calls model(...) with no imgsz at all, i.e. ultralytics' default 640):
#
#     imgsz    cup found    agrees w/ research    ms
#       640        64%              100%          4.3
#       960        62%               94%          5.5
#      1280         8%               47%          9.0
#      1920         0%                --         16.1
#
# Upsampling does NOT buy detail -- it takes the cup OFF-DISTRIBUTION. The detector
# learned its size priors at 640; at 1280 the cup arrives at twice the scale it ever saw
# and the model simply stops finding it. Bigger input is slower AND worse, on both axes
# at once. If you ever retrain at another size, change this to match the TRAINING size.
DEFAULT_IMGSZ = 640


@dataclass
class CupDet:
    """Cup detection in one frame of one camera.

    box: [x1, y1, x2, y2] in pixels, or None if no cup this frame.
    conf: detection confidence (0.0 when no cup).
    center: [cx, cy] box centre — the point triangulated to 3D.
    """
    frame: int
    conf: float
    box: list[float] | None
    center: list[float] | None


def _pick_cup(result):
    """Highest-confidence cup box in a frame, or None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    b = result.boxes
    cls = b.cls.cpu().numpy()
    conf = b.conf.cpu().numpy()
    mask = cls == CUP_CLASS
    if not mask.any():
        return None
    idx = [i for i in range(len(conf)) if mask[i]]
    best = max(idx, key=lambda i: conf[i])
    if conf[best] < MIN_CONF:
        return None
    xyxy = b.xyxy[best].cpu().numpy()
    return [float(v) for v in xyxy], float(conf[best])


def detect_cup(clip: str | Path, model_path: str = DEFAULT_MODEL,
               device: str | int = 0, imgsz: int = DEFAULT_IMGSZ,
               progress_every: int = 60) -> list[CupDet]:
    """Run the cup segmenter over every frame; return one CupDet per frame."""
    from ultralytics import YOLO

    clip = Path(clip)
    if not clip.exists():
        raise FileNotFoundError(clip)

    model = YOLO(model_path)
    out: list[CupDet] = []
    t0 = time.time()
    results = model.predict(source=str(clip), stream=True, device=device,
                            imgsz=imgsz, conf=MIN_CONF, verbose=False)
    for i, r in enumerate(results):
        picked = _pick_cup(r)
        if picked is None:
            out.append(CupDet(frame=i, conf=0.0, box=None, center=None))
        else:
            box, conf = picked
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            out.append(CupDet(frame=i, conf=conf, box=box, center=[cx, cy]))
        if progress_every and (i + 1) % progress_every == 0:
            fps = (i + 1) / (time.time() - t0)
            print(f"  {clip.name}: {i + 1} frames  ({fps:.1f} fps)", flush=True)

    dt = time.time() - t0
    det = sum(1 for d in out if d.box)
    print(f"  {clip.name}: {len(out)} frames, cup in {det} "
          f"({det / max(len(out), 1):.0%}), {len(out) / max(dt, 1e-9):.1f} fps",
          flush=True)
    return out


def to_payload(clip: Path, dets: list[CupDet], model_path: str) -> dict:
    return {
        "clip": str(clip), "model": model_path, "cup_class": CUP_CLASS,
        "min_conf": MIN_CONF, "n_frames": len(dets),
        "frames": [asdict(d) for d in dets],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=0)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = ap.parse_args(argv)

    out = args.out or args.clip.with_suffix(".cup.json")
    print(f"cup: {args.clip.name} -> {out.name} ({args.model})", flush=True)
    dets = detect_cup(args.clip, model_path=args.model, device=args.device,
                      imgsz=args.imgsz)
    out.write_text(json.dumps(to_payload(args.clip, dets, args.model)))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
