"""Body/hand keypoint extraction for the drink task, per camera.

Runs YOLO-pose on a single clip and emits per-frame COCO-17 keypoints for the
task-relevant joints (wrists = reach, nose/eyes/ears = mouth proxy for dwell).
Output is a flat JSON keyed by frame, ready to be triangulated to 3D with the
same multi-view calibration used for the cup.

This is the mocap-free replacement for the QTM head marker: the mouth-proxy
point (nose, falling back to the eye/ear centroid) comes straight from the
cameras instead of the mocap head cluster.

Model choice (2026-07): YOLO-pose now for zero-dep speed; swap to RTMPose
whole-body later if dwell scoring needs a true mouth/lip landmark. The public
interface here (extract_pose -> list[FramePose]) does not change on that swap.

Usage:
    python -m cup_task.pose_keypoints CLIP.mp4 -o out.json
    python -m cup_task.pose_keypoints CLIP.mp4 --model yolo11m-pose.pt --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# COCO-17 keypoint order emitted by YOLO-pose (fixed by the model).
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP_INDEX = {name: i for i, name in enumerate(COCO17)}

# Every keypoint we keep: the full upper body. Knees and ankles are dropped —
# they carry no signal for a seated drinking task. Head points (nose/eyes/ears)
# are the mouth proxy; wrists = reach; elbows/shoulders/hips = arm & trunk
# kinematics for whatever metric turns out to need them.
TASK_KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip"]

# Below this per-keypoint confidence a point is treated as missing (NaN), so a
# low-confidence joint never gets triangulated as if it were solid.
MIN_KP_CONF = 0.30

# Match the model's TRAINING resolution (stock COCO yolo*-pose nets are trained at 640).
# This used to be 1280, which cost ~2x the time for nothing: the person is large in frame
# and was already detected at 100% coverage, so there was no accuracy to gain -- while the
# CUP net at 1280 was catastrophically off-distribution (8% recall vs 64% at 640, see
# cup_detect.DEFAULT_IMGSZ). Upsampling past the training size is not free detail.
DEFAULT_IMGSZ = 640


@dataclass
class FramePose:
    """One person's keypoints in one frame of one camera.

    kps: {joint_name: [x_px, y_px, conf]} for the TASK_KP subset. A joint below
    MIN_KP_CONF is omitted so downstream triangulation can tell it apart from a
    confident detection at (0,0). box_conf is the person-detection confidence.
    """
    frame: int
    box_conf: float
    kps: dict[str, list[float]]


def _pick_person(result):
    """Return (keypoints_xy, keypoints_conf, box_conf) for the best person, or None.

    The task has one subject; pick the highest-confidence person box. (Multi-person
    disambiguation across cameras is a later problem — flagged, not solved here.)
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None
    confs = result.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    kxy = result.keypoints.xy[best].cpu().numpy()       # (17, 2)
    kconf = result.keypoints.conf[best].cpu().numpy()   # (17,)
    return kxy, kconf, float(confs[best])


def extract_pose(clip: str | Path, model_path: str = "yolo11n-pose.pt",
                 device: str | int = 0, imgsz: int = DEFAULT_IMGSZ,
                 progress_every: int = 60) -> list[FramePose]:
    """Run YOLO-pose over every frame of `clip`; return one FramePose per frame.

    Frames with no person detected still yield a FramePose (empty kps) so the
    output is dense and frame indices line up 1:1 with the video.
    """
    from ultralytics import YOLO  # imported lazily so `--help` needs no GPU

    clip = Path(clip)
    if not clip.exists():
        raise FileNotFoundError(clip)

    model = YOLO(model_path)
    out: list[FramePose] = []
    t0 = time.time()

    # stream=True yields one Results per frame without buffering the whole video.
    results = model.predict(source=str(clip), stream=True, device=device,
                            imgsz=imgsz, verbose=False)
    for i, r in enumerate(results):
        picked = _pick_person(r)
        kps: dict[str, list[float]] = {}
        box_conf = 0.0
        if picked is not None:
            kxy, kconf, box_conf = picked
            for name in TASK_KP:
                j = KP_INDEX[name]
                c = float(kconf[j])
                if c >= MIN_KP_CONF:
                    kps[name] = [float(kxy[j, 0]), float(kxy[j, 1]), c]
        out.append(FramePose(frame=i, box_conf=box_conf, kps=kps))

        if progress_every and (i + 1) % progress_every == 0:
            fps = (i + 1) / (time.time() - t0)
            print(f"  {clip.name}: {i + 1} frames  ({fps:.1f} fps)", flush=True)

    dt = time.time() - t0
    det = sum(1 for f in out if f.kps)
    print(f"  {clip.name}: {len(out)} frames, person in {det} "
          f"({det / max(len(out), 1):.0%}), {len(out) / max(dt, 1e-9):.1f} fps",
          flush=True)
    return out


def mouth_proxy(fp: FramePose) -> list[float] | None:
    """Best available 2D mouth-proxy point for one frame: nose, else eye/ear centroid.

    Returns [x, y, conf] or None if the head is not visible this frame. This is
    the single point that stands in for the QTM head marker the old pipeline used.
    """
    if "nose" in fp.kps:
        return fp.kps["nose"]
    pts = [fp.kps[k] for k in ("left_eye", "right_eye", "left_ear", "right_ear")
           if k in fp.kps]
    if not pts:
        return None
    n = len(pts)
    return [sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n,
            min(p[2] for p in pts)]


def to_payload(clip: Path, frames: list[FramePose], model_path: str) -> dict:
    return {
        "clip": str(clip),
        "model": model_path,
        "min_kp_conf": MIN_KP_CONF,
        "task_kp": TASK_KP,
        "n_frames": len(frames),
        "frames": [asdict(f) for f in frames],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output JSON (default: <clip>.pose.json)")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default=0)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = ap.parse_args(argv)

    out = args.out or args.clip.with_suffix(".pose.json")
    print(f"pose: {args.clip.name} -> {out.name} ({args.model}, imgsz={args.imgsz})",
          flush=True)
    frames = extract_pose(args.clip, model_path=args.model, device=args.device,
                          imgsz=args.imgsz)
    out.write_text(json.dumps(to_payload(args.clip, frames, args.model)))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
