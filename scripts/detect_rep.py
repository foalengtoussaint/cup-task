"""Run cup + pose detection across every camera clip of one rep.

Writes per-cam <clip>.cup.json and <clip>.pose.json into an output dir, ready
for triangulate.py. Cache-first: skips a clip's JSON if it already exists.

    python scripts/detect_rep.py CLIPS_DIR REP_PREFIX -o OUT_DIR
    # REP_PREFIX e.g. P07_drinking_left_20240124_142730  (matches .*.mp4)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task.cup_detect import detect_cup, to_payload as cup_payload
from cup_task.pose_keypoints import extract_pose, to_payload as pose_payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips_dir", type=Path)
    ap.add_argument("rep_prefix")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--cup-model", default="cup_clean3d_refill.pt")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default=0)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    clips = sorted(args.clips_dir.glob(f"{args.rep_prefix}.*.mp4"))
    print(f"rep {args.rep_prefix}: {len(clips)} camera clips", flush=True)

    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {clip.name}", flush=True)
        cj = args.out / f"{clip.stem}.cup.json"
        pj = args.out / f"{clip.stem}.pose.json"
        if cj.exists():
            print("   cup: cached", flush=True)
        else:
            dets = detect_cup(clip, model_path=args.cup_model, device=args.device,
                              imgsz=args.imgsz, progress_every=0)
            cj.write_text(json.dumps(cup_payload(clip, dets, args.cup_model)))
        if pj.exists():
            print("   pose: cached", flush=True)
        else:
            frames = extract_pose(clip, model_path=args.pose_model, device=args.device,
                                  imgsz=args.imgsz, progress_every=0)
            pj.write_text(json.dumps(pose_payload(clip, frames, args.pose_model)))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
