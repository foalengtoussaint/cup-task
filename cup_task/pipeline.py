"""End-to-end drink-task pipeline for one rep: videos -> phases.

The BASE pipeline. Runs the stages that exist and are verified, in order:

    per-camera clips ──► cup_detect   ──┐
                    └──► pose_keypoints ┤
                                        ├──► triangulate (3D, mm, world frame)
                                        │
                                        └──► segment (phases: rest / forward /
                                                      drinking / back / rest)

Detection is the LIVE half (one merged model pass per camera, ~98 fps measured); this
driver is the OFFLINE half -- it wants all the frames, so it runs after capture.

Deliberately the same geometric segmentation the research truth uses. Not yet ported
(both are known improvements, see segment.py): TCN gap-fill of the occluded cup track,
and a head-distance feature channel.

    python -m cup_task.pipeline CLIPDIR --calib calibration.toml -o out/
    python -m cup_task.pipeline CLIPDIR --calib calib.toml --stem P07_drinking_right_...

CLIPDIR holds one rep's per-camera clips named <stem>.<cam>.mp4 (the .N suffix is the
calibration's cam_N). Every stage caches to JSON, so re-running skips finished work --
the GPU passes are the expensive part and you should never pay for them twice.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cup_task import cup_detect, pose_keypoints, segment, triangulate
from cup_task.kalman_3d import load_calibration

FPS = 60.0


def find_reps(clipdir: Path) -> dict[str, list[Path]]:
    """{stem: [per-camera clips]} for every rep in a directory."""
    reps: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(clipdir.glob("*.mp4")):
        m = re.match(r"(.+)\.(\d+)\.mp4$", p.name)
        if m:
            reps[m.group(1)].append(p)
    return dict(reps)


def detect_rep(clips: list[Path], out: Path, cup_model: str, pose_model: str,
               device="0") -> None:
    """Cup + pose 2D for every camera of one rep. Skips whatever is already cached."""
    out.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        for kind, fn, model in (("cup", cup_detect, cup_model),
                                ("pose", pose_keypoints, pose_model)):
            jf = out / f"{clip.stem}.{kind}.json"
            if jf.exists():
                print(f"  {jf.name}: cached", flush=True)
                continue
            t0 = time.time()
            if kind == "cup":
                dets = cup_detect.detect_cup(clip, model_path=model, device=device)
                payload = cup_detect.to_payload(clip, dets, model)
            else:
                frames = pose_keypoints.extract_pose(clip, model_path=model, device=device)
                payload = pose_keypoints.to_payload(clip, frames, model)
            jf.write_text(json.dumps(payload))
            print(f"  {jf.name}: {time.time()-t0:.0f}s", flush=True)


def fuse_3d(json_dir: Path, calib: Path, out: Path) -> dict:
    """Triangulate cup + mouth + wrists to 3D. Cached."""
    if out.exists():
        print(f"  {out.name}: cached", flush=True)
        return json.loads(out.read_text())
    cams = load_calibration(calib, target_size=(1920, 1080))
    result = {"calib": str(calib), "targets": {}}
    for tgt in ("cup", "mouth", "left_wrist", "right_wrist"):
        kind, fn = triangulate.POINT_FN[tgt]
        per_cam = triangulate._load_per_cam(json_dir, kind)
        if not per_cam:
            continue
        n = max(len(v) for v in per_cam.values())
        track = triangulate.triangulate_target(per_cam, cams, fn, n)
        got = sum(1 for t in track if t["X"] is not None)
        print(f"  {tgt:12s} 3D in {got}/{n} frames ({got/max(n,1):.0%})", flush=True)
        result["targets"][tgt] = track
    out.write_text(json.dumps(result))
    return result


def phases_from_3d(tracks: dict, fps=FPS) -> dict:
    """Segment the drink task from the 3D cup track (the verified base method)."""
    cup_track = tracks["targets"].get("cup")
    if not cup_track:
        raise SystemExit("no cup 3D track -- cannot segment")
    cup, _ = segment.track_confidence(cup_track)
    seg = segment.segment_cup_only(cup, fps=fps)
    return seg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clipdir", type=Path, help="dir of one rep's <stem>.<cam>.mp4 clips")
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--stem", help="which rep (default: all reps found)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--cup-model", default=cup_detect.DEFAULT_MODEL)
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)

    reps = find_reps(a.clipdir)
    if not reps:
        raise SystemExit(f"no <stem>.<cam>.mp4 clips in {a.clipdir}")
    if a.stem:
        reps = {a.stem: reps[a.stem]} if a.stem in reps else {}
        if not reps:
            raise SystemExit(f"stem {a.stem!r} not found; have: {sorted(find_reps(a.clipdir))}")
    print(f"{len(reps)} rep(s) in {a.clipdir}", flush=True)

    for stem, clips in reps.items():
        print(f"\n=== {stem}  ({len(clips)} cams) ===", flush=True)
        d = a.out / stem
        detect_rep(clips, d, a.cup_model, a.pose_model, a.device)
        tracks = fuse_3d(d, a.calib, d / "tracks3d.json")
        seg = phases_from_3d(tracks)

        iv = [(n, s, e) for n, s, e in seg["intervals"]]
        print(f"  phases:", flush=True)
        for n, s, e in iv:
            print(f"    {n:18s} {s/FPS:6.2f}-{e/FPS:6.2f}s  ({(e-s)/FPS:.2f}s)", flush=True)
        drink = [(s, e) for n, s, e in iv if n == "drinking"]
        if drink:
            s, e = drink[0]
            print(f"  DRINK DWELL: {(e-s)/FPS:.2f}s", flush=True)
        else:
            print(f"  DRINK DWELL: none found", flush=True)
        (d / "phases.json").write_text(json.dumps(
            {"stem": stem, "fps": FPS, "intervals": iv,
             "drink_s": (drink[0][1] - drink[0][0]) / FPS if drink else None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
