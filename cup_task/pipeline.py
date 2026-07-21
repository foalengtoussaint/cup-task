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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cup_task import cup_detect, pose_keypoints, pose_smooth, segment, triangulate
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
    """Cup + pose 2D for every camera of one rep. Skips whatever is already cached.

    The two nets run CONCURRENTLY on separate CUDA streams (see run_both_streams): they are
    independent -- neither reads the other's output -- and the GPU is launch-bound at these
    sizes, so overlapping them is close to free. Measured 1.42x at batch=1 vs running them
    back to back.
    """
    out.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        want = {k: out / f"{clip.stem}.{k}.json" for k in ("cup", "pose")}
        todo = {k: p for k, p in want.items() if not p.exists()}
        for k, p in want.items():
            if k not in todo:
                print(f"  {p.name}: cached", flush=True)
        if not todo:
            continue

        t0 = time.time()
        results = run_both_streams(clip, cup_model, pose_model, device, set(todo))
        for kind, payload in results.items():
            todo[kind].write_text(json.dumps(payload))
            print(f"  {todo[kind].name}: {time.time()-t0:.0f}s", flush=True)


def run_both_streams(clip: Path, cup_model: str, pose_model: str, device: str,
                     kinds: set[str]) -> dict:
    """Run the cup and pose nets on SEPARATE CUDA STREAMS, concurrently.

    Two threads alone are not enough: threads that both call predict() submit into the SAME
    default CUDA stream, so the kernels still serialize on the device and you only overlap
    the CPU-side preprocessing. A stream each lets them interleave on the GPU.

    NOTE the stream context is THREAD-LOCAL -- it must be entered inside the worker, not on
    the caller. Entering it on the main thread and submitting to a pool silently falls back
    to the default stream.
    """
    import torch

    def work(kind: str):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if kind == "cup":
                dets = cup_detect.detect_cup(clip, model_path=cup_model, device=device)
                payload = cup_detect.to_payload(clip, dets, cup_model)
            else:
                frames = pose_keypoints.extract_pose(clip, model_path=pose_model,
                                                     device=device)
                payload = pose_keypoints.to_payload(clip, frames, pose_model)
        stream.synchronize()
        return payload

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {k: ex.submit(work, k) for k in kinds}
        return {k: f.result() for k, f in futs.items()}


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
    """Segment the drink task: cup-only gate, then fix the grasp onset with the pose.

    The cup-only onset gate fires on triangulation jitter (a still cup has a ~30-50mm/s
    noise floor vs a 15mm/s gate), so the pose refinement is not optional -- without it the
    reach window is truncated and every reach-scoped measure is wrong.
    """
    t = tracks["targets"]
    if not t.get("cup"):
        raise SystemExit("no cup 3D track -- cannot segment")
    cup, _ = segment.track_confidence(t["cup"])
    seg = segment.segment_cup_only(cup, fps=fps)

    if t.get("mouth") and (t.get("left_wrist") or t.get("right_wrist")):
        mouth, _ = segment.track_confidence(t["mouth"])
        hand, _ = segment.track_confidence(t[dominant_wrist(t)])
        seg = segment.refine_grasp_with_pose(seg, cup, hand, mouth, fps=fps)
    return seg


def wrist_speed_v3(json_dir: Path, tracks: dict, calib: Path, wrist: str,
                   flow_dir: Path | None = None) -> "np.ndarray | None":
    """v3 wrist SPEED = blend(PyrLK flow speed, SmoothNet position speed).

    Flow is measured ONLINE (see docs/PIPELINE_V3.md) -- live, the frames are already in hand. This
    offline entry point replays it from cached per-camera flow when present, and falls back to
    decoding the clips only if it must. Returns None when there is no flow to work with, in which
    case the caller should use the SmoothNet speed alone.
    """
    from cup_task import flow_speed as FS
    from cup_task import speed_blend
    from cup_task.kalman_3d import load_calibration

    track = tracks["targets"].get(wrist)
    if not track:
        return None
    X = np.array([t["X"] if t.get("X") else [np.nan] * 3 for t in track], dtype=float)
    n = len(X)

    # per-camera wrist pixels from the cached pose JSONs
    px: dict[str, np.ndarray] = {}
    for pj in sorted(json_dir.glob("*.pose.json")):
        cam = f"cam_{pj.name.split('.')[1]}"
        frames = json.loads(pj.read_text())["frames"]
        p = np.full((len(frames), 2), np.nan)
        for i, fr in enumerate(frames):
            k = fr.get("kps", {}).get(wrist)
            if k and k[2] > FS.CONF_THR:
                p[i] = k[:2]
        px[cam] = p

    flow_dir = flow_dir or (json_dir / "flow")
    flow: dict[str, np.ndarray] = {}
    for cam, p in px.items():
        f = flow_dir / f"{cam}__pyrlk.npy"
        if f.exists():
            v = np.load(f)
            if len(v) == len(p):
                flow[cam] = v
    if not flow:
        return None

    cams = load_calibration(calib, target_size=(1920, 1080))
    sp_flow = FS.speed_from_cached_flow(px, flow, cams, n)

    d = np.linalg.norm(np.diff(X, axis=0), axis=1) * FPS
    sp_sn = np.r_[np.nan, d]
    return speed_blend.blend(sp_flow, sp_sn)


def dominant_wrist(targets: dict) -> str:
    """Which wrist did the task: the one with the larger 3D motion range."""
    def rng(key):
        tr = targets.get(key)
        if not tr:
            return 0.0
        x, _ = segment.track_confidence(tr)
        x = x[np.isfinite(x).all(1)]
        return float(np.linalg.norm(x.max(0) - x.min(0))) if len(x) else 0.0
    return "right_wrist" if rng("right_wrist") >= rng("left_wrist") else "left_wrist"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clipdir", type=Path, help="dir of one rep's <stem>.<cam>.mp4 clips")
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--stem", help="which rep (default: all reps found)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--cup-model", default=cup_detect.DEFAULT_MODEL)
    ap.add_argument("--pose-model", default=str(Path(__file__).resolve().parents[1]
                                                / "models" / "yolo26n-pose.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--smooth-pose", action="store_true",
                    help="v2: refine triangulated pose with SmoothNet before segmentation "
                         "(jitter -93%%, peak-vel restored; see docs/PIPELINE_V2_PLAN.md)")
    ap.add_argument("--cup-track", action="store_true",
                    help="v2: detect-once UETrack cup tracking instead of every-frame triangulation "
                         "(needs clips + GPU; the offline results path uses cached tracker points)")
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
        if a.smooth_pose:
            tracks = pose_smooth.smooth_tracks(tracks)
            print(f"  pose refined (SmoothNet): {tracks['pose_smoothed']}", flush=True)
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
