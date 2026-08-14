"""Detect-once cup tracking stage (v2 pipeline).

Per camera: run YOLO only until it first finds the cup, seed a UETrack-B tracker with that box, then
the tracker alone produces the cup point for the rest of the trial -- NO re-detection, NO re-anchoring
(re-anchoring is harmful: re-seeding from a shared leave-out consensus lets one bad camera poison the
good ones). Feed the per-camera tracked centres into the greedy >=2-cam consensus with a 150mm
continuity gate to get one 3D cup track.

Validated 2026-07-21 on the 18 DELTA c3d trials: median trajectory corr 0.9995 vs OMC, 17/18 >= 0.998,
worst P13 t15 = 0.931 (a genuine 2-good-camera rig floor). ~60x cheaper on detection than every-frame
YOLO. See docs/PIPELINE_V2_PLAN.md, project_tracker_shootout_uetrack.

Output is the SAME shape as triangulate_target so it drops into fuse_3d:
    [{frame, X:[x,y,z]|None, n_cams, kept:[cam,...]}]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import json

from cup_task import consensus

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


def _tracker():
    """UETrack-B via the scripts wrapper (checkpoint is 1.3GB, lives in the session scratchpad;
    not copied into the repo). One shared model, per-camera state is swapped by init()."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from uetrack_wrap import UETrackB
    return UETrackB()


def _center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def track_cup_3d(clip_frames: dict[str, "np.ndarray|list"], yolo_boxes: dict[int, dict],
                 calib: dict, n_frames: int, tracker_factory=_tracker) -> list[dict]:
    """clip_frames: {cam: iterable/list of BGR frames} -- per-camera decoded video (index = frame).
    yolo_boxes: {frame: {cam: [x,y,w,h] | None}} -- cached cup detections (the seed source).
    calib: {cam: CamCalib}. Returns the 3D cup track in fuse_3d format.

    One UETrack model is shared; each camera keeps its own template via re-init on its seed frame.
    Because UETrack holds per-track state internally, we keep ONE tracker instance PER CAMERA.
    """
    cams = list(clip_frames)
    trackers = {c: None for c in cams}          # created on that camera's seed frame
    seeded = {c: False for c in cams}
    prev3d = None
    track = []
    for f in range(n_frames):
        obs = {}
        for c in cams:
            frame = clip_frames[c][f] if f < len(clip_frames[c]) else None
            if frame is None:
                continue
            rgb = frame[:, :, ::-1]             # BGR -> RGB (UETrack wants RGB)
            if not seeded[c]:
                bx = yolo_boxes.get(f, {}).get(c)
                if bx is not None:
                    trackers[c] = tracker_factory()
                    trackers[c].init(rgb, tuple(float(v) for v in bx))
                    seeded[c] = True
                    obs[c] = _center(bx)
            else:
                x, y, w, h = trackers[c].update(rgb)
                obs[c] = (x + w / 2.0, y + h / 2.0)
        X, kept, _ = consensus.consensus3(obs, calib, prev=prev3d)
        if X is not None:
            prev3d = X
        track.append({
            "frame": f,
            "X": None if X is None else [round(float(v), 1) for v in X],
            "n_cams": len(kept),
            "kept": sorted(kept),
        })
    return track


def track_cup_3d_batched(clip_frames: dict[str, "np.ndarray|list"], yolo_boxes: dict[int, dict],
                         calib: dict, n_frames: int) -> list[dict]:
    """Same as track_cup_3d but with ONE batched UETrack forward per rig-frame across all cameras
    (4-5x faster at 5-10 cams; byte-identical to the sequential tracker at N=1, <=2px GPU-numeric
    drift at N>=2, well under the 30px consensus gate). Seeds each camera from its first YOLO box.
    """
    from uetrack_wrap import UETrackBatch      # scripts/ is on sys.path via _SCRIPTS
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    cams = list(clip_frames)
    cam_idx = {c: i for i, c in enumerate(cams)}
    trk = UETrackBatch(len(cams))
    seeded = [False] * len(cams)
    prev3d = None
    track = []
    for f in range(n_frames):
        rgbs = [None] * len(cams)
        for c in cams:
            frame = clip_frames[c][f] if f < len(clip_frames[c]) else None
            i = cam_idx[c]
            if frame is None:
                continue
            rgb = frame[:, :, ::-1]
            if not seeded[i]:
                bx = yolo_boxes.get(f, {}).get(c)
                if bx is not None:
                    trk.init(i, rgb, tuple(float(v) for v in bx))
                    seeded[i] = True
            else:
                rgbs[i] = rgb
        boxes = trk.update(rgbs)                         # one batched forward for all seeded cams
        obs = {}
        for c in cams:
            i = cam_idx[c]
            b = boxes[i]
            if b is not None:
                obs[c] = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
            elif not seeded[i]:                          # just-seeded this frame: use the YOLO box
                bx = yolo_boxes.get(f, {}).get(c)
                if bx is not None:
                    obs[c] = (bx[0] + bx[2] / 2.0, bx[1] + bx[3] / 2.0)
        X, kept, _ = consensus.consensus3(obs, calib, prev=prev3d)
        if X is not None:
            prev3d = X
        track.append({"frame": f, "X": None if X is None else [round(float(v), 1) for v in X],
                      "n_cams": len(kept), "kept": sorted(kept)})
    return track


def track_cup_3d_from_cache(cache_json: str | Path, calib: dict) -> list[dict]:
    """Build the 3D cup track from a cached per-camera tracker dump (cache_tracks.py format:
    {frame: {cam: {"yolo", "trk":[cx,cy], "seeded"}}}) + greedy consensus. No clips / no GPU.

    This is the offline path used for the OMC-truth results: the per-camera UETrack points were cached
    once by the tracker thread; here we only run the geometry (consensus3) over them.
    """
    rec = json.loads(Path(cache_json).read_text())
    frames = sorted((int(k) for k in rec), )
    n = (max(frames) + 1) if frames else 0
    prev3d = None
    last_acc = None                    # frame index of the last accepted point (for the velocity gate)
    track = []
    for f in range(n):
        row = rec.get(str(f)) or rec.get(f) or {}
        obs = {c: tuple(v["trk"]) for c, v in row.items()
               if v.get("trk") is not None and c in calib}
        gap = 1 if last_acc is None else (f - last_acc)   # frames since last accept -> velocity budget
        X, kept, _ = consensus.consensus3(obs, calib, prev=prev3d, gap=gap)
        if X is not None:
            prev3d = X; last_acc = f
        track.append({
            "frame": f,
            "X": None if X is None else [round(float(v), 1) for v in X],
            "n_cams": len(kept),
            "kept": sorted(kept),
        })
    return track
