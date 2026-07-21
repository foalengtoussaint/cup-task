"""Wrist SPEED from optical flow — direct 2D pixel velocity, triangulated to 3D velocity.

WHY this exists: the pipeline's speed used to be d(position)/dt. Differentiation AMPLIFIES the
residual position jitter (YOLO's ~2.5mm) into velocity noise — that is why raw pos-diff speed is
~43mm/s off and even SmoothNet is 13.5. PyrLK measures the pixel displacement between two frames
DIRECTLY, with no differentiation, so its off-peak speed is clean (4.1mm/s vs OMC).

The 3D lift is the non-obvious part. We do NOT differentiate a 3D position. Per frame we triangulate
TWO points — the wrist pixel {p} and the flow-advanced pixel {p+flow} — and take their 3D difference:

    v3d = (triangulate{p_c + flow_c}  -  triangulate{p_c}) * fps

Both triangulations use the same cameras and the same calibration, so the common-mode calibration
error cancels in the difference and what survives is the motion. This is a VELOCITY measurement, not
a position derivative.

ONLINE vs OFFLINE: this stage is the reason the design puts flow ONLINE (see docs/PIPELINE_V3_PLAN).
PyrLK needs the raw frame PAIR (t-1, t). Live, both frames are already in hand — the cost is 0.5ms
per wrist per camera on top of a decode we have already paid. Offline it forces a SECOND full video
decode, which dominates everything else in the post-processing budget. Same numbers either way; the
online placement is strictly cheaper.

Known limitation (measured, do not re-litigate): flow OVER-shoots the fast peak by +61mm/s because
motion blur smears the wrist patch. That is fundamental to flow-triangulation, not a tuning problem
(RAFT/DIS/tuned-LK all over-shoot too). The fix is to blend with SmoothNet at speed — see
cup_task.speed_blend. Full numbers in docs/SPEED_METRICS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FPS = 60.0
CONF_THR = 0.30

# PyrLK settings. Chosen by shootout (docs/SPEED_METRICS.md): beats DIS (5x slower, same error),
# tuned-LK (smaller window, worse: 9.5), and RAFT-small (deep, worse: 18.1). A 21px window is wide
# enough to hold the blurred wrist patch at peak speed without averaging in the background.
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3)


def _lk_criteria():
    import cv2
    return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def flow_at(prev_gray, cur_gray, px) -> np.ndarray | None:
    """PyrLK the single pixel px from prev_gray to cur_gray. Returns the (2,) pixel displacement,
    or None if the tracker lost it. This is the whole per-camera online cost (~0.5ms)."""
    import cv2
    if px is None or not np.isfinite(px).all():
        return None
    p0 = np.asarray(px, dtype=np.float32).reshape(1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                         criteria=_lk_criteria(), **LK_PARAMS)
    if st[0, 0] != 1:
        return None
    return np.asarray(p1[0, 0] - p0[0, 0], dtype=float)


class FlowSpeedOnline:
    """Streaming wrist-flow for the live loop: feed it one rig-frame at a time.

    Holds only the previous grayscale frame per camera, so memory is O(cameras) and it never needs
    the clip. `update` returns the 3D speed for the frame just consumed (mm/s), or NaN when fewer
    than two cameras produced a usable flow vector.
    """

    def __init__(self, calib: dict, fps: float = FPS, min_cams: int = 2):
        self.calib = calib
        self.fps = fps
        self.min_cams = min_cams
        self._prev: dict[str, np.ndarray] = {}

    def update(self, gray_by_cam: dict[str, np.ndarray],
               wrist_px_by_cam: dict[str, np.ndarray]) -> float:
        """One rig-frame. gray_by_cam = {cam: HxW uint8}, wrist_px_by_cam = {cam: (2,) or None}."""
        obs_p, obs_pv = {}, {}
        for cam, gray in gray_by_cam.items():
            prev = self._prev.get(cam)
            px = wrist_px_by_cam.get(cam)
            if prev is not None and px is not None and np.isfinite(px).all():
                fl = flow_at(prev, gray, px)
                if fl is not None and np.isfinite(fl).all():
                    obs_p[cam] = np.asarray(px, dtype=float)
                    obs_pv[cam] = np.asarray(px, dtype=float) + fl
            self._prev[cam] = gray
        return speed_from_flow_obs(obs_p, obs_pv, self.calib, self.fps, self.min_cams)


def speed_from_flow_obs(obs_p: dict, obs_pv: dict, calib: dict,
                        fps: float = FPS, min_cams: int = 2) -> float:
    """Triangulate {p} and {p+flow} and difference them -> scalar 3D speed (mm/s), NaN if too few
    cameras. Shared by the online and offline paths so both compute the identical number."""
    from cup_task.kalman_3d import triangulate_dlt
    cams = [c for c in obs_p if c in obs_pv and c in calib]
    if len(cams) < min_cams:
        return float("nan")
    Xp = triangulate_dlt([calib[c] for c in cams], [np.asarray(obs_p[c]) for c in cams])
    Xv = triangulate_dlt([calib[c] for c in cams], [np.asarray(obs_pv[c]) for c in cams])
    if Xp is None or Xv is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(Xv) - np.asarray(Xp)) * fps)


def speed_from_cached_flow(wrist_px: dict[str, np.ndarray], flow: dict[str, np.ndarray],
                           calib: dict, n: int, fps: float = FPS,
                           min_cams: int = 2) -> np.ndarray:
    """OFFLINE path: per-camera wrist pixels + precomputed per-camera flow -> (n,) 3D speed track.

    Calls the SAME speed_from_flow_obs as the online class, so the offline replay of a live session
    reproduces the live numbers exactly (the number and its check must share one implementation).
    """
    out = np.full(n, np.nan)
    cams = [c for c in flow if c in wrist_px and c in calib]
    for f in range(n):
        obs_p, obs_pv = {}, {}
        for c in cams:
            if (f < len(wrist_px[c]) and f < len(flow[c])
                    and np.isfinite(wrist_px[c][f]).all() and np.isfinite(flow[c][f]).all()):
                obs_p[c] = wrist_px[c][f]
                obs_pv[c] = wrist_px[c][f] + flow[c][f]
        out[f] = speed_from_flow_obs(obs_p, obs_pv, calib, fps, min_cams)
    return out


def flow_track_from_clip(clip: Path, wrist_px: np.ndarray, cache_dir: Path | None = None,
                         method: str = "pyrlk") -> np.ndarray:
    """Decode a clip and PyrLK the wrist through it -> (T,2) pixel velocity. Offline/replay only.

    Cached to <cache_dir>/<clip.stem>__<method>.npy — the decode is the expensive part and the
    metrics on top are free to recompute, so a cached clip is never decoded twice.
    """
    import cv2
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ck = cache_dir / f"{Path(clip).stem}__{method}.npy"
        if ck.exists():
            v = np.load(ck)
            if len(v) == len(wrist_px):
                return v
    T = len(wrist_px)
    vel = np.full((T, 2), np.nan)
    cap = cv2.VideoCapture(str(clip))
    prev, f = None, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if prev is not None and f < T and np.isfinite(wrist_px[f - 1]).all():
            fl = flow_at(prev, gray, wrist_px[f - 1])
            if fl is not None:
                vel[f - 1] = fl
        prev = gray
        f += 1
    cap.release()
    if cache_dir is not None:
        np.save(ck, vel)
    return vel
