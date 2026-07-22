"""Tracked surface cloud -> 3D translation AND rotation, without ever matching appearance across views.

WHY THIS EXISTS (and why cloud_velocity.py does not work). The obvious build is: detect sub-features
per camera, match them across cameras by NCC along epipolar lines, triangulate, then Kabsch between
consecutive clouds. Measured on DELTA P07 that fails at the FIRST step, before any time is involved:
1-2 matches out of ~15 candidates per pair. The reason is not the detector and not the threshold --
at KNOWN-TRUE correspondences, NCC between two cameras has median 0.01 and clears a 0.65 gate only
5% of the time. A curved specular cup photographed from two directions simply does not produce
matchable patches. Cross-view appearance matching is a dead end on this rig, at any threshold.

WHAT THIS DOES INSTEAD -- correspondence by CONSTRUCTION, never by appearance:

  1. SEED geometrically. The cup's 3D position is known (consensus tracker) and its radius is known.
     Generate points ON that surface model in 3D, project them into every camera. Point k in cam A
     and point k in cam B are the same physical point BECAUSE THEY WERE THE SAME 3D POINT -- no
     matching, no NCC, no epipolar search.
  2. TRACK forward with PyrLK, independently per camera. This is what the user pointed out: PyrLK
     already gives t -> t+1 correspondence for free, within a camera, which is the only place
     correspondence is actually obtainable.
  3. LIFT each track to 3D by triangulating its own per-camera observations (>=2 cams), reusing the
     repo's DLT. Track identity is preserved through the lift, so the 3D cloud at t and at t+1 is
     THE SAME SET OF POINTS in the same order -- exactly what Kabsch requires and what re-detection
     could never supply.
  4. KABSCH + RANSAC between consecutive clouds -> linear and angular velocity.

The forward-backward check (track the point back and require it to return within `fb_tol` px) is
what keeps a drifting or occluded track from silently poisoning the fit; PyrLK's own status flag
does not catch a confident-but-wrong track.

    from cup_task.cloud_track import CloudTracker
    trk = CloudTracker(calib)                       # repo CamCalib objects
    res = trk.update(gray_by_cam, cup_xyz, dt=1/60) # cup_xyz from the consensus tracker
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from cup_task.cloud_velocity import FrameResult, compute_3d_velocity   # reuse the solver

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FB_TOL = 1.0          # px, forward-backward consistency
MIN_CAMS = 2          # cameras a track needs to be liftable to 3D
RESEED_BELOW = 8      # rebuild the seed when fewer than this many tracks survive


def surface_seed(centre: np.ndarray, radius: float = 40.0, height: float = 95.0,
                 n: int = 24, seed: int = 0) -> np.ndarray:
    """Points on a cup-sized cylinder centred at `centre` -> (n,3) world points.

    A CYLINDER, not a sphere: the cup is one, and the seed only has to be close enough that the
    projected points land on the real object's pixels. Being wrong about the exact surface costs
    nothing here -- the seeds are only a STARTING PLACE for PyrLK, which then follows whatever
    real texture is actually at that pixel. What matters is that the same 3D point defines the
    seed in every camera, which is what makes the cross-view correspondence exact by construction.
    """
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-height / 2, height / 2, n)
    return np.asarray(centre, float) + np.stack(
        [radius * np.cos(th), radius * np.sin(th), z], 1)


def _project(cal, X: np.ndarray) -> np.ndarray | None:
    """World point -> pixel, or None if behind the camera."""
    Y = np.asarray(cal.R) @ np.asarray(X, float) + np.asarray(cal.t).ravel()
    if Y[2] <= 1:
        return None
    u = np.asarray(cal.K) @ Y
    return u[:2] / u[2]


def _visible(cal, X: np.ndarray, centre: np.ndarray) -> bool:
    """Keep only the near hemisphere: a point on the far side of the cup is occluded BY the cup.

    Without this, half the seeds project onto the silhouette from behind, PyrLK tracks whatever
    front-surface texture happens to sit there, and those tracks report the wrong motion under
    rotation -- the exact signal we are trying to measure.
    """
    C = -np.asarray(cal.R).T @ np.asarray(cal.t).ravel()
    return float((np.asarray(X) - np.asarray(centre)) @ (C - np.asarray(centre))) > 0


class CloudTracker:
    """Tracks a seeded surface cloud and reports per-frame linear + angular velocity."""

    def __init__(self, calib: dict, radius: float = 40.0, height: float = 95.0,
                 n_seed: int = 24, fb_tol: float = FB_TOL, min_cams: int = MIN_CAMS,
                 units_per_metre: float = 1000.0, reseed_below: int = RESEED_BELOW,
                 min_inliers: int = 8):
        self.calib = calib
        self.radius, self.height, self.n_seed = radius, height, n_seed
        self.fb_tol, self.min_cams = fb_tol, min_cams
        self.units_per_metre = units_per_metre
        self.reseed_below = reseed_below
        # REFUSE TO ANSWER below this many inliers rather than emit a confident wrong number.
        # Measured (P07 trial_10, per-frame error vs OMC): 5-7 inliers -> 531 mm/s median,
        # 8-11 -> 113, 12+ -> 109. The blow-ups (up to 17 m/s) live almost entirely in the
        # thin-evidence frames, and a rigid fit on a handful of points on a ~40mm object is
        # nearly unconstrained. Returning NaN loses coverage honestly; returning the number
        # loses accuracy silently.
        self.min_inliers = min_inliers
        self._px: dict[str, np.ndarray] = {}      # cam -> (K,2) current pixel per track (NaN = lost)
        self._prev_gray: dict[str, np.ndarray] = {}
        self._prev_cloud: np.ndarray | None = None
        self._prev_ids: np.ndarray | None = None

    # -- seeding ------------------------------------------------------------------------------
    def seed(self, gray: dict, centre: np.ndarray) -> None:
        """(Re)seed the cloud from the cup's 3D position. Correspondence is by construction."""
        P = surface_seed(centre, self.radius, self.height, self.n_seed)
        self._px = {}
        for cam, cal in self.calib.items():
            if cam not in gray:
                continue
            pts = np.full((len(P), 2), np.nan)
            h, w = gray[cam].shape
            for k, X in enumerate(P):
                if not _visible(cal, X, centre):
                    continue
                u = _project(cal, X)
                if u is not None and 5 <= u[0] < w - 5 and 5 <= u[1] < h - 5:
                    pts[k] = u
            self._px[cam] = pts
        self._prev_gray = {c: g.copy() for c, g in gray.items()}
        self._prev_cloud = None       # a reseed breaks track identity; do not difference across it
        self._prev_ids = None

    # -- per-frame ----------------------------------------------------------------------------
    def _track_forward(self, gray: dict) -> None:
        """PyrLK every camera's tracks one frame, with a forward-backward consistency check."""
        for cam, pts in self._px.items():
            if cam not in gray or cam not in self._prev_gray:
                self._px[cam] = np.full_like(pts, np.nan)
                continue
            ok = np.isfinite(pts).all(1)
            if not ok.any():
                continue
            p0 = pts[ok].astype(np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray[cam], gray[cam], p0, None, **LK)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray[cam], self._prev_gray[cam], p1, None, **LK)
            fb = np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_tol)
            new = np.full((ok.sum(), 2), np.nan)
            new[good] = p1.reshape(-1, 2)[good]
            out = np.full_like(pts, np.nan)
            out[ok] = new
            self._px[cam] = out
        self._prev_gray = {c: g.copy() for c, g in gray.items()}

    def _lift(self) -> tuple[np.ndarray, np.ndarray]:
        """Triangulate each surviving track -> (cloud (M,3), ids (M,)). Identity preserved."""
        from cup_task.kalman_3d import triangulate_dlt
        cams = [c for c in self._px if c in self.calib]
        if not cams:
            return np.empty((0, 3)), np.empty(0, int)
        K = len(next(iter(self._px.values())))
        pts, ids = [], []
        for k in range(K):
            use = [c for c in cams if np.isfinite(self._px[c][k]).all()]
            if len(use) < self.min_cams:
                continue
            X = triangulate_dlt([self.calib[c] for c in use],
                                [np.asarray(self._px[c][k]) for c in use])
            if X is not None and np.isfinite(X).all():
                pts.append(X); ids.append(k)
        return (np.array(pts) if pts else np.empty((0, 3))), np.array(ids, int)

    def update(self, gray: dict, cup_xyz: np.ndarray, dt: float) -> FrameResult | None:
        """One frame. `cup_xyz` is the cup's 3D position (consensus tracker) for seeding only."""
        if not self._px or not np.isfinite(cup_xyz).all():
            if np.isfinite(cup_xyz).all():
                self.seed(gray, cup_xyz)
            return None
        self._track_forward(gray)
        cloud, ids = self._lift()

        res = None
        if len(cloud) >= 3:
            res = FrameResult(centroid_3d=cloud.mean(0), active_3d_points_count=len(cloud),
                              cloud=cloud)
            if self._prev_cloud is not None and self._prev_ids is not None:
                # INTERSECT on track id -- the whole point: same id == same physical point,
                # so the two clouds are corresponded with no matching whatsoever.
                common, ia, ib = np.intersect1d(self._prev_ids, ids, return_indices=True)
                if len(common) >= 3:
                    v = compute_3d_velocity(self._prev_cloud[ia], cloud[ib], dt,
                                            self.units_per_metre)
                    if (v["linear_velocity"] is not None
                            and v["n_inliers"] >= self.min_inliers):
                        res.linear_velocity = v["linear_velocity"]
                        res.linear_speed = v["linear_speed"]
                        res.angular_velocity = v["angular_velocity"]
                        res.angular_speed = v["angular_speed"]
                        res.rotation_matrix = v["rotation_matrix"]
                    res.active_3d_points_count = v["n_inliers"]
            self._prev_cloud, self._prev_ids = cloud, ids

        # tracks die (occlusion, blur, drift); reseed from the current 3D position when too few
        if len(cloud) < self.reseed_below and np.isfinite(cup_xyz).all():
            self.seed(gray, cup_xyz)
        return res
