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

from cup_task.cloud_velocity import (FrameResult, compute_3d_velocity,   # reuse the solver
                                     kabsch_ransac, umeyama)

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
                 min_inliers: int = 8, gate_consensus: bool = True,
                 anchor_px: float | None = 30.0, max_scale_dev: float | None = 0.01,
                 retire_deformed: bool = True):
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
        self.gate_consensus = gate_consensus
        # ANCHOR: drop a track whose pixel drifts further than this from the frame's own detected
        # keypoint. We HAVE a keypoint every frame, so a track that wanders off the object is
        # detectable without any appearance model -- it is simply too far from where the object is.
        # None disables. Expressed in px because that is where the drift happens.
        # Measured n=12 (cup MOVING error vs OMC / coverage), with the consensus gate on:
        #     no anchor  22.2 / 97.7%      40px  18.0 / 97.7%
        #     30px       16.0 / 96.6%      25px  15.2 / 93.5%   (15px kills every track)
        # Monotonic and replicating (30px beats no-anchor on 11/12 trials), unlike the n_seed
        # sweep that looked good on one trial and reversed across the cohort. 30px is the default
        # because 25px buys 0.8mm/s for 3 points of coverage.
        self.anchor_px = anchor_px
        # RIGIDITY GATE. Fit a SIMILARITY (Umeyama) alongside the rigid fit and refuse the frame
        # when the fitted scale strays more than this from 1. A rigid cup cannot change size, so
        # s != 1 measures the cloud DEFORMING (tracks sliding), with no ground truth needed.
        # Measured n=12, cup, moving frames -- error by |s-1| quartile:
        #     most rigid 11.3 | 13.1 | 18.9 | most deformed 36.3 mm/s   (3.2x spread, rho +0.360)
        # As a gate at 0.01: keeps 75.6% of frames, median 16.83 -> 14.05, p90 79.8 -> 54.7,
        # >100mm/s 7.4% -> 3.4%.
        # ⚠ This is the ONLY quality signal here that works: rho(n_inliers, err) = -0.009, i.e.
        # min_inliers has NO predictive power, and raising it actively hurts (>=12 keeps 27% of
        # frames and makes the median WORSE, 19.96 vs 16.20). Scale finds what inlier count misses.
        # None disables.
        self.max_scale_dev = max_scale_dev
        # On a frame the rigidity gate rejects, kill the individual tracks that RANSAC found
        # inconsistent, rather than only refusing the frame. Sliding is per-track noise, not
        # accumulated drift (measured: per-track radius drift jumps to ~4mm in the first 30 frames
        # then PLATEAUS, so retiring by age would not help), which is why the removal is targeted
        # at the offending tracks instead of the oldest ones.
        self.retire_deformed = retire_deformed
        self._track_prev3d: dict[int, np.ndarray] = {}
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
        self._track_prev3d = {}       # ids are REUSED after a reseed -- stale continuity would
                                      # let consensus3 accept a new track against an old position

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
        """Triangulate each surviving track -> (cloud (M,3), ids (M,)). Identity preserved.

        ⚠ EACH TRACK IS CONSENSUS-GATED INDEPENDENTLY, which the first version did not do and
        which is why its cloud was not rigid. A track that drifts in ONE camera still gets
        triangulated from every camera that has it, and that one bad 2D point drags the 3D
        position -- measured, points that should sit at a fixed radius on a 40mm cup wandered with
        29mm std. `consensus3` is the repo's existing answer to exactly this ("which cameras agree
        about where this point is"), applied here PER TRACK rather than once for the whole object.
        A track whose cameras cannot agree contributes nothing instead of contributing a lie.
        """
        from cup_task import consensus as _cons
        from cup_task.kalman_3d import triangulate_dlt
        cams = [c for c in self._px if c in self.calib]
        if not cams:
            return np.empty((0, 3)), np.empty(0, int)
        K = len(next(iter(self._px.values())))
        pts, ids = [], []
        for k in range(K):
            obs = {c: tuple(self._px[c][k]) for c in cams if np.isfinite(self._px[c][k]).all()}
            if len(obs) < self.min_cams:
                continue
            if self.gate_consensus and len(obs) >= 2:
                X, kept, _ = _cons.consensus3(obs, self.calib, prev=self._track_prev3d.get(k))
                if X is None or len(kept) < self.min_cams:
                    continue
            else:
                use = list(obs)
                X = triangulate_dlt([self.calib[c] for c in use],
                                    [np.asarray(obs[c]) for c in use])
            if X is not None and np.isfinite(X).all():
                X = np.asarray(X, float)
                self._track_prev3d[k] = X
                pts.append(X); ids.append(k)
        return (np.array(pts) if pts else np.empty((0, 3))), np.array(ids, int)

    def _anchor(self, kp_by_cam: dict) -> None:
        """Kill tracks that have drifted too far from THIS frame's detected keypoint.

        The detection is available every frame, so a track that has slid off the object announces
        itself geometrically -- no appearance model, no template, no threshold on similarity. This
        is the cheap half of "we already have a mechanism to filter bad tracks".
        """
        if self.anchor_px is None:
            return
        for cam, pts in self._px.items():
            kp = kp_by_cam.get(cam)
            if kp is None or not np.isfinite(np.asarray(kp, float)).all():
                continue
            d = np.linalg.norm(pts - np.asarray(kp, float), axis=1)
            pts[d > self.anchor_px] = np.nan

    def _retire(self, ids) -> None:
        """Permanently drop these track ids in every camera (they will be replaced at the next
        reseed). Killing the point in the PIXEL state is what makes it stay dead -- the 3D lift
        reads from there, so a track removed here cannot re-enter the cloud."""
        for k in np.atleast_1d(np.asarray(ids, int)).ravel():
            for pts in self._px.values():
                if 0 <= k < len(pts):
                    pts[k] = np.nan
            self._track_prev3d.pop(int(k), None)

    def update(self, gray: dict, cup_xyz: np.ndarray, dt: float,
               kp_by_cam: dict | None = None) -> FrameResult | None:
        """One frame. `cup_xyz` = cup 3D position (seeding + reseeding).

        `kp_by_cam` = this frame's detected 2D keypoint per camera. Optional but recommended: it
        is what lets `anchor_px` cull tracks that have slid off the object.
        """
        if not self._px or not np.isfinite(cup_xyz).all():
            if np.isfinite(cup_xyz).all():
                self.seed(gray, cup_xyz)
            return None
        self._track_forward(gray)
        if kp_by_cam:
            self._anchor(kp_by_cam)
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
                    A, B = self._prev_cloud[ia], cloud[ib]
                    v = compute_3d_velocity(A, B, dt, self.units_per_metre)
                    # Rigidity check on the SAME inliers the motion was fitted to, so the number
                    # describes the points that actually produced the answer.
                    scale_ok = True
                    if self.max_scale_dev is not None and len(A) >= 4:
                        _, _, mask = kabsch_ransac(A, B, thresh=5.0)
                        if mask is not None and mask.sum() >= 4:
                            _, _, s = umeyama(A[mask], B[mask])
                            res.scale = s
                            scale_ok = abs(s - 1.0) <= self.max_scale_dev
                            # A deformed frame is not correctable -- measured, no power of s
                            # recovers it (36.81 -> 36.64 at best) because the deformed frames
                            # carry the SAME 0.938 ratio as the rigid ones. But the deformation is
                            # not shared equally: retire the tracks that individually violate
                            # rigidity, so the NEXT frame is built from better points instead of
                            # being refused too.
                            if not scale_ok and self.retire_deformed:
                                self._retire(common[~mask] if len(mask) == len(common) else [])
                        else:
                            scale_ok = False
                    if (v["linear_velocity"] is not None and scale_ok
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


def fit_cylinder_axis(cloud: np.ndarray, radius: float = 40.0
                      ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Surface points on a cup -> (centre, axis) of the best-fit cylinder of KNOWN radius.

    WHY THIS IS THE RIGHT MEASURAND. Every other "cup position" in this repo is an arbitrary point:
    the OMC marker cluster sits on one patch of the cup, the UETrack point is wherever a box centre
    landed, and the cloud's own centroid is biased to the visible near hemisphere. None is the
    cup's centre, so speeds compared between them differ by omega x r for an unknown r. Fitting a
    cylinder of the cup's KNOWN radius to observed surface points recovers a centre defined by the
    OBJECT rather than by the sensor -- the same physical point every frame, in every method.

    ⚠ THE NEAR-HEMISPHERE PROBLEM IS EXACTLY WHAT THIS FIXES. The cloud only ever sees the camera-
    facing surface, so its centroid sits ~R/2 off-centre and MOVES as the cup rotates (the visible
    patch changes). A cylinder fit pushes inward by the known radius from the observed surface, so
    a one-sided patch still recovers the axis it curves around.

    Method: the axis direction is the SMALLEST-variance direction of the surface normals' spread --
    for points on a cylinder, the direction along the axis is the one in which the points do NOT
    curve. Estimated here as the third principal component of the centred cloud only when the cloud
    is elongated enough to see it; otherwise the dominant spread direction is used. Then the centre
    is the point minimising sum(|dist_perp(p) - radius|^2), solved in the plane perpendicular to
    the axis by a linear (Pratt-style) circle fit.

    Returns (None, None) when there are too few points to constrain the fit.
    """
    P = np.asarray(cloud, float)
    if len(P) < 6:
        return None, None
    c0 = P.mean(0)
    Q = P - c0
    # axis = the principal direction with the LEAST curvature signature. For a cup viewed from the
    # side the points wrap around the axis, so the axis is the eigenvector whose spread is most
    # "flat" -- in practice the one with the smallest |curvature| of the projected points.
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    best, best_res = None, np.inf
    for ax in Vt:                      # try each principal direction as the axis
        ax = ax / np.linalg.norm(ax)
        # project into the plane perpendicular to ax and fit a circle of the KNOWN radius
        e1 = np.cross(ax, [1.0, 0, 0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(ax, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        u = Q @ e1
        v = Q @ e2
        # linear least squares for the circle centre: |p - c|^2 = r^2 expands to
        # 2*u*cu + 2*v*cv + (r^2 - cu^2 - cv^2) = u^2 + v^2, linear in (cu, cv, k)
        A = np.stack([2 * u, 2 * v, np.ones_like(u)], 1)
        b = u ** 2 + v ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cu, cv = sol[0], sol[1]
        res = float(np.mean((np.hypot(u - cu, v - cv) - radius) ** 2))
        if res < best_res:
            best_res, best = res, (c0 + cu * e1 + cv * e2, ax)
    return best if best is not None else (None, None)
