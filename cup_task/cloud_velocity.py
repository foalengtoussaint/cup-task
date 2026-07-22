"""Multi-camera 3D point-cloud velocity: decoupled linear + angular speed from 2D keypoints.

Takes ONE 2D keypoint per camera (you already detect those) and turns it into a small LOCAL 3D
point cloud, by sampling sub-features on the object surface around the keypoint and matching them
across views along epipolar lines. Two consecutive clouds then give a rigid motion (Kabsch/SVD),
which decomposes into a linear velocity and an angular velocity.

WHY A CLOUD AND NOT JUST THE KEYPOINT. A single triangulated point gives you translation only --
rotation is invisible to it, because a rotating object whose centre is still produces zero motion
at its centre. The cloud is what makes angular velocity observable at all.

⚠⚠ THIS MODULE'S PIPELINE DOES NOT WORK ON REAL DATA. USE `cup_task.cloud_track` INSTEAD.
It is kept because it is the recorded NEGATIVE RESULT that motivates the replacement, and because
its solver half (kabsch / kabsch_ransac / compute_3d_velocity) is what cloud_track reuses.

  synthetic rig  linear 4-11% error, angular 3-791% (tests/test_cloud_velocity.py)
  REAL DELTA cup 13667 mm/s vs OMC, against the shipping flow path's 17.3 -- ~780x worse

The synthetic test was far EASIER than reality: it renders crisp isolated blobs on a clean
background, which are trivially matchable across views. A real cup is not. THE KILLER, measured at
KNOWN-TRUE correspondences on DELTA P07: cross-view NCC has median **0.01** and clears the 0.65
gate only **5%** of the time, so match_features_epipolar returns 1-2 matches out of ~15 candidates
and the cloud collapses to ~3 points. That is not a threshold to tune -- a curved specular cup lit
differently in each view does not produce cross-view-matchable patches at all. Ruled out first:
the cup spans 43-69px (LARGER than the 35px ROI, so the ROI is entirely on the object), and wide
baselines match slightly BETTER not worse (corr(baseline angle, matches) = +0.44).

THE FIX is to never compare appearance across cameras: seed points on a known 3D surface and let
PyrLK carry them within each camera, so cross-view correspondence is by CONSTRUCTION. That is
`cup_task.cloud_track`, which reaches 29.7 mm/s on the same data.

    from cup_task.cloud_velocity import CloudVelocityTracker
    trk = CloudVelocityTracker(cameras, units_per_metre=1000.0)   # mm calibration
    res = trk.update(frames_gray, keypoints_2d, dt=1/60)
    if res is not None:
        print(res.linear_speed, res.angular_speed)

UNITS. `cameras[...]['t']` fixes the world unit. This repo's calibration is in MILLIMETRES, while
the requested outputs are m/s and rad/s -- so `units_per_metre` is EXPLICIT and has no default that
silently assumes metres. Pass 1000.0 for mm, 1.0 for m. Angular velocity is unit-free.

ON scipy's `Rotation.align_vectors` FOR THE ROTATION STEP: it is CORRECT here. Despite the
"vectors" framing in its docs it is magnitude-weighted -- it minimises sum|b_i - R a_i|^2, the
Kabsch objective -- and it handles the reflection case internally. Verified equal to the SVD
implementation below to machine precision, noiseless and under noise (test_cloud_velocity.py).
This module still uses `np.linalg.svd` directly as the default, for two practical reasons: it
returns the translation in the same call, and it avoids a scipy round-trip through quaternions in
the inner loop. `kabsch_align_vectors` is exposed and is a drop-in alternative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --- defaults, all overridable per call ------------------------------------------------------
ROI_PX = 35              # side of the local patch sampled around the keypoint
MAX_FEATURES = 15        # Shi-Tomasi corners per view
MIN_FEATURES = 10        # below this a view contributes nothing useful
NCC_MIN = 0.65           # normalized cross-correlation floor for a cross-view match
EPILINE_MAX_PX = 2.0     # max distance from the epipolar line
PATCH_HALF = 5           # half-side of the NCC template (11x11)
CLOUD_RADIUS = 150.0     # outlier cut: distance from the centre keypoint, in WORLD units (mm)
REPROJ_MAX_PX = 3.0      # outlier cut: mean reprojection error


@dataclass
class FrameResult:
    """One frame's motion state. Velocities are None on the first frame (nothing to difference)."""
    centroid_3d: np.ndarray                     # (3,) world units
    active_3d_points_count: int
    linear_velocity: np.ndarray | None = None   # (3,) m/s
    linear_speed: float | None = None           # m/s
    angular_velocity: np.ndarray | None = None  # (3,) rad/s
    angular_speed: float | None = None          # rad/s
    rotation_matrix: np.ndarray | None = None   # (3,3) frame t -> t+1
    cloud: np.ndarray = field(default=None, repr=False)   # (N,3), kept for the next frame


# ================================================================================================
# 1. Local patch feature sampler
# ================================================================================================
def extract_local_features(gray: np.ndarray, keypoint: tuple[float, float],
                           roi_px: int = ROI_PX, max_features: int = MAX_FEATURES,
                           quality: float = 0.01, min_distance: float = 2.0) -> np.ndarray:
    """Shi-Tomasi corners inside a roi_px x roi_px box centred on `keypoint` -> (N,2) FULL-IMAGE px.

    Returns full-image coordinates, not ROI-local ones: every downstream step (epipolar lines,
    triangulation, reprojection) is defined in image coordinates, and converting once here means no
    other function has to remember the offset. Empty (0,2) if the ROI is degenerate or featureless.
    """
    if gray is None or gray.ndim != 2:
        return np.empty((0, 2), np.float32)
    h, w = gray.shape
    cx, cy = float(keypoint[0]), float(keypoint[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return np.empty((0, 2), np.float32)

    half = roi_px // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + roi_px, y0 + roi_px
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return np.empty((0, 2), np.float32)

    roi = gray[y0:y1, x0:x1]
    corners = cv2.goodFeaturesToTrack(roi, maxCorners=max_features, qualityLevel=quality,
                                      minDistance=min_distance)
    if corners is None:
        return np.empty((0, 2), np.float32)
    pts = corners.reshape(-1, 2).astype(np.float32)
    pts[:, 0] += x0
    pts[:, 1] += y0
    return pts


# ================================================================================================
# 2. Epipolar cross-view matcher
# ================================================================================================
def fundamental_from_calib(cam_a: dict, cam_b: dict) -> np.ndarray:
    """Fundamental matrix mapping a point in A to its epipolar line in B.

    Built from the RELATIVE pose (B with respect to A) rather than from point correspondences, so
    it inherits the calibration's accuracy instead of estimating anything.
    """
    Ka, Ra, ta = np.asarray(cam_a["K"]), np.asarray(cam_a["R"]), np.asarray(cam_a["t"]).reshape(3)
    Kb, Rb, tb = np.asarray(cam_b["K"]), np.asarray(cam_b["R"]), np.asarray(cam_b["t"]).reshape(3)
    R_rel = Rb @ Ra.T                      # A-camera frame -> B-camera frame
    t_rel = tb - R_rel @ ta
    tx = np.array([[0, -t_rel[2], t_rel[1]],
                   [t_rel[2], 0, -t_rel[0]],
                   [-t_rel[1], t_rel[0], 0]])
    E = tx @ R_rel
    return np.linalg.inv(Kb).T @ E @ np.linalg.inv(Ka)


def _patch(gray: np.ndarray, x: float, y: float, half: int = PATCH_HALF) -> np.ndarray | None:
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > gray.shape[1] or yi + half + 1 > gray.shape[0]:
        return None
    return gray[yi - half:yi + half + 1, xi - half:xi + half + 1]


def match_features_epipolar(gray_a: np.ndarray, gray_b: np.ndarray,
                            feats_a: np.ndarray, feats_b: np.ndarray,
                            cam_a: dict, cam_b: dict,
                            ncc_min: float = NCC_MIN,
                            epiline_max_px: float = EPILINE_MAX_PX,
                            patch_half: int = PATCH_HALF) -> list[tuple[int, int, float]]:
    """Match A's sub-features to B's along epipolar lines -> [(idx_a, idx_b, ncc), ...].

    TWO INDEPENDENT TESTS, and both must pass:
      * GEOMETRY -- candidate must lie within `epiline_max_px` of the epipolar line. This is a hard
        constraint from calibration; nothing on the object can violate it.
      * APPEARANCE -- normalized cross-correlation >= `ncc_min`, computed with matchTemplate on the
        11x11 patch. NCC (TM_CCOEFF_NORMED) is used rather than SSD because the same surface point
        differs in brightness and contrast between cameras (different exposure/white balance), and
        NCC is invariant to exactly that affine intensity change.

    Geometry alone is not enough -- the epipolar line is a LINE, and many points lie on it. Each B
    feature is claimed by at most one A feature (best NCC wins), so the result is a partial
    one-to-one matching rather than a fan-out.
    """
    if len(feats_a) == 0 or len(feats_b) == 0:
        return []
    F = fundamental_from_calib(cam_a, cam_b)
    lines = cv2.computeCorrespondEpilines(feats_a.reshape(-1, 1, 2).astype(np.float64), 1, F)
    lines = lines.reshape(-1, 3)

    best: dict[int, tuple[int, float]] = {}     # idx_b -> (idx_a, ncc)
    for ia, (pa, ln) in enumerate(zip(feats_a, lines)):
        pat = _patch(gray_a, pa[0], pa[1], patch_half)
        if pat is None:
            continue
        a, b_, c = ln                            # ax + by + c = 0, normalised by OpenCV
        denom = np.hypot(a, b_)
        if denom < 1e-9:
            continue
        for ib, pb in enumerate(feats_b):
            dist = abs(a * pb[0] + b_ * pb[1] + c) / denom
            if dist > epiline_max_px:
                continue
            win = _patch(gray_b, pb[0], pb[1], patch_half)
            if win is None or win.shape != pat.shape:
                continue
            ncc = float(cv2.matchTemplate(win, pat, cv2.TM_CCOEFF_NORMED)[0, 0])
            if ncc < ncc_min:
                continue
            if ib not in best or ncc > best[ib][1]:
                best[ib] = (ia, ncc)
    return [(ia, ib, n) for ib, (ia, n) in best.items()]


# ================================================================================================
# 3. Triangulation + outlier rejection
# ================================================================================================
def projection_matrix(cam: dict) -> np.ndarray:
    """3x4 P = K [R|t]. Uses a supplied 'P' when present, else builds it."""
    if cam.get("P") is not None:
        return np.asarray(cam["P"], float)
    K, R, t = np.asarray(cam["K"]), np.asarray(cam["R"]), np.asarray(cam["t"]).reshape(3, 1)
    return K @ np.hstack([R, t])


def _reprojection_error(X: np.ndarray, cams: list[dict], pts: list[np.ndarray]) -> float:
    """Mean reprojection error of one 3D point over the views that saw it (px)."""
    errs = []
    Xh = np.append(X, 1.0)
    for cam, p in zip(cams, pts):
        u = projection_matrix(cam) @ Xh
        if abs(u[2]) < 1e-9:
            return np.inf
        errs.append(np.hypot(u[0] / u[2] - p[0], u[1] / u[2] - p[1]))
    return float(np.mean(errs)) if errs else np.inf


def triangulate_and_filter(matches: list[tuple[np.ndarray, np.ndarray]],
                           cam_a: dict, cam_b: dict,
                           centre_3d: np.ndarray | None = None,
                           cloud_radius: float = CLOUD_RADIUS,
                           reproj_max_px: float = REPROJ_MAX_PX
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Matched pixel pairs -> (points_3d (N,3), reproj_errors (N,)) after both outlier cuts.

    `cloud_radius` is in WORLD units -- the same units as cam['t']. The default 150 corresponds to
    the requested 15cm on a MILLIMETRE calibration; pass 0.15 for a metre one.
    """
    if not matches:
        return np.empty((0, 3)), np.empty((0,))
    pa = np.array([m[0] for m in matches], np.float64).T      # 2xN
    pb = np.array([m[1] for m in matches], np.float64).T
    Xh = cv2.triangulatePoints(projection_matrix(cam_a), projection_matrix(cam_b), pa, pb)
    w = Xh[3]
    ok = np.abs(w) > 1e-9
    X = np.full((Xh.shape[1], 3), np.nan)
    X[ok] = (Xh[:3, ok] / w[ok]).T

    keep, errs = [], []
    for i, Xi in enumerate(X):
        if not np.isfinite(Xi).all():
            continue
        if centre_3d is not None and np.linalg.norm(Xi - centre_3d) > cloud_radius:
            continue
        e = _reprojection_error(Xi, [cam_a, cam_b], [pa[:, i], pb[:, i]])
        if e > reproj_max_px:
            continue
        keep.append(Xi); errs.append(e)
    if not keep:
        return np.empty((0, 3)), np.empty((0,))
    return np.array(keep), np.array(errs)


# ================================================================================================
# 4. Rigid motion + speed
# ================================================================================================
def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform taking cloud A onto cloud B: returns (R, t) with B ~= (R @ A.T).T + t.

    Correspondence-based Kabsch: row i of A must be the same physical point as row i of B. The
    determinant correction is what keeps this a ROTATION -- without it a noisy near-planar cloud
    can produce a REFLECTION (det = -1), which is not a physical rigid motion and would show up as
    a nonsensical angular velocity.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def kabsch_align_vectors(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same fit via `scipy.spatial.transform.Rotation.align_vectors` -- an equivalent alternative.

    align_vectors is MAGNITUDE-WEIGHTED (it minimises sum|b_i - R a_i|^2, the Kabsch objective),
    not direction-only as its name suggests, and it applies the same reflection correction. The
    test suite asserts it agrees with `kabsch` to machine precision, including under noise and on
    strongly anisotropic clouds. Use whichever reads better; `kabsch` is the default only because
    it also returns the translation and skips a quaternion round-trip.
    """
    from scipy.spatial.transform import Rotation
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    rot, _ = Rotation.align_vectors(B - cb, A - ca)
    R = rot.as_matrix()
    return R, cb - R @ ca


def compute_3d_velocity(cloud_a: np.ndarray, cloud_b: np.ndarray, dt: float,
                        units_per_metre: float = 1000.0,
                        ransac: bool = True, ransac_thresh: float = 5.0) -> dict:
    """Two corresponded clouds -> decoupled linear and angular velocity.

    Returns m/s and rad/s. `units_per_metre` converts the world unit (1000 for mm, 1 for m);
    angular velocity is unit-free so it is unaffected. `n_inliers` reports how many pairs actually
    supported the motion -- read it, because a small inlier count means the rest of the numbers
    are a fit to a handful of points.

    `ransac=True` (default) because re-detected clouds contain genuinely non-corresponding pairs;
    see kabsch_ransac. Pass False only when correspondence is known exact (e.g. tracked features).

    The rotation is converted to a rotation VECTOR (axis * angle) and divided by dt. That is exact
    for a per-frame delta as long as the rotation is under pi -- beyond that the axis-angle form
    aliases to the short way round, which no per-frame measurement can distinguish. At 60fps that
    corresponds to >180 rad/s, far outside anything physical here.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    from scipy.spatial.transform import Rotation
    n_in = len(cloud_a)
    mask = np.ones(len(cloud_a), bool)
    if ransac and len(cloud_a) >= 4:
        R, t_vec, mask = kabsch_ransac(cloud_a, cloud_b, ransac_thresh)
        if R is None:
            return {"rotation_matrix": None, "linear_velocity": None, "linear_speed": None,
                    "angular_velocity": None, "angular_speed": None, "n_inliers": 0}
        n_in = int(mask.sum())
    else:
        R, t_vec = kabsch(cloud_a, cloud_b)
    rotvec = Rotation.from_matrix(R).as_rotvec()

    # ⚠ REPORT THE CENTROID'S MOTION, NOT KABSCH's t_vec.
    # Kabsch returns t = centroid_B - R @ centroid_A, which is the translation of the WORLD ORIGIN
    # under the fitted motion. The cloud sits ~1700mm from that origin here, so that origin-
    # referred translation carries a LEVER ARM: a small rotation error swings R @ centroid_A by
    # (rotation error) x (distance to origin), and shows up as an enormous phantom translation.
    # Measured on DELTA P07 moving frames: t_vec read 2.10x the true cup speed while the cloud's
    # own centroid displacement read 0.77x -- the tracking was fine, the reference point was not.
    # The object's velocity is the motion of a point ON the object, so use the centroid.
    ca = np.asarray(cloud_a, float)[mask].mean(0)
    cb = np.asarray(cloud_b, float)[mask].mean(0)
    lin = (cb - ca) / units_per_metre / dt
    ang = rotvec / dt
    return {
        "rotation_matrix": R,
        "linear_velocity": lin,
        "linear_speed": float(np.linalg.norm(lin)),
        "angular_velocity": ang,
        "angular_speed": float(np.linalg.norm(ang)),
        "n_inliers": n_in,
    }


# ================================================================================================
# Orchestration
# ================================================================================================
class CloudVelocityTracker:
    """Stateful per-frame driver: holds the previous cloud so `update` can difference against it.

    The cloud is rebuilt from scratch each frame and matched to the previous one by NEAREST
    NEIGHBOUR, because Shi-Tomasi gives no persistent identity across frames -- corner k on frame t
    is not corner k on frame t+1. Kabsch NEEDS true correspondence, so a wrong pairing is a wrong
    rotation, and this is the weakest link in the chain: it assumes the object moved less between
    frames than the spacing between its own sub-features. For fast motion, track the features
    (PyrLK) instead of re-detecting them -- see `cup_task.flow_speed`, which measures velocity
    directly from flow and sidesteps correspondence entirely.
    """

    def __init__(self, cameras: dict, units_per_metre: float = 1000.0,
                 roi_px: int = ROI_PX, ncc_min: float = NCC_MIN,
                 epiline_max_px: float = EPILINE_MAX_PX,
                 cloud_radius: float = CLOUD_RADIUS,
                 reproj_max_px: float = REPROJ_MAX_PX,
                 min_features: int = MIN_FEATURES):
        self.cameras = cameras
        self.units_per_metre = units_per_metre
        self.roi_px = roi_px
        self.ncc_min = ncc_min
        self.epiline_max_px = epiline_max_px
        self.cloud_radius = cloud_radius
        self.reproj_max_px = reproj_max_px
        self.min_features = min_features
        self._prev: np.ndarray | None = None

    # -- cloud construction -------------------------------------------------------------------
    def build_cloud(self, frames_gray: dict, keypoints_2d: dict) -> tuple[np.ndarray, np.ndarray]:
        """All views -> (cloud (N,3), centre_3d (3,)). Uses the FIRST view as the reference.

        The centre is triangulated from the KEYPOINTS across all available views (not from the
        cloud), so the distance filter is anchored to something independent of the sub-features it
        is filtering -- otherwise an outlier would drag the centre toward itself and survive.
        """
        cams = [c for c in keypoints_2d
                if c in self.cameras and c in frames_gray
                and keypoints_2d[c] is not None
                and np.isfinite(np.asarray(keypoints_2d[c], float)).all()]
        if len(cams) < 2:
            return np.empty((0, 3)), np.full(3, np.nan)

        centre = self._triangulate_centre(cams, keypoints_2d)
        feats = {c: extract_local_features(frames_gray[c], keypoints_2d[c], self.roi_px)
                 for c in cams}
        ref = cams[0]
        if len(feats[ref]) < 2:
            return np.empty((0, 3)), centre

        clouds = []
        for other in cams[1:]:
            if len(feats[other]) < 2:
                continue
            m = match_features_epipolar(frames_gray[ref], frames_gray[other],
                                        feats[ref], feats[other],
                                        self.cameras[ref], self.cameras[other],
                                        self.ncc_min, self.epiline_max_px)
            if not m:
                continue
            pairs = [(feats[ref][ia], feats[other][ib]) for ia, ib, _ in m]
            X, _ = triangulate_and_filter(pairs, self.cameras[ref], self.cameras[other],
                                          centre if np.isfinite(centre).all() else None,
                                          self.cloud_radius, self.reproj_max_px)
            if len(X):
                clouds.append(X)
        if not clouds:
            return np.empty((0, 3)), centre
        return np.vstack(clouds), centre

    def _triangulate_centre(self, cams: list, keypoints_2d: dict) -> np.ndarray:
        """DLT over all views on the CENTRE keypoint alone."""
        A = []
        for c in cams:
            P = projection_matrix(self.cameras[c])
            x, y = float(keypoints_2d[c][0]), float(keypoints_2d[c][1])
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.array(A))
        X = Vt[-1]
        return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)

    # -- per-frame ----------------------------------------------------------------------------
    def update(self, frames_gray: dict, keypoints_2d: dict, dt: float) -> FrameResult | None:
        """One frame in, one FrameResult out (None if the cloud is too small to be usable).

        Velocities are None on the first usable frame: a velocity needs two frames, and reporting
        zero there would be a fabricated measurement rather than a missing one.
        """
        cloud, centre = self.build_cloud(frames_gray, keypoints_2d)
        if len(cloud) < self.min_features:
            self._prev = None      # a gap breaks correspondence; do not difference across it
            return None

        res = FrameResult(centroid_3d=cloud.mean(0), active_3d_points_count=len(cloud),
                          cloud=cloud)
        if self._prev is not None:
            a, b = _robust_pair(self._prev, cloud, self.cloud_radius)
            if len(a) >= 3:
                v = compute_3d_velocity(a, b, dt, self.units_per_metre)
                res.linear_velocity = v["linear_velocity"]
                res.linear_speed = v["linear_speed"]
                res.angular_velocity = v["angular_velocity"]
                res.angular_speed = v["angular_speed"]
                res.rotation_matrix = v["rotation_matrix"]
                # the count that matters is the INLIER count -- the pairs that actually supported
                # the motion, not the pairs that were offered to the solver
                res.active_3d_points_count = v["n_inliers"]
        self._prev = cloud
        return res


def correspond_clouds(A: np.ndarray, B: np.ndarray,
                      max_dist: float = CLOUD_RADIUS,
                      shift: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mutual-nearest-neighbour pairing between two clouds -> (A_sub, B_sub), row-aligned.

    MUTUAL is the point: a plain nearest-neighbour lets several A points claim one B point, which
    silently duplicates evidence and biases the fit. Requiring the choice to be reciprocated
    discards ambiguous pairs instead, which is the right failure for an estimator that assumes
    true correspondence.

    ⚠ `shift` MATTERS AND IS NOT OPTIONAL IN PRACTICE. Nearest-neighbour assumes the object moved
    LESS than the spacing between its own sub-features; when it moves more, each point's true
    partner is no longer its nearest one and the pairing biases the displacement TOWARD ZERO
    (measured: 8.3mm recovered from a true 12mm step, a 30% under-read, because the wrong partner
    sits partway along the motion). Passing the centroid shift -- which needs no correspondence --
    removes the bulk motion first, so the matching only has to resolve the residual deformation.
    """
    if len(A) == 0 or len(B) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    from scipy.spatial import cKDTree
    A_q = A if shift is None else A + np.asarray(shift, float)
    ta, tb = cKDTree(A_q), cKDTree(B)
    da, ia = tb.query(A_q)        # for each A, nearest B
    _, ib = ta.query(B)           # for each B, nearest A
    ra, rb = [], []
    for i, (j, d) in enumerate(zip(ia, da)):
        if d <= max_dist and ib[j] == i:
            ra.append(A[i]); rb.append(B[j])      # NB: original A, not the shifted query copy
    return np.array(ra), np.array(rb)


def kabsch_ransac(A: np.ndarray, B: np.ndarray, thresh: float = 5.0,
                  iters: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Largest subset of pairs explained by ONE rigid motion -> (R, t, inlier_mask).

    ⚠ THIS IS LOAD-BEARING, NOT A REFINEMENT. The cloud is RE-DETECTED each frame, so its points
    are not stable physical landmarks: measured on the synthetic rig, only 5 of 16 true object
    points were hit, and over half the cloud sat 11-27mm from ANY real point. Kabsch assumes row i
    of A and row i of B are the same physical point; here a large fraction of pairs simply are not.
    Plain least squares over all of them fits the mismatches as hard as the good pairs, which
    reads LOW when the wrong partner sits partway along the motion and HIGH when it sits beyond.

    RANSAC replaces the assumption with a test: sample 3 pairs, fit, count how many other pairs
    that motion explains within `thresh`, keep the largest consistent set. It needs no threshold
    on the DATA (only on the residual) and degrades to "no answer" rather than a confident wrong
    one -- which is the correct failure for a measurement.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    n = len(A)
    if n < 3:
        return None, None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R, t = kabsch(A[idx], B[idx])
        res = np.linalg.norm((R @ A.T).T + t - B, axis=1)
        inl = res < thresh
        if inl.sum() > best.sum():
            best = inl
        if best.sum() == n:
            break
    if best.sum() < 3:
        return None, None, best
    R, t = kabsch(A[best], B[best])          # refit on all inliers
    return R, t, best


def _robust_pair(A: np.ndarray, B: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Correspondence with the bulk translation removed before matching.

    Nearest-neighbour assumes the object moved LESS than the spacing between its own sub-features.
    When it moves more, each point's true partner is no longer its nearest and the pairing biases
    the displacement toward zero (measured: 8.3mm from a true 12mm step). The centroid difference
    needs no correspondence, so removing it first is free and makes the matching resolve only the
    residual deformation.

    Deliberately ONE pass: a second pass re-estimating the shift from the surviving pairs
    compounds their bias instead of correcting it (measured: over-read 12mm as 17.9mm). RANSAC
    downstream is what handles the pairs this still gets wrong.
    """
    return correspond_clouds(A, B, max_dist, B.mean(0) - A.mean(0))
