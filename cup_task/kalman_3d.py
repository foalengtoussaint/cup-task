"""3D Kalman filter for multi-camera live object tracking.

Loads an aniposelib TOML calibration, then fuses per-camera 2D detections
into a 6D world-frame state (position + velocity, mm) via an EKF — each
camera's pixel observation enters as a measurement linearized around the
current state.

Used by `live_track.py`. Independent from `kalman.py`, which is the 2D
single-camera filter used by the offline pseudo-label / evaluate pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import toml


# Chi-squared gates by DoF (99.9% threshold)
GATE_2D = 13.82    # 2 DoF, used by 2D image-plane measurement updates
GATE_3D = 16.27    # 3 DoF, used by 3D world-position measurement updates


@dataclass
class CamCalib:
    """One camera's calibration, with intrinsics rescaled to the live size."""
    name: str            # e.g. "cam_1"
    K: np.ndarray        # 3x3
    dist: np.ndarray     # (5,) Brown-Conrady
    R: np.ndarray        # 3x3, world -> camera
    t: np.ndarray        # (3,) mm, world -> camera
    size: tuple[int, int]  # (W, H) of the target frame


def load_calibration(toml_path: str | Path,
                     target_size: tuple[int, int] = (1280, 720),
                     ) -> dict[str, CamCalib]:
    """Load an aniposelib TOML, rescale intrinsics, return {cam_name: CamCalib}.

    The TOML's [cam_N] blocks carry a `name` field (e.g. "1".."5"); the
    returned dict is keyed by f"cam_{name}", matching the ZMQ stream names
    from `recording/cam_streams.py`.
    """
    data = toml.load(toml_path)

    out: dict[str, CamCalib] = {}
    for key, block in data.items():
        if not key.startswith("cam_") or not isinstance(block, dict):
            continue
        cal_w, cal_h = block["size"]
        sx = target_size[0] / cal_w
        sy = target_size[1] / cal_h
        K = np.array(block["matrix"], dtype=float)
        K[0, :] *= sx
        K[1, :] *= sy
        dist = np.array(block["distortions"], dtype=float)
        R, _ = cv2.Rodrigues(np.array(block["rotation"], dtype=float))
        t = np.array(block["translation"], dtype=float)
        # Three TOML naming conventions in the wild — all normalized to "cam_<n>":
        #   refit_extrinsics:        name="cam_1" → "cam_1"
        #   recording/run_calibration: name="cam1"  → "cam_1"
        #   iMOVE app:               name="3"     → "cam_3"
        name = block["name"]
        if name.startswith("cam_"):
            cam_key = name
        elif name.startswith("cam"):
            cam_key = f"cam_{name[3:]}"
        else:
            cam_key = f"cam_{name}"
        out[cam_key] = CamCalib(name=cam_key, K=K, dist=dist, R=R, t=t,
                                 size=tuple(target_size))
    return out


def project_points(cam: CamCalib, X_world: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: project an Nx3 array of world points into one cam.

    Returns (uvs Nx2, valid N-bool). valid[i] is False when the point is
    behind the camera (Z_cam <= 0). Uses cv2.projectPoints (C path) — much
    faster than calling `project()` in a Python loop.
    """
    if X_world.size == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    Xc = X_world @ cam.R.T + cam.t
    valid = Xc[:, 2] > 0
    if not np.any(valid):
        return np.zeros((len(X_world), 2)), valid
    rvec, _ = cv2.Rodrigues(cam.R)
    proj, _ = cv2.projectPoints(X_world.reshape(-1, 1, 3).astype(np.float64),
                                rvec, cam.t.reshape(3, 1).astype(np.float64),
                                cam.K, cam.dist)
    return proj.reshape(-1, 2), valid


def project(cam: CamCalib, X_world: np.ndarray) -> tuple[np.ndarray, bool]:
    """Project a 3D world point into one camera's pixel coords (with distortion).

    Returns ((u, v), in_front). in_front is False if the point sits behind the
    camera (Z_cam <= 0); in that case (u, v) is meaningless.
    """
    Xc = cam.R @ X_world + cam.t
    if Xc[2] <= 0:
        return np.zeros(2), False
    x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = cam.dist[:5]
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = cam.K[0, 0] * x_d + cam.K[0, 2]
    v = cam.K[1, 1] * y_d + cam.K[1, 2]
    return np.array([u, v]), True


def triangulate_dlt(cams: Sequence[CamCalib],
                    pts: Sequence[np.ndarray]) -> np.ndarray:
    """Triangulate one 3D world point from N>=2 camera observations (mm).

    pts[i] is the (cx, cy) pixel coord in cams[i]. Observations are
    undistorted to normalized image coords first, then a DLT linear system
    is solved by SVD.
    """
    if len(cams) < 2:
        raise ValueError("need >=2 cams to triangulate")

    A = np.zeros((2 * len(cams), 4))
    for i, (cam, pt) in enumerate(zip(cams, pts)):
        und = cv2.undistortPoints(
            np.array([[pt]], dtype=np.float32), cam.K, cam.dist)[0, 0]
        P = np.hstack([cam.R, cam.t.reshape(3, 1)])  # 3x4 in camera frame
        x, y = float(und[0]), float(und[1])
        A[2 * i]     = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


class KalmanFilter3D:
    """6D constant-velocity Kalman filter (3D pos + 3D vel, mm world coords)
    updated by per-camera 2D detections via EKF.

    State: [X, Y, Z, Vx, Vy, Vz] (positions in mm, velocities in mm/s).
    Each `update()` call expects a wall-clock timestamp; the filter computes
    dt internally and applies the prediction step itself.
    """

    def __init__(self, process_noise: float = 200.0,
                 meas_noise_px: float = 8.0):
        # process_noise is in mm/s^2 (continuous-time accel std-dev).
        # meas_noise_px ~ 1σ pixel noise for 2D image-plane measurements;
        # the Mahalanobis gate radius scales with this. 8 px ≈ ±24 px
        # accept window with 99.9% chi-sq gate, big enough to absorb sharp
        # direction changes that the constant-velocity model didn't predict.
        self.q = process_noise ** 2
        self.r = meas_noise_px ** 2
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t_last: float | None = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def init(self, X: np.ndarray, t: float) -> None:
        self.x = np.zeros(6)
        self.x[:3] = X
        # ~50 mm position uncertainty (DLT noise), velocity unknown (~500 mm/s)
        self.P = np.diag([50.0, 50.0, 50.0, 500.0, 500.0, 500.0]) ** 2
        self.t_last = t

    def predict(self, t: float) -> None:
        assert self.x is not None and self.t_last is not None
        dt = t - self.t_last
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        # Continuous white-acceleration model
        Q = np.zeros((6, 6))
        Q[:3, :3] = self.q * dt ** 3 / 3 * np.eye(3)
        Q[:3, 3:] = self.q * dt ** 2 / 2 * np.eye(3)
        Q[3:, :3] = self.q * dt ** 2 / 2 * np.eye(3)
        Q[3:, 3:] = self.q * dt * np.eye(3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_last = t

    def _project_and_jacobian(self, cam: CamCalib
                              ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Pinhole projection + 2x6 Jacobian wrt state (no distortion in H)."""
        assert self.x is not None
        Xc = cam.R @ self.x[:3] + cam.t
        if Xc[2] <= 0:
            return np.zeros(2), np.zeros((2, 6)), False
        x, y, z = Xc
        fx, fy = cam.K[0, 0], cam.K[1, 1]
        cx0, cy0 = cam.K[0, 2], cam.K[1, 2]
        z_pred = np.array([fx * x / z + cx0, fy * y / z + cy0])
        J_xc = np.array([
            [fx / z, 0.0,    -fx * x / (z * z)],
            [0.0,    fy / z, -fy * y / (z * z)],
        ])
        H = np.zeros((2, 6))
        H[:, :3] = J_xc @ cam.R
        return z_pred, H, True

    def update(self, cam: CamCalib, z_pix: np.ndarray, t: float,
               gate: float = GATE_2D) -> tuple[bool, float]:
        """EKF update with one camera's 2D detection. Returns (accepted, d^2).

        Rejects (no state change) on gate failure or if the projection puts
        the state behind the camera.
        """
        # Undistort measurement back to pinhole pixel coords (P=K) so it
        # matches the distortion-free projection used in z_pred / H.
        und = cv2.undistortPoints(
            np.array([[z_pix]], dtype=np.float32), cam.K, cam.dist,
            P=cam.K)[0, 0]
        z_meas = np.array(und, dtype=float)

        self.predict(t)
        z_pred, H, ok = self._project_and_jacobian(cam)
        if not ok:
            return False, float("inf")
        S = H @ self.P @ H.T + self.r * np.eye(2)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, float("inf")
        y = z_meas - z_pred
        d2 = float(y @ S_inv @ y)
        if d2 > gate:
            return False, d2
        K_gain = self.P @ H.T @ S_inv
        self.x = self.x + K_gain @ y
        self.P = (np.eye(6) - K_gain @ H) @ self.P
        return True, d2

    def update_3d(self, X_meas: np.ndarray, t: float,
                  R: np.ndarray | None = None,
                  meas_noise_mm: float = 60.0,
                  gate: float = GATE_3D) -> tuple[bool, float]:
        """Linear KF update with a 3D world-frame position measurement.

        `R`: 3x3 world-frame measurement noise covariance. Pass anisotropic
        per-cam R (small perpendicular, large along depth ray) to keep
        biased depth estimates from dragging the state — perpendicular
        components from different cams then triangulate depth correctly.
        If None, falls back to isotropic `meas_noise_mm² · I`.
        """
        assert self.x is not None
        self.predict(t)
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        if R is None:
            R = (meas_noise_mm ** 2) * np.eye(3)
        y = X_meas - self.x[:3]
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, float("inf")
        d2 = float(y @ S_inv @ y)
        if d2 > gate:
            return False, d2
        K_gain = self.P @ H.T @ S_inv
        self.x = self.x + K_gain @ y
        self.P = (np.eye(6) - K_gain @ H) @ self.P
        return True, d2


    def project_into(self, cam: CamCalib) -> tuple[np.ndarray, bool]:
        """Project current state into one camera's pixel coords."""
        assert self.x is not None
        return project(cam, self.x[:3])


def measurement_noise_world(cam: CamCalib,
                            sigma_perp_mm: float = 5.0,
                            sigma_depth_mm: float = 100.0) -> np.ndarray:
    """3x3 world-frame covariance for a 3D point back-projected from a 2D
    detection in `cam`. Cigar-shaped: small perpendicular (image plane is
    well observed), large along the cam's depth ray (depth from bbox/mask
    is noisy and shape-bias prone).

    R_world = R_cam_to_world @ diag(σ_⊥², σ_⊥², σ_depth²) @ R_cam_to_world^T
    where R_cam_to_world = cam.R^T (since cam.R maps world→cam).
    """
    R_local = np.diag([sigma_perp_mm ** 2, sigma_perp_mm ** 2,
                       sigma_depth_mm ** 2])
    R_c2w = cam.R.T
    return R_c2w @ R_local @ R_c2w.T


def box_corners(X_center: np.ndarray, W: float, H: float, D: float,
                yaw: float) -> np.ndarray:
    """8 corners of an oriented 3D box, in world frame.

    Box is centered at X_center, sized (W,D,H) along its local (x,y,z),
    rotated by `yaw` around world Z (vertical). Order:
        0..3 = bottom face (CCW from above), 4..7 = top face.
    """
    hw, hd, hh = W / 2.0, D / 2.0, H / 2.0
    local = np.array([
        [-hw, -hd, -hh], [ hw, -hd, -hh], [ hw,  hd, -hh], [-hw,  hd, -hh],
        [-hw, -hd,  hh], [ hw, -hd,  hh], [ hw,  hd,  hh], [-hw,  hd,  hh],
    ], dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ Rz.T + X_center


def align_z_to(axis: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the canonical Z = [0,0,1] direction onto
    a unit vector `axis`. Identity if axis is already +Z, 180° flip about
    X if it's −Z, Rodrigues formula otherwise.
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    a = axis / n
    z = np.array([0.0, 0.0, 1.0])
    cos_a = float(np.dot(z, a))
    if cos_a > 1.0 - 1e-9:
        return np.eye(3)
    if cos_a < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    rot_axis = np.cross(z, a)
    sin_a = float(np.linalg.norm(rot_axis))
    rot_axis = rot_axis / sin_a
    K = np.array([[0.0, -rot_axis[2], rot_axis[1]],
                  [rot_axis[2], 0.0, -rot_axis[0]],
                  [-rot_axis[1], rot_axis[0], 0.0]])
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


def cylinder_points(X_center: np.ndarray, R: float, H: float,
                    axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
                    n_around: int = 8) -> np.ndarray:
    """Sample 2·n_around surface points on a cylinder whose axis is `axis`
    (unit vector, world frame), centered at X_center. Returns top ring
    followed by bottom ring.
    """
    theta = np.linspace(0.0, 2 * np.pi, n_around, endpoint=False)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z_top = np.full(n_around, H / 2.0)
    z_bot = np.full(n_around, -H / 2.0)
    top_local = np.stack([x, y, z_top], axis=1)
    bot_local = np.stack([x, y, z_bot], axis=1)
    Rot = align_z_to(axis)
    top = top_local @ Rot.T + X_center
    bot = bot_local @ Rot.T + X_center
    return np.vstack([top, bot])


def project_cylinder(X_center: np.ndarray, R: float, H: float,
                     cam: CamCalib,
                     axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
                     ) -> tuple[float, float, float, float] | None:
    """Project a cylinder with arbitrary axis direction into one camera,
    return the bounding 2D bbox of its silhouette. Vectorized via
    cv2.projectPoints; ~10× faster than the per-point loop, which matters
    for the scipy refit's hot path.
    """
    pts = cylinder_points(X_center, R, H, axis)
    uvs, valid = project_points(cam, pts)
    if valid.sum() < 4:
        return None
    arr = uvs[valid]
    return (float(arr[:, 0].min()), float(arr[:, 1].min()),
            float(arr[:, 0].max()), float(arr[:, 1].max()))


def cylinder_silhouette_area_mm2(R: float, H: float, cam: CamCalib,
                                 axis: np.ndarray = np.array([0.0, 0.0, 1.0])
                                 ) -> float:
    """Real-world silhouette area (mm²) of a cylinder viewed by `cam`.
    Rectangle 2R·H·sin_tilt + circle cap π·R²·cos_tilt, where tilt is the
    angle between the cam's optical axis and the cylinder's axis (so
    pure-side view → rectangle, pure-end view → circle).
    """
    cam_axis_world = cam.R.T @ np.array([0.0, 0.0, 1.0])
    axis_n = np.asarray(axis, dtype=float)
    axis_n = axis_n / max(float(np.linalg.norm(axis_n)), 1e-12)
    cos_tilt = abs(float(np.dot(cam_axis_world, axis_n)))
    sin_tilt = float(np.sqrt(max(0.0, 1.0 - cos_tilt ** 2)))
    return 2.0 * R * H * sin_tilt + np.pi * R * R * cos_tilt


def project_box(X_center: np.ndarray, W: float, H: float, D: float,
                yaw: float, cam: CamCalib
                ) -> tuple[float, float, float, float] | None:
    """Project an oriented 3D box into a camera. Returns the tight 2D
    bounding rectangle (x0, y0, x1, y1) of the projected corners that
    land in front of the camera, or None if fewer than 4 corners are
    visible.
    """
    uvs = []
    for X in box_corners(X_center, W, H, D, yaw):
        uv, ok = project(cam, X)
        if ok:
            uvs.append(uv)
    if len(uvs) < 4:
        return None
    arr = np.array(uvs)
    return (float(arr[:, 0].min()), float(arr[:, 1].min()),
            float(arr[:, 0].max()), float(arr[:, 1].max()))


def cam_depth_of(X_world: np.ndarray, cam: CamCalib) -> float:
    """Z coordinate of a world point in this cam's frame (mm in front)."""
    return float((cam.R @ X_world + cam.t)[2])


def back_project(cam: CamCalib, cx: float, cy: float, Z_cam: float
                 ) -> np.ndarray:
    """Convert a (cx, cy) pixel observed at known depth in `cam` into a
    world-frame 3D point. Depth Z_cam comes from the bbox-size scaling
    trick (`Z = Z_ref * w_ref / w_now`).
    """
    und = cv2.undistortPoints(
        np.array([[(cx, cy)]], dtype=np.float32), cam.K, cam.dist,
        P=cam.K)[0, 0]
    x_norm = (float(und[0]) - cam.K[0, 2]) / cam.K[0, 0]
    y_norm = (float(und[1]) - cam.K[1, 2]) / cam.K[1, 1]
    X_cam = np.array([x_norm * Z_cam, y_norm * Z_cam, Z_cam])
    # World <- camera: X_world = R^T (X_cam - t)
    return cam.R.T @ (X_cam - cam.t)
