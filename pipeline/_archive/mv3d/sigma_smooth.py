"""Sigma-aware temporal smoothing of a triangulated 3D trajectory.

A plain low-pass (Butterworth) is BLIND: it smooths every frame equally, so it rounds off REAL fast
peaks as hard as it kills jitter (why it costs +11% at the wrist peak, see project_smoothnet_pose).
YOLO's per-frame sigma tells us WHICH frames are actually noisy. A constant-velocity Kalman + RTS
smoother uses that natively: measurement noise R_t scales with the triangulation's per-frame
uncertainty, so the filter TRUSTS sharp frames (keeps the real motion) and LEANS ON the constant-
velocity prior only where the measurement is blurry. That is the principled version of "smooth where
uncertain, follow where confident" — impossible for a fixed-cutoff low-pass.

We derive the per-frame 3D measurement covariance by propagating the per-view, per-axis pixel sigma
through the DLT linearization (first-order). This keeps the smoothing tied to the SAME sigma we use
for the weighted-DLT fusion — one uncertainty, used end to end.
"""
import numpy as np


def triangulation_cov(uv, sig_px, P):
    """First-order 3D covariance of a DLT point from per-view per-axis pixel sigma.

    Linearize reprojection r_i = proj_i(X) - uv_i around the solution; J = d(proj)/dX (2V x 3),
    pixel noise cov = diag(sig_px^2). Then Cov(X) ~= (J^T W J)^-1 with W = diag(1/sig_px^2).
    uv (V,2) px, sig_px (V,2) px (per-axis), P (V,3,4). Returns X (3,), Cov (3,3).
    """
    from pipeline.mv3d.dlt import weighted_dlt_axis
    import torch
    w = 1.0 / np.clip(sig_px, 1e-3, None) ** 2                      # (V,2) inverse-var weights
    X = weighted_dlt_axis(torch.tensor(uv, dtype=torch.float32),
                          torch.tensor(w, dtype=torch.float32),
                          torch.tensor(P, dtype=torch.float32)).numpy()
    # Jacobian of projection at X
    Xh = np.r_[X, 1.0]
    JtWJ = np.zeros((3, 3))
    for i, Pi in enumerate(P):
        d = Pi @ Xh                                                 # (3,) [su, sv, s]
        s = d[2]
        if abs(s) < 1e-6:
            continue
        # d(u,v)/dX where u = (P0.Xh)/(P2.Xh)
        Ju = (Pi[0, :3] * s - d[0] * Pi[2, :3]) / s ** 2            # (3,)
        Jv = (Pi[1, :3] * s - d[1] * Pi[2, :3]) / s ** 2
        JtWJ += np.outer(Ju, Ju) * w[i, 0] + np.outer(Jv, Jv) * w[i, 1]
    Cov = np.linalg.pinv(JtWJ + 1e-9 * np.eye(3))
    return X, Cov


def rts_smooth_cv(meas, meas_cov, fps=60.0, q=1e3):
    """Constant-velocity Kalman + RTS smoother on a 3D trajectory with per-frame measurement cov.

    State = [pos(3), vel(3)]. Per-frame R_t = meas_cov[t] (from triangulation_cov) -> the filter
    trusts low-sigma frames and coasts on the CV prior through high-sigma ones. Process noise q sets
    how much genuine acceleration we allow (higher q = follows fast real motion more, smooths less).
    meas (T,3), meas_cov (T,3,3). Returns smoothed pos (T,3).
    """
    T = len(meas); dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    # CV process noise (white-acceleration model)
    Qb = np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]) * q
    Q = np.zeros((6, 6))
    for a in range(3):
        Q[np.ix_([a, a+3], [a, a+3])] = Qb
    xf = np.zeros((T, 6)); Pf = np.zeros((T, 6, 6))
    xp = np.zeros((T, 6)); Pp = np.zeros((T, 6, 6))
    x = np.r_[meas[0], np.zeros(3)]; Pcur = np.eye(6) * 1e4
    for t in range(T):
        if t > 0:
            x = F @ x; Pcur = F @ Pcur @ F.T + Q
        xp[t] = x; Pp[t] = Pcur
        R = meas_cov[t]
        S = H @ Pcur @ H.T + R
        K = Pcur @ H.T @ np.linalg.pinv(S)
        x = x + K @ (meas[t] - H @ x)
        Pcur = (np.eye(6) - K @ H) @ Pcur
        xf[t] = x; Pf[t] = Pcur
    # RTS backward pass
    xs = xf.copy(); Ps = Pf.copy()
    for t in range(T - 2, -1, -1):
        C = Pf[t] @ F.T @ np.linalg.pinv(Pp[t+1])
        xs[t] = xf[t] + C @ (xs[t+1] - xp[t+1])
        Ps[t] = Pf[t] + C @ (Ps[t+1] - Pp[t+1]) @ C.T
    return xs[:, :3]
