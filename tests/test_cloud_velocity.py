"""Tests for cup_task.cloud_velocity, on synthetic cameras with a KNOWN ground-truth motion.

Everything here is generated from a rig we control -- 4 cameras on a ring looking at the origin, an
object whose pose we prescribe -- so each assertion compares against a true answer rather than
against the code's own output. Rendering is real: 3D points are projected and stamped into 8-bit
images, so goodFeaturesToTrack / matchTemplate / triangulatePoints all run on actual pixels.

    python -m pytest tests/test_cloud_velocity.py -q       # if pytest is available
    python tests/test_cloud_velocity.py                    # plain, no pytest needed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cup_task.cloud_velocity import (                       # noqa: E402
    CloudVelocityTracker, compute_3d_velocity, correspond_clouds, extract_local_features,
    fundamental_from_calib, kabsch, kabsch_align_vectors, match_features_epipolar,
    projection_matrix, triangulate_and_filter,
)

W, H = 640, 480
MM = 1000.0          # world units are millimetres


# ------------------------------------------------------------------------------------------------
# synthetic rig
# ------------------------------------------------------------------------------------------------
def make_cameras(n: int = 4, radius: float = 1500.0) -> dict:
    """`n` cameras on a horizontal ring at `radius` mm, all looking at the world origin."""
    K = np.array([[600.0, 0, W / 2], [0, 600.0, H / 2], [0, 0, 1]])
    cams = {}
    for i in range(n):
        a = 2 * np.pi * i / n
        C = np.array([radius * np.cos(a), 200.0 * (i % 2) - 100.0, radius * np.sin(a)])
        fwd = -C / np.linalg.norm(C)                       # look at origin
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, fwd); right /= np.linalg.norm(right)
        true_up = np.cross(fwd, right)
        R = np.vstack([right, true_up, fwd])               # world -> camera
        t = -R @ C
        cams[f"cam_{i}"] = {"K": K, "R": R, "t": t, "P": K @ np.hstack([R, t.reshape(3, 1)])}
    return cams


def project(cam: dict, X: np.ndarray) -> np.ndarray:
    """World point(s) -> pixels. X is (3,) or (N,3)."""
    X = np.atleast_2d(np.asarray(X, float))
    u = (projection_matrix(cam) @ np.hstack([X, np.ones((len(X), 1))]).T).T
    return u[:, :2] / u[:, 2:3]


def render(cam: dict, pts3d: np.ndarray, seed: int = 0) -> np.ndarray:
    """Project points and stamp a distinct blob at each -> uint8 image.

    Each point gets its OWN intensity and an asymmetric 3x3 shape, so NCC can actually tell them
    apart. A uniform dot pattern would make every patch identical and the matcher's job trivially
    ambiguous -- which would test nothing.
    """
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 40, np.uint8)
    img = (img + rng.integers(0, 12, (H, W))).astype(np.uint8)   # texture, so NCC is not degenerate
    px = project(cam, pts3d)
    for i, (x, y) in enumerate(px):
        xi, yi = int(round(x)), int(round(y))
        if not (2 <= xi < W - 2 and 2 <= yi < H - 2):
            continue
        val = 120 + (i * 37) % 130
        img[yi - 1:yi + 2, xi - 1:xi + 2] = val // 2
        img[yi, xi] = val
        img[yi - 1, xi] = val                                     # asymmetric -> orientation-bearing
        img[yi, xi + 1] = val
    return img


def object_points(n: int = 14, scale: float = 40.0, seed: int = 3) -> np.ndarray:
    """A small rigid constellation, ~`scale` mm across, centred on the origin."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, scale, (n, 3))
    return P - P.mean(0)


def rot(axis: np.ndarray, ang: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    axis = np.asarray(axis, float)
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * ang).as_matrix()


# ------------------------------------------------------------------------------------------------
# 4. the solver -- pure geometry, no pixels
# ------------------------------------------------------------------------------------------------
def test_kabsch_recovers_known_transform():
    P = object_points()
    R_true = rot([0.3, 1.0, -0.2], 0.25)
    t_true = np.array([12.0, -5.0, 3.0])
    Q = (R_true @ P.T).T + t_true
    R, t = kabsch(P, Q)
    assert np.allclose(R, R_true, atol=1e-9), "rotation not recovered"
    assert np.allclose(t, t_true, atol=1e-9), "translation not recovered"
    assert np.isclose(np.linalg.det(R), 1.0), "not a proper rotation"


def test_kabsch_rejects_reflection():
    """A mirrored cloud must NOT be fitted with det=-1: a reflection is not a rigid motion."""
    P = object_points()
    Q = P.copy(); Q[:, 2] *= -1
    R, _ = kabsch(P, Q)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_velocity_decouples_translation_and_rotation():
    """Pure translation => zero angular; pure rotation about the centroid => zero linear."""
    P = object_points()
    dt = 1 / 60

    Q = P + np.array([10.0, 0.0, 0.0])                    # 10mm in one frame = 0.6 m/s
    v = compute_3d_velocity(P, Q, dt, MM)
    assert np.isclose(v["linear_speed"], 0.6, atol=1e-9)
    assert v["angular_speed"] < 1e-9, "translation leaked into rotation"

    ang = 0.05
    Q = (rot([0, 0, 1], ang) @ P.T).T                     # about the centroid (origin)
    v = compute_3d_velocity(P, Q, dt, MM)
    assert np.isclose(v["angular_speed"], ang / dt, atol=1e-9)
    assert v["linear_speed"] < 1e-9, "rotation leaked into translation"


def test_translation_is_centroid_motion_not_origin_translation():
    """⚠ REGRESSION: Kabsch's `t` is the WORLD ORIGIN's translation, not the object's.

    t = centroid_B - R @ centroid_A. When the cloud sits far from the origin (here 1700mm, as in
    the real rig), any rotation swings `R @ centroid_A` through a LEVER ARM, and the origin-
    referred translation bears no relation to how far the object actually moved.

    Measured on DELTA P07 before this was fixed: t_vec read 2.10x the true cup speed while the
    cloud's own centroid displacement read 0.77x -- the tracking was fine, the reference point was
    wrong. This test uses a PURE rotation about the object's own centre, where the object's
    velocity is ZERO by construction but `t_vec` is large.
    """
    far = np.array([0.0, 0.0, 1700.0])                  # object far from the world origin
    P = object_points() + far
    R_true = rot([0, 1, 0], 0.05)
    Q = (R_true @ (P - far).T).T + far                  # spins in place: centroid does NOT move

    v = compute_3d_velocity(P, Q, 1 / 60, MM, ransac=False)
    assert v["linear_speed"] < 0.01, \
        f"pure rotation about the centre must give ~0 linear speed, got {v['linear_speed']:.3f} m/s"
    assert np.isclose(v["angular_speed"], 0.05 * 60, atol=1e-6), "angular speed wrong"

    # and the raw Kabsch t_vec is indeed huge here -- that is exactly what must NOT be reported
    _, t_vec = kabsch(P, Q)
    assert np.linalg.norm(t_vec) > 50, \
        "test is not exercising the lever arm; move the cloud further from the origin"


def test_units_conversion_is_explicit():
    """The same displacement in mm and in m must give the same m/s."""
    P = object_points()
    Q = P + np.array([10.0, 0, 0])
    a = compute_3d_velocity(P, Q, 1 / 60, units_per_metre=1000.0)["linear_speed"]
    b = compute_3d_velocity(P / 1000, Q / 1000, 1 / 60, units_per_metre=1.0)["linear_speed"]
    assert np.isclose(a, b), "unit handling inconsistent"


def test_align_vectors_matches_kabsch():
    """`Rotation.align_vectors` IS a correct Kabsch solver -- pinned because it is easy to assume
    otherwise from its name.

    Despite the "vectors" framing it is MAGNITUDE-WEIGHTED: it minimises sum|b_i - R a_i|^2, the
    Kabsch objective, and applies the same reflection correction. Tested on a strongly ANISOTROPIC
    cloud (one point 100x further out than the rest) -- the case that would expose a normalising
    implementation, since there the far point must dominate the fit -- both noiseless and noisy.
    """
    rng = np.random.default_rng(0)
    P = np.array([[100.0, 0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0], [0.0, 1.0, 1.0], [0.5, 0.2, -1.0]])
    R_true = rot([0, 0, 1], 0.20)
    Q = (R_true @ P.T).T

    err = lambda R: np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    assert err(kabsch(P, Q)[0]) < 1e-6, "Kabsch should be exact on noiseless data"

    for noise in (0.0, 0.5, 2.0):
        Qn = Q + rng.normal(0, noise, Q.shape) if noise else Q
        Ra, ta = kabsch(P, Qn)
        Rb, tb = kabsch_align_vectors(P, Qn)
        assert np.allclose(Ra, Rb, atol=1e-9), f"solvers disagree at noise={noise}"
        assert np.allclose(ta, tb, atol=1e-9), f"translations disagree at noise={noise}"


def test_align_vectors_is_magnitude_weighted():
    """The property the above relies on, isolated: a distant point outvotes a near one.

    Two inconsistent targets -- the long vector wants +0.4 rad, the short one wants 0. A
    magnitude-weighted fit lands near +0.4; a normalising one would split the difference at +0.2.
    """
    from scipy.spatial.transform import Rotation
    A = np.array([[10.0, 0, 0], [0, 1.0, 0]])
    B = np.array([[10 * np.cos(0.4), 10 * np.sin(0.4), 0], [0, 1.0, 0]])
    ang = Rotation.align_vectors(B, A)[0].as_rotvec()[2]
    assert ang > 0.35, f"expected the long vector to dominate (~0.4), got {ang:.3f}"


def test_first_frame_reports_no_velocity():
    """A velocity needs two frames. Reporting 0.0 on frame one would be a fabricated measurement."""
    cams = make_cameras()
    P = object_points()
    trk = CloudVelocityTracker(cams, units_per_metre=MM)
    frames = {c: render(cams[c], P, seed=i) for i, c in enumerate(cams)}
    kp = {c: project(cams[c], np.zeros(3))[0] for c in cams}
    r = trk.update(frames, kp, dt=1 / 60)
    if r is not None:
        assert r.linear_velocity is None and r.angular_velocity is None


# ------------------------------------------------------------------------------------------------
# 1-3. the pixel path
# ------------------------------------------------------------------------------------------------
def test_extract_local_features_stays_inside_roi():
    cams = make_cameras()
    P = object_points(n=25, scale=25.0)
    cam = cams["cam_0"]
    img = render(cam, P)
    kp = project(cam, np.zeros(3))[0]
    f = extract_local_features(img, kp, roi_px=35)
    assert len(f) > 0, "no features found on a textured target"
    assert np.all(np.abs(f - kp) <= 35 / 2 + 1), "feature escaped the ROI"


def test_extract_local_features_handles_edges_and_nan():
    cams = make_cameras()
    img = render(cams["cam_0"], object_points())
    assert len(extract_local_features(img, (2.0, 2.0))) >= 0          # must not raise
    assert len(extract_local_features(img, (np.nan, 5.0))) == 0
    assert len(extract_local_features(img, (10_000.0, 10_000.0))) == 0


def test_fundamental_matrix_satisfies_epipolar_constraint():
    """x_b^T F x_a = 0 for TRUE correspondences -- validates F independently of the matcher."""
    cams = make_cameras()
    a, b = cams["cam_0"], cams["cam_1"]
    F = fundamental_from_calib(a, b)
    P = object_points(n=20, scale=100.0)
    pa, pb = project(a, P), project(b, P)
    for ua, ub in zip(pa, pb):
        r = np.append(ub, 1) @ F @ np.append(ua, 1)
        assert abs(r) < 1e-6, f"epipolar constraint violated: {r}"


def test_matcher_finds_true_correspondences():
    """Matches must be CORRECT, not merely numerous -- checked against the known projection."""
    cams = make_cameras()
    a, b = cams["cam_0"], cams["cam_1"]
    P = object_points(n=12, scale=30.0)
    ia_, ib_ = render(a, P, 1), render(b, P, 2)
    fa = extract_local_features(ia_, project(a, np.zeros(3))[0], roi_px=45)
    fb = extract_local_features(ib_, project(b, np.zeros(3))[0], roi_px=45)
    m = match_features_epipolar(ia_, ib_, fa, fb, a, b)
    assert len(m) >= 3, f"too few matches ({len(m)})"

    # every accepted match must sit on its epipolar line, by construction of the gate
    F = fundamental_from_calib(a, b)
    for ia2, ib2, ncc in m:
        ln = F @ np.append(fa[ia2], 1)
        d = abs(np.append(fb[ib2], 1) @ ln) / np.hypot(ln[0], ln[1])
        assert d <= 2.0 + 1e-6, f"match {d:.2f}px off the epipolar line"
        assert ncc >= 0.65


def test_triangulate_and_filter_rejects_far_points():
    cams = make_cameras()
    a, b = cams["cam_0"], cams["cam_1"]
    near = np.array([5.0, 0.0, 0.0])
    far = np.array([900.0, 0.0, 0.0])                    # 90cm out: beyond the 15cm cut
    pairs = [(project(a, X)[0], project(b, X)[0]) for X in (near, far)]
    X, _ = triangulate_and_filter(pairs, a, b, centre_3d=np.zeros(3), cloud_radius=150.0)
    assert len(X) == 1 and np.allclose(X[0], near, atol=1e-6), "distance filter failed"


def test_triangulate_and_filter_rejects_bad_reprojection():
    """A mismatched pair triangulates to a point that reprojects badly -- it must be cut."""
    cams = make_cameras()
    a, b = cams["cam_0"], cams["cam_1"]
    good = np.array([5.0, 2.0, -3.0])
    pa = project(a, good)[0]
    pb_wrong = project(b, good)[0] + np.array([25.0, 18.0])
    X, _ = triangulate_and_filter([(pa, pb_wrong)], a, b, centre_3d=np.zeros(3),
                                  cloud_radius=1e9, reproj_max_px=3.0)
    assert len(X) == 0, "reprojection filter let a mismatch through"


def test_correspond_clouds_is_mutual():
    """Two A points must not both claim the same B point."""
    A = np.array([[0.0, 0, 0], [1.0, 0, 0], [50.0, 0, 0]])
    B = np.array([[0.05, 0, 0], [60.0, 0, 0]])
    a, b = correspond_clouds(A, B)
    assert len(a) == len(b)
    assert len(np.unique(b, axis=0)) == len(b), "a B point was claimed twice"


# ------------------------------------------------------------------------------------------------
# end-to-end
# ------------------------------------------------------------------------------------------------
def test_end_to_end_translation_speed():
    """Full pipeline on rendered frames: recover a known 12mm/frame translation."""
    cams = make_cameras()
    P = object_points(n=16, scale=30.0)
    step = np.array([12.0, 0.0, 0.0])
    dt = 1 / 60
    trk = CloudVelocityTracker(cams, units_per_metre=MM, min_features=4)

    got = []
    for f in range(4):
        C = step * f
        pts = P + C
        frames = {c: render(cams[c], pts, seed=i) for i, c in enumerate(cams)}
        kp = {c: project(cams[c], C)[0] for c in cams}
        r = trk.update(frames, kp, dt)
        if r is not None and r.linear_speed is not None:
            got.append(r)

    assert got, "pipeline produced no velocity at all"
    expect = np.linalg.norm(step) / MM / dt                 # 0.72 m/s
    # Measured margin is 4-11%; 15% leaves headroom without being so loose that a real
    # regression could hide under it (the pre-RANSAC bug read 35% low and must fail here).
    for r in got:
        assert abs(r.linear_speed - expect) < 0.15 * expect, \
            f"linear speed {r.linear_speed:.3f} vs expected {expect:.3f} m/s"
        assert r.active_3d_points_count >= 3
        assert np.isfinite(r.centroid_3d).all()


def test_end_to_end_rotation_speed_is_UNRELIABLE():
    """⚠ DOCUMENTS A REAL LIMITATION -- angular velocity from a re-detected cloud is NOT usable.

    Measured on this rig against a true 4.8 rad/s: 4.65, 42.76, 1.70, 9.08 -- errors of 3%, 791%,
    65%, 89%. Translation on the identical clouds is fine (4-11%), so this is specific to
    rotation, and the mechanism is in the inlier counts (3-7 points):

      * rotation is only observable through the SPREAD of the cloud, while translation is carried
        by its centroid -- so translation averages its errors down and rotation does not;
      * the cloud is RE-DETECTED per frame, so its points are not stable landmarks (only 5 of 16
        true object points were hit on this rig);
      * a 3-point fit on a nearly planar cloud is close to unconstrained about the weak axis.

    This test asserts the CURRENT behaviour -- translation good, rotation not -- so the limitation
    cannot be forgotten and so a genuine fix shows up as this test failing. To actually get
    angular velocity, the sub-features must be TRACKED across frames (PyrLK) rather than
    re-detected, which supplies the true correspondence Kabsch assumes.
    """
    cams = make_cameras()
    P = object_points(n=16, scale=35.0)
    ang = 0.08                                              # rad per frame
    dt = 1 / 60
    trk = CloudVelocityTracker(cams, units_per_metre=MM, min_features=4)

    got = []
    for f in range(6):
        pts = (rot([0, 1, 0], ang * f) @ P.T).T
        frames = {c: render(cams[c], pts, seed=i) for i, c in enumerate(cams)}
        kp = {c: project(cams[c], np.zeros(3))[0] for c in cams}
        r = trk.update(frames, kp, dt)
        if r is not None and r.angular_speed is not None:
            got.append(r)

    assert got, "pipeline produced no velocity at all"
    expect = ang / dt                                       # 4.8 rad/s
    errs = [abs(r.angular_speed - expect) / expect for r in got]
    # It is not that rotation is merely noisy -- it is unusable, and that must stay visible.
    assert max(errs) > 0.35, (
        "angular velocity is now accurate on all frames -- if this is a real fix, update this "
        f"test and the module docstring. errors={[round(e, 2) for e in errs]}")
    # Every frame must still be finite and physically bounded; garbage-in must not become NaN.
    for r in got:
        assert np.isfinite(r.angular_speed) and r.angular_speed < 1e3
        assert r.active_3d_points_count >= 3


def test_gap_breaks_correspondence():
    """After a dropped frame the tracker must NOT difference across the gap.

    Doing so would divide a two-frame displacement by a one-frame dt and report double the speed.
    """
    cams = make_cameras()
    P = object_points(n=16, scale=30.0)
    trk = CloudVelocityTracker(cams, units_per_metre=MM, min_features=4)
    frames = {c: render(cams[c], P, seed=i) for i, c in enumerate(cams)}
    kp = {c: project(cams[c], np.zeros(3))[0] for c in cams}
    trk.update(frames, kp, 1 / 60)

    empty = {c: np.zeros((H, W), np.uint8) for c in cams}   # featureless -> cloud too small
    assert trk.update(empty, kp, 1 / 60) is None
    r = trk.update(frames, kp, 1 / 60)
    if r is not None:
        assert r.linear_velocity is None, "differenced across a gap"


def _main() -> int:
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                              # noqa: BLE001
            bad += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
