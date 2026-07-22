"""Tests for cup_task.cloud_track -- the tracked-cloud (correspondence-by-construction) pipeline.

Synthetic rig with known motion, as in test_cloud_velocity.py, but the object is a textured
CYLINDER so PyrLK has real surface texture to follow and rotation is actually observable.

    python tests/test_cloud_track.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cup_task.cloud_track import CloudTracker, surface_seed, _visible   # noqa: E402
from test_cloud_velocity import H, MM, W, make_cameras, project, rot    # noqa: E402


class _Cal:
    """Minimal stand-in for the repo's CamCalib (cloud_track uses .K/.R/.t only)."""

    def __init__(self, d):
        self.K, self.R, self.t = np.asarray(d["K"]), np.asarray(d["R"]), np.asarray(d["t"]).ravel()
        self.dist = np.zeros(5)


def cyl_points(n=200, radius=40.0, height=95.0, seed=1):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-height / 2, height / 2, n)
    return np.stack([radius * np.cos(th), radius * np.sin(th), z], 1)


def render_cyl(cal, pts, centre, seed=0):
    """Project a textured cylinder; only the NEAR half is drawn (the far half is self-occluded)."""
    import cv2
    rng = np.random.default_rng(seed)
    img = (np.full((H, W), 45) + rng.integers(0, 10, (H, W))).astype(np.uint8)
    C = -cal.R.T @ cal.t
    for i, P in enumerate(pts):
        if (P - centre) @ (C - centre) <= 0:
            continue
        Y = cal.R @ P + cal.t
        if Y[2] <= 1:
            continue
        u = cal.K @ Y
        x, y = u[0] / u[2], u[1] / u[2]
        xi, yi = int(round(x)), int(round(y))
        if not (2 <= xi < W - 2 and 2 <= yi < H - 2):
            continue
        v = 90 + (i * 53) % 150
        img[yi - 1:yi + 2, xi - 1:xi + 2] = v // 2
        img[yi, xi] = v
        img[yi - 1, xi] = v
    return img


def _rig():
    cams = {k: _Cal(v) for k, v in make_cameras().items()}
    return cams


def test_seed_lands_on_the_surface():
    """Seeds must be on a cup-sized shell around the given centre, not scattered."""
    c = np.array([100.0, -50.0, 1500.0])
    P = surface_seed(c, radius=40.0, height=95.0, n=60)
    r = np.linalg.norm(P[:, :2] - c[:2], axis=1)
    assert np.allclose(r, 40.0, atol=1e-9), "seeds not on the cylinder radius"
    assert np.all(np.abs(P[:, 2] - c[2]) <= 95 / 2 + 1e-9), "seeds outside the cylinder height"


def test_visibility_rejects_the_far_side():
    """A point on the far side of the object from the camera must be culled."""
    cams = _rig()
    cal = cams["cam_0"]
    centre = np.zeros(3)
    C = -cal.R.T @ cal.t
    near = centre + 40.0 * (C - centre) / np.linalg.norm(C - centre)
    far = centre - 40.0 * (C - centre) / np.linalg.norm(C - centre)
    assert _visible(cal, near, centre)
    assert not _visible(cal, far, centre)


def test_correspondence_is_by_construction():
    """The SAME seed index must refer to the SAME 3D point in every camera.

    This is the property the whole design rests on -- it is what replaces the cross-view NCC
    matching that measured 5% recall at true correspondences on real data.
    """
    cams = _rig()
    centre = np.zeros(3)
    pts = cyl_points(80)
    gray = {c: render_cyl(cams[c], pts, centre, seed=i) for i, c in enumerate(cams)}
    trk = CloudTracker(cams, n_seed=40, units_per_metre=MM)
    trk.seed(gray, centre)

    P = surface_seed(centre, trk.radius, trk.height, trk.n_seed)
    for cam, cal in cams.items():
        for k, px in enumerate(trk._px[cam]):
            if not np.isfinite(px).all():
                continue
            Y = cal.R @ P[k] + cal.t
            u = cal.K @ Y
            assert np.allclose(px, u[:2] / u[2], atol=1e-6), \
                f"{cam} seed {k} is not the projection of shared 3D point {k}"


def test_translation_recovered():
    """A known translation must come back through seed -> track -> lift -> Kabsch."""
    cams = _rig()
    pts = cyl_points(200)
    step = np.array([6.0, 0.0, 0.0])
    dt = 1 / 60
    trk = CloudTracker(cams, n_seed=48, units_per_metre=MM, min_inliers=6)

    got = []
    for f in range(8):
        centre = step * f
        gray = {c: render_cyl(cams[c], pts + centre, centre, seed=i) for i, c in enumerate(cams)}
        r = trk.update(gray, centre, dt)
        if r is not None and r.linear_speed is not None:
            got.append(r.linear_speed)

    assert got, "no velocity produced at all"
    expect = np.linalg.norm(step) / MM / dt
    med = float(np.median(got))
    assert abs(med - expect) < 0.30 * expect, \
        f"median linear speed {med:.3f} vs expected {expect:.3f} m/s (all: {np.round(got, 3)})"


def test_rotation_is_observable():
    """The reason a cloud exists at all: a rotating object with a STILL centre must show angular
    velocity and ~no linear velocity. A single tracked point cannot do this."""
    cams = _rig()
    pts = cyl_points(200)
    ang = 0.05
    dt = 1 / 60
    trk = CloudTracker(cams, n_seed=48, units_per_metre=MM, min_inliers=6)

    w, lin = [], []
    for f in range(8):
        R_ = rot([0, 0, 1], ang * f)
        gray = {c: render_cyl(cams[c], (R_ @ pts.T).T, np.zeros(3), seed=i)
                for i, c in enumerate(cams)}
        r = trk.update(gray, np.zeros(3), dt)
        if r is not None and r.angular_speed is not None:
            w.append(r.angular_speed); lin.append(r.linear_speed)

    assert w, "no angular velocity produced at all"
    expect = ang / dt
    med = float(np.median(w))
    assert abs(med - expect) < 0.45 * expect, \
        f"median angular {med:.2f} vs expected {expect:.2f} rad/s (all: {np.round(w, 2)})"
    # the centre is stationary, so linear speed must stay far below the rim's tangential speed
    assert float(np.median(lin)) < 0.5, f"rotation leaked into linear speed: {np.median(lin):.2f}"


def test_thin_evidence_is_refused_not_guessed():
    """Below min_inliers the tracker must report NO velocity rather than a confident wrong one.

    Measured on real data: 5-7 inliers gives 531 mm/s median error against 113 at 8-11, with
    blow-ups to 17 m/s. Refusing loses coverage honestly; answering loses accuracy silently.
    """
    cams = _rig()
    pts = cyl_points(200)
    trk = CloudTracker(cams, n_seed=48, units_per_metre=MM, min_inliers=1000)  # unreachable
    for f in range(3):
        centre = np.array([6.0, 0, 0]) * f
        gray = {c: render_cyl(cams[c], pts + centre, centre, seed=i) for i, c in enumerate(cams)}
        r = trk.update(gray, centre, 1 / 60)
        if r is not None:
            assert r.linear_speed is None, "answered despite insufficient inliers"


def test_reseed_does_not_difference_across_identity_break():
    """A reseed creates new track ids; differencing across it would compare unrelated points."""
    cams = _rig()
    pts = cyl_points(200)
    trk = CloudTracker(cams, n_seed=48, units_per_metre=MM, min_inliers=6)
    centre = np.zeros(3)
    gray = {c: render_cyl(cams[c], pts, centre, seed=i) for i, c in enumerate(cams)}
    trk.update(gray, centre, 1 / 60)
    trk.seed(gray, centre)
    assert trk._prev_cloud is None and trk._prev_ids is None, \
        "reseed left stale cloud state that a later frame would difference against"


def _main() -> int:
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1; print(f"  FAIL  {name}: {e}")
        except Exception as e:                              # noqa: BLE001
            bad += 1; print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
