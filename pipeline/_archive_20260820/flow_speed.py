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
pipeline.speed_blend. Full numbers in docs/SPEED_METRICS.md.
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

    def __init__(self, calib: dict, fps: float = FPS, min_cams: int = 2,
                 gate_consensus: bool = True, gate_flow: bool = False,
                 fuser: str = "l1"):
        self.calib = calib
        self.fps = fps
        self.min_cams = min_cams
        self.gate_consensus = gate_consensus       # GEOMETRIC gate on POSITIONS -- still ON
        # LOO gate on the FLOW VECTORS -- now OFF by default: the l1 fuser does this job with no
        # `tol` and no per-target `max_drop`. Stacking both is redundant (they fix the same thing).
        self.gate_flow = gate_flow
        self.fuser = fuser
        self._prev: dict[str, np.ndarray] = {}
        self._prev3d = None

    def update(self, gray_by_cam: dict[str, np.ndarray],
               wrist_px_by_cam: dict[str, np.ndarray],
               target_xyz=None, occluder_xyz=None) -> float:
        """One rig-frame. gray_by_cam = {cam: HxW uint8}, wrist_px_by_cam = {cam: (2,) or None}.

        Pass `target_xyz` + `occluder_xyz` (this frame's 3D points, e.g. cup and acting wrist) to
        enable the occlusion mask -- see `occluded_by` and `speed_from_cached_flow`.
        """
        avail = {}
        for cam, gray in gray_by_cam.items():
            px = wrist_px_by_cam.get(cam)
            if self._prev.get(cam) is not None and px is not None and np.isfinite(px).all():
                avail[cam] = np.asarray(px, dtype=float)
            self._prev[cam] = gray

        if target_xyz is not None and occluder_xyz is not None:
            avail = {c: p for c, p in avail.items()
                     if c in self.calib
                     and not occluded_by(target_xyz, occluder_xyz, self.calib[c])}

        use = avail
        if self.gate_consensus and len(avail) >= 2:
            from pipeline import consensus as _cons
            X, kept, _ = _cons.consensus3({c: tuple(p) for c, p in avail.items()},
                                          self.calib, prev=self._prev3d)
            if X is not None:
                self._prev3d = X
            if len(kept) >= self.min_cams:
                use = {c: avail[c] for c in kept if c in avail}

        obs_p, obs_pv = {}, {}
        for cam, px in use.items():
            fl = flow_at(self._prev[cam], gray_by_cam[cam], px)
            if fl is not None and np.isfinite(fl).all():
                obs_p[cam] = px
                obs_pv[cam] = px + fl
        if self.gate_flow and len(obs_p) >= 3:
            keep = flow_consensus_cams(obs_p, obs_pv, self.calib, self.fps)
            if len(keep) >= self.min_cams:
                obs_p = {c: obs_p[c] for c in keep}
                obs_pv = {c: obs_pv[c] for c in keep}
        return speed_from_flow_obs(obs_p, obs_pv, self.calib, self.fps, self.min_cams, self.fuser)


OCCLUDER_CONE_DEG = 10.0     # angular radius around the camera->target ray


def camera_center(cal) -> np.ndarray:
    """Camera centre in world coordinates: C = -Rᵀ t."""
    return -np.asarray(cal.R).T @ np.asarray(cal.t).ravel()


def flow_consensus_cams(obs_p: dict, obs_pv: dict, calib: dict, fps: float = FPS,
                        tol: float = 20.0, max_drop: int | None = None) -> list:
    """LEAVE-ONE-OUT consensus on the FLOW VECTORS -> the cameras to keep.

    Drop the camera whose removal most changes the fused velocity, while that change exceeds `tol`
    mm/s.

    WHY IT HELPS -- and it is NOT an occlusion detector. Measured against the geometric occlusion
    test it has only 33% recall on occluded camera-frames, and 44% of what it drops is not occluded
    at all. What it actually does is make the 3D fusion ROBUST: triangulating flow is a
    least-squares fit across cameras, which has a breakdown point of zero, so a single bad vector
    shifts the answer. This replaces it with an outlier-resistant estimator. It never needs to know
    WHY a camera is wrong (occlusion, motion blur, a texture mismatch) -- only that it disagrees in
    3D. That is why it generalises to both targets while the occlusion mask does not.

    It is genuinely robustness and not "reject big motion": dropping the largest-|flow| camera
    instead makes the cup WORSE (34.0 vs 25.3 mm/s ungated), because that discards the camera with
    the best view of the motion. A median over camera PAIRS scores the same as this (19.6 vs 19.4),
    confirming the mechanism is robust fusion rather than anything specific to leave-one-out.

    Preferred over the geometric `occluded_by` mask: no 3D tracks, no angular threshold, no
    assumption about which object is the occluder. Measured (n=12):
        CUP    MOVING err 25.3 -> 19.4 mm/s, rest p95 81.3 -> 14.8, onset 633 -> 300ms
        WRIST  MOVING err 21.7 -> 18.7 mm/s (the occlusion mask instead gives 92.3 -- see below)

    THE TWO TARGETS FAIL FOR DIFFERENT REASONS, which is why one symmetric gate is needed:

                    CUP                          WRIST
      cause         OCCLUSION (hand covers it)   MOTION BLUR (fast wrist smears the patch)
      structure     one camera, sustained        ALL cameras, transient (median run 1 frame,
                    (cam_1 drove the whole tail) 59% single-frame; per-camera 12-19%, per-trial
                                                 14-20% -- no bad camera, no bad trial)
      when          cup at REST, hand passing    drops scale with SPEED: rest 0.2%, slow 16.4%,
                                                 mid 47.7%, fast 56.0%; by phase drinking 2.5%
                                                 vs reaching 25.8% / returning 25.7%
      median works? YES (19.6 ~ LOO's 19.4)      NO (22.8 -- WORSE than no gate at all)

    The wrist needs LOO specifically: blur hits SEVERAL cameras on the same frame, so 3-of-5
    disagree on 21% of frames (13% for the cup) and a median is then itself one of the bad values.
    Iterative targeted removal survives that; a median does not. Per-frame error by how many
    cameras disagree (wrist, moving frames): 0 -> LOO 13.4 / median 12.4; 1 -> 18.4 / 21.1;
    2 -> 24.7 / 42.8; 3 -> 46.9 / 49.3. LOO's advantage is concentrated at 2 disagreeing cameras.
    ⚠ At 3+ nothing works (46.9 mm/s) -- too few good cameras remain.

    ⚠ IT IS A BET, NOT A GUARANTEE. On the frames where it drops something it improves the estimate
    only 60% of the time on the wrist (72% on the cup) -- it is worth it because the wins are much
    bigger than the losses (median |err| 46.8 -> 30.9 wrist, 51.3 -> 19.1 cup). The win rate is flat
    across every speed band (59-61%), so it is not exploiting one regime. Direction: it LOWERS the
    reported speed (wrist -13.0, cup -4.9 mm/s median) because the dropped cameras read TOO FAST
    (+51.3 mm/s on the wrist) -- exactly what blur predicts, since a smeared patch makes PyrLK
    over-shoot. (The cup's dropped cameras are biased the other way, -18.3, consistent with its
    different failure mode.)

    `max_drop` caps how many cameras may be removed. Measured MOVING-frame error by cap:
        WRIST  0:21.7  1:19.4  **2:18.1**  3:18.7   <- 3 drops is a 55% coin flip, so cap at 2
        CUP    0:25.3  1:23.7  2:20.2  **3:19.4**   <- keeps improving; one sustained bad camera
    """
    cams = [c for c in obs_p if c in obs_pv and c in calib]
    n0 = len(cams)
    while len(cams) >= 3:
        if max_drop is not None and n0 - len(cams) >= max_drop:
            break
        # NB: "dlt2" on purpose -- `tol` below was tuned against the two-triangulation fuser, so
        # this gate must keep using it or the threshold silently changes meaning.
        base = velocity_from_flow_obs({c: obs_p[c] for c in cams},
                                      {c: obs_pv[c] for c in cams}, calib, fps, 2, "dlt2")
        if not np.isfinite(base).all():
            break
        worst, wc = -1.0, None
        for c in cams:
            sub = [x for x in cams if x != c]
            v = velocity_from_flow_obs({k: obs_p[k] for k in sub},
                                       {k: obs_pv[k] for k in sub}, calib, fps, 2, "dlt2")
            if not np.isfinite(v).all():
                continue
            d = float(np.linalg.norm(v - base))
            if d > worst:
                worst, wc = d, c
        if wc is None or worst < tol:
            break
        cams = [x for x in cams if x != wc]
    return cams


def occluded_by(target_xyz, occluder_xyz, cal,
                cone_deg: float = OCCLUDER_CONE_DEG) -> bool:
    """Is `occluder_xyz` between this camera and `target_xyz`?

    ⚠ CUP ONLY -- never pass the cup as an occluder of the WRIST. The relationship is not
    symmetric: a hand occludes a resting cup, but a HELD cup does not meaningfully occlude the
    wrist. Measured, cup-as-occluder-of-wrist fires on 72.5% of camera-frames (the cup is near the
    wrist for most of the task because it is being held) and starves the triangulation: wrist
    MOVING error 21.7 -> 92.3 mm/s. Prefer `flow_consensus_cams`, which is symmetric and needs no
    3D input.

    ANGULAR test, not a pixel distance. Measure the angle at the camera centre between the ray to
    the target and the ray to the occluder, and require BOTH:
        (a) that angle is within `cone_deg`, and
        (b) the occluder is NEARER to the camera than the target.

    Angle rather than pixels because a pixel threshold does not scale with distance -- the same
    150px means a very different physical separation at 1m and at 2m, and that mis-classification is
    exactly what hid the effect at first (the contaminated frames sat at ~9deg = ~215px at 1358mm,
    beyond a 150px cutoff, so they looked "not occluded").

    Both halves are load-bearing. Measured on the cup at rest, flow magnitude by band:
        angle      occluder IN FRONT        occluder BEHIND
        0-5 deg    median 0.879 px          0.016 px
        5-10 deg   median 0.800 px          0.021 px
        10-15 deg  median 0.010 px          0.018 px
    Contamination is ~45x the clean floor inside 10deg with the occluder in front, and vanishes
    outside it; BEHIND is clean at every angle. Hence cone_deg=10 and the depth check.
    """
    t = np.asarray(target_xyz, float)
    o = np.asarray(occluder_xyz, float)
    if not (np.isfinite(t).all() and np.isfinite(o).all()):
        return False
    C = camera_center(cal)
    vt, vo = t - C, o - C
    nt, no = np.linalg.norm(vt), np.linalg.norm(vo)
    if nt < 1e-6 or no < 1e-6:
        return False
    if no >= nt:                                    # occluder is behind the target
        return False
    ang = np.degrees(np.arccos(np.clip(vt @ vo / (nt * no), -1.0, 1.0)))
    return bool(ang < cone_deg)


def projection_jacobian(cal, X) -> np.ndarray | None:
    """d(pixel)/d(world position) at X for this camera -- the 2x3 projection Jacobian.

    This is what makes a camera's contribution HONEST: a camera looking along the direction of
    motion sees almost no pixel displacement, and J encodes exactly that, so the fit stops treating
    its (tiny, noise-dominated) reading as equally informative. Measured: this is the single largest
    driver of per-camera flow error -- a camera looking ALONG the motion has ~2x the relative error
    of one looking ACROSS it.
    """
    R_ = np.asarray(cal.R); t = np.asarray(cal.t).ravel(); K = np.asarray(cal.K)
    Xc = R_ @ np.asarray(X) + t
    x, y, z = Xc
    if z <= 1:
        return None
    fx, fy = K[0, 0], K[1, 1]
    return np.array([[fx / z, 0, -fx * x / z ** 2],
                     [0, fy / z, -fy * y / z ** 2]]) @ R_


def solve_velocity(rows, mode: str = "l1", fps: float = FPS,
                   huber_k: float = 1.5, iters: int = 8) -> np.ndarray | None:
    """rows = [(J_2x3, flow_2), ...] -> 3D velocity (mm/s), solving u_dot = J v.

    `mode` selects ONLY the loss, never the model:
        plain    least squares. Breakdown point 0 -- one bad camera moves the answer arbitrarily.
        l1       DEFAULT. IRLS with w = 1/|r|, so minimising sum(w*|r|^2) == minimising sum|r|:
                 a geometric-median-like solution. A camera's INFLUENCE is capped by its own
                 error (twice as wrong => half the weight), so nothing is ever hard-dropped and
                 there is NO threshold anywhere. The only constant is a 1e-3 floor against
                 division by zero.
        huber    quadratic near zero, linear in the tail. Needs a tuning constant (huber_k).
        trimmed  drop the single worst-residual camera. No threshold, but a hard decision.
    """
    if len(rows) < 2:
        return None
    A = np.vstack([r[0] for r in rows])
    b = np.concatenate([r[1] for r in rows])
    if mode == "plain":
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        return v * fps
    if mode == "trimmed" and len(rows) >= 3:
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (A @ v - b).reshape(-1, 2)
        keep = [i for i in range(len(rows)) if i != int(np.argmax(np.linalg.norm(res, axis=1)))]
        A = np.vstack([rows[i][0] for i in keep])
        b = np.concatenate([rows[i][1] for i in keep])
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        return v * fps
    v, *_ = np.linalg.lstsq(A, b, rcond=None)
    for _ in range(iters):
        r = (A @ v - b).reshape(-1, 2)
        rn = np.linalg.norm(r, axis=1)
        if mode == "huber":
            s = 1.4826 * np.median(rn) + 1e-6
            w = np.where(rn <= huber_k * s, 1.0, huber_k * s / np.maximum(rn, 1e-9))
        else:                                    # l1
            w = 1.0 / np.maximum(rn, 1e-3)
        W = np.repeat(w, 2)[:, None]
        v, *_ = np.linalg.lstsq(A * W, b * W[:, 0], rcond=None)
    return v * fps


def velocity_from_flow_obs(obs_p: dict, obs_pv: dict, calib: dict,
                           fps: float = FPS, min_cams: int = 2,
                           fuser: str = "l1") -> np.ndarray:
    """Per-camera flow vectors -> ONE 3D VELOCITY VECTOR (mm/s), robustly fused.

    The vector, not just its magnitude. Direction is free here and it is what lets a caller ask "is
    the target moving TOWARD or AWAY from somewhere" without ever differentiating a position track.
    That matters because differentiating position is precisely what injects the noise this whole
    module exists to avoid (a ~1mm per-frame positional wobble becomes ~60mm/s of phantom speed).

    HOW: solve u_dot = J(X) v across cameras (see projection_jacobian), with an L1/IRLS loss (see
    solve_velocity). Both parts are general geometric/statistical mechanisms with NO tuned
    constants. `fuser="plain"` recovers ordinary least squares.

    WHY NOT the older two-triangulation difference (triangulate {p} and {p+flow}, subtract): it
    weights every camera equally regardless of whether that camera can SEE the motion, and it is a
    least-squares fit with breakdown point 0. Measured 20.53 vs 21.71 mm/s for the Jacobian form.
    Pass fuser="dlt2" if you need the old behaviour for a back-to-back comparison.

    ⚠ The L1 fuser replaced the `flow_consensus_cams` LOO gate as the default robustifier
    (2026-07-22). They are NOT DISTINGUISHABLE on this cohort -- paired bootstrap over 12 trials,
    l1 - loo(cap2): wrist -4.03 mm/s CI [-14.32, +3.25], cup +1.46 CI [-4.27, +9.17] -- so the
    tie-break is mechanistic: l1 has no `tol` and no per-target `max_drop`, while the LOO gate has
    both and its own docstring put the optimal cap at 2 for the wrist but 3 for the cup.
    """
    from pipeline.kalman_3d import triangulate_dlt
    nan3 = np.full(3, np.nan)
    cams = [c for c in obs_p if c in obs_pv and c in calib]
    if len(cams) < min_cams:
        return nan3
    if fuser == "dlt2":
        Xp = triangulate_dlt([calib[c] for c in cams], [np.asarray(obs_p[c]) for c in cams])
        Xv = triangulate_dlt([calib[c] for c in cams], [np.asarray(obs_pv[c]) for c in cams])
        if Xp is None or Xv is None:
            return nan3
        return (np.asarray(Xv, float) - np.asarray(Xp, float)) * fps

    X = triangulate_dlt([calib[c] for c in cams], [np.asarray(obs_p[c]) for c in cams])
    if X is None:
        return nan3
    rows = []
    for c in cams:
        J = projection_jacobian(calib[c], X)
        if J is not None:
            rows.append((J, np.asarray(obs_pv[c], float) - np.asarray(obs_p[c], float)))
    if len(rows) < min_cams:
        return nan3
    v = solve_velocity(rows, fuser, fps)
    return nan3 if v is None else np.asarray(v, float)


def speed_from_flow_obs(obs_p: dict, obs_pv: dict, calib: dict,
                        fps: float = FPS, min_cams: int = 2, fuser: str = "l1") -> float:
    """Scalar 3D speed (mm/s), NaN if too few cameras. Magnitude of velocity_from_flow_obs, so
    the online and offline paths compute the identical number from one implementation."""
    v = velocity_from_flow_obs(obs_p, obs_pv, calib, fps, min_cams, fuser)
    return float(np.linalg.norm(v)) if np.isfinite(v).all() else float("nan")


def velocity_from_cached_flow(px: dict[str, np.ndarray], flow: dict[str, np.ndarray],
                              calib: dict, n: int, fps: float = FPS,
                              min_cams: int = 2, fuser: str = "l1") -> np.ndarray:
    """OFFLINE path -> (n,3) 3D VELOCITY track (mm/s), NaN rows where too few cameras.

    Same geometry as speed_from_cached_flow, but keeps the direction. Use this when a caller needs
    to know WHERE the target is heading -- e.g. the segmenter asking whether the cup is travelling
    away from or back toward its rest position, which scalar speed cannot answer.
    """
    out = np.full((n, 3), np.nan)
    cams = [c for c in flow if c in px and c in calib]
    for f in range(n):
        obs_p, obs_pv = {}, {}
        for c in cams:
            if (f < len(px[c]) and f < len(flow[c])
                    and np.isfinite(px[c][f]).all() and np.isfinite(flow[c][f]).all()):
                obs_p[c] = px[c][f]
                obs_pv[c] = px[c][f] + flow[c][f]
        out[f] = velocity_from_flow_obs(obs_p, obs_pv, calib, fps, min_cams, fuser)
    return out


def radial_velocity(vel_xyz: np.ndarray, pos_xyz: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Component of a 3D velocity along the outward radial direction from `origin` (mm/s).

    Positive = moving AWAY from origin, negative = moving BACK toward it. Feed it the flow
    velocity and you get a signed "is it leaving or returning" signal with NO differentiation of
    position anywhere in the chain -- unlike d(displacement)/dt, which inherits every bit of the
    position track's frame-to-frame noise.
    """
    v = np.asarray(vel_xyz, float)
    r = np.asarray(pos_xyz, float) - np.asarray(origin, float)
    nrm = np.linalg.norm(r, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = r / nrm[:, None]
    out = np.einsum("ij,ij->i", v, u)
    out[~np.isfinite(nrm) | (nrm < 1e-6)] = np.nan
    return out


def speed_from_cached_flow(wrist_px: dict[str, np.ndarray], flow: dict[str, np.ndarray],
                           calib: dict, n: int, fps: float = FPS,
                           min_cams: int = 2, gate_consensus: bool = True,
                           gate_flow: bool = False, fuser: str = "l1",
                           target_xyz=None, occluder_xyz=None,
                           cone_deg: float = OCCLUDER_CONE_DEG) -> np.ndarray:
    """OFFLINE path: per-camera 2D points + precomputed per-camera flow -> (n,) 3D speed track.

    Calls the SAME speed_from_flow_obs as the online class, so the offline replay of a live session
    reproduces the live numbers exactly (the number and its check must share one implementation).

    `gate_consensus` (default ON): use only the cameras the geometric consensus KEEPS on that frame.
    Without it, a camera tracking the wrong object still contributes its flow vector with full
    weight -- the consensus exists precisely to reject those, and flow was ignoring that rejection.
    Measured, cup speed on MOVING frames: 24.7 -> 19.7 mm/s (-20%), and it costs almost nothing
    (mean cameras 5.00 -> 4.80). The WRIST barely moves (8.3 -> 8.3; blend peak 20.9 -> 19.8, kept
    4.99/5.00) -- the known asymmetry: a cup false positive is a DIFFERENT OBJECT and reprojects
    far, so the gate catches it, while a pose error is the right person's slightly-wrong joint and
    reprojects plausibly.

    `target_xyz` + `occluder_xyz` (both (n,3), optional) enable the OCCLUSION mask: a camera is
    dropped on frames where the occluder passes between it and the target (see `occluded_by`).
    For the cup, pass the cup track and the acting wrist -- when the hand covers the cup, PyrLK
    tracks the HAND'S texture and reports the hand's motion as the cup's. Measured on the cup:
        rest p95   81.3 -> 11.9 mm/s      (7x; now UNDER the 15mm/s onset gate)
        MOVING err 25.3 -> 22.0 mm/s      (also improves)
    That single mask is what makes flow usable as a segmenter gate at all: first crossing of
    FWD_ON vs OMC goes 633ms -> 250ms, beating SmoothNet's 400ms.

    `gate_flow` (default OFF since 2026-07-22): the LOO consensus on the flow vectors. Superseded
    by the `l1` fuser, which robustifies the same fit with no `tol` and no per-target `max_drop`.
    They are not distinguishable numerically (paired bootstrap over 12 trials, l1 - loo(cap2):
    wrist -4.03 mm/s CI [-14.32,+3.25], cup +1.46 CI [-4.27,+9.17]), so the tie-break is the
    absence of tuned constants. Stacking both is redundant -- they fix the same failure.

    NOTE the points are the RAW DETECTED pixels (YOLO keypoint / UETrack point), never a
    reprojection of the 3D consensus. Flow must be measured where the image evidence is; a
    reprojected point would inherit the triangulation's own error and defeat the purpose of
    measuring velocity independently of position.
    """
    out = np.full(n, np.nan)
    cams = [c for c in flow if c in wrist_px and c in calib]
    tgt = None if target_xyz is None else np.asarray(target_xyz, float)
    occ = None if occluder_xyz is None else np.asarray(occluder_xyz, float)
    prev3d = None
    for f in range(n):
        avail = {c: wrist_px[c][f] for c in cams
                 if f < len(wrist_px[c]) and np.isfinite(wrist_px[c][f]).all()}
        if tgt is not None and occ is not None and f < len(tgt) and f < len(occ):
            avail = {c: p for c, p in avail.items()
                     if not occluded_by(tgt[f], occ[f], calib[c], cone_deg)}
        use = avail
        if gate_consensus and len(avail) >= 2:
            from pipeline import consensus as _cons
            X, kept, _ = _cons.consensus3({c: tuple(p) for c, p in avail.items()},
                                          calib, prev=prev3d)
            if X is not None:
                prev3d = X
            if len(kept) >= min_cams:
                use = {c: avail[c] for c in kept if c in avail}
        obs_p, obs_pv = {}, {}
        for c, p in use.items():
            if f < len(flow[c]) and np.isfinite(flow[c][f]).all():
                obs_p[c] = p
                obs_pv[c] = p + flow[c][f]
        if gate_flow and len(obs_p) >= 3:
            keep = flow_consensus_cams(obs_p, obs_pv, calib, fps)
            if len(keep) >= min_cams:
                obs_p = {c: obs_p[c] for c in keep}
                obs_pv = {c: obs_pv[c] for c in keep}
        out[f] = speed_from_flow_obs(obs_p, obs_pv, calib, fps, min_cams, fuser)
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
