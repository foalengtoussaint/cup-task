"""Fuse per-camera 2D detections of a rep into 3D world points (mm).

For each frame, gather every camera that saw the target (cup center, or a body
keypoint), then triangulate a single 3D point. Detections are NOT trusted
blindly: one camera locking onto the wrong object (the classic failure in this
rig — see drink_study notes on cam_4 / cam_10) would drag a naive all-camera DLT
metres off. So we triangulate the largest geometrically-agreeing subset:

  1. DLT over all available cams -> candidate X
  2. reproject X, drop cams whose reprojection error > REPROJ_PX
  3. re-triangulate on the survivors; require >= MIN_CAMS

This is the same consensus principle the research pipeline uses, distilled to a
single-frame robust triangulation (no temporal KF yet — that's the next stage).

Inputs are the per-camera JSONs from cup_detect.py / pose_keypoints.py. The
clip-suffix `.N.mp4` maps to calib key `cam_N`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.kalman_3d import CamCalib, load_calibration, project, triangulate_dlt

MIN_CAMS = 3          # >=3 agreeing cams — the hard floor from the robustness study
REPROJ_PX = 30.0      # a cam is an inlier if it reprojects within this many px

# The gate above is load-bearing for the CUP and REDUNDANT for POSE. Measured (11 reps,
# wrists vs MeTRAbs 3D): it fires on 24% of frames and moves the median 16.8 -> 16.6mm,
# never killing a frame. A cup FP is a DIFFERENT OBJECT (the side-desk glass) — it sits
# elsewhere in the world, reprojects far off, and the gate catches it. A pose error is the
# right person's slightly-wrong joint: still roughly in place, so it reprojects plausibly
# and passes the gate anyway, while a 10-cam DLT already averages the jitter out.
# Don't bother tightening this to "improve" pose — and note MIN_CAMS would start DROPPING
# pose frames on a rig with fewer cameras, for no accuracy gain. See archive/docs_20260820/WORKLOG.md.


def kf_rts_smooth(cons: np.ndarray, fps: float = 60.0,
                  q: float = 200.0 ** 2, r: float = 30.0 ** 2) -> np.ndarray:
    """Consensus -> constant-velocity KF -> RTS smoother. Fills gaps, kills jitter.

    THIS REPLACES LINEAR INTERPOLATION, and the difference is not cosmetic. The cup is
    OCCLUDED for ~24% of frames -- and not at random: it is occluded precisely at the mouth,
    during the drink, which is the part we are trying to measure. Something has to invent
    those positions. Drawing a straight line through them puts kinks in the trajectory
    exactly where the interesting curvature is.

    Measured in the research pipeline over 355 reps (trajectory-cleanliness):
        consensus + linear interp   : 274/355 clean
        consensus -> KF -> RTS      : 354/355 clean
    The KF wins because a constant-velocity model coasts through a gap the way an arm
    actually moves, and the RTS backward pass then uses the frames AFTER the gap to correct
    the coast -- so the filled segment is consistent with both sides, not just extrapolated.

    It also moves the DWELL BOUNDARIES the right way, checked frame-by-frame on the raw
    video (P07): drink onset KF 2.90s vs linear 3.08s (at 2.90s the cup is already at her
    lips); drink offset KF 3.78s vs linear 4.20s (at 3.78s it has tilted off them). Linear
    is late at both ends because a straight line between the last sighting before the
    occlusion and the first after CUTS THE CORNER of the cup's arc to the mouth -- the
    chord sits farther from the face than the true path, so "near the mouth" fires late
    and clears early.

    NO GATING. An earlier pipeline fed the KF raw 2D detections behind a Mahalanobis gate;
    one bad detection kicked the state, and then the gate rejected every TRUE detection as
    "too far" and it coasted away to metres. Feeding the CONSENSUS in as a direct 3D
    measurement with no gate makes divergence impossible: every consensus frame pulls the
    state back. It was the gating that was harmful, not the filter.

    cons: (T,3), NaN where no consensus. Returns (T,3) smoothed, gaps filled.
    """
    cons = np.asarray(cons, float)
    T = len(cons)
    dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = q * dt ** 3 / 3 * np.eye(3); Q[:3, 3:] = q * dt ** 2 / 2 * np.eye(3)
    Q[3:, :3] = q * dt ** 2 / 2 * np.eye(3); Q[3:, 3:] = q * dt * np.eye(3)
    R = r * np.eye(3)

    valid = np.isfinite(cons).all(1)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((T, 3), np.nan)

    x = np.zeros(6); x[:3] = cons[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0]) ** 2
    xp, Pp, xu, Pu = [], [], [], []
    for t in range(T):
        x = F @ x; P = F @ P @ F.T + Q                     # predict (= the gap fill)
        xp.append(x.copy()); Pp.append(P.copy())
        if valid[t]:                                        # update, ungated
            y = cons[t] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        xu.append(x.copy()); Pu.append(P.copy())

    xs = [None] * T; xs[-1] = xu[-1]                        # RTS backward pass
    for t in range(T - 2, -1, -1):
        C = Pu[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xu[t] + C @ (xs[t + 1] - xp[t + 1])
    return np.array([s[:3] for s in xs])


def kf_fill_gaps(cons: np.ndarray, fps: float = 60.0, max_gap_s: float = 0.75,
                 min_coverage: float = 0.30, **kw) -> np.ndarray:
    """kf_rts_smooth, but ONLY where filling is defensible. Returns NaN elsewhere.

    A constant-velocity coast across a 0.5s occlusion is a fill; the same coast across most of a
    trial is invention -- and it does not look broken, it produces a smooth curve that simply never
    goes where the cup went (measured: a trial 87% without consensus coasts 300-600mm from the mouth
    through the entire drink). Three guards:
      * coverage: if fewer than `min_coverage` of frames have a real observation, fill nothing.
      * gap length: interior gaps longer than `max_gap_s` stay NaN.
      * no extrapolation: nothing before the first or after the last real observation is filled.
    Observed frames keep the smoother's output (that is the denoising half of the job).
    """
    cons = np.asarray(cons, float)
    T = len(cons)
    valid = np.isfinite(cons).all(1)
    if valid.sum() < 2 or valid.mean() < min_coverage:
        return np.full((T, 3), np.nan)
    out = kf_rts_smooth(cons, fps=fps, **kw)
    keep = valid.copy()
    idx = np.flatnonzero(valid)
    lo, hi = idx[0], idx[-1]
    max_gap = int(round(max_gap_s * fps))
    g0 = None
    for t in range(lo, hi + 1):
        if not valid[t]:
            if g0 is None:
                g0 = t
        elif g0 is not None:
            if t - g0 <= max_gap:
                keep[g0:t] = True                      # short interior gap -> fill
            g0 = None
    res = np.full((T, 3), np.nan)
    res[keep] = out[keep]
    return res


def fill_cup_from_wrist(cup: np.ndarray, wrist: np.ndarray, min_pairs: int = 20,
                        fallback_offset: np.ndarray | None = None, hold_tol_mm: float = 60.0,
                        hold_only: bool = False) -> tuple[np.ndarray, dict]:
    """Fill frames with NO cup consensus from the WRIST, which is what still carries the cup.

    While the cup is held, cup and wrist translate together, so cup ~= wrist + d with d roughly
    constant over the hold. d is estimated per trial as the median (cup - wrist) over the frames where
    the cup DOES have consensus, so nothing outside the trial is assumed; with fewer than `min_pairs`
    such frames a caller-supplied `fallback_offset` (e.g. the participant median) is used, and failing
    that the frames stay NaN. This is the position-level version of the cup/wrist fusion already used
    for phases (drink_study/analysis/fuse_phases.py, log-odds of cup->head and wrist->head distance):
    the hand's distance to the mouth stands in for the cup's when the cup is unobserved.

    hold_only: restrict the fill to the HOLD SPAN (first to last observed hold frame). The offset is
    only valid while the hand actually has the cup -- the same argument this function already makes
    for ESTIMATING d applies to APPLYING it. Measured 2026-08-20: without it, 44.5% of filled frames
    (11168/25085) lie outside the hold span, 16.7% of them in rest_pre where the cup is on the table
    and the hand is at rest, so `wrist + d` puts the cup wherever the hand is. Constant-offset
    residual is 51-65 mm inside the hold against 132-141 mm in reaching/returning.

    Returns (filled, info) where info records the offset, how many frames were filled and from what.
    """
    cup = np.asarray(cup, float).copy()
    wrist = np.asarray(wrist, float)
    T = min(len(cup), len(wrist))
    cup, wrist = cup[:T], wrist[:T]
    have = np.isfinite(cup).all(1) & np.isfinite(wrist).all(1)
    need = (~np.isfinite(cup).all(1)) & np.isfinite(wrist).all(1)
    # Estimate d ONLY over frames where the hand actually HAS the cup. Over the whole trial the
    # average is a blend of holding and not-holding: before grasp and after release the cup sits on
    # the table while the hand is elsewhere, contributing offsets of several hundred mm in an
    # arbitrary direction. The hold is identified from the OBSERVED frames alone (wrist->cup within
    # `hold_tol_mm` of its own minimum), so this never depends on the segmentation it feeds.
    hold = np.zeros(len(cup), bool)
    if have.any():
        d_wc = np.linalg.norm(cup[have] - wrist[have], axis=1)
        thr = float(np.min(d_wc)) + hold_tol_mm
        hold[np.flatnonzero(have)[d_wc <= thr]] = True
    info = {"n_pairs": int(have.sum()), "n_hold": int(hold.sum()), "n_filled": 0,
            "offset_mm": None, "source": "none"}
    if hold.sum() >= min_pairs:
        d = np.median(cup[hold] - wrist[hold], axis=0)
        info["source"] = "per-trial-hold"
    elif fallback_offset is not None:
        d = np.asarray(fallback_offset, float)
        info["source"] = "fallback"
    else:
        return cup, info
    if hold_only:
        span = np.zeros(len(cup), bool)
        hi = np.flatnonzero(hold)
        if len(hi):
            span[hi[0]:hi[-1] + 1] = True
        need = need & span
        info["hold_only"] = True
    cup[need] = wrist[need] + d
    info["offset_mm"] = [float(v) for v in d]
    info["n_filled"] = int(need.sum())
    return cup, info


def camera_jitter(per_cam: dict[str, list], point_fn, n_frames: int) -> dict[str, float]:
    """Per-camera 2D jitter (median |3rd difference|, px) of one target's keypoint.

    THE WEIGHT SOURCE for weighted_triangulate. Why the THIRD difference: successive
    differences peel off what a real arm can legitimately do. The 1st is velocity (a moving
    hand has plenty), the 2nd is acceleration (the reach has that too), but the 3rd is JERK
    -- and a real arm, driven by muscle through soft tissue, cannot change its acceleration
    abruptly at 60fps. So what survives the third difference is almost entirely ESTIMATION
    NOISE. It is a high-pass filter: real motion is low-frequency and differences away,
    detector noise is high-frequency and remains.

    Why jitter and NOT the two obvious alternatives:

      * NOT per-keypoint CONFIDENCE. Measured across cameras, confidence is INVERTED:
        corr(median conf, median jitter) = +0.56 (yolo26s) / +0.79 (x). The most confident
        cameras are the SHAKIEST -- cam_9 is 0.995 confident and the jitteriest (4.6px),
        cam_8 is 0.981 and among the steadiest. Weighting by confidence would up-weight the
        worst cameras. (WITHIN one camera confidence IS a valid per-frame signal, r = -0.3
        to -0.64 vs jitter -- but that is a different question from ranking cameras.)

      * NOT REPROJECTION RESIDUAL. It measures disagreement with the consensus, and thereby
        assumes the consensus is TRUE. On this rig cam_4 has the worst wrist residual
        (20.9mm vs cam_8's 7.5mm) -- yet its 2D jitter is BETTER than average (2.25px), and
        DROPPING it makes the 3D noisier (3.34 -> 3.77mm). Watching the video, cam_4's own
        2D keypoint sits on the wrist better than the reprojected consensus does. A camera
        that is steady and RIGHT but outvoted looks identical, under a residual metric, to a
        camera that is wrong. Do not build a weight on it.

    Known limitation: jitter rewards SMOOTH WRONGNESS. A tracker that lags, or glides
    confidently onto the wrong point, scores beautifully. This measures noise, not
    correctness -- and it cannot settle whether the consensus itself is biased. That needs an
    INDEPENDENT reference (MeTRAbs), not another statistic from the same cameras.
    """
    out: dict[str, float] = {}
    for ckey, frames in per_cam.items():
        xy = []
        for f in range(min(n_frames, len(frames))):
            p = point_fn(frames[f])
            xy.append(p if p is not None else np.array([np.nan, np.nan]))
        xy = np.asarray(xy, float)
        if len(xy) < 4:
            continue
        d3 = np.linalg.norm(np.diff(xy, n=3, axis=0), axis=1)
        j = np.nanmedian(d3)
        if np.isfinite(j) and j > 0:
            out[ckey] = float(j)
    return out


def weighted_triangulate(cams: list[CamCalib], pts: list[np.ndarray],
                         weights: list[float]) -> np.ndarray:
    """DLT with per-camera weights. Each camera contributes 2 rows; scale them by sqrt(w)
    so the least-squares solution minimises the WEIGHTED sum of squared residuals."""
    A = []
    for cam, p, w in zip(cams, pts, weights):
        P = cam.K @ np.hstack([cam.R, cam.t.reshape(3, 1)])
        s = np.sqrt(max(w, 1e-9))
        A.append(s * (p[0] * P[2] - P[0]))
        A.append(s * (p[1] * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    h = Vt[-1]
    return h[:3] / h[3]


def _cam_key_from_clip(clip_name: str) -> str | None:
    """`P07_..._142730.4.mp4` -> 'cam_4'. Returns None if no .N. suffix."""
    m = re.search(r"\.(\d+)\.mp4$", clip_name)
    return f"cam_{int(m.group(1))}" if m else None


def robust_triangulate(cams: list[CamCalib], pts: list[np.ndarray]
                       ) -> tuple[np.ndarray | None, list[int], float]:
    """Consensus triangulation. Returns (X_mm | None, inlier_indices, median_reproj_px).

    None if fewer than MIN_CAMS survive the reprojection gate.
    """
    if len(cams) < 2:
        return None, [], float("inf")
    X = triangulate_dlt(cams, pts)                      # 1: all-cam candidate
    errs = np.array([np.linalg.norm(project(c, X)[0] - p)
                     for c, p in zip(cams, pts)])
    keep = [i for i in range(len(cams)) if errs[i] <= REPROJ_PX]  # 2: gate
    if len(keep) < MIN_CAMS:
        # RELAXED (2026-08): a MISCALIBRATED camera reprojects >REPROJ_PX even on a GOOD high-conf
        # detection (P19 right_shoulder: 3 cams at 43/28/46px, 2D conf 0.99). Returning None dropped
        # the whole joint -> deleted the affected-arm trials of miscalibrated participants wholesale
        # (P19: 83 trials, incl. every low-coordination reach). Instead FALL BACK to the all-cam DLT
        # so a miscalibrated frame is FORCED rather than lost. Validated (paper/relax_gate_*): recovers
        # all 691 trials on all 11 measures (interjoint 0.25->0.51), neutral-to-better on clean data
        # (the gate fires on 0.05% of P07 frames, so this is a no-op there). Still requires >=2 cams.
        keep = list(range(len(cams)))                   # X already = all-cam DLT computed above
    if len(keep) < len(cams):                           # 3: refit on inliers
        X = triangulate_dlt([cams[i] for i in keep], [pts[i] for i in keep])
    med = float(np.median([np.linalg.norm(project(cams[i], X)[0] - pts[i])
                           for i in keep]))
    return X, keep, med


def _load_per_cam(json_dir: Path, kind: str) -> dict[str, list]:
    """Map cam_key -> that camera's per-frame list, from *.{kind}.json files."""
    out: dict[str, list] = {}
    for jf in sorted(json_dir.glob(f"*.{kind}.json")):
        payload = json.loads(jf.read_text())
        ckey = _cam_key_from_clip(Path(payload["clip"]).name)
        if ckey:
            out[ckey] = payload["frames"]
    return out


def _cup_point(fr: dict) -> np.ndarray | None:
    c = fr.get("center")
    return np.array(c, dtype=float) if c else None


def _mouth_point(fr: dict) -> np.ndarray | None:
    """Mouth-proxy 2D from a pose frame: nose, else eye/ear centroid."""
    kps = fr.get("kps", {})
    if "nose" in kps:
        return np.array(kps["nose"][:2], dtype=float)
    pts = [kps[k][:2] for k in ("left_eye", "right_eye", "left_ear", "right_ear")
           if k in kps]
    return np.mean(pts, axis=0) if pts else None


def _wrist_point(fr: dict, side: str) -> np.ndarray | None:
    kps = fr.get("kps", {})
    k = f"{side}_wrist"
    return np.array(kps[k][:2], dtype=float) if k in kps else None


def _trunk_point(fr: dict) -> np.ndarray | None:
    """Upper-back proxy = shoulder midpoint. The Murphy trunk-displacement measure wants
    an upper-back site; COCO has no spine, and the shoulder midpoint is the closest rigid
    stand-in that both shoulders vote on (so one bad shoulder can't swing it far)."""
    kps = fr.get("kps", {})
    pts = [kps[k][:2] for k in ("left_shoulder", "right_shoulder") if k in kps]
    return np.mean(pts, axis=0) if len(pts) == 2 else None


POINT_FN = {
    "cup": ("cup", _cup_point),
    "mouth": ("pose", _mouth_point),
    "left_wrist": ("pose", lambda fr: _wrist_point(fr, "left")),
    "right_wrist": ("pose", lambda fr: _wrist_point(fr, "right")),
    "trunk": ("pose", _trunk_point),
}


def triangulate_target(per_cam: dict[str, list], cams: dict[str, CamCalib],
                       point_fn, n_frames: int, jitter_weight: bool = True) -> list[dict]:
    """Per-frame robust 3D for one target. Returns [{frame, X, n_cams, reproj_px}].

    With jitter_weight (default), cameras are weighted by 1/jitter^2 -- inverse-variance,
    the right weighting when the errors are independent and you know their scale. A camera
    whose 2D keypoint is shaky contributes proportionally less to the fit. Measured on the
    P07 wrist: 3D jitter 3.34 -> 3.07mm (-8%), for free. See camera_jitter for why the weight
    is JITTER and not confidence (inverted across cameras) or reprojection residual (assumes
    the consensus is truth, and would wrongly discard cam_4).
    """
    weights = {}
    if jitter_weight:
        jit = camera_jitter(per_cam, point_fn, n_frames)
        weights = {c: 1.0 / (j ** 2) for c, j in jit.items()}

    track = []
    for f in range(n_frames):
        use_cams, use_pts, use_w = [], [], []
        for ckey, frames in per_cam.items():
            if ckey not in cams or f >= len(frames):
                continue
            p = point_fn(frames[f])
            if p is not None:
                use_cams.append(cams[ckey])
                use_pts.append(p)
                use_w.append(weights.get(ckey, 1.0))
        X, keep, med = robust_triangulate(use_cams, use_pts)
        if X is not None and jitter_weight and len(keep) >= MIN_CAMS:
            # refit the surviving inliers, this time weighted
            X = weighted_triangulate([use_cams[i] for i in keep],
                                     [use_pts[i] for i in keep],
                                     [use_w[i] for i in keep])
            med = float(np.median([np.linalg.norm(project(use_cams[i], X)[0] - use_pts[i])
                                   for i in keep]))
        track.append({
            "frame": f,
            "X": None if X is None else [round(float(v), 1) for v in X],
            "n_cams": len(keep),
            "reproj_px": None if X is None else round(med, 2),
        })
    return track


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_dir", type=Path, help="dir of per-cam *.cup.json / *.pose.json")
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--targets", nargs="+",
                    default=["cup", "mouth", "left_wrist", "right_wrist"],
                    choices=list(POINT_FN))
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cams = load_calibration(args.calib, target_size=(1920, 1080))
    print(f"calib: {len(cams)} cams {sorted(cams)}", flush=True)

    result = {"calib": str(args.calib), "targets": {}}
    for tgt in args.targets:
        kind, fn = POINT_FN[tgt]
        per_cam = _load_per_cam(args.json_dir, kind)
        if not per_cam:
            print(f"  {tgt}: no {kind} JSONs found, skipping", flush=True)
            continue
        n = max(len(v) for v in per_cam.values())
        track = triangulate_target(per_cam, cams, fn, n)
        got = sum(1 for t in track if t["X"] is not None)
        med_px = np.median([t["reproj_px"] for t in track if t["reproj_px"] is not None]) \
            if got else float("nan")
        print(f"  {tgt:12} {len(per_cam)} cams, 3D in {got}/{n} frames "
              f"({got/n:.0%}), median reproj {med_px:.1f}px", flush=True)
        result["targets"][tgt] = track

    args.out.write_text(json.dumps(result))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
