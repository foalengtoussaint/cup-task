"""The ACTUAL Murphy position-measures, MMC vs OMC, DELTA cohort.

Not the signals (compare_pose_omc_delta.py) but the scalar SCORES the clinician reads. Uses
cup_task.score.compute_position_measures -- the position-derived subset cup-task ports (the
angle measures need the MuJoCo qpos IK fit, which we don't have; score.py documents the split).

Both sides get the SAME phase boundaries, segmented from the OMC cup (clean mocap cup, so the
DELTA cup-detector's poor recall does NOT confound the pose-scoring question). Then each side's
own hand + trunk 3D drives the measures. Difference = the pose source, phases held fixed.

  MMC hand  = right_wrist triangulated (yolo26s), mm, calib world
  OMC hand  = wrist_inner/outer_R midpoint, mm, mocap frame
  trunk     = shoulder-midpoint (COCO _trunk_point proxy / OMC shoulder markers)
  cup       = OMC cup cluster (both sides, for the shared segmentation only)

    python scripts/score_omc_delta.py --part P14 --trial trial_1_R_unaffected
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import segment
from cup_task.score import compute_position_measures

# reuse the loaders + fixes from the signal comparison
from scripts.compare_pose_omc_delta import (_load_mmc, _load_omc, _load_calib_mm,
                                            _find_lag, VIDEO_FPS, JOINTS, DELTA, _despike,
                                            _resample, C3D_RATE, _lp)
from cup_task.triangulate import kf_rts_smooth
import ezc3d


def _kfrts(xyz):
    """Consensus-anchored KF+RTS on an (T,3) track (fills gaps, kills jitter). The MMC per-joint
    3D IS the consensus here (robust-triangulated), so feed it straight in."""
    return kf_rts_smooth(np.asarray(xyz, float), fps=VIDEO_FPS)


SKELETON = [("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("right_shoulder", "left_shoulder"), ("right_hip", "left_hip"),
            ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip")]


def _bonelock_pbd(joints, bones=SKELETON, iters=8):
    """Fast whole-skeleton consistency by CONSTRAINT PROJECTION (position-based-dynamics style).

    The chain version (--bonelock) anchors the shoulder and re-places each joint from its parent,
    so every upstream correction is dumped downstream (the elbow's error leaks into the forearm,
    which is what shifted the elbow angle). Here instead each bone constraint moves BOTH endpoints
    half-way toward satisfying the median length, iterated a few times (Gauss-Seidel). No anchor,
    no chaining -- the correction is DISTRIBUTED, so each joint moves as little as possible and no
    single joint absorbs the whole error. O(T x iters x bones), no optimiser.

    Locks the whole skeleton (arm + shoulder span + pelvis + torso sides), not just the arm, so the
    trunk axis that shoulder flexion/abduction are measured against is stabilised too.
    """
    P = {j: np.array(v, float).copy() for j, v in joints.items()}
    L = {}
    for a, b in bones:
        if a in P and b in P:
            L[(a, b)] = np.nanmedian(np.linalg.norm(P[b] - P[a], axis=1))
    for _ in range(iters):
        for (a, b), Lab in L.items():
            d = P[b] - P[a]
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = (dist - Lab) / np.where(dist > 1e-9, dist, np.nan) * d
            ok = np.isfinite(corr).all(1)
            P[a][ok] += 0.5 * corr[ok]
            P[b][ok] -= 0.5 * corr[ok]
    return P


def _smooth_dir(d, hz=8.0, fps=VIDEO_FPS):
    """Temporally smooth a unit-direction track: lowpass each component, re-normalise.

    Smoothing the DIRECTION (not the position) is the point. The per-frame geometric fix
    (bone-lock/PBD) is memoryless: it re-places a joint along whatever direction that frame
    happened to observe, at a fixed radius -- so direction noise gets baked in at full lever arm
    and the joint jitters MORE than raw (measured: wrist jerk 6.2 -> 10.3). Smoothing the
    direction first removes the frame-to-frame angular wobble while leaving the bone rigid and the
    real swing intact. 8Hz is well above the ~4Hz the measures keep, so real motion survives.
    """
    from scipy.signal import butter, filtfilt
    d = np.asarray(d, float)
    out = np.full_like(d, np.nan)
    ok = np.isfinite(d).all(1)
    if ok.sum() < 12:
        return d
    idx = np.flatnonzero(ok)
    b, a = butter(2, hz / (fps / 2))
    tmp = np.empty_like(d)
    for k in range(3):
        tmp[:, k] = np.interp(np.arange(len(d)), idx, d[idx, k])
        tmp[:, k] = filtfilt(b, a, tmp[:, k])
    nrm = np.linalg.norm(tmp, axis=1, keepdims=True)
    out[ok] = (tmp / np.where(nrm > 1e-9, nrm, np.nan))[ok]
    return out


def _kf_only(cons, fps=VIDEO_FPS, q=200.0 ** 2, r=30.0 ** 2):
    """Forward-only KF (the same const-velocity model as kf_rts_smooth, WITHOUT the backward
    RTS pass). Causal = the real-time-realistic option; it smooths less than the full smoother,
    so it should preserve velocity PEAKS better (RTS's backward pass is what flattens them by
    borrowing future frames). Returns the filtered states xu[:, :3]."""
    cons = np.asarray(cons, float)
    T = len(cons); dt = 1.0 / fps
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = q * dt ** 3 / 3 * np.eye(3); Q[:3, 3:] = q * dt ** 2 / 2 * np.eye(3)
    Q[3:, :3] = q * dt ** 2 / 2 * np.eye(3); Q[3:, 3:] = q * dt * np.eye(3)
    R = r * np.eye(3)
    valid = np.isfinite(cons).all(1); idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.full((T, 3), np.nan)
    x = np.zeros(6); x[:3] = cons[idx[0]]
    P = np.diag([50, 50, 50, 500, 500, 500.0]) ** 2
    out = np.full((T, 3), np.nan)
    for t in range(T):
        x = F @ x; P = F @ P @ F.T + Q
        if valid[t]:
            y = cons[t] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y; P = (np.eye(6) - K @ H) @ P
        out[t] = x[:3]
    return out


def _shift(v, lag):
    out = np.full_like(v, np.nan)
    if lag >= 0:
        out[lag:] = v[:len(v) - lag] if lag else v
    else:
        out[:lag] = v[-lag:]
    return out


def _omc_cup(part, trial, n):
    """OMC cup cluster centroid on the video grid (for the shared segmentation)."""
    c = ezc3d.c3d(str(DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]
    cup = np.mean([P[:3, L.index(f"cluster_cup_{i}"), :].T for i in (1, 2, 3, 4)], axis=0)
    grid = _despike(_resample(cup, C3D_RATE, VIDEO_FPS))
    if len(grid) < n:
        grid = np.vstack([grid, np.full((n - len(grid), 3), np.nan)])
    return grid[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="P14")
    ap.add_argument("--trial", default="trial_1_R_unaffected")
    ap.add_argument("--smooth", choices=["none", "kfrts", "kf", "smoothnet"], default="none",
                    help="MMC joint-track smoothing: none / kf+rts (full) / kf (forward-only) / "
                         "smoothnet (v2 pipeline pose_smooth stage)")
    ap.add_argument("--bonelock", action="store_true",
                    help="hold arm bones (upperarm/forearm) at median length, keeping directions "
                         "-- the FAST skeleton-consistency fix (no IK optimiser)")
    ap.add_argument("--bonelock2", action="store_true",
                    help="cleaner bone-lock: preserve each bone's ORIGINAL direction independently "
                         "(forearm dir from the OLD elbow, not the moved one) so fixing the "
                         "upper-arm length doesn't perturb the elbow angle")
    ap.add_argument("--pbd", action="store_true",
                    help="whole-skeleton constraint projection: every bone (arm+shoulders+pelvis+"
                         "torso) held at median length, corrections DISTRIBUTED over both endpoints "
                         "(no anchor, no chaining). Fast.")
    ap.add_argument("--pbd-iters", type=int, default=8)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--hybridt", action="store_true",
                    help="NEGATIVE RESULT, kept for the record: smooth the bone DIRECTIONS before "
                         "placing. Redundant with score.py's own 4Hz lowpass -> no measure gain, "
                         "and double-smoothing flattens the peaks. Use --hybrid2 instead.")
    ap.add_argument("--dir-hz", type=float, default=8.0)
    ap.add_argument("--hybrid2", action="store_true",
                    help="SKELETON-AWARE SMOOTHER: hybrid -> smooth positions -> hybrid again. The "
                         "smooth kills the jitter but breaks the bones; the 2nd projection restores "
                         "rigidity, and because its input is now smooth it re-injects no noise. "
                         "Gives a skeleton that is BOTH rigid AND mocap-smooth.")
    ap.add_argument("--mid-hz", type=float, default=12.0,
                    help="intermediate smoothing cutoff for --hybrid2. 12Hz removes the jitter with "
                         "NO peak loss (real motion + the peak live below 4Hz).")
    a = ap.parse_args(argv)
    side = "right" if "_R_" in a.trial else "left"

    mmc, n = _load_mmc(a.part, a.trial)
    if a.smooth == "kfrts":
        mmc = {j: _kfrts(v) for j, v in mmc.items()}
        print("[MMC tracks smoothed with consensus KF+RTS (full)]", flush=True)
    elif a.smooth == "kf":
        mmc = {j: _kf_only(v) for j, v in mmc.items()}
        print("[MMC tracks smoothed with forward-only KF]", flush=True)
    elif a.smooth == "smoothnet":
        from cup_task.pose_smooth import smooth_track

        def _sn(xyz):
            tr = [{"frame": f, "X": (None if not np.isfinite(p).all() else [float(v) for v in p])}
                  for f, p in enumerate(np.asarray(xyz, float))]
            out = smooth_track(tr)
            return np.array([o["X"] if o["X"] is not None else [np.nan] * 3 for o in out])
        mmc = {j: _sn(v) for j, v in mmc.items()}
        print("[MMC tracks smoothed with SmoothNet (v2 pose_smooth stage)]", flush=True)
    if a.bonelock:
        sh, el, wr = mmc["right_shoulder"], mmc["right_elbow"], mmc["right_wrist"]
        Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
        Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
        du = el - sh; du = du / (np.linalg.norm(du, axis=1, keepdims=True) + 1e-9)
        el2 = sh + du * Lu
        df = wr - el2; df = df / (np.linalg.norm(df, axis=1, keepdims=True) + 1e-9)
        mmc["right_elbow"], mmc["right_wrist"] = el2, el2 + df * Lf
        print(f"[bone-lock: upperarm={Lu:.0f}mm forearm={Lf:.0f}mm held constant]", flush=True)
    if a.bonelock2:
        sh, el, wr = mmc["right_shoulder"], mmc["right_elbow"], mmc["right_wrist"]
        Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
        Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
        # each bone's direction from its ORIGINAL parent, computed BEFORE anything moves:
        du = el - sh; du = du / (np.linalg.norm(du, axis=1, keepdims=True) + 1e-9)
        df = wr - el; df = df / (np.linalg.norm(df, axis=1, keepdims=True) + 1e-9)   # OLD elbow
        el2 = sh + du * Lu
        mmc["right_elbow"], mmc["right_wrist"] = el2, el2 + df * Lf
        print(f"[bone-lock2 (independent dirs): upperarm={Lu:.0f}mm forearm={Lf:.0f}mm]", flush=True)
    if a.hybrid:
        # BEST variant. The two halves do different jobs, and each is wrong for the other's half:
        #   * TORSO -> PBD (distributed constraint projection). Stabilises the reference frame the
        #     shoulder angles + trunk displacement are measured against. Distributing is right here
        #     because no torso joint should absorb the whole error.
        #   * ARM -> chain-lock. Leaves the WRIST free at the end of the chain, so its speed is not
        #     damped by upstream constraints (whole-skeleton PBD pulls the wrist around and makes
        #     peak_velocity WORSE than baseline: -95 vs -84).
        # Result: peak_velocity -1.3 mm/s and peak_elbow_ang_vel +0.1 deg/s (both ~exact) while
        # keeping PBD's trunk fix.
        TORSO = [("right_shoulder", "left_shoulder"), ("right_hip", "left_hip"),
                 ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip")]
        mmc = _bonelock_pbd(mmc, bones=TORSO, iters=a.pbd_iters)
        sh, el, wr = mmc["right_shoulder"], mmc["right_elbow"], mmc["right_wrist"]
        Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
        Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
        du = el - sh; du = du / (np.linalg.norm(du, axis=1, keepdims=True) + 1e-9)
        el2 = sh + du * Lu
        df = wr - el2; df = df / (np.linalg.norm(df, axis=1, keepdims=True) + 1e-9)
        mmc["right_elbow"], mmc["right_wrist"] = el2, el2 + df * Lf
        print(f"[hybrid: PBD torso (x{a.pbd_iters}) + chain-lock arm "
              f"(upperarm={Lu:.0f} forearm={Lf:.0f}mm)]", flush=True)
    if a.hybridt:
        # hybrid + TEMPORAL consistency. Same two halves, but each bone's DIRECTION is lowpassed
        # before the joint is placed along it -- so the frame-to-frame angular wobble is not baked
        # in at full lever arm (the memoryless version makes wrist jerk 6.2 -> 10.3, WORSE than raw).
        TORSO = [("right_shoulder", "left_shoulder"), ("right_hip", "left_hip"),
                 ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip")]
        mmc = _bonelock_pbd(mmc, bones=TORSO, iters=a.pbd_iters)
        sh, el, wr = mmc["right_shoulder"], mmc["right_elbow"], mmc["right_wrist"]
        Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
        Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
        du = _smooth_dir((el - sh) / (np.linalg.norm(el - sh, axis=1, keepdims=True) + 1e-9),
                         hz=a.dir_hz)
        el2 = sh + du * Lu
        df = _smooth_dir((wr - el2) / (np.linalg.norm(wr - el2, axis=1, keepdims=True) + 1e-9),
                         hz=a.dir_hz)
        mmc["right_elbow"], mmc["right_wrist"] = el2, el2 + df * Lf
        print(f"[hybrid+temporal: PBD torso + dir-smoothed chain @{a.dir_hz}Hz "
              f"(upperarm={Lu:.0f} forearm={Lf:.0f}mm)]", flush=True)
    if a.hybrid2:
        # SKELETON-AWARE SMOOTHER = hybrid -> smooth -> hybrid.
        # Per-point smoothing alone breaks the bones; the geometric projection alone bakes in
        # per-frame direction noise (jerk 6.2 -> 10.3). Composing them fixes both: the smooth
        # kills the high-frequency jitter, then the 2nd projection restores rigidity -- and since
        # its INPUT is now smooth, its directions are smooth, so it re-injects nothing.
        # Measured (right wrist): jerk 10.30 -> 1.10 @12Hz with the peak UNCHANGED (687 -> 687),
        # because the jitter is >12Hz while the real motion and the peak live below 4Hz.
        # A 2nd geometric pass WITHOUT the smooth in between is a literal no-op (1e-6 mm) --
        # the chain projection is exact in one step. The smooth is what makes iterating mean
        # anything. This is the "smoothing that knows about the skeleton" -- it does not treat
        # each joint as independent, because the projection re-couples them.
        from cup_task.score import _smoothed_xyz as _sm

        def _fill_(x):
            x = np.asarray(x, float).copy()
            for k in range(3):
                v = np.isfinite(x[:, k])
                if v.sum() >= 2:
                    x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
            return x

        def _hyb(J):
            P = _bonelock_pbd({k: np.asarray(v, float).copy() for k, v in J.items()},
                              bones=[("right_shoulder", "left_shoulder"), ("right_hip", "left_hip"),
                                     ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip")],
                              iters=a.pbd_iters)
            sh, el, wr = P["right_shoulder"], P["right_elbow"], P["right_wrist"]
            Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
            Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
            du = el - sh; du = du / (np.linalg.norm(du, axis=1, keepdims=True) + 1e-9)
            el2 = sh + du * Lu
            df = wr - el2; df = df / (np.linalg.norm(df, axis=1, keepdims=True) + 1e-9)
            P["right_elbow"], P["right_wrist"] = el2, el2 + df * Lf
            return P

        h1 = _hyb(mmc)
        sm = {k: _sm(_fill_(v), VIDEO_FPS, a.mid_hz, 2) for k, v in h1.items()}
        mmc = _hyb(sm)
        print(f"[hybrid2 (skeleton-aware smoother): hybrid -> smooth@{a.mid_hz}Hz -> hybrid]",
              flush=True)
    if a.pbd:
        before = np.nanstd(np.linalg.norm(mmc["right_wrist"] - mmc["right_elbow"], axis=1))
        mmc = _bonelock_pbd(mmc, iters=a.pbd_iters)
        after = np.nanstd(np.linalg.norm(mmc["right_wrist"] - mmc["right_elbow"], axis=1))
        print(f"[PBD whole-skeleton x{a.pbd_iters}: forearm-len std {before:.1f} -> {after:.1f}mm]",
              flush=True)
    omc = _load_omc(a.part, a.trial, n)
    lag, sc = _find_lag(mmc["right_wrist"], omc["right_wrist"])
    omc = {j: _shift(v, lag) for j, v in omc.items()}
    cup = _shift(_omc_cup(a.part, a.trial, n), lag)
    print(f"sync lag {lag:+d} fr, wrist-speed corr {sc:.3f}", flush=True)

    omc_hand, omc_head = omc["right_wrist"], omc["nose"]

    # ONE COMPLETE CYCLE. This trial is sip1 (a full reach-drink-RETURN) then a truncated sip2.
    # The cycle closes only when BOTH the cup AND the wrist are back at rest -- the cup returns
    # ~1s before the hand does (the hand lingers, then travels home = the `returning` phase), so
    # cutting at cup-return alone would chop `returning` and break total_movement_time. Cut where
    # both settle (+ a small rest buffer), before sip2 starts, and segment only that window.
    def _rest_close(cup, wrist, thr=60.0, buf=20):
        cr = _lp(np.linalg.norm(cup - np.nanmedian(cup[:30], 0), axis=1))
        wr = _lp(np.linalg.norm(wrist - np.nanmedian(wrist[:30], 0), axis=1))
        lift = np.flatnonzero(cr > 150)
        if not len(lift):
            return len(cup)
        both = np.flatnonzero((np.arange(len(cup)) > lift[0]) & (cr < thr) & (wr < thr))
        return min(both[0] + buf, len(cup)) if len(both) else len(cup)

    cut = _rest_close(cup, omc_hand)
    print(f"first complete cycle: frames 0..{cut} ({cut/VIDEO_FPS:.1f}s); sip2 (truncated) dropped",
          flush=True)
    cupC, handC, headC = cup[:cut], omc_hand[:cut], omc_head[:cut]

    # Segment EXACTLY as cup_task/pipeline.py:phases_from_3d does -- the segmenter we already
    # have: segment_cup_only -> refine_grasp_with_pose (the pose step, mouth = OMC head) ->
    # to_murphy_phases. On the single-cycle window this yields the full 7-phase sequence.
    seg = segment.segment_cup_only(cupC, fps=VIDEO_FPS)
    seg = segment.refine_grasp_with_pose(seg, cupC, handC, headC, fps=VIDEO_FPS)
    phases = segment.to_murphy_phases(seg, handC, cupC, fps=VIDEO_FPS)
    print("base intervals:", seg["intervals"], flush=True)
    print("murphy phases:", [(nm, s, e) for nm, s, e in phases], flush=True)
    n = cut  # score on the single-cycle window
    mmc = {j: v[:cut] for j, v in mmc.items()}
    omc = {j: v[:cut] for j, v in omc.items()}

    def trunk(P):
        return (P["right_shoulder"] + P["left_shoulder"]) / 2.0

    def fill(xyz):
        """Interp over small gaps -- score.py's _smoothed_xyz uses filtfilt, which has no NaN
        handling and propagates a single missing frame across the WHOLE speed signal (here 3
        NaN frames NaN'd all of peak_velocity). The pipeline feeds track_confidence output,
        already gap-filled; do the same before scoring."""
        xyz = np.asarray(xyz, float).copy()
        for ax in range(3):
            v = np.isfinite(xyz[:, ax])
            if v.sum() >= 2:
                xyz[:, ax] = np.interp(np.arange(len(xyz)), np.flatnonzero(v), xyz[v, ax])
        return xyz

    m_meas = compute_position_measures(fill(mmc["right_wrist"]), fill(trunk(mmc)), phases, side,
                                       fps=VIDEO_FPS)
    o_meas = compute_position_measures(fill(omc["right_wrist"]), fill(trunk(omc)), phases, side,
                                       fps=VIDEO_FPS)
    md, od = m_meas.to_dict(), o_meas.to_dict()

    print(f"\n{'measure':38} {'MMC':>10} {'OMC':>10} {'Δ':>10}", flush=True)
    print("-" * 72, flush=True)
    for k in md:
        if k == "side":
            continue
        mv, ov = md[k], od[k]
        if isinstance(mv, float) and np.isfinite(mv) and np.isfinite(ov):
            print(f"{k:38} {mv:10.2f} {ov:10.2f} {mv-ov:+10.2f}", flush=True)
        else:
            print(f"{k:38} {str(mv):>10} {str(ov):>10}", flush=True)

    # ---- the ANGLE measures, computed from RAW POINTS on both sides ("not ported" because
    # the container refuses raw-point angles in production -- jitter -- but computable to SEE.
    # Healthy refs from the paper are printed for scale.) ----
    from scripts.compare_pose_omc_delta import _murphy_signals, _lp as _lpf
    def elbow_angle(P):
        u, v = P["right_shoulder"] - P["right_elbow"], P["right_wrist"] - P["right_elbow"]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        return _lpf(np.degrees(np.arccos(np.clip(c, -1, 1))))

    def ph(name):
        for nm, s, e in phases:
            if nm == name:
                return s, e
        return None

    def angle_scalars(P):
        sig = _murphy_signals(P)
        flex, abd = _lpf(sig["shoulder_flexion"]), _lpf(sig["shoulder_abduction"])
        elb = elbow_angle(P)
        eav = np.abs(np.gradient(elb)) * VIDEO_FPS
        r, d = ph("reaching"), ph("drinking")
        def mn(a, w): return float(np.nanmin(a[w[0]:w[1]])) if w else float("nan")
        def mx(a, w): return float(np.nanmax(a[w[0]:w[1]])) if w else float("nan")
        # interjoint_coordination = Pearson(shoulder_flex, elbow_angle) over inner-80% reaching
        ijc = float("nan")
        if r and r[1] - r[0] >= 10:
            mgn = int(0.1 * (r[1] - r[0]))
            a1, b1 = flex[r[0] + mgn:r[1] - mgn], elb[r[0] + mgn:r[1] - mgn]
            if np.std(a1) > 1e-6 and np.std(b1) > 1e-6:
                ijc = float(np.corrcoef(a1, b1)[0, 1])
        # BUG FIX: the container computes max(elbow_angle) over reaching+forward_transport, in the
        # EXTENSION convention (180 = straight). It is "how STRAIGHT did the arm get at full
        # reach", not "how bent". I had min() -- backwards -- and the wrong phase slice.
        rf = r
        fw = ph("forward_transport")
        if r and fw:
            rf = (r[0], fw[1])
        return {
            "elbow_extension_reaching (max)": mx(elb, rf),
            "shoulder_flexion_reaching (max)": mx(flex, r),
            "shoulder_flexion_drinking (max)": mx(flex, d),
            "shoulder_abduction_reaching (max)": mx(abd, r),
            "shoulder_abduction_drinking (max)": mx(abd, d),
            "peak_elbow_ang_vel (deg/s)": mx(eav, (0, len(elb))),
            "interjoint_coordination (r)": ijc,
        }

    ma, oa = angle_scalars(mmc), angle_scalars(omc)
    healthy = {"elbow_extension_reaching (max)": 53.5, "shoulder_flexion_reaching (max)": 45.6,
               "shoulder_flexion_drinking (max)": 51.7, "shoulder_abduction_drinking (max)": 30.1}
    print(f"\n--- ANGLE measures (raw-point; NOT the ported set; healthy ref shown) ---",
          flush=True)
    print(f"{'measure':38} {'MMC':>8} {'OMC':>8} {'Δ':>8} {'healthy':>8}", flush=True)
    for k in ma:
        print(f"{k:38} {ma[k]:8.1f} {oa[k]:8.1f} {ma[k]-oa[k]:+8.1f} "
              f"{healthy.get(k, float('nan')):8.1f}", flush=True)


if __name__ == "__main__":
    main()
