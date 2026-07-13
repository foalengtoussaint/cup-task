"""Drink-task phase segmentation from the 3D cup track.

Ported from the research pipeline (object-tracking `drink_study/lib/segment_cup_only.py`).

    segment_cup_only(cup)   <- THE ONE TO USE. Cup track alone: van Andel glass-velocity
                               gates give the transport window, then the drink dwell is
                               "near peak displacement AND slow" inside it.

Verified port: on 40 research reps this reproduces drink_study's segment_cup_only phase
intervals EXACTLY (frame-for-frame, 40/40). Same code, same answer.

Gate constants are the LOPO-TUNED ones (DRINK_SPEED/DRINK_DISP_PAD = 150/150, not the
original 120/90): tuning moved the dwell-duration bias from -95ms to +14ms and improved
20 of 21 participant folds. Do not "restore" the lower values.

This is the BASE pipeline -- deliberately the same geometric method the research truth
uses. Two known improvements are NOT ported yet:
  - TCN gap-fill of the cup track (the occluded apex is where the cup track is worst)
  - a head-distance feature channel (research `proxy21` = kinematics+occlusion+head
    beats the video-only `base17`, ~85ms vs ~123ms LOPO) -- but there the head comes
    from MOCAP, and swapping in a video head landmark is an unmeasured substitution.

EXPERIMENTAL, not the default: segment_fused() below adds a log-odds cup+pose fusion.
Measured here on 123 research reps it RECOVERED 0 dwells and DESTROYED 6 vs cup-only --
because on the general cohort the cup-only gate already finds a dwell in 100% of reps,
so there is nothing to recover. Its "recovered 13/20" result came from a deliberately
adversarial subset (the 20 worst reps, where the cup track was failing). Both numbers are
real; they describe different populations. Don't ship it as the default on the strength
of the wrong one.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

FPS = 60.0

# --- transport window (van Andel glass-velocity gates) ---
BUTTER_HZ = 6.0        # 2nd-order zero-phase Butterworth on the POSITION path
FWD_ON = 15.0          # mm/s: forward-transport onset
BACK_OFF = 10.0        # mm/s: back-transport end
CUP_MIN_RUN = max(int(0.15 * FPS), 3)

# --- drink dwell (geometric gate) ---
# LOPO-tuned in the research pipeline: 120/90 -> 150/150 moved the duration bias from
# -95ms to +14ms and improved 20 of 21 folds. Do not "restore" the lower values.
DRINK_SPEED = 150.0    # mm/s: cup nearly still at the mouth
DRINK_DISP_PAD = 150.0 # mm: "near peak" = within this of max displacement from rest
MIN_PHASE = max(int(0.20 * FPS), 5)

# --- fusion ---
DRINK_FRAC = 0.15      # van Andel: drink when distance < 15% of steady state
SOFTNESS_MM = 60.0     # logistic softness, near-mouth membership
SPD_SOFT = 80.0        # logistic softness, slow membership
E_ON, E_OFF = 0.55, 0.40      # hysteresis on the fused evidence
MIN_DRINK_FR = max(int(0.20 * FPS), 5)

PHASE_NAMES = ["rest_pre", "forward_transport", "drinking", "back_transport", "rest_post"]
P_REST_PRE, P_FWD, P_DRINK, P_BACK, P_REST_POST = range(5)


# ---------------------------------------------------------------- helpers

def _median_smooth(x, w=7):
    x = np.asarray(x, float)
    if w < 2 or len(x) < w:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + w]) for i in range(len(x))])


def _butter_lp(x, fps=FPS, hz=BUTTER_HZ):
    """Zero-phase 2nd-order low-pass on each axis. No-op on very short clips."""
    x = np.asarray(x, float)
    if len(x) < 15:
        return x
    b, a = butter(2, hz / (0.5 * fps), btype="low")
    return filtfilt(b, a, x, axis=0)


def _runs(mask):
    """Contiguous True runs as [start, end) index pairs."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def _intervals(phase):
    out, s = [], 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[s]:
            out.append((PHASE_NAMES[phase[s]], s, i)); s = i
    return out


def _interp_nan(a):
    a = np.asarray(a, float).copy()
    idx = np.flatnonzero(np.isfinite(a))
    if len(idx) < 2:
        return a
    a[:] = np.interp(np.arange(len(a)), idx, a[idx])
    return a


def _interp_nan_xyz(xyz):
    out = np.asarray(xyz, float).copy()
    for ax in range(3):
        out[:, ax] = _interp_nan(out[:, ax])
    return out


def _logi(x, soft):
    """->1 when x << 0, ->0 when x >> 0."""
    return 1.0 / (1.0 + np.exp(np.clip(x / soft, -30, 30)))


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -30, 30)))


# ---------------------------------------------------------------- cup-only

def segment_cup_only(xyz, fps=FPS, *, fwd_on=FWD_ON, back_off=BACK_OFF,
                     drink_speed=DRINK_SPEED, drink_disp_pad=DRINK_DISP_PAD,
                     butter_hz=BUTTER_HZ, min_phase=MIN_PHASE) -> dict:
    """Phases from the 3D cup track alone. xyz: (T,3) mm, NaN where untracked."""
    xyz = np.asarray(xyz, float)
    T = len(xyz)
    phase = np.full(T, P_REST_PRE, dtype=np.int8)
    valid = np.isfinite(xyz).all(1)
    empty = {"phase": phase, "intervals": _intervals(phase), "speed": np.zeros(T),
             "disp": np.zeros(T), "grasp": (0, 0), "drink_runs": []}
    if valid.sum() < 10:
        return empty

    filled = _butter_lp(_interp_nan_xyz(xyz), fps, butter_hz)
    speed = np.r_[0.0, np.linalg.norm(np.diff(filled, axis=0), axis=1)] * fps

    rw = max(int(0.5 * fps), 10)
    rest_pos = np.median(filled[:min(rw, T)], axis=0)
    disp = np.linalg.norm(filled - rest_pos, axis=1)

    # transport window: hysteresis on the glass-velocity gates
    onset_runs = [(s, e) for s, e in _runs(speed > fwd_on) if e - s >= CUP_MIN_RUN]
    if not onset_runs:
        return {**empty, "speed": speed, "disp": disp}
    grasp_start, grasp_end = onset_runs[0][0], onset_runs[-1][1]
    loose = np.flatnonzero(speed > back_off)
    tail = loose[loose >= grasp_end - 1]
    if tail.size:
        grasp_end = int(tail[-1]) + 1

    in_win = np.zeros(T, bool)
    in_win[grasp_start:grasp_end] = True

    peak_disp = disp[in_win].max() if in_win.any() else 0.0
    near_peak = disp > (peak_disp - drink_disp_pad)
    drink_runs = [(s, e) for s, e in _runs(in_win & near_peak & (speed < drink_speed))
                  if e - s >= min_phase]

    phase[grasp_start:grasp_end] = P_FWD
    phase[grasp_end:] = P_REST_POST
    if drink_runs:
        d0, d1 = drink_runs[0][0], drink_runs[-1][1]
        phase[grasp_start:d0] = P_FWD
        phase[d0:d1] = P_DRINK
        phase[d1:grasp_end] = P_BACK
    else:
        apex = (grasp_start + int(np.argmax(disp[grasp_start:grasp_end]))
                if grasp_end > grasp_start else grasp_start)
        phase[grasp_start:apex] = P_FWD
        phase[apex:grasp_end] = P_BACK

    return {"phase": phase, "intervals": _intervals(phase), "speed": speed, "disp": disp,
            "grasp": (grasp_start, grasp_end), "drink_runs": drink_runs,
            "peak_disp": float(peak_disp)}


# ---------------------------------------------------------------- fusion

def drink_evidence(dist_to_mouth, speed, fps=FPS):
    """Per-frame drink evidence in [0,1] from one source's distance-to-mouth + speed.

    NEAR-mouth is the evidence; SLOW is only a GATE. A fast frame cannot be a dwell, so
    speed caps the evidence -- but a slow frame must not DILUTE a clear near-mouth
    reading (multiplying two soft memberships caps the product near 0.5 and it never
    crosses threshold; that was a real bug in the first version).

    The trigger is self-normalised per trial: van Andel fire the drink phase when the
    face-to-glass distance drops below 15% of steady state. No reference point here
    (nose proxy, or a mocap head centroid) actually reaches the glass, so an absolute
    15% is unreachable. We apply the 15% to the rest->closest EXCURSION instead: drink
    = 85% of the way in from rest. Same physical event, independent of how far inside
    the head the reference point sits.
    """
    d = _interp_nan(dist_to_mouth)
    steady = np.nanpercentile(d, 90)       # resting (far) distance ~ steady state
    dmin = np.nanpercentile(d, 5)          # closest approach
    thresh = dmin + DRINK_FRAC * (steady - dmin)
    near = _logi(d - thresh, SOFTNESS_MM)
    gate = _logi(_median_smooth(speed) - DRINK_SPEED, SPD_SOFT)
    gate = np.clip(gate * 1.6, 0.0, 1.0)   # widen the 'slow' plateau: a gate, not a dilutor
    return near * gate, d


def _hysteresis(E, on=E_ON, off=E_OFF, min_run=MIN_DRINK_FR):
    m = np.zeros(len(E), bool)
    state = False
    for i, e in enumerate(E):
        state = (e >= on) if not state else (e >= off)
        m[i] = state
    return [(s, e) for s, e in _runs(m) if e - s >= min_run]


def track_confidence(track: list[dict], min_cams: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(xyz, conf) from a triangulate.triangulate_target() track.

    conf = agreement x tightness:
      - agreement: how many cameras survived the gate, saturating at min_cams
      - tightness: how well they agreed, from the median reprojection error
                   (3px -> 1.0, 28px -> 0.15)
    A frame with no 3D at all gets conf 0, which is the point: the fusion must be able
    to tell "the cup is here and I'm sure" from "the cup is here-ish because I coasted".
    This is what collapses during the occluded mouth dwell and lets the pose take over.
    """
    T = len(track)
    xyz = np.full((T, 3), np.nan)
    conf = np.zeros(T)
    for i, fr in enumerate(track):
        X, n, px = fr.get("X"), fr.get("n_cams", 0), fr.get("reproj_px")
        if X is None:
            continue
        xyz[i] = X
        if px is not None:
            tight = float(np.clip(1.0 - (px - 3.0) / 25.0, 0.15, 1.0))
            conf[i] = min(n / float(min_cams), 1.0) * tight
    return xyz, conf


def pose_confidence(frames: list[dict], joints, min_conf: float = 0.30) -> np.ndarray:
    """Per-frame pose confidence = geometric mean of the named joints' 2D confidences,
    taken across cameras. Both joints must be good for the pose vote to count -- a
    confident wrist next to an invisible mouth cannot locate a sip."""
    T = len(frames)
    out = np.zeros(T)
    for i, fr in enumerate(frames):
        kps = fr.get("kps", {})
        cs = [kps[j][2] for j in joints if j in kps and kps[j][2] >= min_conf]
        out[i] = float(np.prod(cs) ** (1.0 / len(joints))) if len(cs) == len(joints) else 0.0
    return out


def segment_fused(cup, mouth, wrist, cup_conf=None, pose_conf=None, fps=FPS) -> dict:
    """EXPERIMENTAL -- not the default. Phases from the cup track AND the pose.

    On the general cohort this is a small REGRESSION vs segment_cup_only (measured: 123
    reps, recovered 0, destroyed 6) because cup-only already finds a dwell in 100% of
    them. It earns its keep only where the cup track is actually failing. Use
    segment_cup_only unless you have measured that fusion helps on YOUR reps.

    cup, mouth, wrist : (T,3) mm in the same 3D world frame, NaN where untracked.
    cup_conf, pose_conf : (T,) in [0,1]; per-frame trust in each source. Default 1.0.

    The transport window still comes from the cup (it is the clean end-effector); only
    the DRINK DWELL is fused, because that is the part the cup gets wrong.
    """
    cup = np.asarray(cup, float)
    mouth = np.asarray(mouth, float)
    wrist = np.asarray(wrist, float)
    T = min(len(cup), len(mouth), len(wrist))
    cup, mouth, wrist = cup[:T], mouth[:T], wrist[:T]
    w_cup = np.ones(T) if cup_conf is None else np.asarray(cup_conf, float)[:T]
    w_pose = np.ones(T) if pose_conf is None else np.asarray(pose_conf, float)[:T]

    # interpolate dropped frames BEFORE differentiating: a single (0,0,0) zero-fill
    # frame otherwise poisons distance-to-mouth and spikes the speed into a phantom
    cup_i = _butter_lp(_interp_nan_xyz(cup), fps)
    mouth_i = _butter_lp(_interp_nan_xyz(mouth), fps)
    wrist_i = _butter_lp(_interp_nan_xyz(wrist), fps)

    # one physical signal (distance to mouth), two independent sources
    cup_mouth = np.linalg.norm(cup_i - mouth_i, axis=1)
    wrist_mouth = np.linalg.norm(wrist_i - mouth_i, axis=1)
    cup_spd = np.r_[0, np.linalg.norm(np.diff(cup_i, axis=0), axis=1)] * fps
    wrist_spd = np.r_[0, np.linalg.norm(np.diff(wrist_i, axis=0), axis=1)] * fps

    e_cup, cup_mouth = drink_evidence(cup_mouth, cup_spd, fps)
    e_pose, wrist_mouth = drink_evidence(wrist_mouth, wrist_spd, fps)

    # LOG-ODDS fusion: each source votes, scaled by its own confidence. Votes add, so
    # two confident sources agreeing exceed either alone; an unconfident source barely
    # moves the needle. Neutral prior keeps E ~ 0.5 when nobody is confident.
    E = _median_smooth(_sigmoid(w_cup * _logit(e_cup) + w_pose * _logit(e_pose)), 5)
    drink_runs = _hysteresis(E)
    drink = (drink_runs[0][0], drink_runs[-1][1]) if drink_runs else None

    # transport window from the cup (clean end-effector), dwell from the fusion
    seg = segment_cup_only(cup, fps=fps)
    onset, offset = seg["grasp"]

    phase = np.full(T, P_REST_PRE, dtype=np.int8)
    onset = onset or 0
    offset = offset or T
    phase[onset:offset] = P_FWD
    phase[offset:] = P_REST_POST
    if drink:
        d0, d1 = drink[0], min(drink[1], offset) if offset > drink[0] else drink[1]
        phase[onset:d0] = P_FWD
        phase[d0:d1] = P_DRINK
        phase[d1:offset] = P_BACK

    return {"phase": phase, "intervals": _intervals(phase), "E": E,
            "e_cup": e_cup, "e_pose": e_pose, "w_cup": w_cup, "w_pose": w_pose,
            "cup_mouth": cup_mouth, "wrist_mouth": wrist_mouth,
            "grasp": (onset, offset), "drink": drink, "drink_runs": drink_runs}
