"""Murphy measures for the drink task — the POSITION-derived subset.

Ports `imove_extensions/murphy_measures.py` (the iMOVE container's translation of the
17-measure Murphy protocol for the iARAT drink task) to cup-task's mocap-free 3D.

WHAT IS HERE AND WHAT IS NOT
----------------------------
The 17 measures split by what they need, and only half of them survive the trip:

  PORTED (position-derived) — computed from the 3D hand/trunk positions, which we have:
    total_movement_time, peak_velocity, time_to_peak_velocity (+ _percent),
    time_to_first_peak_velocity (+ _percent), number_of_movement_units,
    max_trunk_displacement

  NOT PORTED (angle-derived) — elbow_extension_reaching, shoulder_flexion_*,
    shoulder_abduction_*, peak_elbow_angular_velocity, interjoint_coordination.
    These read MuJoCo `qpos` — the body model's classical IK fit. We have COCO-17
    keypoints, no body model, no IK. The container's authors explicitly REFUSED to
    compute these from raw keypoint geometry ("would inherit pose jitter"), so we
    don't either. Computing them from points is possible; it is a separate decision
    that needs its own measurement, not a freebie.

Returning half the protocol and saying so beats returning all 17 with 8 quietly wrong.

FAITHFULNESS
------------
The subtle parts are copied, not re-derived, because the container's comments record
bugs they already paid for:
  * peak_velocity / time_to_peak are scoped to the REACHING phase, and the time is
    relative to REACH START — using the global frame index "silently adds rest_pre's
    duration" (their words) and inflates a 0.47s healthy value past 1s.
  * movement units are counted over reaching + forward_transport + back_transport +
    returning — everything EXCEPT drinking.
  * smooth POSITION then differentiate. Filtering velocity post-hoc smears the spikes
    but they still dominate peak_velocity.
  * video-tuned constants, NOT the mocap notebook's: 4Hz lowpass (not 6), movement-unit
    amplitude 60mm/s (not 20), 150ms time gap.
    MEASURED 2026-08-22: the markerless wrist moves 13.4 mm/s at rest (IQR 8.5-22.4) against
    4.0 mm/s for the optical markers -- NOT the "~30-50mm/s" this comment used to assert, which
    was never measured. The 60 mm/s value is not critical either way: swept over 20-80 mm/s on
    both systems, movement-unit agreement moves only between r_s 0.80 and 0.82
    (scratchpad/mu_sweep.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt

DEFAULT_FPS = 60.0
DEFAULT_LOWPASS_HZ = 4.0
DEFAULT_BUTTER_ORDER = 2
DEFAULT_MU_AMPLITUDE_MMPS = 60.0
DEFAULT_MU_TIME_GAP_S = 0.15


@dataclass
class MurphyPositionMeasures:
    """The position-derived subset of the Murphy measures. Angles are absent by design
    (no body-model IK here) -- see the module docstring."""
    side: str
    total_movement_time: float
    peak_velocity: float
    time_to_peak_velocity: float
    time_to_peak_velocity_percent: float
    time_to_first_peak_velocity: float
    time_to_first_peak_velocity_percent: float
    number_of_movement_units: int
    max_trunk_displacement: float

    def to_dict(self) -> dict:
        return asdict(self)


def _butter_lowpass(sig, fs, cutoff, order):
    """Zero-phase Butterworth lowpass; identity on signals too short to filter.

    NaN-SAFE. filtfilt propagates a single NaN across the WHOLE output, so two missing frames in a
    596-frame trial silently nulled every measure for that trial (measured: 2/12 trials returned
    NaN peak_velocity for exactly this reason). Interpolate the gaps, filter, then restore NaN
    only where the input was actually missing -- so a gap can never poison the frames around it.
    """
    sig = np.asarray(sig, float)
    if len(sig) < 3 * (order + 1):
        return sig.astype(np.float32)
    bad = ~np.isfinite(sig)
    if bad.all():
        return sig.astype(np.float32)
    if bad.any():
        idx = np.arange(len(sig))
        sig = np.interp(idx, idx[~bad], sig[~bad])
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    out = filtfilt(b, a, sig).astype(np.float32)
    if bad.any():
        out[bad] = np.nan
    return out


def _smoothed_xyz(xyz, fs, cutoff, order):
    """Median-of-3 per axis, THEN lowpass. The medfilt is not optional: a lowpass
    SMEARS a single-frame position spike into a multi-frame bump that survives even a
    2Hz cutoff, and that bump then masquerades as a 200+mm/frame velocity outlier and
    dominates peak_velocity. Dropping it shifted peak_velocity by up to 2.4% here."""
    xyz = np.asarray(xyz, float)
    out = np.empty_like(xyz, dtype=np.float32)
    for ax in range(xyz.shape[1]):
        out[:, ax] = _butter_lowpass(medfilt(xyz[:, ax], kernel_size=3), fs, cutoff, order)
    return out


def _hand_speed_mmps(hand_xyz_smoothed, fps):
    """mm/s from ALREADY-SMOOTHED positions, zero-padded at frame 0."""
    d = np.linalg.norm(np.diff(hand_xyz_smoothed, axis=0), axis=1) * fps
    return np.concatenate(([0.0], d)).astype(np.float32)


def _phase_slice(phase_intervals, *names):
    sel = [(s, e) for (n, s, e) in phase_intervals if n in names]
    if not sel:
        return None
    return min(s for s, _ in sel), max(e for _, e in sel)


def _count_movement_units(velocity, amplitude_thr_mmps, time_gap_frames):
    """Count min->max oscillations: for each local min, take the next local max; count it
    if that max exceeds the amplitude threshold and is >= time_gap_frames later."""
    if len(velocity) < 3:
        return 0
    min_idx, _ = find_peaks(-velocity)
    max_idx, _ = find_peaks(velocity)
    if len(min_idx) == 0 or len(max_idx) == 0:
        return 0
    n = 0
    for mn in min_idx:
        for mx in max_idx:
            if mx <= mn:
                continue
            if velocity[mx] > amplitude_thr_mmps and (mx - mn) >= time_gap_frames:
                n += 1
            break
    return n


def compute_position_measures(
    hand_xyz: np.ndarray,          # (T,3) dominant-hand 3D, mm, world frame
    trunk_xyz: np.ndarray,         # (T,3) trunk/upper-back 3D, mm (for trunk displacement)
    phase_intervals: list,         # [(name, start, end_exclusive)] -- needs 7-phase names
    side: str,                     # 'right' | 'left'
    *,
    fps: float = DEFAULT_FPS,
    lowpass_hz: float = DEFAULT_LOWPASS_HZ,
    butter_order: int = DEFAULT_BUTTER_ORDER,
    mu_amplitude_mmps: float = DEFAULT_MU_AMPLITUDE_MMPS,
    mu_time_gap_s: float = DEFAULT_MU_TIME_GAP_S,
) -> MurphyPositionMeasures:
    """The position-derived Murphy measures for one rep.

    phase_intervals must use the CONTAINER's 7-phase names (rest_pre, reaching,
    forward_transport, drinking, back_transport, returning, rest_post). The windows are
    load-bearing: `reaching` in particular is a real phase (hand travelling to the cup
    BEFORE the cup moves), not a synonym for forward_transport -- conflating the two
    "doubles the effective window and shifts peak-velocity timing".
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    hand_xyz = np.asarray(hand_xyz, float)
    T = len(hand_xyz)

    # smooth the POSITION, then differentiate (see docstring)
    hand_speed = _hand_speed_mmps(
        _smoothed_xyz(hand_xyz, fps, lowpass_hz, butter_order), fps)

    reaching = _phase_slice(phase_intervals, "reaching")
    returning_ph = _phase_slice(phase_intervals, "returning")
    rest_post = _phase_slice(phase_intervals, "rest_post")

    # total_movement_time = returning end - reaching start
    movement_start = reaching[0] if reaching else 0
    if rest_post is not None:
        movement_end = rest_post[0]
    elif returning_ph is not None:
        movement_end = returning_ph[1]
    else:
        movement_end = T
    total_movement_time = (movement_end - movement_start) / fps

    # peak velocity + timing, scoped to REACHING, timed from REACH START
    nan = float("nan")
    peak_velocity = 0.0
    t_peak = t_peak_pct = t_first = t_first_pct = nan
    if reaching is not None and reaching[1] > reaching[0]:
        rs, re = reaching
        seg = hand_speed[rs:re]
        if len(seg):
            peak_velocity = float(np.max(seg))
            pk = int(np.argmax(seg))
            # ABSOLUTE time_to_peak is measured from the TRIAL START (frame 0), not the reaching
            # window start -- verified vs AutoMQ (reproduces their stored scalar to ~0; measuring
            # from reach start was off by the per-trial-varying ~0.7s pre-reach, which collapsed the
            # correlation to r=0.21). The PERCENT stays fraction-of-reaching (pk/reach_len), which
            # DOES match AutoMQ. (2026-08-05)
            t_peak = (rs + pk) / fps
            t_peak_pct = pk / max(re - rs, 1) * 100.0
            # first peak: the raw derivative still wiggles enough that find_peaks fires
            # on ~50mm/s blips in the first 100ms. Smooth harder and demand prominence.
            seg_pk = _butter_lowpass(seg, fps, min(3.0, lowpass_hz), butter_order)
            prom = max(50.0, 0.15 * peak_velocity)
            first, _ = find_peaks(seg_pk, prominence=prom)
            if len(first):
                idx = int(first[0])
            else:
                # find_peaks only sees INTERIOR local maxima, so a reach whose speed peaks at the
                # very edge of the window yields nothing at all (measured: 2/12 trials here, one
                # with argmax at frame 0). A single-peaked reach is the NORMAL healthy case -- it
                # must not report "no first peak". Fall back to the argmax, which for a
                # single-peaked profile IS the first peak.
                idx = int(np.nanargmax(seg_pk)) if np.isfinite(seg_pk).any() else -1
            if idx >= 0:
                t_first = (rs + idx) / fps          # from frame 0 (see t_peak note)
                t_first_pct = idx / max(re - rs, 1) * 100.0

    # movement units: everything EXCEPT drinking
    gap = max(int(mu_time_gap_s * fps), 1)
    n_units = 0
    for name in ("reaching", "forward_transport", "back_transport", "returning"):
        sl = _phase_slice(phase_intervals, name)
        if sl is None:
            continue
        n_units += _count_movement_units(hand_speed[sl[0]:sl[1]], mu_amplitude_mmps, gap)

    # trunk displacement: max |deviation| from the starting position, on the y axis
    trunk_y = np.asarray(trunk_xyz, float)[:, 1]
    trunk_y = _butter_lowpass(trunk_y, fps, lowpass_hz, butter_order)
    max_trunk_displacement = float(np.max(np.abs(trunk_y - trunk_y[0]))) if len(trunk_y) else 0.0

    return MurphyPositionMeasures(
        side=side,
        total_movement_time=float(total_movement_time),
        peak_velocity=float(peak_velocity),
        time_to_peak_velocity=float(t_peak),
        time_to_peak_velocity_percent=float(t_peak_pct),
        time_to_first_peak_velocity=float(t_first),
        time_to_first_peak_velocity_percent=float(t_first_pct),
        number_of_movement_units=int(n_units),
        max_trunk_displacement=max_trunk_displacement,
    )
