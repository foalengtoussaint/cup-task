# Movement-quality measures — how each is computed

This document defines every Murphy movement-quality measure reported in the MMC↔OMC validation
(Figure 4 / Table III), exactly as our scorer computes it. **All measures are computed from OUR
markerless 3D pose keypoints only.** AutoMQ (the DELTA study's optical-mocap processing) is used
*only* as the ground-truth value to correlate against, never inside our computation.

Two ground rules carried through the whole pipeline:
- **Smooth position, then differentiate.** Velocity/angular-velocity are derived from a low-passed
  *position/angle* signal, not by filtering a raw derivative (which smears spikes but leaves them
  dominant). Butterworth, zero-phase (`filtfilt`), NaN-safe (gaps interpolated → filtered → gaps
  restored, so two missing frames can't null a whole trial).
- **Windows come from phases.** Peak/interjoint/angle measures are scoped to the `reaching` and/or
  `drinking` phase. For the pose-isolated table these phases are AutoMQ's OMC phases mapped onto the
  video timeline (100 Hz → 60 Hz + wrist-speed lag), so both systems see the *same* window and the
  comparison isolates pose error from phase-classification error.

Pipeline: **BA + SmoothNet** — robust-reprojection bundle adjustment (Huber, `λ_bone = 0`, refines
all 9 joints) → SmoothNet temporal smoother → measures. `fps = 60`.

| constant | value | note |
|---|---|---|
| low-pass (position→speed) | 4 Hz | video-tuned; mocap notebook used 6 Hz |
| low-pass (peak_velocity operator) | 6 Hz on the *speed* | AutoMQ's own operator (diff first, then 6 Hz) |
| Butterworth order | 2 | zero-phase |
| movement-unit amplitude | 60 mm/s | video jitter floor ~30–50 mm/s (mocap used 20) |
| movement-unit time gap | 0.15 s | |

---

## Velocity / timing / smoothness (position-derived)

Source of the wrist speed: `s(t) = ‖d/dt · lowpass(wrist_xyz, 4 Hz)‖` in mm/s.

### peak_velocity — mm/s
`max( lowpass( |diff(raw wrist)|·fps, 6 Hz ) )` over the **reaching** window only.
Uses AutoMQ's exact operator (differentiate the *raw* wrist, then low-pass the *speed* at 6 Hz) —
verified strictly better than low-pass-position-4Hz-then-diff on a 685-trial sweep
(bias −26.8→−17.9, |err| 46.2→44.4, r 0.81→0.90). Scoped to reaching, not the whole trial (a fast
back-transport motion was a spurious "peak" AutoMQ never sees). Residual ≈ −18 mm/s bias = the
genuine markerless-vs-marker + 60-vs-100 Hz effect.

### time_to_peak_velocity — s
`(reach_start + argmax(speed[reaching])) / fps`. **Measured from frame 0 (trial start)**, not from
reach-start — AutoMQ stores it that way, and measuring from reach-start collapsed r to 0.21 (the
per-trial-varying ~0.7 s pre-reach dominated).

### time_to_first_peak_velocity — s
First *prominent* local speed maximum in reaching (`find_peaks`, prominence = max(50, 0.15·PV), on a
harder 3 Hz low-pass). A single-peaked healthy reach has no interior local max, so it falls back to
the argmax. Timed from frame 0, same as above.

### number_of_movement_units — count
Smoothness proxy: count of min→max speed oscillations where the rise exceeds 60 mm/s within ≥0.15 s,
over reaching + forward_transport + back_transport + returning (drinking excluded). This is a noisy
measure by nature (r_s ≈ 0.48); AutoMQ's authors flag their own implementation as not fully settled
and suggest log-dimensionless-jerk as a more reliable alternative.

### total_movement_time — s
`(returning_end − reaching_start) / fps` (falls back through rest_post/returning end). **Phase-derived**
— because the pose-isolated table uses OMC phases for *both* systems, this measure is near-tautological
(r_s = 1.00) and is reported as a consistency check, not an independent validation of pose.

---

## Trunk

### max_trunk_displacement — mm
Robust-reference **3D excursion** of the shoulder midpoint:
`p98( ‖sh_mid(t) − median(sh_mid[first 15 frames])‖ )`, where `sh_mid = (L_shoulder + R_shoulder)/2`.

This is *not* an anatomical forward-projection. The forward normal (side × down) uses the occluded
seated hips, so it collapsed to ~0 on ~half of some participants' trials (P07: forward-projection
r_s = 0.06 while the raw motion matched OMC 1:1 — "moves as much but uncorrelated" was the tell).
The 3D excursion works because drink-task trunk motion is dominated by forward lean; cohort r_s
0.65 → 0.93, P07 0.06 → 0.98. Cost: a small over-read (+~12 mm) from including lateral/vertical sway.

---

## Angles (geometric, no IK)

All angles are computed from three 3D keypoints — **the same geometric construction AutoMQ uses**
(AutoMQ computes angles geometrically from raw markers; there is **no** MuJoCo / OpenSim IK on the
ground-truth side of *this* validation, so the definitions are directly comparable). Each angle series
is low-passed before reduction.

### max_elbow_angle (elbow extension) — deg
Angle at the elbow from (shoulder→elbow) and (wrist→elbow):
`arccos( û·v̂ )`, `u = shoulder−elbow`, `v = wrist−elbow`. Whole-trial **max**.
Reported as extension (larger = straighter). r_s ≈ 0.86.

### peak_elbow_angular_velocity — deg/s
`max( |gradient(elbow_angle)|·fps )` over **reaching**. r_s ≈ 0.83.

### max_shoulder_flexion — deg
Angle of the upper-arm vector below the trunk axis: `arccos( arm · down_c )`,
`arm = (elbow−shoulder)`, `down_c` = a **per-trial CONSTANT** trunk-down axis (median of
(hip_mid − shoulder_mid) over finite frames). Reduced by **max** over the reaching→drinking span.

Why constant, not per-frame: participants are seated with **occluded hips**, so a per-frame
hip-based down axis is pure jitter and flexion trial-corr collapses to 0.16; the seated trunk axis
is ~static, so the per-trial median lifts it to r_s ≈ 0.47. The formula itself is correct (corr +1.00
vs AutoMQ on clean markers) — the residual weakness is a per-participant offset from the occluded
seated-hip keypoint being placed ~150 mm anterior (at the waist). This is the weakest angle measure
and is reported honestly as such.

### max_shoulder_abduction — deg
Component of the arm along the **lateral** shoulder-line direction:
`arcsin( arm · side )`, `side = (R_shoulder − L_shoulder)` normalized (the arm's-own shoulder pointing
outward). Reduced by **max** over reaching→drinking. The lateral (not medial) direction is critical —
medial sign-flipped the measure (corr −0.91, −55.6° bias); lateral gives corr +0.89 vs AutoMQ. r_s ≈ 0.61.

### interjoint_coordination — Pearson r
`corr( shoulder_flexion , elbow_angle )` over the **inner 80% of reaching** (crop 10% off each end —
the leading/trailing 10% straddle the pre-reach and grasp pauses, which are zero-variance / opposite-trend
samples). Reproduces AutoMQ's stored value on 84% of trials within 0.05.

**Reporting note:** in a healthy-dominant cohort the OMC value is near-constant (~1.0), so a rank
correlation across trials has almost no true variance to track and reads low (r_s ≈ 0.25). The honest
framing is *agreement* not rank: MMC agrees on 76% of trials (bias −0.03); the tail is concentrated in
P07/P14 and is driven by shoulder-flexion shape error (smoothing flexion alone recovers 76%→89%;
smoothing elbow alone does nothing).

---

## Measures NOT reported (and why)

- **time_to_peak_velocity_percent, time_to_first_peak_velocity_percent** — AutoMQ stores them, but
  Unger et al. do not validate them (phase duration is identical for MMC and OMC in the pose-isolated
  setup, making the percent variants redundant with the seconds versions). Dropped to match the paper.
- **shoulder_flexion (drinking phase)** — Unger's 12th panel. AutoMQ does not store a separate
  drinking-phase flexion column, so there is no ground-truth value to validate against, and computing
  one would require deriving an OMC quantity (excluded by rule). Hence our figure has 11 panels, not 12.

---

## Summary — final BA + SmoothNet agreement (pose-isolated, OMC phases)

| measure | r_s | note |
|---|---|---|
| total_movement_time | 1.00 | phase-derived (circular) |
| trunk_displacement | 0.93 | 3D excursion |
| time_to_peak_velocity | 0.89 | |
| time_to_first_peak_velocity | 0.88 | |
| max_elbow_angle (extension) | 0.86 | |
| peak_velocity | 0.86 | |
| peak_elbow_angular_velocity | 0.83 | |
| shoulder_abduction | 0.61 | |
| number_of_movement_units | 0.48 | intrinsically noisy |
| shoulder_flexion | 0.47 | seated-hip offset floor |
| interjoint_coordination | 0.25 | OMC near-constant; report as agreement (76%) not rank |
