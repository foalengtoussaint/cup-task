# cup-task — pipeline specification

**What this is:** how the pipeline works — stages, signals, definitions, and the rules each boundary
uses. No results here; every measured number lives in [RESULTS.md](RESULTS.md).

The task is the **iARAT drink task**: reach for a glass, lift it, drink, put it back, return the hand
to rest. The pipeline turns synchronised multi-camera video into the 3D tracks, movement phases, and
Murphy clinical measures for one repetition — with **no motion capture**.

---

## 1. Stages

The pipeline splits at **the point where a stage stops needing raw pixels**.

```
 ═══ ONLINE — per frame, N cameras, while the pixels are in hand ════════════════════════
        (names under each box = the cup_task module that implements it)
  ┌──────────────┐
  │ N × capture  │  1080p @60fps, BGR→RGB once per frame
  └──────┬───────┘  (shared by every consumer below)
         │
         ├──────────────────────┬─────────────────────────┬──────────────────────┐
         ▼                      ▼                         ▼                      ▼
  ┌─────────────┐      ┌──────────────────┐      ┌───────────────┐      ┌──────────────┐
  │ YOLO-pose   │      │ YOLO cup DETECT  │      │ PyrLK flow    │      │ PyrLK flow   │
  │ batched N   │      │  — first frame   │      │ @ wrist px    │      │ @ cup px     │
  │             │      │      only —      │      │               │      │              │
  │             │      │        ↓         │      │ CPU threads,  │      │ CPU threads  │
  │             │      │ UETrack batched  │      │ overlaps GPU  │      │              │
  │             │      │ N (every frame)  │      │               │      │              │
  └──────┬──────┘      └────────┬─────────┘      └───────┬───────┘      └──────┬───────┘
   pose_keypoints          cup_detect               flow_speed            flow_speed
         │                 cup_track                     │                      │
         ▼                      ▼                        ▼                      ▼
    2D keypoints          2D cup points        2D wrist Δpx/frame      2D cup Δpx/frame
         │                      │                        │                      │
 ════════╪══════════════════════╪════════════════════════╪══════════════════════╪═══════
         │      OFFLINE — per trial, numbers only (no pixels, no GPU needed)     │
         ▼                      ▼                        ▼                      ▼
  ┌─────────────┐      ┌──────────────────┐      ┌──────────────────────────────────┐
  │ triangulate │      │ greedy consensus │      │ solve u̇ = J(X)·v across cams,   │
  │ (DLT, ≥2    │      │ ≥2 cams, 150mm   │      │ L1/IRLS → 3D VELOCITY VECTOR     │
  │  cameras)   │      │ continuity gate  │      │ (never differentiates position)  │
  └──────┬──────┘      └────────┬─────────┘      └────────────────┬─────────────────┘
    triangulate            consensus                         flow_speed
         ▼                      ▼                                 │
   pose 3D (mm)           cup 3D (mm)                             │
         │                      │                                 │
         ▼                      ▼                                 │
  ┌─────────────┐      ┌──────────────────┐                       │
  │ SmoothNet   │      │ SmoothNet        │                       │
  │ window 32   │      │ (same filter,    │                       │
  │ per joint,  │      │  cup track)      │                       │
  │ centred     │      │                  │                       │
  └──────┬──────┘      └────────┬─────────┘                       │
    pose_smooth            pose_smooth                            │
         │                      │                                 │
         ├──────────────────────┼─────────────────────────────────┤
         ▼                      ▼                                 ▼
  ┌────────────────────────────────────┐              ┌────────────────────────┐
  │ SEGMENTATION → 7 phases            │              │ speed BLEND            │
  │  segment_cup_only                  │              │ sigmoid gate on speed: │
  │  → refine_grasp_with_pose          │              │  slow → flow           │
  │  → to_murphy_phases                │              │  fast → SmoothNet      │
  └────────────────┬───────────────────┘              └───────────┬────────────┘
              segment                                       speed_blend
                   └──────────────────┬───────────────────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │ MURPHY SCORING               │
                       │ 8 position measures          │
                       │ peak_velocity ← the blend    │
                       └──────────────┬───────────────┘
                                    score
                                      ▼
                              per-trial measures
```

| stage | module | where | why there |
|---|---|---|---|
| decode, YOLO-pose | `pose_keypoints` | online | needs the frame; is the capture loop |
| cup detect (once) + UETrack | `cup_detect`, `cup_track` | online | same |
| PyrLK flow (wrist + cup) | `flow_speed` | online | needs the frame **pair** — see §5 |
| triangulation | `triangulate` | offline | geometry on 2D points |
| cup consensus | `consensus` | offline | same |
| flow 2D→3D velocity | `flow_speed` | offline | geometry on the online flow vectors |
| SmoothNet (pose **and** cup) | `pose_smooth` | offline | **non-causal**: a ±16-frame window needs 0.27 s of future |
| speed blend | `speed_blend` | offline | needs SmoothNet |
| segmentation | `segment` | offline | a phase is an interval — needs the whole trial |
| Murphy scoring | `score` | offline | needs the phases |

---

## 2. The two 3D targets

**Pose** — YOLO-pose per camera → DLT triangulation (≥2 cameras) → SmoothNet.

**Cup** — YOLO detects the cup **once**, on the first frame it can; UETrack then carries it for the
rest of the trial with no re-detection. Per-camera tracked points go through a **greedy
biggest-agreeing-subset consensus** (minimum 2 cameras, with a 150 mm temporal-continuity gate that
rejects a spurious 2-camera agreement far from the previous point). Re-anchoring the tracker from the
consensus is **off**: it lets one bad camera poison the good ones.

The cup gets the **same SmoothNet filter** as the pose. This is easy to miss —
`pose_smooth.smooth_tracks` excludes the cup by default because the cup has its own tracker — but a
tracked cup carries ~1 mm of per-frame positional wobble, which is invisible in position and becomes
~60 mm/s once differentiated.

---

## 3. Speed: why there are two sources

Differentiating a position track amplifies its noise. Optical flow measures the pixel displacement
between two frames **directly**, so it never sees that noise:

```
X       = triangulate{p_c}                    # where the target is, this frame
u̇_c = J_c(X) · v                              # what camera c would see if it moved at v
v3d(t)  = ‖ argmin_v Σ_c ‖J_c(X)·v − flow_c‖ ‖ × fps      # L1/IRLS, no threshold
```

It is a **velocity measurement, not a position derivative** — nothing is differenced across time,
and common-mode calibration error cancels because every camera is referred to the same `X`.

**Why the Jacobian rather than differencing two triangulations** (`triangulate{p+flow} −
triangulate{p}`, the pre-2026-07-22 form, still available as `fuser="dlt2"`): a camera looking
*along* the direction of motion sees almost no pixel displacement, and the two-triangulation form
weights its near-zero, noise-dominated reading equally with a camera looking across the motion.
`J` encodes exactly that visibility. Viewing geometry is the **largest** driver of per-camera flow
error — a camera looking along the motion has ~2× the relative error — so this is the one estimator
that models the dominant term. Measured 20.53 vs 21.71 mm/s.

**Why L1 rather than least squares:** least squares has breakdown point 0, so one camera reading an
occluder's motion moves the answer. IRLS with `w = 1/‖r‖` minimises `Σ‖r‖` (a geometric-median-like
solution), capping each camera's *influence* by its own error. It replaced a leave-one-out
consensus gate that did the same job with two tuned constants; they measure the same on this cohort,
so the tie-break was the absence of knobs. See RESULTS §10.

**Which cameras, and which points:**

- **Points are the RAW DETECTED pixels** — YOLO's keypoint, UETrack's tracked cup point. Never a
  reprojection of the 3D consensus: flow must be measured where the image evidence is, and a
  reprojected point would inherit the triangulation's own error, defeating the purpose of measuring
  velocity independently of position.
- **Cameras are gated by the consensus** (`gate_consensus=True`, default). Only cameras the
  geometric consensus *keeps* on that frame contribute. Without it a camera tracking the wrong
  object contributes its flow vector at full weight — the consensus exists to reject exactly those.

That gate fixes errors **while the target moves** but not **at rest**, and the reason is worth
knowing: at rest the consensus rejects ~0 cameras (they all agree *where* the cup is) yet each still
reports a sub-pixel spurious *motion* (p95 ≈ 0.9 px), which triangulates into tens of mm/s. The two
failure modes are orthogonal — geometric disagreement is gateable, sub-pixel flow noise is not.

The two sources fail in complementary regimes:

| | flow (PyrLK) | SmoothNet (d position/dt) |
|---|---|---|
| off-peak / at rest | **clean** — direct velocity | phantom speed from differentiation |
| at the peak | over-shoots — motion **blur** smears the patch | **accurate** — position is unblurred |

So the wrist speed is a **speed-gated blend**:

```python
wb    = 1 / (1 + exp(-(flow_speed - 350) / 120))   # sigmoid on the CURRENT speed
blend = (1 - wb) * flow + wb * smoothnet
```

The gate is soft on purpose: a hard threshold injects a discontinuity exactly where peak-detection
looks. ⚠ **350/120 are hand-set** on P07+P08 — `speed_blend.fit_gate` exists to LOPO-tune them.

**Which signal drives what:**

| use | signal | reason |
|---|---|---|
| wrist speed reporting (`peak_velocity`) | **blend** | best at the peak *and* off-peak |
| cup speed reporting | **flow** | best where the cup actually moves; the blend adds nothing here |
| **segmenter gates** | **SmoothNet** | flow's rest-speed **tail** trips a 15 mm/s threshold before the cup moves; and the segmenter needs a position track, which flow cannot give |

---

## 4. Segmentation — the 7 phases

Definitions follow **van Andel et al.** (PMC5933268, Table 1), the source protocol for the drink task.

| phase | starts when | signal |
|---|---|---|
| `rest_pre` | trial start | — |
| `reaching` *(includes grasping)* | hand velocity exceeds 2 % of peak | hand |
| `forward_transport` | **glass velocity > 15 mm/s** | cup |
| `drinking` | face–glass distance < 15 % of steady state | cup + head |
| `back_transport` *(includes release of grasp)* | glass leaves the mouth | cup |
| `returning` | **glass velocity < 10 mm/s** (glass is down) | cup |
| `rest_post` | hand velocity back to 2 % of peak | hand |

Two consequences that are easy to get backwards:

- **`reaching` INCLUDES the grasp** — it ends when the *glass* starts moving, not when the hand
  arrives. There is a real ~370 ms gap between the two (the hand settles its grip before lifting).
- **`back_transport` INCLUDES the release** — it ends when the *glass* stops, not when the hand
  lets go.

### How the boundaries are found

Three passes, in order:

1. **`segment_cup_only`** — cup speed + displacement-from-rest give a first transport window and the
   drink dwell (cup near the mouth **and** slow). The drink gates (`DRINK_SPEED`/`DRINK_DISP_PAD` =
   150/150) were LOPO-tuned; do not restore the older 120/90.
2. **`refine_grasp_with_pose`** — **replaces both transport boundaries**. While the hand holds the
   cup they are one rigid body, so the **wrist→cup distance** is flat; it *closes* during the reach
   and *opens* at the release. The grasp is the end of the one big closing run; the release is the
   start of the one big opening run. Scale-free — it needs no threshold on the distance itself, only
   that a run travels ≥30 % of the observed span. **This stage is not optional.**
3. **`to_murphy_phases`** — splits the 5 cup phases into the container's 7, adding `reaching` and
   `returning` from the hand's direction of travel.

**Why the plateau and not a cup rule.** By set-down the cup displacement has already flattened into a
noise band, so a cup-only rule is choosing between wobbles; the wrist→cup distance meanwhile ramps
~150 → ~450 mm and both tracks ride that ramp together. In mocap the pre-release wobble *is* the
hand — it collapses ~17× the instant the hand lets go — but a video tracker's own noise floor is
comparable to that wobble, so **for the tracker there is no contrast at the release to detect**. The
event is a hand event; look at the hand.

---

## 5. Online / offline placement

The rule: a stage is online only if it needs raw pixels **and** running it there is cheaper.

- **Flow is online** because it needs the frame *pair*. Online, both frames are already decoded and
  the work overlaps the GPU on a CPU thread pool. Offline it would force a **second full decode** of
  every camera — several times the entire offline budget.
- **SmoothNet is offline and cannot move.** It is a symmetric ±16-frame window filter, so producing
  frame *t* requires frame *t+16*: online it would add ~0.27 s of latency and still not be faster.
- **Segmentation and scoring are offline by definition** — a phase is an interval.

---

## 6. Conventions

- **Units** are millimetres and mm/s throughout. DELTA calibration ships translations in metres and
  is rescaled on load (`_load_calib_mm`).
- **Displacement** means **origin-relative**: `d(t) = ‖X(t) − X(t₀)‖`, distance travelled from the
  track's own start. Like speed it is rotation-invariant, so it needs no rigid alignment and does not
  inherit the rig↔mocap calibration offset. A per-axis MMC−OMC difference is **not** meaningful: the
  two frames are rotated relative to each other.
- **Smoothing order**: filter POSITION, then differentiate. Filtering velocity afterwards smears
  spikes but leaves them dominating the peak.
- **Filters must be NaN-safe.** `filtfilt` propagates a single NaN across its whole output, so gaps
  are interpolated before filtering and restored after.
- **SmoothNet windows are mean-centred** before the forward pass and re-offset after. The pretrained
  h36m model expects root-relative metres near the origin; absolute world coordinates are
  off-distribution and its learned bias then translates the whole track.

## 7. Where things live

| | |
|---|---|
| `cup_task/` | the pipeline modules (the boxes in the schema) |
| `scripts/` | active harnesses: `results_v3_delta.py` (metrics), `bench_v3.py` (speed), `cup_flow_probe.py` |
| `scripts/archive/` | settled investigations, kept — see its README |
| `models/` | pose + SmoothNet + UETrack weights (repo-persistent on purpose) |
| `cache/` | cached detections/tracks/flow — the whole offline path runs from here with no GPU |
| `archive/` | settled outputs and checkpoints, kept — see its README |
| [RESULTS.md](RESULTS.md) | every measured number |
| [WORKLOG.md](WORKLOG.md) | chronological record, including what was tried and rejected |
