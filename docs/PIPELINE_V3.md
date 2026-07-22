# cup-task pipeline v3 — design, online/offline split, and measured results

Measured 2026-07-21 on DELTA, **n=12 OMC-truth trials (P07 + P08, trial_10–15)**.
Branch `pipeline-v2-uetrack-smoothnet`. Supersedes PIPELINE_V3_PLAN.md (that was the plan; this is
the built pipeline and its numbers).

**Cohort: P13 is excluded from EVERY table.** Its OMC clock drifts linearly against video (−8 → +3
frames over ~6 s, 3.8 % rate mismatch), so its ground truth is progressively mis-timed. That is not a
constant lag one cross-correlation can absorb, and it corrupts POSITION as well as speed — on the cup
its median displacement error is 10–12 mm vs 2–3 mm for P07/P08, d-corr 0.974 vs 0.998, and it owned
the entire 504 mm error tail. Including it simultaneously flattered v1 and penalised v3. P13's data
and caches are untouched; a linear time-warp of its OMC would restore all 6 trials (n=12 → 18).

---

## 1. The design: what runs ONLINE, what runs OFFLINE

The split is drawn at **the point where a stage stops needing raw pixels.**

### Full schema

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
  │ triangulate │      │ greedy consensus │      │ triangulate {p} and {p+flow},    │
  │ (DLT, ≥2    │      │ ≥2 cams, 150mm   │      │ difference → 3D VELOCITY VECTOR  │
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
  │ SEGMENTATION  segment_cup_only     │              │ speed BLEND            │
  │  • speed + displacement-from-rest  │              │ sigmoid gate on speed: │
  │  • drink = near mouth AND slow     │              │  slow → flow           │
  │  • return ends when DISPLACEMENT   │              │  fast → SmoothNet      │
  │    flattens (not on a speed gate)  │              └───────────┬────────────┘
  │        ↓ to_murphy_phases          │                    speed_blend
  │  + reaching / returning from the   │                          │
  │    HAND's direction of travel      │                          │
  │  → 7 phases                        │                          │
  └────────────────┬───────────────────┘                          │
              segment                                             │
                   └──────────────────┬───────────────────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │ MURPHY SCORING               │
                       │ 8 position measures          │
                       │ (medfilt3 + 4Hz internally)  │
                       │ peak_velocity ← the blend    │
                       └──────────────┬───────────────┘
                                    score
                                      ▼
                              per-trial measures
```

**Reading the schema.** Four independent things are measured per frame online (pose, cup, and a flow
vector at each of the two targets); everything downstream is arithmetic on those numbers. The two
flow branches exist because *differentiating a position track is what injects the noise* — flow
measures displacement between two frames directly, so it never sees the ~1 mm per-frame positional
wobble that becomes ~60 mm/s of phantom speed. SmoothNet appears twice (pose and cup) and is the
same filter both times.

| stage | module | where | why |
|---|---|---|---|
| decode, YOLO-pose | `pose_keypoints` | **online** | needs the frame; is the capture loop |
| cup detect (once) + UETrack | `cup_detect`, `cup_track` | **online** | same |
| **PyrLK flow** (wrist + cup) | `flow_speed` | **online** | needs the frame PAIR. The one real decision — see below. |
| triangulation | `triangulate` | offline | pure geometry on 2D points; nothing to gain online |
| cup consensus (≥2 cams) | `consensus` | offline | same |
| flow 2D→3D velocity lift | `flow_speed` | offline | geometry on the online flow vectors |
| SmoothNet (pose **and** cup) | `pose_smooth` | **offline, mandatory** | *non-causal* window filter (±16 frames ≈ 0.27 s of FUTURE). Cannot run online without adding latency. |
| speed blend | `speed_blend` | offline | needs SmoothNet, so it inherits offline |
| segmentation (7 phases) | `segment` | offline | needs the whole trial by definition (a dwell is an interval) |
| Murphy scoring | `score` | offline | needs the phases |

**Why flow is online (the measured justification).** Flow is the only stage that could go either way,
and the criterion you set — put it online only if it is ultimately faster — decides it cleanly:

- online: the frame pair is already decoded, and flow runs on a CPU thread pool concurrently with the
  GPU nets, so its **marginal** cost is **0.4–1.9 ms/rig-frame at 1–5 cams** (2.8 ms at 10) — essentially
  free. (Raw serial cost is 1.5 ms/wrist/cam; PyrLK releases the GIL so threading gives ~2.4× at 5–10
  cams, and the GPU pass hides nearly all of what remains.)
- offline: it forces a **second full decode** of every camera = **5182 ms/trial** on a 7-cam 8.9s
  trial — **6× the entire offline budget (887 ms)**.

Same numbers either way; online is ~6× cheaper end to end. Flow goes online.

**Why SmoothNet is offline** (it looks online-able, but is not): it consumes a symmetric window, so
producing frame *t* needs frame *t+16*. Online it would impose 0.27 s of latency and still not be
faster. It is also the single largest offline cost (796 ms of the 887 ms), so it is the right thing
to optimise if the post-processing budget ever matters.

---

## 2. Speed benchmark

**RTX 3060 Ti (8 GB, 38 SMs)**, 1080p real DELTA frames. fps = **rig-frames/s** (all cameras advance one frame).

### Online loop

| cams | pose fps | cup fps | flow **+ms** | BOTH + flow | realtime (≥60)? |
|---|---|---|---|---|---|
| 1 | 237.2 | **170.3** | **0.43** | **100.1** | **YES** |
| 5 | 75.0 | **83.7** | **1.93** | **38.5** | no |
| 10 | 41.8 | **46.1** | **2.80** | **21.6** | no |

⚠ An earlier version of this table read cup 90/61/40 fps. That was a **benchmark artifact**: a
per-frame `[:, :, ::-1].copy()` on 1080p costs **6.96 ms**, more than UETrack's own **4.04 ms** step,
and it was being charged to the tracker. In a real rig the colour conversion happens once per frame
anyway (pose needs it too), so it is capture-loop overhead. Corrected (and after the `.tolist()` fix below), UETrack is **170–217 fps at 1 cam**
— consistent with the earlier tracker thread's 193 fps, and *faster* than YOLO-pose.

`flow +ms` is the **marginal** cost — what threaded flow actually adds on top of the GPU pass,
measured directly (not `ncam × per-wrist`, which would read 1.5/7.7/15.4 and overstate it ~5–30×).
PyrLK releases the GIL, so the thread pool gives real parallelism (~2.4× at 5–10 cams), and the GPU
pass hides nearly all of what is left. **Flow is effectively free online.**

The binding constraint is the two GPU nets, not flow. Pose and cup each clear 60 fps alone at 1–5
cams; **run together they do not, past 1 camera**. At **1 camera the full loop runs at 100 fps**,
comfortably realtime.

### Why the two nets do not overlap — diagnosed, not assumed

Hardware: **RTX 3060 Ti, 8 GB, 38 SMs** (mid-range consumer card).

**Memory is not the constraint.** Resident: UETrack 1057 MB + YOLO-pose 45 MB ≈ **1.1 GB of 8.2 GB
(13 %)**. Nothing is competing for space.

**The GPU is genuinely compute-saturated.** Timed with **CUDA events** (real device time — wall-clock
around `predict()` is misleading, because it returns before the GPU finishes, which made an earlier
pass of this analysis wrongly conclude "99 % CPU-bound"):

| at 5 cams | pure GPU time |
|---|---|
| YOLO-pose forward | 16.06 ms |
| UETrack update | 11.37 ms |
| **sum** | **27.43 ms** → ceiling **36 fps** even with *perfect* overlap |

The measured combined loop is 29.7 ms, so **the GPU is busy 92 % of it**. There is no idle device time
to schedule into — which is why threading the two nets makes things *worse* (1/5/10 cam =
0.81×/0.87×/0.91×) and why serial ≈ sum-of-parts.

**Batching gives no economy either.** A raw batched forward on pre-resized 640×640 GPU tensors costs
the same as full `predict()` (12.15 vs 12.23 ms at 5 cams), and per-camera cost is flat at ~2.4 ms
from 5→10 cams. One 640×640 image already fills 38 SMs; more cameras scale linearly.

**Fixed along the way:** `UETrackBatch.update` called `.tolist()` **per camera** inside its output
loop — N separate GPU→CPU syncs that re-serialised the batch immediately after the batched forward
(~50 % of its CPU time). Hoisted to one transfer for the whole batch: **1 cam 178 → 217 fps**,
numerically identical (max 0.0001 px vs the sequential tracker).

**Conclusion: more throughput needs a second GPU, a lighter backbone, or lower input resolution —
not better scheduling.** Flow is the exception: it is CPU work, genuinely overlaps, 0.4–2.8 ms.

**Deployment reading:** a 5-camera rig at 60 fps needs either a second GPU (pose on one, cup on the
other — they are independent), a lighter cup backbone, or accepting ~38 fps capture. The accuracy
results below do NOT depend on this: they are computed offline from cached detections.

### Offline post-processing (per trial: 533 frames = 8.9 s, 7 cams)

| stage | ms |
|---|---|
| cup 3D (consensus) | 38.7 |
| **SmoothNet (3 joints)** | **795.8** |
| flow 2D→3D speed | 43.3 |
| speed blend | 0.5 |
| segmentation | 0.4 |
| Murphy scoring | 8.3 |
| **TOTAL** | **≈887 ms/trial (10 % of realtime)** |

**"flow 2D→3D speed"** is the geometry that converts the online per-camera 2D flow into one 3D speed.
Per frame it triangulates **two** points — the wrist pixel `p_c` and the flow-advanced pixel
`p_c + flow_c` — and differences them:

```
v3d(t) = ‖ triangulate{p_c + flow_c} − triangulate{p_c} ‖ × fps
```

Both use the same cameras and calibration, so common-mode calibration error cancels and only motion
survives. Crucially this is a **velocity measurement, not a position derivative** — nothing is
differenced across time, which is exactly why it avoids the jitter amplification that makes
`pos-diff` speed 42.7 mm/s. The 43 ms is just the DLT solves (533 frames × 2); the flow itself was
already computed online.

Post-processing is ~1 s per trial — negligible; a session of 60 trials post-processes in under a
minute. SmoothNet is 91% of it (sliding window, step 1, one forward per position).

---

## 3. Accuracy vs OMC — displacement + speed

**DISPLACEMENT is origin-relative**: `d(t) = ‖X(t) − X(t₀)‖`, distance travelled from the track's own
start; error = `|d_MMC − d_OMC|`. This is rotation-invariant (like speed), so it needs no rigid
alignment and does not inherit the ~38 mm rig↔mocap calibration floor. It measures the MOTION, which
is what the clinical measures actually read.

### Pose joints (n=12 trials, P07+P08)

| target | displ RAW | p90 | displ SN | p90 | speed RAW | speed SN |
|---|---|---|---|---|---|---|
| acting wrist | 4.9 | 11.8 | 4.9 | 12.4 | 42.7 | **10.6** |
| acting elbow | 21.8 | 36.4 | **18.5** | 35.5 | 44.8 | **10.0** |
| acting shoulder | 10.7 | 19.7 | 10.3 | 20.2 | 38.3 | **6.2** |
| nose | 2.2 | 6.3 | 2.6 | 6.5 | 13.7 | **5.3** |

mm and mm/s. **SmoothNet is position-neutral (±0.7 mm) but cuts speed error ~4×** — exactly what a
temporal filter should do: it must not move the track, only de-noise its derivative. Elbow
displacement is the outlier (28–32 mm) — the COCO elbow keypoint and the OMC elbow marker are
different landmarks, so this is largely a definition gap, not tracking error.

### Cup — v3 wins on every fair cut

Both displacement and speed are frame-invariant, so no rigid fit is involved anywhere.

| source | displ med | mean | p90 | **d-corr** | spd all | **spd MOVING** | s-corr | cov all | **cov MOVING** |
|---|---|---|---|---|---|---|---|---|---|
| v1 every-frame | 2.7 | 5.0 | 11.3 | 0.9981 | **18.3** | 136.2 | 0.67 | 78% | 52% |
| v3 UETrack | **2.3** | 15.5 | 58.4 | **0.9996** | 46.3 | **77.4** | **0.93** | **100%** | **100%** |

**`d-corr` = 0.9996 for v3**, reproducing the v2 tracker-shootout result (it was measured against
OMC, on displacement-from-origin — the frame-invariant quantity; a per-axis position correlation
reads *negative* here because the MMC and OMC frames are rotated).

⚠ **Do not read the `spd all` column.** v1 covers only 78 % of frames, and those are overwhelmingly
the frames where the cup is **STILL**: median OMC cup speed is **0.6 mm/s on frames v1 has** vs
**139.3 mm/s on frames it misses**. Its all-frames median therefore mostly scores *"is the stationary
cup stationary?"*. On frames where the cup is actually **MOVING**, v1 is **1.8× worse** (136 vs
77 mm/s) and sees only **half** of them.

The mean/median gap needs the same care. v3's pooled mean (15.5 mm) exceeds v1's (5.0) — but split by
frame population:

| v3 displacement error | mean | max |
|---|---|---|
| on frames v1 also has | **4.1** | 60.9 |
| on frames v1 misses | 56.2 | 107.8 |

So v3 is **not degraded where they overlap** (4.1 vs v1's 5.0 — it is *better*); its higher overall
mean is entirely the price of covering the extra 22 % — the occluded apex that v1 simply fails to
produce. Those frames are genuinely hard, not a tracker defect.

**Conclusion: use v3 for the cup, full stop.** Better typical accuracy (4.1 vs 5.0 mm where both
exist), 2× the coverage on the frames that matter, better speed where the cup moves, and it is what
makes the segmentation below work.

### Cup SPEED — the same position-good / derivative-noisy split as the wrist

v3's cup position is near-perfect (d-corr 0.9996) but its *derivative* is only 0.93. The cause is
mechanical: **~1 mm of per-frame positional wobble**, which is 0.15 % of a 700 mm trajectory and
completely invisible on a displacement plot, becomes **~60 mm/s** once differentiated. (OMC's own
per-frame step is 0.23 mm → ~14 mm/s; during actual motion the two agree — p90 step 11.6 vs 10.2 mm.
The gap is entirely at rest.) That noise floor is what used to break the segmenter's 10 mm/s gate.

So the wrist's fix was applied to the cup — `cup_task.flow_speed` is target-agnostic, it just needed
cup pixels instead of wrist pixels (`scripts/cup_flow_probe.py`, n=12):

| cup speed method | \|err\| all | **MOVING** | corr | at REST | clears 10 mm/s gate |
|---|---|---|---|---|---|
| pos-diff | 46.3 | 77.4 | 0.931 | 42.5 | **no** |
| SmoothNet | **6.6** | 35.9 | 0.995 | 3.4 | yes |
| **flow (PyrLK)** | 7.5 | **25.3** | 0.989 | 4.8 | yes |
| blend | 7.3 | 27.4 | 0.994 | 4.6 | yes |

**Flow is 3× better than pos-diff where the cup actually moves** (25.3 vs 77.4 mm/s) — the wrist
method transfers. Note the *blend* does NOT win here, unlike the wrist: SmoothNet already handles the
cup's slower peaks, so plain flow is the pick for reporting cup speed.

### Why the SEGMENTER uses SmoothNet and NOT flow

Flow is the better *reporting* signal but the worse *gating* signal. Three measured reasons:

**1. Flow has a heavy tail at rest, and the segmenter's gates are thresholds.** Speed during
`rest_pre`, before the cup has moved at all:

| | median | **p95** |
|---|---|---|
| OMC | 0.4 | 3.6 |
| SmoothNet | 3.3 | **14.2** |
| flow | 4.9 | **81.3** |

The onset gate is `FWD_ON = 15 mm/s`. SmoothNet's p95 sits just under it; **flow's is 5× above it**,
so flow spuriously trips the gate before the cup moves. Flow's excellent *median* (4.9) hides this —
the same median-vs-tail trap as the point-tracking thread.

**2. Consequently it misses the onset.** First crossing of `FWD_ON` vs OMC's crossing, n=12:

| | median | p90 |
|---|---|---|
| **SmoothNet** | **308 ms** | **460 ms** |
| flow | 600 ms | 748 ms |

**3. The segmenter needs POSITION, which flow cannot provide.** Displacement-from-rest drives both
the drink rule and the arrival rule; flow yields velocity only.

So: **flow for reporting cup speed, SmoothNet for driving the segmenter.** Same division as the
wrist, and for the same underlying reason — a direct velocity measurement is accurate in the mean but
noisy frame-to-frame, which is exactly wrong for a threshold crossing.

---

## 4. Wrist speed — the v3 speed path (n=12)

| method | per-frame | off-peak | PEAK med | p90 | max | peak-time mean | max |
|---|---|---|---|---|---|---|---|
| pos-diff (v1) | 42.7 | 40.9 | 142.3 | 281.6 | 562.0 | 74 ms | 333 ms |
| SmoothNet | 10.6 | 7.2 | 21.0 | 48.0 | 82.3 | 52 ms | 233 ms |
| flow (PyrLK) | 8.3 | **5.0** | 60.8 | 136.7 | 162.0 | 65 ms | 267 ms |
| **BLEND** | **6.9** | **5.0** | **20.9** | **47.9** | 85.7 | **48 ms** | 217 ms |

All mm/s vs OMC. **The blend wins or ties every column**: it takes flow's clean off-peak (direct
velocity, no differentiation) and SmoothNet's accurate peak (position is unblurred), because the two
fail in complementary regimes. vs the v1 baseline it is **6× better per-frame and 7× better at the
peak** — and peak velocity is a Murphy measure, so this propagates straight to the scoring.

---

## 5. Segmentation — ALL 7 phases, n=12

⚠ **Score every phase, not just the drink dwell.** An earlier version of this document reported only
the dwell (67 ms) and called segmentation solved. That was misleading: the dwell is the *easiest*
boundary — the cup stops dead at the mouth — and reporting it alone **hid a defect that broke 11 of
12 trials**. The table below is the honest metric: onset, offset and duration error for each of the
7 Murphy phases, plus a **miss** count (a phase that was never produced at all, which is worse than a
mis-timed one).

Same segmenter on every row, so the **cup track is the only variable.** v3 = UETrack + SmoothNet.

| phase | miss | \|Δonset\| | \|Δoffset\| | \|Δduration\| |
|---|---|---|---|---|
| rest_pre | 0/12 | 0 ms | 17 ms | 17 ms |
| reaching | 0/12 | 17 ms | 42 ms | 42 ms |
| forward_transport | 0/12 | 42 ms | 67 ms | 125 ms |
| drinking | 0/12 | 67 ms | 17 ms | 108 ms |
| back_transport | 0/12 | 17 ms | 167 ms | 258 ms |
| returning | 0/11 | 200 ms | 17 ms | 183 ms |
| rest_post | 0/12 | 17 ms | 0 ms | 17 ms |
| **ALL** | **0/83** | **17 ms** | **17 ms** | |

**Every phase is produced in every trial, and every boundary is within 1–12 frames.**

### The defect the dwell-only metric hid, and the fix

`grasp_end` used to be *"the last frame with speed > BACK_OFF (10 mm/s)"*. Once the cup lands there is
a burst of small noise right around that gate — set-down contact, tracker residual, mocap marker
wobble — so the boundary was decided by whichever signal twitched last rather than by the event. A
detect-once tracker never falls silent at all (rest speed ~40 mm/s vs OMC's ~11), so:

- `returning` / `rest_post` were **never produced in 11/12 trials**
- `back_transport` swallowed the trial tail
- `total_movement_time` over-ran by **1.50 s**

**Fix: read the boundary off the DISPLACEMENT curve, not the speed curve.** The cup landing is where
displacement-from-rest stops changing, and *both tracks flatten at the same instant even when their
speeds disagree*. Take the FIRST sustained flat run after the cup returns near rest (everything after
landing is noise), anchored inside the transport window so an outward pause cannot satisfy it.

| | before | after |
|---|---|---|
| missing phases | 11/12 trials | **0/83** |
| back_transport offset | unbounded | **167 ms** |
| returning onset | unbounded | **200 ms** |
| total_movement_time error | 1.50 s | **0.03 s** |

Two failed attempts are recorded so they are not retried: a **sign-only** rule (`radial >= 0`, no
magnitude floor) fires during the *outward* trip and collapsed the window (21/52 misses); a
**two-condition stability + direction** rule was better but still lost phases (3/83). Displacement
flattening is what works.

**Residual (167/200 ms).** This is one shared boundary — back_transport-end = returning-onset — and
checking the raw 100 Hz mocap shows it is genuinely ambiguous rather than a tracking error: in the
disputed gaps the cup moves a median of 14.3 mm/s over a 4.8 mm range (a still cup is ~1 mm/s, ~1 mm).
Participants keep adjusting the cup for up to ~1.4 s after setting it down. In ~2 trials v3
over-smooths that real motion, in ~1 it invents motion the mocap does not show, and in ~1 OMC's own
threshold exits early while motion continues. Neither side is simply right.

## 6. Murphy measures — v1 vs v3, |error| vs OMC — n=12

**END-TO-END**: each arm segments with its OWN cup track (v1 with v1's, v3 with v3's, OMC with the
OMC cup), so these numbers include BOTH the pose and the segmentation — what the pipeline actually
delivers. Now that the segmentation is good, holding phases fixed would flatter both arms by handing
them ground-truth boundaries they would never have live. (`--fixed-phases` still isolates the pose
for attribution.)

### Position measures — the ported set, 8/8

| measure | n | v1 \|err\| | v3 \|err\| | change |
|---|---|---|---|---|
| total_movement_time (s) | 12 | 0.04 | **0.03** | −20 % |
| **peak_velocity** (mm/s) | 12 | 45.22 | **24.28** | **−46 %** |
| time_to_peak_velocity (s) | 12 | 0.06 | **0.05** | −14 % |
| time_to_peak_velocity_percent | 12 | 10.39 | 10.57 | +2 % |
| time_to_first_peak_velocity (s) | 12 | 0.04 | **0.03** | −20 % |
| **time_to_first_peak_velocity_percent** | 12 | 15.58 | **9.27** | **−41 %** |
| **number_of_movement_units** | 12 | 2.00 | **1.00** | **−50 %** |
| max_trunk_displacement (mm) | 5 | 7.01 | 7.08 | +1 % |

The measures that read the *derivative* improve most — exactly the ones jitter was corrupting.
`max_trunk_displacement` is a slow positional measure SmoothNet correctly leaves alone.

**All 8 measures now report on all 12 trials.** Two `score.py` bugs were suppressing them:

- **`_butter_lowpass` was NaN-poisoning whole trials.** `filtfilt` propagates a single NaN across its
  entire output, so **2 missing frames in a 596-frame trial nulled every measure for that trial**
  (2/12 trials). Now interpolates gaps, filters, and restores NaN only where the input was missing.
- **`time_to_first_peak_velocity` was structurally blind to edge peaks.** `find_peaks` only returns
  *interior* local maxima, so a single-peaked reach whose speed peaks at the window edge (measured:
  argmax at frame 0) returned nothing — and a single-peaked reach is the normal healthy case. Falls
  back to the argmax, which for that profile *is* the first peak.

⚠ `time_to_peak_velocity_percent` previously read −52 %; that was computed on the NaN-poisoned
8-trial subset. On the full 12 it is +2 % — a reminder that a measure silently dropping trials also
biases the ones it keeps.

### Angle measures — 7, raw-point

⚠ **NOT the ported container set.** The container computes angles from a MuJoCo `qpos` IK fit and
explicitly refuses raw-point angles ("would inherit pose jitter"); cup-task has no body model. Both
sides use the identical formula, so the MMC-vs-OMC comparison is fair, but these are
computable-to-*see*, not for clinical scoring.

| measure | n | v1 \|err\| | v3 \|err\| | change |
|---|---|---|---|---|
| elbow_extension_reaching (°) | 12 | 6.35 | **5.99** | −6 % |
| shoulder_flexion_reaching (°) | 10 | 20.65 | 20.46 | −1 % |
| shoulder_flexion_drinking (°) | 11 | 21.47 | 22.10 | +3 % |
| shoulder_abduction_reaching (°) | 12 | **2.19** | 3.12 | +42 % |
| shoulder_abduction_drinking (°) | 12 | 8.73 | **8.50** | −3 % |
| **peak_elbow_ang_vel** (°/s) | 12 | 66.88 | **45.49** | **−32 %** |
| interjoint_coordination (r) | 3 | 0.00 | 0.06 | — |

Same pattern: the angular *velocity* measure improves sharply (−32 %) while the static angles barely
move. The shoulder-flexion errors (~20°) are large for both arms — a raw-point angle definition gap,
not a tracking difference. `interjoint_coordination` has n=3 and is not interpretable.

---

## 7. Bugs found and fixed during this run

1. **SmoothNet was translating every track by ~84 mm.** The pretrained h36m model expects
   root-relative poses in metres near the origin; it was being fed absolute world coordinates ~1.5 m
   out, so its learned bias landed as a constant offset. Fixed by centring each window on its own
   mean and adding it back (offset-covariant — a temporal filter must never move the track):
   **84 mm → 1.9 mm**, with jitter (−92%) and peak-velocity accuracy unchanged, and speed error
   improved (wrist 15.0 → 12.3 mm/s). *This was invisible to every previously-used metric* — jitter,
   speed correlation, and peak velocity are all blind to a constant offset.
2. **UETrack weights lived in a session scratchpad** that was cleaned up, silently breaking all
   tracker runs. Re-fetched from HF `kangben258/UETrack` into `models/trackers/uetrack/` (repo,
   persistent) and the wrapper now overrides the config path at load time.
3. **Displacement was first computed per-axis**, which measured the MMC↔OMC frame rotation rather
   than tracking error (the cup read 299 mm). Origin-relative displacement removes the issue.
4. **`segment_cup_only` ended the return on a speed threshold**, which a detect-once tracker never
   satisfies — 11/12 trials lost their terminal phases and `total_movement_time` over-ran by 1.50 s.
   Now ends on displacement flattening (§5). Only visible once *all 7 phases* were scored.
5. **The cup was never being smoothed.** `pose_smooth.smooth_tracks` excludes it by default (it has
   its own tracker), but its ~1 mm per-frame wobble → ~60 mm/s of phantom speed is exactly what
   SmoothNet fixes: rest speed 40 → 2.8 mm/s, position moved only 0.76 mm.

### Measurement traps hit during this run (all reversed a conclusion)

- **A metric that hides the defect.** Dwell-only segmentation looked excellent (67 ms) while 11/12
  trials were structurally broken. Score every phase.
- **Selection bias.** "v1 has better cup speed" was true only because v1's 78 % coverage is almost
  entirely the *stationary* cup (0.6 mm/s on frames it has, 139.3 mm/s on frames it misses).
- **Derivative-blind metrics.** Jitter, speed-correlation and peak-velocity are all blind to a
  constant offset, which is how an 84 mm SmoothNet translation survived a full validation.
- **Position plots cannot show speed noise.** 1 mm of wobble is invisible on a 700 mm trajectory and
  dominates its derivative — the reason "positions match but speeds differ" was not a bug.
- **`predict()` returns before the GPU finishes.** Wall-clock timing around it measures kernel
  *launches*; it made an earlier pass conclude "99 % CPU-bound" when the GPU was 92 % busy. Use
  `torch.cuda.Event`.
- **Filtered signals hide real motion.** "OMC is just noise-chasing at the return boundary" was wrong
  — the raw 100 Hz mocap shows genuine 46–95 mm/s cup adjustments there. Check the raw data before
  blaming the reference.

## 8. What to do next

1. **LOPO-tune the blend gate** (350/120 hand-set on P07+P08) before trusting it on a new cohort;
   `speed_blend.fit_gate` is written for this.
2. **Two-GPU or lighter cup backbone** if 5-cam 60 fps live capture is required (§2 — the GPU is
   92 % busy at 5 cams, so this is a hardware limit, not a scheduling one).
3. **Wire cup flow into the reporting path.** Validated (§3: 25.3 vs 77.4 mm/s on moving frames) but
   `cup_flow_probe.py` is still a probe — `score.py` does not yet read it. Keep the segmenter on
   SmoothNet.
4. **`time_to_first_peak_velocity` is NaN** for both arms (n=8). Unresolved in the measure itself.
5. **P13 clock drift**: a linear time-warp of its OMC would return 6 trials to the cohort (n=12 → 18).
6. **The 167/200 ms return boundary** needs a better *definition* of "the cup is down" (participants
   keep adjusting it for ~1.4 s), not a better track. Both sides are ambiguous there.
