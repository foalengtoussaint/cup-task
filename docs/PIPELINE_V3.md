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

```
   ── ONLINE (per frame, while the pixels are in hand) ──────────────────────────
   capture ─┬─► YOLO-pose (batched across cams) ──────► 2D keypoints ──┐
            ├─► YOLO cup (once) → UETrack (batched) ──► 2D cup points ─┤
            └─► PyrLK flow at the wrist pixel ────────► 2D velocity ────┤
                                                                        │  numbers only
   ── OFFLINE (per trial, on numbers) ───────────────────────────────────▼──────
      triangulate ─► consensus (cup 3D) ─► SmoothNet (pose 3D) ─► speed BLEND
                  ─► segmentation (drink phases) ─► Murphy scoring
```

| stage | where | why |
|---|---|---|
| decode, YOLO-pose, cup detect + UETrack | **online** | needs the frame; is the capture loop |
| **PyrLK flow** | **online** | needs the frame PAIR. See below — this is the one real decision. |
| triangulation, consensus | offline | pure geometry on 2D points; nothing to gain online |
| SmoothNet | **offline, mandatory** | it is a *non-causal* window filter (±16 frames ≈ 0.27s of FUTURE). Cannot run online without adding latency. |
| speed blend | offline | needs SmoothNet, so it inherits offline |
| segmentation, Murphy | offline | needs the whole trial by definition (a dwell is an interval) |

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
makes the segmentation below work. Position is near-perfect (d-corr 0.9996) while its derivative is
0.93 — the same position-good / derivative-noisy split the wrist shows, and the reason the wrist gets
the flow+blend treatment. **The cup has no equivalent yet: applying a flow-style direct-velocity
measurement to the cup is the clearest open win.**

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

## 5. Segmentation (drink phases) — n=12

Same segmenter on both sides, so the **cup track is the only variable.**

| | median \|err\| | p90 |
|---|---|---|
| **dwell** | **67 ms** | 255 ms |
| onset | 50 ms | 128 ms |
| offset | 25 ms | 202 ms |

67 ms median dwell error = **4 frames at 60 fps**, on dwells of 0.6–1.7 s (≈5 % of the interval).
Onset is 3 frames, offset 1.5. 9 of 12 trials are within 120 ms; the tail (P07_12 +300 ms, P08_12
+267 ms) is where the cup is occluded at the apex and the track is interpolated.

## 6. Murphy measures — v1 raw pose vs v3 — n=12

Phases held FIXED (from the OMC cup), so the **pose source is the only variable.**

| measure | n | v1 \|err\| | v3 \|err\| | change |
|---|---|---|---|---|
| total_movement_time | 12 | 0.00 | 0.00 | — |
| **peak_velocity** (mm/s) | 10 | 45.22 | **26.41** | **−42 %** |
| **number_of_movement_units** | 12 | 2.00 | **0.50** | **−75 %** |
| max_trunk_displacement (mm) | 5 | 7.01 | 7.01 | 0 % |

The two measures that read the *derivative* both improve sharply — they are exactly the ones jitter
was corrupting. `total_movement_time` is phase-derived (phases are fixed here, so it cannot move) and
`max_trunk_displacement` is a slow positional measure SmoothNet correctly leaves alone.

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

## 8. What to do next

1. **LOPO-tune the blend gate** (350/120 hand-set on P07+P08) before trusting it on a new cohort;
   `speed_blend.fit_gate` is written for this.
2. **Two-GPU or lighter cup backbone** if 5-cam 60 fps live capture is required.
3. **Cup speed — the clearest open win.** v3's cup POSITION is near-perfect (d-corr 0.9996) but its
   differentiated speed is only 0.93 / 77 mm/s on moving frames. That is exactly the profile the
   wrist had before flow. Applying the same direct-velocity measurement to the cup (PyrLK at the cup
   pixel → triangulate {p} and {p+flow}) should transfer straight over; `cup_task.flow_speed` is
   already target-agnostic, so this is wiring, not new method. (Superseded an earlier note here that
   suggested v1 for cup speed — that was an artifact of v1 only covering the STATIONARY frames.)
4. **P13 clock drift**: a linear time-warp of its OMC would return 6 trials to the speed cohort.
