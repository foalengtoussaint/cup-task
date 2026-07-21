# cup-task pipeline v3 — design, online/offline split, and measured results

Measured 2026-07-21 on DELTA, n=18 OMC-truth trials (P07/P08/P13 trial_10–15).
Branch `pipeline-v2-uetrack-smoothnet`. Supersedes PIPELINE_V3_PLAN.md (that was the plan; this is
the built pipeline and its numbers).

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
  GPU nets, so its **marginal** cost is **0.2 ms/rig-frame at 1–5 cams** (2.3 ms at 10) — essentially
  free. (Raw serial cost is 1.5 ms/wrist/cam; PyrLK releases the GIL so threading gives ~2.4× at 5–10
  cams, and the GPU pass hides nearly all of what remains.)
- offline: it forces a **second full decode** of every camera = **5353 ms/trial** on a 7-cam 8.9s
  trial — **6× the entire offline budget (892 ms)**.

Same numbers either way; online is ~6× cheaper end to end. Flow goes online.

**Why SmoothNet is offline** (it looks online-able, but is not): it consumes a symmetric window, so
producing frame *t* needs frame *t+16*. Online it would impose 0.27 s of latency and still not be
faster. It is also the single largest offline cost (810 ms of the 889 ms), so it is the right thing
to optimise if the post-processing budget ever matters.

---

## 2. Speed benchmark

RTX (7.6 GB), 1080p real DELTA frames. fps = **rig-frames/s** (all cameras advance one frame).

### Online loop

| cams | pose fps | cup fps | flow **+ms** | BOTH + flow | realtime (≥60)? |
|---|---|---|---|---|---|
| 1 | 249.3 | 90.0 | **0.37** | **66.7** | **YES** |
| 5 | 76.2 | 61.4 | **1.42** | **34.5** | no |
| 10 | 44.9 | 39.6 | **3.40** | **20.3** | no |

`flow +ms` is the **marginal** cost — what threaded flow actually adds on top of the GPU pass,
measured directly (not `ncam × per-wrist`, which would read 1.5/7.7/15.4 and overstate it ~5–30×).
PyrLK releases the GIL, so the thread pool gives real parallelism (~2.4× at 5–10 cams), and the GPU
pass hides nearly all of what is left. **Flow is effectively free online.**

The binding constraint is the two GPU nets, not flow. Pose and cup each clear 60 fps alone at 1–5
cams; **run together they do not, past 1 camera** — on one GPU they contend for the same SMs, so the
combined rate lands below `min(pose, cup)` instead of overlapping.

**Deployment reading:** a 5-camera rig at 60 fps needs either a second GPU (pose on one, cup on the
other — they are independent), a lighter cup backbone, or accepting ~34 fps capture. The accuracy
results below do NOT depend on this: they are computed offline from cached detections.

### Offline post-processing (per trial: 533 frames = 8.9 s, 7 cams)

| stage | ms |
|---|---|
| cup 3D (consensus) | 36.7 |
| **SmoothNet (3 joints)** | **802.4** |
| flow 2D→3D speed | 43.0 |
| speed blend | 0.5 |
| segmentation | 0.5 |
| Murphy scoring | 8.4 |
| **TOTAL** | **≈892 ms/trial (10% of realtime)** |

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

### Pose joints (n=18 trials)

| target | displ RAW | p90 | displ SN | p90 | speed RAW | speed SN |
|---|---|---|---|---|---|---|
| acting wrist | 6.5 | 14.2 | 7.2 | 14.0 | 48.6 | **12.3** |
| acting elbow | 31.8 | 51.8 | **28.0** | 52.0 | 46.3 | **11.6** |
| acting shoulder | 5.3 | 15.1 | 5.7 | 14.8 | 42.2 | **6.5** |
| nose | 3.2 | 10.3 | 3.4 | 10.2 | 19.4 | **7.2** |

mm and mm/s. **SmoothNet is position-neutral (±0.7 mm) but cuts speed error ~4×** — exactly what a
temporal filter should do: it must not move the track, only de-noise its derivative. Elbow
displacement is the outlier (28–32 mm) — the COCO elbow keypoint and the OMC elbow marker are
different landmarks, so this is largely a definition gap, not tracking error.

### Cup

| source | displ | p90 | speed | coverage |
|---|---|---|---|---|
| v1 every-frame YOLO | 3.1 | 14.6 | **23.4** | 74% |
| v3 detect-once UETrack | 3.3 | 48.9 | 54.8 | **100%** |

⚠ **These are not apples-to-apples** — v3 is scored on 26% more frames, and they are the HARD ones
(occluded at the drink apex) that v1 simply failed to produce. Restricted to shared frames:

| on frames BOTH produce | v1 | v3 |
|---|---|---|
| displacement | 3.1 | **1.6** |
| speed | **14.5** | 48.6 |

So v3's cup track is **more accurate in position** where both exist, and it fills the apex gap
(100% vs 74% coverage), but its extra frames carry ~49 mm error and its **speed is worse** — a
tracked point moves smoothly-but-slightly-wrong, and differentiation punishes that.

**Recommendation:** use v3 for **position/coverage** (segmentation needs an unbroken track through
the apex — and the dwell numbers below confirm it works), use v1/every-frame detection where cup
SPEED is the quantity of interest. This is the same split already established for the wrist.

---

## 4. Wrist speed — the v3 speed path (P07+P08; P13 excluded, clock drift)

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

## 5. Segmentation (drink phases) — n=18

Same segmenter on both sides, so the **cup track is the only variable.**

| | median \|err\| | p90 |
|---|---|---|
| **dwell** | **67 ms** | 185 ms |
| onset | 33 ms | 138 ms |
| offset | 33 ms | 112 ms |

67 ms median dwell error = **4 frames at 60 fps**, on dwells of 0.6–1.7 s (≈5% of the interval).
Onset and offset are each 2 frames. 14 of 18 trials are within 100 ms; the tail (P07_12 +300 ms,
P08_12 +267 ms) is where the cup is occluded at the apex and the track is interpolated.

## 6. Murphy measures — v1 raw pose vs v3

Phases held FIXED (from the OMC cup), so the **pose source is the only variable.**

| measure | n | v1 \|err\| | v3 \|err\| | change |
|---|---|---|---|---|
| total_movement_time | 18 | 0.00 | 0.00 | — |
| **peak_velocity** (mm/s) | 10 | 45.22 | **26.41** | **−42%** |
| **number_of_movement_units** | 18 | 1.00 | **0.50** | **−50%** |
| max_trunk_displacement (mm) | 5 | 7.01 | 7.01 | 0% |

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
3. **Cup speed**: keep every-frame detection where cup speed matters; v3 tracking is for coverage.
4. **P13 clock drift**: a linear time-warp of its OMC would return 6 trials to the speed cohort.
