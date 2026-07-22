# cup-task v3 — measured results

Every number the pipeline has been validated on. How it works is in [SPEC.md](SPEC.md); the
chronological record, including rejected ideas, is in [WORKLOG.md](WORKLOG.md).

**Cohort: P07 + P08, trial_10–15, n = 12**, DELTA, against OMC (Qualisys) ground truth.
Measured 2026-07-21/22. Hardware: RTX 3060 Ti (8 GB, 38 SMs).

> **P13 is excluded from every table.** Its OMC clock drifts linearly against video (−8 → +3 frames
> over ~6 s, a 3.8 % rate mismatch), so its ground truth is progressively mis-timed. That is not a
> constant lag one cross-correlation can absorb and it corrupts POSITION as well as timing — on the
> cup, median displacement error 10–12 mm vs 2–3 mm for P07/P08, d-corr 0.974 vs 0.998, and it owned
> the entire 504 mm error tail. Including it simultaneously flattered v1 and penalised v3. Its data
> and caches are untouched; a linear time-warp of its OMC would restore all 6 trials (n → 18).

---

## 1. Speed

### Online loop — rig-frames/s (all cameras advance one frame)

| cams | pose | cup | flow **+ms** | **both + flow** | realtime (≥60)? |
|---|---|---|---|---|---|
| 1 | 237.2 | 170.3 | 0.43 | **100.1** | **yes** |
| 5 | 75.0 | 83.7 | 1.93 | **38.5** | no |
| 10 | 41.8 | 46.1 | 2.80 | **21.6** | no |

`flow +ms` is the **marginal** cost — what threaded flow adds on top of the GPU pass, measured
directly. It is not `ncam × per-wrist` (which would read 1.5/7.7/15.4 and overstate it 5–30×): PyrLK
releases the GIL, so the thread pool gives real parallelism and the GPU pass hides the rest.
**Flow is effectively free online.**

### The GPU is the limit — diagnosed, not assumed

| at 5 cams | pure GPU time (CUDA events) |
|---|---|
| YOLO-pose forward | 16.06 ms |
| UETrack update | 11.37 ms |
| **sum** | **27.43 ms** → ceiling ~36 fps with *perfect* overlap |

The measured combined loop is 29.7 ms, so **the GPU is busy 92 % of it**. There is no idle device
time to schedule into, which is why threading the two nets is *worse* than serial
(1/5/10 cam = 0.81/0.87/0.91×) and why batching gives no economy past 1 camera (a raw batched forward
on pre-resized 640×640 GPU tensors costs the same as full `predict()`: 12.15 vs 12.23 ms). Memory is
not a constraint — 1.1 GB of 8.2 GB resident.

**More throughput needs a second GPU, a lighter backbone, or lower input resolution — not better
scheduling.**

### Offline post-processing — per trial (533 frames = 8.9 s, 7 cameras)

| stage | ms |
|---|---|
| cup 3D (consensus) | 38.7 |
| **SmoothNet (3 joints)** | **795.8** |
| flow 2D→3D speed | 43.3 |
| speed blend | 0.5 |
| segmentation | 0.4 |
| Murphy scoring | 8.3 |
| **total** | **≈887 ms/trial (10 % of realtime)** |

A 60-trial session post-processes in under a minute. SmoothNet is 90 % of it.

---

## 2. Accuracy vs OMC

Displacement is **origin-relative** (see SPEC §6): rotation-invariant, no rigid fit, so it does not
inherit the ~38 mm rig↔mocap calibration floor.

### Pose joints

| target | displ RAW | p90 | displ SN | p90 | speed RAW | speed SN |
|---|---|---|---|---|---|---|
| acting wrist | 4.9 | 11.8 | 4.9 | 12.4 | 42.7 | **10.6** |
| acting elbow | 21.8 | 36.4 | **18.5** | 35.5 | 44.8 | **10.0** |
| acting shoulder | 10.7 | 19.7 | 10.3 | 20.2 | 38.3 | **6.2** |
| nose | 2.2 | 6.3 | 2.6 | 6.5 | 13.7 | **5.3** |

mm and mm/s. **SmoothNet is position-neutral (±0.7 mm) but cuts speed error ~4×** — exactly what a
temporal filter should do: never move the track, only de-noise its derivative. The elbow's larger
displacement is a landmark-definition gap (COCO elbow keypoint vs OMC elbow marker), not tracking.

### Cup — v3 (detect-once UETrack) vs v1 (every-frame YOLO)

| source | displ med | mean | p90 | **d-corr** | spd all | **spd MOVING** | s-corr | cov all | **cov MOVING** |
|---|---|---|---|---|---|---|---|---|---|
| v1 every-frame | 2.7 | 5.0 | 11.3 | 0.9981 | **18.3** | 136.2 | 0.67 | 78 % | 52 % |
| v3 UETrack | **2.3** | 15.5 | 58.4 | **0.9996** | 46.3 | **77.4** | **0.93** | **100 %** | **100 %** |

**d-corr 0.9996** reproduces the v2 tracker-shootout result.

⚠ **Do not read `spd all`.** v1 covers only 78 % of frames and those are overwhelmingly the
*stationary* cup — median OMC cup speed is **0.6 mm/s on frames v1 has** vs **139.3 mm/s on frames it
misses** — so its all-frames median mostly scores *"is the still cup still?"*. Where the cup actually
**moves**, v1 is 1.8× worse and sees only half the frames.

The mean/median gap needs the same care:

| v3 displacement error | mean | max |
|---|---|---|
| on frames v1 also has | **4.1** | 60.9 |
| on frames v1 misses | 56.2 | 107.8 |

v3 is **better where they overlap** (4.1 vs 5.0 mm); its higher pooled mean is entirely the price of
covering the extra 22 % — the occluded apex v1 fails to produce.

**Use v3 for the cup.**

---

## 3. Wrist speed — the blend

| method | per-frame | off-peak | PEAK med | p90 | max | peak-time mean | max |
|---|---|---|---|---|---|---|---|
| pos-diff (v1) | 42.7 | 40.9 | 142.3 | 281.6 | 562.0 | 74 ms | 333 ms |
| SmoothNet | 10.6 | 7.2 | 21.0 | 48.0 | 82.3 | 52 ms | 233 ms |
| flow (PyrLK) | 8.3 | **5.0** | 60.8 | 136.7 | 162.0 | 65 ms | 267 ms |
| **blend** | **6.9** | **5.0** | **20.9** | **47.9** | 85.7 | **48 ms** | 217 ms |

All mm/s vs OMC. **The blend wins or ties every column** — flow's clean off-peak plus SmoothNet's
accurate peak. Versus the v1 baseline: **6× better per-frame, 7× better at the peak**, and peak
velocity is a Murphy measure.

## 4. Cup speed — flow transfers from the wrist

| method | \|err\| all | **MOVING** | corr | at REST | clears the 10 mm/s gate |
|---|---|---|---|---|---|
| pos-diff | 46.3 | 77.4 | 0.931 | 42.5 | **no** |
| SmoothNet | **6.6** | 35.9 | 0.995 | 3.4 | yes |
| **flow (PyrLK)** | 7.5 | **25.3** | 0.989 | 4.8 | yes |
| blend | 7.3 | 27.4 | 0.994 | 4.6 | yes |

**Flow is 3× better than pos-diff where the cup moves.** Note the blend does *not* win here (unlike
the wrist) — SmoothNet already handles the cup's slower peaks — so **plain flow** is the pick for
reporting cup speed.

### Gating the flow cameras — two gates, two different failure modes

**The two targets fail for different reasons**, which is why the winning gate is the symmetric one:

| | **CUP** | **WRIST** |
|---|---|---|
| cause | **occlusion** — the hand covers it | **motion blur** — a fast wrist smears the patch |
| structure | one camera, sustained (cam_1 drove the whole tail) | **all** cameras, transient — median run **1 frame**, 59 % single-frame |
| spread | concentrated | uniform: per-camera 12–19 %, per-trial 14–20 % — no bad camera, no bad trial |
| when | cup at REST, hand passing | scales with **speed**: rest 0.2 %, slow 16.4 %, mid 47.7 %, fast 56.0 % |
| by phase | — | drinking 2.5 % vs reaching 25.8 %, returning 25.7 %, back_transport 28.3 % |
| a median fixes it? | **yes** (19.6 ≈ LOO's 19.4) | **no** (22.8 — *worse than no gate*) |

### Motion blur: detectable, but nothing can be done with it directly

**How PyrLK fails.** It assumes the patch is only *translated* between frames and solves for that
shift from spatial gradients, via the structure tensor `[[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]`. Its minimum
eigenvalue says whether the patch has gradients in *both* directions; if not, the displacement is
unconstrained. Blur destroys exactly those gradients — sharpness and conditioning correlate at
**+0.765** — so the failure mode is predicted by the algorithm itself.

**Blur is directly detectable in the image** — independent confirmation of the mechanism. Sharpness
of the tracked patch (61×61 px), kept vs dropped cameras:

| metric | kept | dropped | ratio |
|---|---|---|---|
| **Laplacian variance** | 669.4 | 388.6 | **0.58** |
| gradient energy | 81.4 | 66.3 | 0.81 |
| HF FFT energy | 441.8 | 327.3 | 0.74 |

The consensus drops **42 % less sharp** patches, established from pixels alone without any knowledge
of speed or geometry. But it is **not usable as a gate or a correction**:

- **As a gate: worse than nothing** (24.3 mm/s vs 21.9 ungated, 19.5 for LOO) and it adds nothing on
  top of LOO (19.7).
- **As a speed correction: too weak.** Raw corr(sharpness, flow error) = **−0.060**, and the obvious
  quartile trend is a **confound** — the blurriest quartile also moves 2.4× faster (blur is *caused*
  by speed, corr −0.155). Stratifying by true speed leaves a real but noisy residual: at matched
  speed the blurry half over-reads by ~20 mm/s (200–400 mm/s band: +26.6 vs +4.7), yet per-band
  correlations are −0.005 to −0.132, so a per-frame correction cannot capture it.
- **The speed blend already handles it** — it hands over to SmoothNet exactly where blur bites (fast
  frames), which is why the blend's peak error is 20.9 mm/s against pure flow's 60.8 (§3).

The wrist's blur hits **several cameras on the same frame**, so 3-of-5 disagree on 21 % of frames
(13 % for the cup) — a median is then itself one of the bad values. Per-frame wrist error by how many
cameras disagree: 0 → LOO 13.4 / median 12.4; 1 → 18.4 / 21.1; **2 → 24.7 / 42.8**; 3 → 46.9 / 49.3.
LOO's advantage is concentrated where exactly 2 cameras are bad. ⚠ At 3+ nothing works (46.9 mm/s):
too few good cameras remain. This is the same blur that causes flow's +61 mm/s peak over-shoot (§3).

### Consensus-gating the flow cameras

Flow originally used *every* camera that produced a point. Restricting it to the cameras the
geometric consensus **keeps** on that frame:

| | gate off | **gate on** | cameras kept |
|---|---|---|---|
| cup, MOVING frames | 25.3 | **19.8** mm/s (−20 %) | 5.00 → 4.80 |
| wrist, per-frame | 8.3 | 8.3 mm/s | 5.00 → 4.99 |
| wrist, blend at peak | 20.9 | **19.8** mm/s | — |

Now the default. The cup gains and the wrist does not, which is the known asymmetry: a cup false
positive is a **different object** and reprojects far, so the gate catches it; a pose error is the
right person's slightly-wrong joint and reprojects plausibly.

⚠ **It does not fix the rest-period tail** (p95 81.3 mm/s, unchanged) — and that is why flow still
loses to SmoothNet for segmentation. At rest the consensus rejects **0.00 cameras**: they all agree
*where* the cup is, but each reports a sub-pixel spurious *motion* which triangulates into tens of
mm/s. Geometric disagreement is gateable; flow contamination is not. First crossing of the 15 mm/s
onset gate vs OMC stays **SmoothNet 400 ms, flow 633 ms**.

### What the rest-period tail actually is: the hand dragging the flow window

The cup's flow is **not** inherently noisy — it is the *quieter* target. Per-camera flow magnitude
during rest (P07_11): cup median **0.015–0.053 px** on four of five cameras versus the wrist's
**1.15–1.61 px**. The tail comes from a handful of frames on one camera at a time.

Leave-one-camera-out on that trial's rest window:

| excluded | rest p95 |
|---|---|
| none | 86.6 mm/s |
| **cam_1** | **15.3 mm/s** |
| cam_2 … cam_5 | 99–120 mm/s (all *worse*) |

But cam_1 is not a bad camera — cohort-wide its positional jitter is mid-pack (1.564 px). Inspecting
the excursion frame by frame: the cup pixel moves 6 px in 24 frames (it is still) while flow reports
up to **4.9 px/frame**, and the spurious magnitude rises and falls exactly as the **wrist approaches
in image space** (160 → 91 px away).

**It is OCCLUSION — the hand passing between the camera and the cup.** And the test must be
**angular**, not a pixel distance. A 150 px proxy barely separates anything (within 150 px: median
0.024 px, p95 1.296; farther: 0.021 / 0.239 — *identical medians*), because a fixed pixel threshold
does not scale with distance: the worst contaminated frames sat at ~9° ≈ **215 px** at 1358 mm, i.e.
*outside* the 150 px cut, so they were being classed "not occluded".

Measuring the angle at the camera centre between the ray to the cup and the ray to the wrist, split
by depth order — cup flow at rest, cohort-wide:

| angle | occluder **IN FRONT** | occluder BEHIND |
|---|---|---|
| 0–5° | median **0.879 px** (p95 2.570) | 0.016 px (0.351) |
| 5–10° | median **0.800 px** (p95 4.407) | 0.021 px (0.212) |
| 10–15° | 0.010 px (0.228) | 0.018 px (0.190) |
| 15–25° | 0.049 px (0.232) | — |

**~45× the clean floor inside 10° with the hand in front, and it vanishes past 10°.** Both halves of
the test are load-bearing: *behind* is clean at every angle. So PyrLK is not disturbed by a nearby
object — it is tracking **the hand's texture** where the hand covers the cup's patch and reporting the
hand's motion as the cup's.

It hits the cup and not the wrist because the wrist *is* the moving object — its window's dominant
motion is the true signal — whereas a static cup loses the least-squares fit to whatever now occupies
its patch. This is also why the consensus cannot catch it: cup *position* stays right in every camera
(the tracker still knows where the cup is); only its *motion* is contaminated.

### Dropping occluded cameras — and it reverses the segmentation verdict

`flow_speed.occluded_by`: within a 10° cone of the camera→target ray **and** nearer to the camera.
Now wired into both the online and offline paths (`target_xyz` + `occluder_xyz`).

| | baseline | 150 px proxy | **angular <10°** |
|---|---|---|---|
| rest p95 | 81.3 | 37.2 | **11.9 mm/s** |
| MOVING-frame error | 25.3 | 26.0 | **22.0 mm/s** |

**7× reduction in the tail, and it improves moving-frame accuracy too.** Crucially 11.9 mm/s is now
*below* the 15 mm/s onset gate — the one thing that had disqualified flow as a segmenter signal:

| first crossing of FWD_ON vs OMC | median | p90 |
|---|---|---|
| SmoothNet | 400 ms | 1152 ms |
| **flow + angular occlusion mask** | **250 ms** | 1138 ms |

⚠ **This reverses the earlier conclusion in §4** that flow is the worse gating signal. That was an
artifact of the occlusion contamination, not a property of flow. The segmenter has **not** been
switched over — it still needs a position track for displacement-from-rest, which flow cannot
provide — but flow is now the better *onset* signal and combining the two is an open item.

**But the segmenter uses SmoothNet, not flow.** Flow is the better *reporting* signal and the worse
*gating* signal:

| speed during `rest_pre` | median | **p95** |
|---|---|---|
| OMC | 0.4 | 3.6 |
| SmoothNet | 3.3 | **14.2** |
| flow | 4.9 | **81.3** |

The onset gate is 15 mm/s. SmoothNet's p95 sits just under it; **flow's is 5× above**, so flow trips
the gate before the cup moves — its good *median* hides the tail. First crossing of the gate vs OMC's:
**SmoothNet 308 ms / p90 460**, flow 600 / 748.

---

## 5. Segmentation — all 7 phases

Same segmenter on every row, so the **cup track is the only variable**. v3 = UETrack + SmoothNet.

| phase | miss | \|Δonset\| | \|Δoffset\| | \|Δduration\| |
|---|---|---|---|---|
| rest_pre | 0/12 | 0 ms | 17 ms | 17 ms |
| reaching | 0/12 | 17 ms | 25 ms | 25 ms |
| forward_transport | 0/12 | 25 ms | 67 ms | 83 ms |
| drinking | 0/12 | 67 ms | 17 ms | 108 ms |
| back_transport | 0/12 | 17 ms | 25 ms | 33 ms |
| returning | 0/11 | 33 ms | 17 ms | 33 ms |
| rest_post | 0/12 | 17 ms | 0 ms | 17 ms |
| **ALL** | **0/83** | **17 ms** | **17 ms** | |

**Every phase produced in every trial; every boundary within 0–67 ms (≤4 frames).**

⚠ **Score all 7 phases, not just the drink dwell.** The dwell is the *easiest* boundary — the cup
stops dead at the mouth — and reporting it alone (67 ms) once hid a defect that broke 11 of 12 trials.

### Boundaries vs the published protocol

Checked against van Andel et al. (PMC5933268, Table 1):

| paper's boundary | criterion | our offset |
|---|---|---|
| reaching end (= fwd transport start) | glass velocity > 15 mm/s | **+0 ms** (exact, 12/12) |
| back transport end (= returning start) | glass velocity < 10 mm/s | **+50 ms** |

So **reaching includes the grasp** (our `reach_end` sits +100 ms *after* the grasp) and
**back_transport includes the release**, both as the protocol specifies.

### Why the transport boundaries need the wrist, not the cup

Same rule applied to *both* sides — does OMC agree with the tracker?

| release rule | median | p90 |
|---|---|---|
| **wrist→cup plateau** | **17 ms** | **50 ms** (4 trials agree exactly) |
| any cup-only rule | 167 ms | 643 ms |

Measured on RAW (un-low-passed) tracks, 0.5 s either side of the release — "jitter" is the residual
about a straight-line fit and the 2nd-difference magnitude:

| | OMC before | OMC after | v3 before | v3 after |
|---|---|---|---|---|
| net drift | 3.00 mm | **0.17 mm** | 4.79 mm | 3.96 mm |
| jitter (line fit) | 0.51 mm | **0.03 mm** | 1.03 mm | 0.97 mm |
| jitter (2nd diff) | 0.09 mm | **0.01 mm** | 2.21 mm | 1.59 mm |

**In mocap the pre-release wobble IS the hand** — everything collapses ~17× the instant it lets go.
But **v3 barely changes**: its post-release motion is tracker noise (1.59 mm HF, 50–150× OMC's floor),
comparable to the real pre-release hand wobble. **For the tracker there is no contrast at the release
to detect**, even though the physical event is sharp. The plateau works because it keys on a ~300 mm
ramp, far above any noise floor.

---

## 6. Murphy measures

**End-to-end**: each arm segments with its own cup track (v1 with v1's, v3 with v3's, OMC with the
OMC cup), so these include *both* the pose and the segmentation — what the pipeline actually
delivers. (`--fixed-phases` isolates the pose for attribution.)

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

### Angle measures — 7, raw-point

⚠ **NOT the ported container set.** The container computes angles from a MuJoCo `qpos` IK fit and
explicitly refuses raw-point angles ("would inherit pose jitter"); cup-task has no body model. Both
sides use the identical formula so the comparison is fair, but these are computable-to-*see*, not for
clinical scoring.

| measure | n | v1 \|err\| | v3 \|err\| | change |
|---|---|---|---|---|
| elbow_extension_reaching (°) | 12 | 6.35 | **5.99** | −6 % |
| shoulder_flexion_reaching (°) | 10 | 20.65 | 20.46 | −1 % |
| shoulder_flexion_drinking (°) | 11 | 21.47 | 22.10 | +3 % |
| shoulder_abduction_reaching (°) | 12 | **2.19** | 3.12 | +42 % |
| shoulder_abduction_drinking (°) | 12 | 8.73 | **8.50** | −3 % |
| **peak_elbow_ang_vel** (°/s) | 12 | 66.88 | **45.49** | **−32 %** |
| interjoint_coordination (r) | 3 | 0.00 | 0.06 | — |

Same pattern: the angular *velocity* measure improves sharply while static angles barely move. The
~20° shoulder-flexion errors are a raw-point definition gap, not a tracking difference.
`interjoint_coordination` has n=3 and is not interpretable.

---

## 7. Bugs found and fixed

1. **SmoothNet was translating every track ~84 mm.** The pretrained h36m model expects root-relative
   metres near the origin; it was fed absolute world coordinates ~1.5 m out, so its learned bias
   landed as a constant offset. Fixed by centring each window on its own mean: **84 mm → 1.9 mm**,
   with jitter (−92 %) and peak-velocity accuracy unchanged and speed error improved.
   *Invisible to every metric then in use* — jitter, speed-correlation and peak velocity are all
   blind to a constant offset. Only origin-relative displacement caught it.
2. **`_butter_lowpass` NaN-poisoned whole trials.** `filtfilt` propagates a single NaN across its
   entire output, so **2 missing frames in a 596-frame trial nulled every measure** for that trial
   (2/12). Now interpolates gaps, filters, restores NaN only where the input was missing.
3. **`time_to_first_peak_velocity` was blind to edge peaks.** `find_peaks` returns only *interior*
   local maxima, so a single-peaked reach whose speed peaks at the window edge returned nothing — and
   that is the normal healthy case. Falls back to argmax.
4. **`segment_cup_only` ended the return on a speed threshold**, which a detect-once tracker never
   satisfies — 11/12 trials lost their terminal phases and `total_movement_time` over-ran by 1.50 s.
   The fix already existed (`refine_grasp_with_pose`, the wrist→cup plateau); the harness was simply
   not calling it. Two bugs in the refinement also blocked it: it bailed out when the *onset* already
   agreed (discarding the release fix, 10/12) and clamped the release with `min(offset, ge)` so it
   could only move the boundary earlier (12/12).
5. **The cup was never being smoothed** — `smooth_tracks` excludes it by default.
6. **`UETrackBatch.update` called `.tolist()` per camera**, N GPU→CPU syncs re-serialising the batch
   right after the batched forward (~50 % of its CPU time). One transfer: **178 → 217 fps**,
   numerically identical (0.0001 px).
7. **UETrack weights lived in a deleted session scratchpad**, silently breaking every tracker run.
   Now repo-persistent in `models/trackers/uetrack/`.

## 8. Measurement traps (each reversed a conclusion)

- **A metric that hides the defect.** Dwell-only segmentation looked excellent (67 ms) while 11/12
  trials were structurally broken. Score every phase.
- **Selection bias.** "v1 has better cup speed" was true only because v1's coverage is almost
  entirely the *stationary* cup.
- **Derivative-blind metrics.** Jitter, speed-correlation and peak velocity are all blind to a
  constant offset — which is how an 84 mm translation survived a full validation.
- **Position plots cannot show speed noise.** 1 mm of wobble is invisible on a 700 mm trajectory and
  dominates its derivative. "Positions match but speeds differ" was not a bug.
- **`predict()` returns before the GPU finishes.** Wall-clock around it times kernel *launches*; it
  made an early pass conclude "99 % CPU-bound" when the GPU was 92 % busy. Use `torch.cuda.Event`.
- **Filtered signals hide real motion.** "OMC is just noise-chasing at the return boundary" was wrong
  — the raw 100 Hz mocap shows genuine 46–95 mm/s cup adjustments there. Check the raw data before
  blaming the reference.
- **Medians hide tails.** Twice: CoTracker's drift and flow's rest-speed p95. Report p90/p95
  alongside medians.
- **Check `git log` and memory before diagnosing a bug as new.** The release boundary had already
  been solved and validated on video months earlier.

## 9. Open

1. **LOPO-tune the blend gate** (350/120 hand-set on P07+P08) — `speed_blend.fit_gate` is written.
2. **Second GPU or lighter backbone** for 5-camera 60 fps live capture.
3. **Wire cup flow into `score.py`** — validated but still only a probe.
4. **P13 clock drift** — a linear time-warp of its OMC restores 6 trials (n → 18).
5. **`weighted_triangulate` / `camera_jitter`** exist in `triangulate.py` but are unused — an
   unfinished jitter-weighted-triangulation thread. Either wire it in and measure, or archive it.

---

## 10. Where the flow-speed error actually comes from (diagnosis)

Traced end to end because the obvious explanations kept being wrong. The pipeline is:
per-camera PyrLK (1–10 px) → undistort → DLT triangulate `{p}` and `{p+flow}` → difference × fps.

**The geometry is NOT the bottleneck.** The reprojection residual of the triangulated point is 6.35 px
— larger than the ~2 px flow signal — but it is **common-mode**: both `Xp` and `Xv` use the same
cameras, calibration and nearly the same pixels, so **94 % of it cancels in the difference**. What
survives is **0.40 px**, i.e. 0.20× the signal.

**Cameras SHOULD disagree, and mostly do so legitimately.** Projecting the true (OMC, rotation-only
Kabsch) velocity into each camera predicts a **0.329** cross-camera spread; the observed spread is
**0.450**. So **~73 % of the disagreement is pure geometry** — each camera sees only the component of
motion perpendicular to its ray. Only ~27 % is genuine flow error, and it is biased: observed /
predicted = **1.17**, the blur over-read, now measured directly rather than inferred.

**Per-camera accuracy is stable and characteristic.** Scoring each camera against the OMC-derived
prediction (rotation-only alignment, so the 35 mm Kabsch translation error cancels in a velocity):
median error **0.34–1.16 px** against 1.9–5.9 px predictions. Split-half correlation **+0.738** — a
camera that is inaccurate early stays inaccurate — and per-camera relative error ranges **0.108–0.442**
(a 4× spread) with across-trial SD of only 0.006–0.042.

### What drives the per-camera error — ranked

| factor | corr with relative error | quartile trend (low→high) |
|---|---|---|
| **perpendicularity** (fraction of motion visible) | **−0.215** | 0.270 → 0.190 → 0.170 → **0.147** |
| predicted \|flow\| (px) | −0.152 | 0.255 → 0.218 → 0.174 → **0.148** |
| distance to target | +0.122 | 0.154 → 0.189 → 0.160 → 0.324 |
| patch sharpness (blur) | −0.067 | 0.216 → 0.204 → 0.199 → 0.158 |

**This overturns the motion-blur story.** The dominant factor is **viewing geometry**: a camera
looking *along* the motion has ~2× the relative error of one looking *across* it, because few pixels
move and the same pixel noise is a larger fraction of a smaller signal. **Bigger motion is *more*
accurate in relative terms** (−0.152), the opposite of "blur ruins fast frames" — blur inflates the
magnitude (+17 %) without dominating the relative error. **Sharpness is the weakest factor** (−0.067),
so the blur thread was chasing the smallest term.

### Consequences for what to build

The one estimator that models this is the **Jacobian least-squares**: solve `u̇ = J(X)·v` for the
velocity directly instead of differencing two triangulations, so each camera is weighted by the
perpendicular component it can actually observe. Measured **20.53 vs 21.71 mm/s** for two-point DLT —
a −5 % gain from a *general geometric mechanism*, no tuned constants. It does not stack with the LOO
consensus (18.95 vs 18.73): both fix the same thing, LOO empirically, the Jacobian by construction.

⚠ **Tried and rejected — all mechanism-free tuning that did not survive the full cohort:** a
Laplacian sharpness gate (24.3 vs 21.9 ungated), PyrLK's own `minEigThreshold` (default 1e-4 rejects
0/532 points; every higher value made speed error worse, 19.2 → 22.2+), CLAHE (helps the pooled
median but is **2.1× worse in the top speed band**, 33.5 → 71.5 — the pooled number hid it), and
Wiener deconvolution with a flow-derived motion PSF (helps only at >700 mm/s, 33.5 → 28.9, and is
neutral-to-worse once LOO is applied, at every K from 0.001 to 0.2).

### Is the flow error irreducible? No — but the fixable part is not where it looks

**Magnitudes:** per-camera flow error is **0.34–1.16 px** median, on predicted flows of 1.9–5.9 px.
Sub-pixel throughout.

**The information floor is far below that.** The LK noise floor `σ_noise/√λ_min` (sensor noise over
the patch's structure-tensor conditioning) is **0.025 px**; observed error is **0.708 px** — **28×
above it**. So there is *plenty* of texture: the error is not a lack of features to track.

**It is MODEL error.** PyrLK assumes the patch only translates with constant brightness. Testing that
directly — shift frame *t+1* back by the measured flow and compare the same window, **integer shifts
only so no interpolation is involved**:

| | grey levels |
|---|---|
| static control (target still, no shift) | **0.87** ← true noise floor |
| aligned by the measured flow | 10.51 (12× floor) |
| **best of ANY integer shift (±4 px search)** | **7.09** (8.1× floor) |

The last row is the point: searching every translation within ±4 px, the **best possible** one still
leaves 7.09 — no translation can match these patches. The wrist deforms and rotates, and the window
straddles hand/forearm/background at different depths.

⚠ **But that gap is NOT recoverable, and chasing it makes things worse.** The best-RMS shift is
*further* from the true (OMC) flow than PyrLK: **0.92 vs 0.71 px**, winning on only 40 % of frames.
Appearance-matching is the wrong objective — the optimal-looking shift overfits the deformation.

**Affine LK does not help either** (tested because "the patch deforms, so allow rotation/scale"
sounded principled). ECC on identical patches, so only the model differs:

| | median | p90 |
|---|---|---|
| PyrLK (current) | **0.67** | 3.00 |
| ECC translation-only | 0.75 | 2.86 |
| ECC affine, 6 params | **0.91** | 3.84 |

Affine is **21 % worse than translation-only on the same solver** and beats PyrLK on 39 % of frames.
The extra parameters absorb noise and deformation instead of isolating a cleaner translation — the
same lesson as the best-RMS shift, twice over: **fitting the patch better ≠ estimating motion better.**

*(Methods note: a first attempt had ECC translation-only scoring 7.05 px, i.e. as bad as affine — a
sign/convention bug in extracting the warp. A synthetic known-shift check recovers +2.04/−3.05 for a
true +2.0/−3.0 with no sign flip. Always sanity-check an estimator on synthetic ground truth before
comparing it to anything.)*

**Conclusion:** the translation model is not the limiting factor, and better patch-matching is a dead
end. The dominant term remains **viewing geometry** (perpendicularity, −0.215), and the only estimator
that models it is the **Jacobian least-squares** (−5 %, general mechanism, no tuned constants).
