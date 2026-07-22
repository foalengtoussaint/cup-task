# Worklog

Running record of what was built and what was actually measured. Grand lines only --
decisions, results, and dead ends worth not repeating. Newest last.

The rule this log exists to enforce: **a number quoted here was measured on stated data.**
Where a result came from a different population than the one it gets applied to, that is
said out loud, because that is how a good result silently becomes a wrong claim.

---

## 2026-07-13 — Wire in the research 3D + segmentation

Goal: cup-task had detection but stopped there (`segment.py`, `score.py` didn't exist).
Port the parts of `object-tracking/drink_study` that were already solved, and *measure*
the parts that were only assumed.

### Base pipeline now runs end-to-end
`cup_task/pipeline.py`: per-camera clips -> cup + pose 2D -> triangulate 3D -> phases.
Every stage caches to JSON; the GPU passes are the expensive bit and are paid once.

### Segmentation: ported and verified exactly
`cup_task/segment.py` <- `drink_study/lib/segment_cup_only.py`. Van Andel glass-velocity
gates for the transport window, then "near peak displacement AND slow" for the dwell.

**Verified: reproduces the research segmenter's phase intervals frame-for-frame on 40/40
reps.** Same code, same answer -- not "looks about right".

Gate constants are the LOPO-tuned 150/150 (not the original 120/90): tuning moved dwell
bias -95ms -> +14ms and improved 20 of 21 folds. Don't restore the lower values.

### Cup+pose fusion: ported, measured, NOT shipped as default
`fuse_phases.py`'s log-odds fusion is in `segment.py` as `segment_fused()`, marked
experimental, because when measured here it **lost**:

| on 123 research reps | dwell found | recovered | destroyed |
|---|---|---|---|
| cup-only | 123/123 | -- | -- |
| fused    | 117/123 | **0** | **6** |

Its famous "recovered 13/20, destroyed 0" came from a deliberately adversarial subset --
the 20 *worst* reps, chosen because the cup track was failing there. On the general cohort
cup-only already finds a dwell in 100% of reps, so fusion has nothing to recover and can
only add risk. Both numbers are real; they describe different populations. Ruled out as a
cause: the confidence signal is fine (verified it collapses 1.00 -> 0.14 at the occluded
dwell exactly as designed, and using the real per-frame confidence changes nothing).

### Real-time: re-measured for yolo26
Model inference **10.3 ms/frame = 98 fps** at 1080p, vs the 16.7ms/60fps bar. Real-time
holds. Decode is only 1.9ms/fr.
Caveat: `model.predict(source=<path>)` adds ~2x overhead -- feed it cv2-decoded frames.
**An offline batch job's wall-clock says nothing about the live rate.** Don't conflate.

> **Superseded 2026-07-13 (see below).** The claim "batching buys ~0 (GPU already
> saturated per-frame)" is **WRONG** and was never measured -- it was assumed. A single
> small image leaves a 3060 Ti almost entirely idle; batching 10 cameras costs 4.6x one
> camera, not 10x. It is the single biggest lever we have.

### First end-to-end run (P07_drinking_left_20240124_142730, 10 cams)
Raw video -> phases, no hand-holding:

    rest_pre           0.00-0.97s
    forward_transport  0.97-3.08s
    drinking           3.08-4.20s   <- 1.12s dwell
    back_transport     4.20-7.78s
    rest_post          7.78-8.08s

The 3D coverage is the finding, and it validates the architecture on our own data:

| target | 3D coverage |
|---|---|
| **cup** | **367/485 (76%)** |
| mouth | 485/485 (100%) |
| wrists | 485/485 (100%) |

**The cup vanishes on ~24% of frames while the pose holds at 100%** -- the hand and body
wrap around the cup exactly when it matters (the sip), and the mouth proxy never blinks.
This is the occlusion story from the research pipeline reproducing itself on the first run,
and it is precisely why (a) the TCN gap-fill exists and (b) a head-distance channel earns
its keep. Do not read the 1.12s dwell as validated -- this rep has no ground truth.

### Consensus gate: needed for the CUP, redundant for POSE (measured)
The reprojection gate in `triangulate.py` (drop cams >30px, require >=3) is load-bearing
for the cup. Tested whether it does anything for pose keypoints -- 11 reps, one per
participant, yolo26s-pose wrists triangulated with vs without the gate, scored against
MeTRAbs multicam 3D:

| | plain DLT | consensus gate |
|---|---|---|
| median error | 16.8mm | 16.6mm |
| coverage | 100% | 100% |
| good@50 / @100 | 100% / 100% | 100% / 100% |

**The gate fires on 24% of frames and moves the median by -0.32mm** (~2% of a 17mm signal
= noise). Helped 3 reps, hurt 1, did nothing on 7. It **never once killed a frame**
(`kills 0%` on all 11) -- there is no tail to cut.

WHY, and this is the transferable bit: a **cup** FP is a *different object* (the cam_10
side-desk glass) -- it sits elsewhere in the world, reprojects far from consensus, gate
catches it. A **pose** error is the *right person's slightly-wrong joint* -- a wrist off
by 20px, still broadly in the right place, so it reprojects plausibly, sails through a
30px gate, and gets included anyway. The gate cannot see the error it would need to catch.
Meanwhile a 10-cam DLT already averages that jitter out.

So: not harmful, just **redundant for pose**. Keep it (shared with the cup, ~free), but
it is NOT load-bearing there -- and mind that `MIN_CAMS=3` would start *killing* pose
frames on a low-camera rig for no accuracy gain. Caveat: the reference is MeTRAbs's own
multicam triangulation -- the best 3D pose available and independent of the YOLO
detections under test, but not mocap.

### NOT ported (deliberately, for now)
The research pipeline's best dwell result (`drink_dwell/`, proxy21 ~85ms vs base17 ~123ms
LOPO) is **kinematics + occlusion + head-distance** on a TCN-gap-filled cup track. Two
reasons it isn't here yet:
- the **TCN gap-fill** has *no saved weights* -- `learn_seg.py` is a LOPO evaluation
  harness that trains per fold and discards them. Porting it means retraining.
- **head_distance reads the MOCAP head centroid.** features.py calls it "a stand-in for
  the future VIDEO head landmark" -- which is exactly what cup-task's pose mouth-proxy is.
  Swapping mocap->video head is an unmeasured substitution, so proxy21's 85ms cannot be
  claimed here on faith.

Decision: ship the base geometric pipeline (same method the research *truth* uses), then
measure those two as separate steps.

---

## 2026-07-13 (later) — Fixing the grasp/release, and a detector bug that cost us both speed AND accuracy

### The cup-motion onset: find the GRASP, not the cup's first wobble
The cup-speed gate (`FWD_ON=15mm/s`) fires on the **jitter floor of a stationary cup**
(~30-50mm/s from triangulation noise), so "transport start" landed at 0.97s -- while the
hand was still reaching. Every reach-scoped Murphy measure inherited that error
(`time_to_peak_velocity` read **93%**, i.e. the peak sat at the very end of a window that
was mostly not reaching).

The fix is a signal with no threshold on cup speed at all (**the user's idea**): the
**wrist->cup distance plateau.** The distance *closes* during the reach, goes *flat* the
instant the hand grasps (hand + cup become one rigid body), and *opens* at release. Take
the first closing run that travels >=30% of the distance span; its END is the grasp. The
first opening run after it is the release. Scale-free, no tuning:

    reach = 247mm of travel = 95% of span   |   fidgets/noise = 0-10mm = <4%

Margins are enormous, so the rule is robust rather than tuned. Same signal brackets **both**
ends of the task -- the release fix made `returning` appear at all.

| | before | after |
|---|---|---|
| transport window | 0.97-7.78s | **1.42-5.48s** |
| `time_to_peak_velocity` | 93% | **39%** (healthy band 30-50%) |
| `back_transport` | 3.58s | 1.28s |

### KF+RTS gap-fill: I had it BACKWARDS, and the user was right
I set `smooth=False` after misreading **one** overlay frame, then built a confident
mechanistic story on it ("a constant-velocity model can't represent a dwell"). Wrong.
Re-checked on **raw** frames (not the overlay -- an overlay drawn from the track under test
begs the question):

- **2.90s** (KF's drink onset): the cup is **already at her lips**. Linear's 3.08s is LATE.
- **3.78s** (KF's offset): the cup has **tilted away**. Linear's 4.20s is LATE.

Linear interpolation **cuts the corner** of the cup's arc to the mouth: the chord across the
occlusion sits farther from the face than the true path, so "near the mouth" fires late and
clears early. The KF is better at **both** ends. `smooth=True` is the default.

**Lesson, written down because it keeps recurring:** look at *both* ends of an interval
before condemning a filter, and never judge a track from an overlay rendered *with* it.

### Murphy position measures: ported, exact
`cup_task/score.py` reproduces the iMOVE container's cached measures **exactly (max diff 0)**
given the same sites + phase intervals. The 8 angle measures are deliberately absent (they
need MuJoCo `qpos` IK; the container's authors refused to derive them from raw keypoints and
so do we). The one non-obvious detail: **`medfilt(3)` before the Butterworth.** Without it a
lowpass *smears* a single-frame spike into a multi-frame bump that survives even a 2Hz
cutoff and then dominates `peak_velocity`.

### THE BIG ONE: `imgsz=1280` was a bug -- 2x slower AND 8x worse
`cup_detect.py`/`pose_keypoints.py` defaulted to **imgsz=1280**. The research pipeline
(`cache_dets_model.py:49`) calls `model(buf, conf=...)` with **no imgsz at all** -- i.e.
ultralytics' default **640**. Measured on P07 cam_4, 200 frames, vs the research cache:

| imgsz | cup found | agrees w/ research <10px | ms |
|-------|-----------|--------------------------|-----|
| **640** | **64%** | **100%** | **4.3** |
| 960   | 62%       | 94%                      | 5.5 |
| 1280  | **8%**    | 47%                      | 9.0 |
| 1920  | **0%**    | --                       | 16.1|

**Upsampling past the training size is OFF-DISTRIBUTION, not free detail.** The detector
learned its size priors at 640; at 1280 the cup arrives at twice the scale it ever saw and
the model simply stops finding it. There is no speed/accuracy tradeoff here -- 1280 was
**worse on both axes at once**.

Consequence, end-to-end on P07 (10 cams): **cup 3D coverage 76% -> 91%.** So a large part
of what the last entry called "the cup vanishes on 24% of frames (occlusion!)" was **not
occlusion -- it was this bug.** Real occlusion is the remaining 9%. The drink dwell widened
0.88s -> 1.10s, in the direction `project_mouth_dwell_truth` says is correct (the speed
proxy runs ~1s short).

### What the GPU is actually bound by: LAUNCHES, not pixels
Inference time vs imgsz at batch=1 (either net) is **flat** below the knee:

    imgsz   128   512   640  1024  1280  1920
    ms      2.8   2.9   3.2   4.3   6.6  13.3
            ^------ FLAT (launch-bound) ------^  ^-- compute-bound --^

25x fewer pixels (128 vs 640) changes inference by **4%**. The cost is per-*call* overhead
(~200 CUDA kernel launches + Python round-trip), not arithmetic. Two consequences:

- **The crop-tracking idea is unnecessary.** Cropping to the person to make the cup bigger
  works exactly as theorized (cup is 26.0px on full@1280 vs 26.4px on crop@640) -- but
  running at the native 640 achieves the same thing with no crop, no state machine, no drift
  guard, no coordinate remapping. Cropping *below* 640 buys nothing (all on the floor).
- **Batching across cameras is the real lever** -- it is the only thing that amortizes the
  launch overhead. 10 cameras cost **4.6x** one camera, not 10x.

### Speed, settled: 10 cameras @ 640, batched + threaded
Both nets, decode excluded (a live camera hands you the frame):

| cameras | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| **fps** | **123** | **83** | 52 | 37 | 31 | **25** |

**25fps at 10 cameras, and that is ENOUGH** -- `score.py` lowpasses at 4Hz, so ~12-15fps
fully samples everything downstream. Frames beyond that are smoothed away regardless.
The 60fps target was a requirement we did not actually have.

### Concurrency: use separate CUDA STREAMS, not just threads
Two Python threads calling `predict()` both submit into the **same default CUDA stream**, so
the two nets' kernels still serialize *on the device* -- all you overlap is one net's CPU
work (letterbox, H2D copy) with the other's GPU work. Giving each net its **own
`torch.cuda.Stream`** lets the kernels actually interleave:

| cams | serial | threads | **streams** | streams gain | fps |
|---|---|---|---|---|---|
| 1  | 8.3  | 6.6  | **5.8**  | **1.42x** | 171 |
| 2  | 11.5 | 10.3 | **8.4**  | 1.36x | 119 |
| 4  | 20.2 | 17.8 | **15.5** | 1.30x | **64** |
| 10 | 45.2 | 41.2 | **38.6** | 1.17x | **26** |

**The gain decays monotonically with batch (1.42x -> 1.17x)** -- which *confirms the
launch-bound diagnosis by its shape*: at batch=1 the GPU idles between small kernels and a
second stream fills the gaps; at batch=10 it is genuinely computing and there are no gaps
left. Bonus: 4 cameras now clears 60fps (64.5, was 51.9).

**Gotcha:** `torch.cuda.stream()` is **thread-local**. The context must be entered INSIDE
the worker thread -- set it on the main thread and submit to a pool and you silently get the
default stream back, i.e. you measure your own bug.

This further weakens the merged net's *speed* case (its argument was "one launch sequence
instead of two", and streams already recover much of that for free) -- its real appeal is
architectural, not throughput. **TensorRT** remains the untapped lever (fuses kernels ->
attacks the launch bound directly; `half=True` on a `.pt` is a **no-op**, measured 4.0 vs
4.1ms -- you only get fp16 for real via TRT).

### Crop-tracking: measured, it does NOTHING. Do not propose it again.
The idea (crop to the subject so the cup is bigger on the model's canvas) is mechanically
sound and was tested twice. **It works exactly as theorized and buys nothing**, because the
cup's pixel size was never the binding constraint:

| P07 cam_4, 485 frames, vs research cache | cup found | cup size on canvas | ms |
|---|---|---|---|
| full frame @640 (shipping) | 354 (73%) | 15.6px | 4.2 |
| task-region crop @640      | 358 (**74%**) | **20.1px** | 4.7 |

The crop makes the cup **29% bigger** and finds **4 more frames out of 485.** The misses are
**occlusion** -- the hand wraps the cup at the lips -- and a larger picture of a hidden cup
is still a hidden cup. (Earlier variant, person-crop@640 vs full@1280, was also a null: cup
26.0px vs 26.4px.) Cropping *below* 640 is worse than useless: inference there is
launch-bound and flat, so you shrink the cup for zero speed gain.

**The lesson is about where the errors live, not about crops.** We reached for a resolution
fix twice for a problem that was never resolution.

### Two videos, and they answer different questions
- `*_MODELVIEW_640.mp4` -- the clip **at 640x360, the actual model input**, with that ONE
  camera's raw 2D detections. Use it to judge the DETECTOR. (Cup is a median **15px** here;
  cam_4 finds it in 73% of frames.)
- `*_phases_640.mp4` -- full-res, but the markers are the **3D fused tracks reprojected**
  through that camera's calibration. They come from all 10 cameras + the KF+RTS, so they
  appear even on frames this camera missed. Use it to judge the **3D + phases**; a marker
  off the cup means the NUMBER is wrong, not just the picture (the dumb-player rule).

### Still open
- **Everything above rests on ONE rep** (P07_drinking_left_20240124_142730). Needs a cohort run.
- `max_trunk_displacement` = 4.8mm vs an expected 10-40mm -- the shoulder-midpoint proxy
  barely moves. A tracking issue, not a windowing one.
- **TensorRT** is the one unmeasured speed lever (fuses kernels -> attacks the launch bound).
  Only worth doing if 25fps@10cams ever stops being enough -- it currently is.

---

## 2026-07-16 — DELTA cohort: independent OMC validation set (found, one trial imported)

Goal: the existing OMC validation (`cache/qtm_omc/`) uses mocap **cup + head only**. Find an
**independently annotated** OMC set to (a) re-run the cup->head validation on a cohort we did
not tune on, and (b) extend it to the arm. Found it on SMB `research_analyzed_dataset/DELTA`.

### What DELTA is — and the ID trap
DELTA / iDrink is a **different rig AND different participants** from the BRIO cup-task set.
The IDs collide (there is a "P14" in both) but they are **not the same people**. Everything is
namespaced under `cache/delta/`, and nothing BRIO transfers: no `qtm_align` sync, no BRIO
calib, no reused lags. Treat it as a from-scratch cohort that happens to share a schema.

Its OMC C3Ds are **fully annotated — 46 points**: `cluster_cup_1..4`, head, and the whole
upper-body arm chain (shoulder/elbow/wrist/hand/fingers/thumb L+R, chest, hips) @ 100 Hz, mm.
That is the point — the cup+head we had was a subset. Videos are per-trial per-camera cut
clips (1920x1080 @ 60fps) that join 1:1 to the C3D **on trial number** (the side/cond suffix
sometimes disagrees). 15 participants have clean video (10-cam P14/15/17/19, 7-cam P07, rest
5-cam); calib is BRIO-format but **translations are in METRES, not mm**. Full inventory +
per-participant download plan in `cache/delta/README.md`. Imported ONE test trial (P14
`trial_1_R_unaffected`) to prove the path before bulk.

### FINDING: pose transfers, the cup detector does NOT
Ran batched cup+pose on the P14 test trial (10 cams x 710 frames, imgsz 640, yolo26s-pose +
`cup_clean3d_refill.pt` — the SAME weights validated on BRIO):

| | measured |
|---|---|
| pose (person recall) | **99–100 % every camera** — yolo26-pose generalizes to DELTA cleanly |
| cup recall | **poor** — best 49 % (cam1), most 5–21 %, **cam4 = 0 %** (BRIO was ~64 %+/cam) |

It is a **miss** problem, not confident-wrong: when the cup net fires, confidence is fine
(median 0.30–0.42, max to 0.78) — it just fails to fire, and never on cam4. That reads as a
genuine **domain gap** (DELTA cup / backgrounds / viewpoints are off-distribution for a
BRIO-finetuned detector), not a threshold to lower. Consequence: the MMC cup track on DELTA
will be gappy (triangulation needs >=2-3 agreeing cams), so **the cup is the transfer
bottleneck; the pose is not.** The OMC cup (from the C3D) is unaffected — only the
camera-detected cup is weak. Open: finetune cup on DELTA (OMC cup C3D + calib = free 2D labels
by projection) / validate only on high-recall cams / render cam4 first to see WHY it misses.

### Batched detector with separate CUDA streams
`detect_rep.py` ran each clip serially (~70–90 fps/clip, ~2 min for the trial). Wrote
`scripts/detect_rep_batched.py`: decode each camera once via NVDEC, run **both** nets on the
same frame batches, on **separate CUDA streams** (realtime.md finding #3 — two threads on the
default stream overlap only CPU work; a stream each lets the kernels interleave). Measured on
the trial: serial-nets 48.8 s -> streams 41.6 s (~1.17x, exactly the doc's predicted gain at
batch=10, where two nets leave few idle gaps). Output verified **byte-identical** across all
20 JSONs (concurrency that moved a detection would be a bug, not a speedup).

### Decode is ~2/3 of the offline wall-clock — and it VANISHES live
Measured decode alone (NVDEC, 10 cams x 710 frames, no inference): **27.0 s of the 41.6 s**.
So the 41.6 s is offline **cache** time, not the doc's inference-only fps budget: it includes
H.264 decode + the cam1 720p->1080p upscale + JSON serialize. **Live, the file-decode cost is
not incurred at all** — a sensor emits raw frames, handed over in a capture thread in parallel;
there is no encoded stream to decode. Strip decode and inference is ~15 s, in line with the
26/19 fps @ 10-cam budget. The live bottleneck is a *different, still-unmeasured* one: whether
10 simultaneous camera streams (USB/CSI bandwidth, grab sync) can feed frames at 60 fps.

### Still open
- Cup transfer must be resolved before any DELTA cup->head comparison is meaningful.
- **cam1 is 720p** while its calib says 1080p (all other 9 are 1080p). `staged/` upscales cam1
  to 1080p; do this per DELTA participant — check each camera's real resolution.
- DELTA video<->c3d **sync from scratch** (no qtm_align): reuse the BRIO cup-speed
  cross-correlation method (re-confirmed accurate this session via the distance + zero-bias
  onset argument on the lag=176 P11 case; a visual overlay was deemed unnecessary).

---

## 2026-07-16 (cont.) — DELTA pose comparison + Murphy scoring + the jitter fix

Continued the DELTA cohort work (see the 2026-07-16 entry above and `cache/delta/README.md`).
Established: **pose transfers, cup does not; the mocap-free weak point is the velocity/smoothness
measure family; and a FAST skeleton fix (bone-lock) recovers most of it.**

### MMC-vs-OMC pose comparison is possible (frames differ; no P14 mapping to reuse)
The DELTA OMC C3D is in the mocap LAB frame (mm); our MMC triangulates to the CALIB-camera world
(metres). Not reconciled, and DELTA's own Pose2Sim video pipeline only ran the 5-cam participants
(P07/P08/P10-P12), never P14 — so there is no ready-made lab->world transform to reuse. Two paths:
frame-INVARIANT signals (distances/angles/speeds — no alignment) and ABSOLUTE mm (needs one rigid
Kabsch, reproduced from the BRIO biomech align). Fixed two metres-vs-mm bugs: (1) `triangulate.py`
rounds X to 1 decimal = 0.1 mm in BRIO but **100 mm** in DELTA metres, quantising the wrist to a
100 mm grid -> scale calib translations x1000 (`_load_calib_mm`); (2) `np.roll` sync-shift is
circular, wrapping the far end onto the near end and fabricating a phantom 3669 mm/s peak -> shift
with NaN-fill.

### Signals transfer; scoring needed real segmentation + a truncation fix
Frame-invariant signal agreement (P14 sip 1): reach |wrist-nose| corr **0.992 / 8 mm**, elbow
angle 0.909 / 5.6 deg, shoulder flexion 0.937, trunk disp 0.951; wrist SPEED the weak one (0.508
raw). The trial is ONE complete reach-drink-return (sip 1) + a TRUNCATED sip 2. Scoring needed:
drink detection by cup->HEAD distance not the displacement proxy (distinguishes a sip from a
set-down); cut to the first complete cycle at BOTH cup AND wrist at rest (the hand returns ~1 s
AFTER the cup — that gap IS the `returning` phase, cutting at cup-return alone drops it); and a
gap-fill before `score._smoothed_xyz` (its filtfilt has no NaN handling — 3 missing frames NaN'd
the whole speed signal). Segment the CANONICAL way: pipeline.py's `segment_cup_only` ->
`refine_grasp_with_pose` -> `to_murphy_phases`.

### All 15 Murphy scalars, MMC vs OMC (9 position-derived + 6 angle-derived)
Protocol is 17; the container's dataclass exposes 15. Computed ALL 15 (the 6 angle ones from raw
points — NOT the ported set; production takes them from the MuJoCo qpos IK fit). Complete-cycle
result: total_movement_time exact (5.25 s), positions/angles/durations transfer (~6-8 deg offsets,
mostly OMC-marker-vs-COCO convention). **Everything DIFFERENTIATED degrades from keypoint jitter:**
peak_velocity 13 % low, movement_units 2 vs 1 (the phantom = a jitter min at the start of the
return, traced frame-by-frame), peak_elbow_angular_velocity 37 % HIGH. (`shoulder_abduction` proxy
has a sign flip -> absolute off-scale, Δ still valid.) Renders: `out/measures_*`,
`out/movement_units_*` (per-phase MU with the scorer's exact rule so picture==number).

### Smoothing: three-way, no single winner; then the real fix
none / KF-only / KF+RTS on the joint tracks (`--smooth`). timing -> KF+RTS (exact); peak MAGNITUDE
-> KF-only (RTS's backward pass is what flattens peaks; causal KF keeps them, and is the
real-time-realistic option); movement units -> none help (KF-only WORST at 4, its lag phase-shifts
jitter into fake oscillations). OMC always sits BETWEEN too-spiky (none/kf) and too-smooth (kfrts).
Render `out/kfrts_*`. Also confirmed KF+RTS was run PER-KEYPOINT (no skeleton constraint); it
didn't distort bones but can't fix the real cause.

### The real cause + the FAST fix (validated a user hypothesis)
The jitter is NOT random noise: the COCO "wrist" is a semantic label, so as the arm rotates each
camera places the keypoint on a DIFFERENT physical part of the wrist -> rays miss a stable joint
-> the triangulated wrist wanders -> **forearm length wobbles 25.7 mm std on a 207 mm bone (12 %)**.
Temporal smoothing can't fix a non-zero-mean, pose-correlated error (hence the slow pipeline's IK
fit). **bone-lock** (`--bonelock`): hold upperarm/forearm at MEDIAN length, keep observed joint
directions, re-place elbow/wrist. O(T), no optimiser. Attacks the wander at its source instead of
low-passing, so real peaks survive:

    peak_elbow_ang_vel  37 % high -> 4 % low        peak_velocity  13 % low -> 3 % low

Full-set tally vs baseline: **4 better (all velocity/timing), 7 same, 1 WORSE**
(elbow_extension_reaching 7.9->11.7 deg — re-placing the elbow along a noisy DIRECTION shifts its
angle). So a strong NET win but NOT strictly dominant; it helps DERIVATIVE measures far more than
STATIC angles, and the residual static-angle gap is the convention offset it can't touch.
bone-lock ALONE beats bone-lock+KF (KF re-adds lag). This is the fast stand-in for the slow IK fit.
Rerun log of all variants (orbit to see the wander): `out/mmc_P14.rrd`.

### Open
- ALL of the above is ONE trial (P14 sip 1) — directional, needs a cohort run to confirm.
- bone-lock's elbow-extension regression: try locking forearm only, or not re-placing the elbow
  direction; or a proper (still fast) bone-length-constrained least-squares.
- cup detector still doesn't transfer to DELTA (separate blocker for any cup-driven measure).

### Faster skeleton fixes explored: chain vs whole-skeleton PBD vs HYBRID (the winner)
The first bone-lock was a CHAIN (shoulder anchors, elbow placed from it, wrist from the elbow), so
every upstream correction is dumped downstream -- that leak is what shifted the elbow angle. Tried
three more, all fast:
  * **v2** -- forearm direction from the ORIGINAL elbow, so the elbow angle is preserved exactly.
    Kills the elbow-extension regression (+7.9, = baseline) but then does NOT fix
    peak_elbow_ang_vel (+56.6, no help). The tension is real and clarifying:
    **peak_elbow_ang_vel IS the derivative of the elbow angle** -- preserve the angle and you
    preserve its jitter. You cannot geometrically denoise a derivative while keeping its integral
    untouched. The angular-vel jitter is DIRECTIONAL (bone-pointing wobble), not just length.
  * **PBD** (`--pbd`) -- constraint projection over the WHOLE skeleton (arm + shoulder span +
    pelvis + torso sides), each bone at median length, correction DISTRIBUTED over both endpoints,
    no anchor, 8 Gauss-Seidel iters. Forearm-length std 25.7 -> 0.0mm.
  * **HYBRID** (`--hybrid`) -- PBD on the TORSO + chain-lock on the ARM. **Best.**

| measure | none | v1 chain | PBD whole | HYBRID |
|---|---|---|---|---|
| peak_velocity | -84 | -18 | **-95 (worse!)** | **-1.3 (exact)** |
| peak_elbow_ang_vel | +56.6 | -6.8 | +9.6 | **+0.1 (exact)** |
| max_trunk_displacement | -15.5 | -15.5 | **+5.4** | +4.8 |
| shoulder_flexion_reaching | -6.0 | -6.0 | -2.9 | -3.3 |

**The two halves need OPPOSITE treatments, and that is the finding.** TORSO wants PBD: it
stabilises the reference frame the shoulder angles and trunk displacement are measured against,
and it is what finally fixed **max_trunk_displacement (-15.5 -> +4.8)** -- a gap every arm-only
variant left stuck and which we had written off as "proxy mismatch". It was skeleton
inconsistency. ARM wants the CHAIN, not PBD: whole-skeleton PBD makes peak_velocity WORSE than
baseline (-95 vs -84) because distributing corrections pulls the WRIST around and damps its speed;
the chain leaves the wrist free at the end so real speed survives.

HYBRID tally vs baseline: **6 better, 2 same, 4 worse.** peak_velocity -84 -> -1.3 mm/s and
peak_elbow_ang_vel +56.6 -> +0.1 deg/s are essentially EXACT. Cost: the abduction measures regress
(-1.8 -> +6.7) and elbow_extension slightly (7.9 -> 9.0). (The abduction proxy has a known
sign-convention flaw, so those are the least trustworthy numbers in the set.) All O(T) vector
math -- no optimiser, no body model. Still ONE trial; a cohort run must confirm before any of this
is trusted, and the median bone lengths need checking across trials/participants.

### The full picture: all 13 measures x 7 variants -- NO variant wins across the board

```
measure                            OMC    none      kf   kfrts v1chain v2indep    PBD  HYBRID
---------------------------------------------------------------------------------------------
total_movement_time                5.2   +0.0*   +0.0*   +0.0*   +0.0*   +0.0*   +0.0*   +0.0*
peak_velocity                    623.6  -84.4   -56.2  -200.8   -18.0   -34.9   -95.2    -1.3*
time_to_peak_velocity              0.2   +0.1    +0.1    +0.0*   -0.0    -0.0    -0.0    -0.0
time_to_peak_velocity_percent     21.8   +9.1    +7.3    +0.0*   -3.6    -3.6    -1.8    -3.6
number_of_movement_units           1.0   +1.0*   +3.0    -1.0*   +1.0*   +2.0    +1.0*   +1.0*
max_trunk_displacement            32.9  -15.5   -15.7   -16.7   -15.5   -15.5    +5.4    +4.8*
elbow_extension_reaching          74.7   +7.9*   +8.7   +16.3   +11.7    +7.9*   +9.2    +9.0
shoulder_flexion_reaching         45.8   -6.0    -5.1    -6.9    -6.0    -6.0    -2.9*   -3.3
shoulder_flexion_drinking         43.9   -7.8    -7.4*   -8.3    -7.8    -7.8    -8.9    -9.0
shoulder_abduction_reaching      -20.1   -5.3    -4.4*   -6.0    -5.3    -5.0    -4.8    -6.7
shoulder_abduction_drinking      -21.5   -1.8    -1.0*   -1.7    -1.8    -2.0    +5.9    +6.7
peak_elbow_ang_vel               153.7  +56.6   +46.1   -30.3    -6.8   +56.6    +9.6    +0.1*
interjoint_coordination            1.0   -0.1    -0.2    -0.0*   -0.1    -0.1    -0.1    -0.1
```
(* = closest to OMC. Delta from OMC, 0 = perfect.)

**The measures split into FAMILIES that want different treatments** -- HYBRID wins the 3 headline
ones (peak_velocity, peak_elbow_ang_vel, max_trunk_displacement); kfrts wins timing +
interjoint; **kf wins shoulder_flexion_drinking and BOTH abductions -- exactly where HYBRID is
WORST**; and plain baseline/v2 win elbow_extension (every geometric fix hurts it). A production
choice would either compute each measure from its own best-processed track (legitimate -- they are
independent) or accept HYBRID's trade.

**`number_of_movement_units`: NOTHING fixes it.** OMC=1; none/v1/PBD/HYBRID->2, kfrts->0, kf->4,
v2->3 -- every variant off by >=1. It stays the genuine mocap-free casualty. The rule is also
brittle in its own right: the amplitude threshold tests the max's ABSOLUTE speed, not the wiggle
size, so a **7 mm/s dip** followed by the main 770 mm/s peak counts as a unit; and the `break`
tests only the FIRST max after each min, so a dip followed by a small bump is rejected even when a
big peak follows immediately. Fragile by construction, not just under jitter.

### How HYBRID works (recorded because the WHY is the transferable part)
* **Torso -> PBD (distributed).** Torso bones form a closed quad (sh<->sh, hip<->hip, R-sh<->R-hip,
  L-sh<->L-hip). Median length per bone over the trial; then per frame per bone,
  `corr = ((|d|-L)/|d|)*d`, and **each endpoint moves HALF, opposite ways**. Iterate 8x -- required
  because the bones SHARE joints (fixing the shoulder span moves both shoulders and breaks the
  shoulder->hip bones). The 50/50 split means **no anchor**, so no joint absorbs the whole error ->
  stable trunk frame -> fixes max_trunk_displacement.
* **Arm -> chain-lock.** Shoulder (now stable) anchors; elbow re-placed along its observed
  direction at median Lu; wrist along its direction from the new elbow at median Lf.
* **Why the arm must NOT use PBD:** in the chain the wrist KEEPS ITS DIRECTION, so only the RADIAL
  distance is clamped -- and during a reach nearly all wrist speed is TANGENTIAL, which passes
  through untouched. Only the radial in-and-out (the fake wander) is removed. Surgical. PBD instead
  nudges the wrist by half of every constraint error plus corrections propagating from the torso,
  8x over, perturbing the tangential motion too -- which is why whole-skeleton PBD makes
  peak_velocity WORSE than baseline (-95 vs -84). **PBD for stability, chain for speed.**

### CORRECTION: HYBRID makes frame-level JITTER worse -- the geometric fix needs TEMPORAL consistency
Spotted by eye on the render ("I feel like it increases the jitter"), then measured. Correct. The
per-frame geometric fix is **memoryless**: it re-places a joint along whatever direction that
single frame observed, at a fixed radius, so direction noise is baked in at FULL LEVER ARM. Jerk
(median |3rd difference|, mm -- peels off velocity+acceleration, leaves jitter):

    joint            raw    HYBRID   HYB+temp    OMC
    right_wrist     6.17   10.30      6.26      0.10
    right_elbow     6.42    9.27      6.39      0.08
    right_shoulder  3.59    6.31      6.31      0.05

**Why the measures improved anyway, and why that is uncomfortable:** score.py computes them AFTER a
4Hz lowpass, so frame-level jitter never reaches them. Bone-lock removed the radial wander that
distorted the peak -> the SMOOTHED trajectory improved while the RAW trajectory got noisier. **We
were optimising a number that could not see the damage being done.** Lesson: check a jitter metric
alongside the measures, and look at the render -- the eye caught what every scalar missed.

**Fix: smooth the DIRECTION, not the position** (`--hybridt`, `_smooth_dir`: lowpass the unit
direction at 8Hz, re-normalise, then place the joint). Keeps the bone rigid and the real swing,
removes the angular wobble. Costs almost nothing:

    peak_velocity      -84 -> -1.3 (hybrid) -> -9.9 (hyb+temp, still 8x better than baseline)
    peak_elbow_ang_vel +56.6 -> +0.1 -> -1.8
    time_to_peak_%     +9.1 -> -3.6 -> +1.8 (temporal is BETTER)
    shoulder_abduction_reaching  -5.3 -> -6.7 -> -8.6 (worse)

**Prefer `--hybridt`.** Two residuals stay honest: the SHOULDER is still worse than raw (6.31 vs
3.59) because the torso PBD is still memoryless and needs the same treatment; and even fixed we
only get BACK TO raw's jitter, never below. **OMC jerk is 0.05-0.10 -- mocap is ~60x smoother than
any variant we have.** Neither geometry nor this smoothing closes that gap.

### ...AND THE CORRECTION TO THAT CORRECTION: the temporal smooth is POINTLESS (measures already lowpass)
Challenged with "there's no measure that improves with the temp, and what's the point if we already
have temporal smoothing" -- both correct. **score.py ALREADY lowpasses at 4Hz (`_smoothed_xyz`)
before computing every measure.** Right-wrist jerk (mm):

    variant      raw track    AFTER score.py's 4Hz lowpass
    raw               6.17         0.0852
    HYBRID           10.30         0.1042
    HYB+temp          6.26         0.1035
    OMC               0.10         0.0169

**The 4Hz lowpass erases the entire difference** -- the raw-track gap I was "fixing" (10.30 vs
6.26) collapses to 0.1042 vs 0.1035, a **0.7%** difference. Consequently NO measure meaningfully
improves with `--hybridt`, and peak_velocity (-1.3 -> -9.9) and peak_elbow_ang_vel (+0.1 -> -1.8)
get WORSE: smoothing at 8Hz then 4Hz is double-smoothing = the same peak-flattening as KF+RTS.
**Use `--hybrid`; `--hybridt` is kept only as a recorded negative result.**

**The real finding is a RENDER/METRIC MISMATCH, not a jitter problem.** HYBRID's jitter increase is
COSMETIC -- it never reaches a measure. It looked alarming only because
`render_mmc_jitter_delta.py` draws the RAW 3D while every measure reads the 4Hz-SMOOTHED 3D, i.e.
**the video shows something no measure ever sees** -- a dumb-player violation (the render must draw
what the numbers are computed from). The fix is to the RENDER, not to add smoothing the pipeline
already applies. Two process lessons: (1) I "fixed" a metric that could not see the damage, then
"fixed" damage that no metric could see -- check where in the chain a number is actually consumed
before optimising it; (2) an eye-catch is a reason to MEASURE, not a reason to immediately patch.

Residual that IS real: **OMC jerk 0.05-0.10 vs raw MMC ~6 (~60x smoother); even after the 4Hz
lowpass, 0.017 vs 0.085 (~5x).** Nothing we tried closes that gap.

### Skeleton-aware smoother (hybrid -> smooth -> hybrid): beautiful jitter, ZERO measure gain
Idea (user's): "smoothing + a second iteration", and "a smoothing that takes into account the
skeleton and doesn't treat each point as independent". Both correct in mechanism, and the
composition works spectacularly ON THE TRACK:

    variant                      forearm-len std   wrist jerk
    raw                             25.75 mm          6.17
    hybrid                           0.000 mm        10.30   (projection INJECTS jitter)
    hybrid x2 (no smooth between)    0.000 mm        10.30   (2nd pass = literal NO-OP, 1e-6mm --
                                                              the chain projection is exact in one
                                                              step; the SMOOTH is what makes
                                                              iterating mean anything)
    hybrid -> smooth@12Hz -> hybrid  0.000 mm         1.10   (9x better, peak UNCHANGED 687->687)
    hybrid -> smooth@4Hz  -> hybrid  0.000 mm         0.11   (= OMC's 0.10, but peak 687->651)

Per-point smoothing alone breaks the bones; projection alone bakes in per-frame direction noise.
Composing fixes both -- the smooth kills the high-freq jitter, the 2nd projection restores
rigidity, and since its INPUT is now smooth it re-injects nothing. That IS a skeleton-aware
smoother: the projection re-couples the joints the per-point filter treated independently. At 12Hz
it is free (jitter is >12Hz; real motion and the peak live <4Hz).

**BUT: it does not improve a single measure.**

    measure                HYBRID   hyb2@12
    peak_velocity           -1.28    +1.49   (wash, both ~exact)
    peak_elbow_ang_vel      +0.10    +2.30   (slightly WORSE)
    max_trunk_displacement  +4.79    +7.50   (WORSE)
    shoulder_flex_drinking  -9.00    -8.40   (slightly better)
    ...rest same

**Same trap as the previous entry, walked into twice:** score.py lowpasses at 4Hz, so the >12Hz
jitter never reached the measures -- neither the jitter HYBRID injects (so the earlier alarm was
misplaced too) nor the jitter this removes. **USE PLAIN `--hybrid`.** `--hybrid2` is kept because
it has real value for (a) the RENDER -- a rigid, mocap-smooth skeleton is what a clinician should
see, and it fixes the dumb-player mismatch (the video currently draws raw 3D no measure reads) --
and (b) any future consumer that does not lowpass.

**Transferable lesson: check WHERE a number is consumed before optimising it.** Two rounds of
excitement (a 60x jitter win landing exactly at the mocap floor) over a quantity nothing
downstream reads.

### COHORT (n=11) OVERTURNS THE SINGLE-TRIAL CONCLUSION -- HYBRID's "exact" was luck
Processed 10 more P14 R_unaffected trials (video + c3d, ~40s/trial batched) and re-ran the
comparison. **Everything concluded from trial_1 alone was wrong, in both directions.**

    measure                        none          HYBRID        v1chain      (bias +/- trial noise)
    peak_velocity            -41.9 +/-40.6   -88.7 +/-49.1  -107.3 +/-50.3
    peak_elbow_ang_vel       +69.2 +/-28.8   +36.9 +/-28.5   +27.9 +/-17.4
    max_trunk_displacement   -21.1 +/- 8.2    -9.0 +/-12.3   -21.1 +/- 8.2
    elbow_extension (max)     +4.0 +/- 1.9    +3.6 +/- 3.0    +2.8 +/- 1.6
    shoulder_flexion_reach    -5.8 +/- 1.6    -4.2 +/- 1.9    -5.9 +/- 1.6

**trial_1 gave HYBRID peak_velocity -1.3 and peak_elbow_ang_vel +0.1 -- both LUCK.** Across 11
trials HYBRID's peak_velocity bias is **-88.7, twice as bad as the -41.9 baseline**: HYBRID does
not fix peak_velocity, it HURTS it, and plain `none` is the best variant for that measure. The
entire "HYBRID is the winner" story was built on one unrepresentative trial -- precisely the
failure the project's own rule ("never conclude from one scalar") exists to prevent.

**What SURVIVES the cohort:**
* The geometric bone-lock DOES help angular velocity, consistently: +69.2 -> +27.9 (v1chain),
  and with LOWER trial noise (+/-17.4 vs +/-28.8). Real.
* **v1chain (the simple arm bone-lock) is the most reliable** -- best elbow_extension
  (+2.8 +/-1.6), best ang_vel, lowest noise. The elaborate HYBRID/PBD adds noise for little gain.
* Torso PBD does halve the trunk bias (-21.1 -> -9.0) but adds noise (+/-8.2 -> +/-12.3).

**BIAS vs NOISE at the MEASURE level (the question n=1 cannot answer):** only the shoulder-flexion
measures are bias-dominated (69%/82% -> would cancel within-subject); everything else is ~half
trial-to-trial NOISE, which does NOT cancel. The non-cancelling part as a fraction of the clinical
(affected-vs-unaffected) effect: elbow_extension 35% (best), shoulder_flexion_reach 47%, trunk 69%,
peak_velocity 83%, shoulder_flexion_drink 95%, peak_elbow_ang_vel 112%, abduction 384%.
**Nothing is below 35%.** So the "it is just a calibratable bias that cancels" story is much weaker
than the single trial suggested -- the honest position is that no measure is comfortably usable yet
on this evidence.

Harness: `scripts/bias_vs_noise_delta.py`. STILL one participant -- the clinical effects are P14's
own impairment pattern, and the n=11 noise estimates are same-session same-side.

### THE CLINICAL TEST: affected vs unaffected -- 2 SIGN FLIPS, fully explained by L-R bias asymmetry
Fetched + processed 10 L_affected trials (now 11 unaff + 10 aff, `scripts/affected_vs_unaffected_delta.py`,
using `--bonelock`/v1chain, the cohort's most reliable variant). Phases from the OMC cup on both
sides, so cup detection is not a confound. Result:

    measure                     OMC (mocap)          MMC (ours)        verdict
    total_movement_time     +0.61 (d=+1.59)     +0.47 (d=+1.18)      recovers
    shoulder_flexion_reach  +2.68 (d=+1.79)     +2.44 (d=+0.84)      recovers
    number_of_movement_units +0.31 (d=+0.44)    +0.39 (d=+0.30)      recovers (both weak)
    elbow_extension         -8.83 (d=-3.35)     -1.81 (d=-0.68)      WEAK
    peak_elbow_ang_vel     -28.08 (d=-3.87)    -13.93 (d=-0.83)      WEAK
    peak_velocity          -83.16 (d=-1.14)    +17.66 (d=+0.19)      **SIGN FLIP**
    max_trunk_displacement -13.10 (d=-1.56)     +0.47 (d=+0.12)      **SIGN FLIP**

Cruel pattern: the measures mocap is BEST at (elbow_extension d=-3.35, peak_elbow_ang_vel d=-3.87)
are the ones we degrade MOST (to -0.68/-0.83). total_movement_time recovers best -- because it
depends only on PHASE BOUNDARIES, not on any landmark position.

**FULLY EXPLAINED: the left and right arms carry DIFFERENT landmark biases, and comparing arms
injects that difference into the effect. The arithmetic is exact:**

    measure              true OMC effect   + (L-R bias gap)  = predicted   measured
    peak_velocity            -83.2            +100.8            +17.6       +17.66  ✓
    max_trunk_displacement   -13.1             +13.6             +0.5        +0.47  ✓
    elbow_extension           -8.8              +7.0             -1.8        -1.81  ✓
    peak_elbow_ang_vel       -28.1             +14.2            -13.9       -13.93  ✓
    total_movement_time      +0.61              -0.1             +0.5        +0.47  ✓

    => MMC effect = true effect + (left bias - right bias)      [every measure, to 2dp]

The right wrist carries a -98.0 mm/s peak_velocity bias, the left +2.8 -- a **+100.8 artefact,
LARGER THAN and OPPOSITE TO the real -83.2 effect**, so the impairment is buried and reversed.
This is deterministic, not noise: no filter, smoother or geometry can touch it.
total_movement_time survives precisely because its L-R gap is ~0.

**INTERPRETATION -- this was the HARDEST version of the question, and arguably the WRONG one.**
Comparing a LEFT arm to a RIGHT arm requires the left/right landmark offsets to match; they do not.
The actual clinical use-case is **the SAME arm over time** (pre vs post therapy), where the
IDENTICAL bias applies to both timepoints and **cancels exactly** -- the L-R gap does not exist in
that design. So: **the pipeline cannot do cross-side comparison without per-side landmark
calibration; this test says nothing against within-side longitudinal use.** That is the design to
test next, and DELTA cannot answer it (single session per participant).

### CONCURRENT VALIDITY: do the measures CORRELATE across trials? (the test that matters most)
Everything above measured agreement in ABSOLUTE value. This asks the question that decides
usefulness: when mocap says trial A > trial B, does MMC agree? If yes, the bias is JUST an offset
to calibrate and the measure is usable. Computed WITHIN each side (pooling sides would fake a
correlation out of the group difference).

    measure                    RIGHT r (n=11)   LEFT r (n=9)   verdict
    total_movement_time            +0.95*          +0.91*      TRACKS
    elbow_extension_reaching       +0.88*          +0.72*      TRACKS
    shoulder_flexion_reaching      +0.81*          +0.78*      TRACKS
    max_trunk_displacement         +0.71*          +0.55       partial
    peak_velocity                  +0.60*          +0.71*      partial
    number_of_movement_units       -0.22           -0.37       NOISE
    peak_elbow_ang_vel             +0.07           -0.54       NOISE
    (* p<0.05)

**THIS REVERSES THE PESSIMISM of the affected-vs-unaffected test.** That test failed on the L-R
landmark asymmetry -- an artefact of comparing a LEFT arm to a RIGHT arm. The WITHIN-side
correlation maps to the real clinical use-case (same arm, pre vs post therapy) and says **3
measures are usable today with a calibration offset**: total_movement_time, elbow_extension,
shoulder_flexion_reaching. Best of all, **elbow_extension is mocap's STRONGEST discriminator
(d=-3.35) AND we track it at r=0.88/0.72** -- the most promising measure in the set.

**THE SHARPEST LESSON OF THE SESSION -- reducing bias != tracking signal.**
`peak_elbow_ang_vel`: the bone-lock "improved" its bias +69 -> +28, which I reported as a win. But
its across-trial correlation is **+0.07 / -0.54 -- we do not track it AT ALL.** The fix was
COSMETIC: it centred the mean while every per-trial value stayed noise. Hours were spent optimising
biases (bone-lock, PBD, HYBRID, smoothers) without ever checking whether the per-trial values
CORRELATED. This one 20-line test would have exposed that on day one. **Check concurrent validity
BEFORE optimising agreement.**

`number_of_movement_units` confirmed dead from a 4th independent angle (r=-0.22/-0.37).

Harness: inline in this entry's run; see `scripts/affected_vs_unaffected_delta.py::_one` for the
per-trial measure extraction. STILL P14 only.

### FINAL VERDICT ON THE PROCESSING VARIANTS: raw wins. Everything built today was worthless or harmful.
Concurrent validity r(OMC,MMC) across trials, within-side, by variant (n=11 R + 9 L):

    measure                      raw   bonelock     pbd    hybrid
    total_movement_time        +0.94     +0.93    +0.93     +0.92
    elbow_extension_reaching   +0.81     +0.80    +0.77     +0.67
    shoulder_flexion_reaching  +0.80     +0.80    +0.80     +0.79
    max_trunk_displacement     +0.63     +0.63    +0.31     +0.38
    peak_velocity              +0.59     +0.66    +0.73     +0.67
    peak_elbow_ang_vel         -0.25     -0.24    -0.10     -0.21

**RAW -- no processing at all -- is best or tied-best on 4 of 6 measures.** PBD and HYBRID
actively DESTROY validity where it existed: max_trunk 0.63 -> 0.31/0.38, and HYBRID drops
elbow_extension 0.81 -> 0.67. **The single genuine win from the entire day of engineering is PBD on
peak_velocity (0.59 -> 0.73). One cell.**

So the whole technical arc of this session -- bone-lock (v1/v2), whole-skeleton PBD, HYBRID,
direction-smoothing, KF, KF+RTS, the skeleton-aware smoother -- was **worthless or harmful by the
only metric that decides usefulness.** Each was validated against BIAS REDUCTION, which is an
irrelevant target: `peak_elbow_ang_vel` had its bias "fixed" +69 -> +28 by bone-lock while its
across-trial correlation is -0.25 (we never tracked it at all). HYBRID, championed hardest and
written up as "the best variant", is near-worst on almost everything.

**RECOMMENDATION: ship RAW.** Optionally PBD if peak_velocity specifically matters. Delete the
rest, or keep only as recorded negative results.

**THE META-LESSON, stated for whoever reads this next:** agreement-in-mean and per-trial tracking
are nearly UNRELATED, and only the second one determines whether a measure is usable. Check
concurrent validity FIRST -- it is a 20-line test -- and only then consider whether any processing
is worth building. Hours went into geometry that a single correlation would have pre-empted.

### THE ACTUAL FIX: the ESTIMATORS are broken, not the pipeline. max() is a noise detector.
Chasing why `time_to_peak_velocity` and `peak_elbow_ang_vel` fail led to the finding that reframes
the whole day. **Both signals are excellent; both measures are destroyed by their estimator.**

**peak_elbow_ang_vel** (shared phases, n=20):

    statistic    r(OMC,MMC)    OMC    MMC   ratio   OMC d   MMC d
    max (today)      +0.07    127.0  163.0   1.30   -3.87   -0.46   <- USELESS
    p99              +0.18    126.6  158.9   1.27   -3.94   -0.67
    p95              +0.77    122.2  135.2   1.11   -4.19   -2.46
    p90              +0.85    114.7  118.6   1.03   -5.02   -3.29   <- BEST (ratio 1.03 = unbiased)
    p75              +0.82     91.9   90.2   0.98   -3.55   -3.25
    mean             +0.98     57.2   58.1   1.01   -4.44   -5.73   <- r=0.98!

**The p99->p95 jump (r 0.18 -> 0.77) localises the damage: the noise lives in the top ~1-5% of
samples.** max() is GUARANTEED to select a jitter spike. Skip those few samples and the measure
works. The MEAN angular velocity tracks at **r=0.98** and the values match (57.2 vs 58.1) -- we
were always measuring the signal perfectly; `max` was measuring our jitter. (The mean even
out-discriminates mocap's own max: MMC d=-5.73 vs OMC d=-3.87 -- though it is a different
construct, average vs peak, so it needs clinical buy-in not just better statistics.)

**time_to_peak_velocity**: the reach velocity profile is a broad bell -- **7.8 frames sit within 5%
of the max** on a 62-frame window. `argmax` picks ONE arbitrary sample off that plateau, so it
inherits the plateau width as error (4.8 fr) -- while the real between-trial signal is only 4.6 fr.
SNR ~= 1 => r caps at 0.70 by arithmetic. Estimator sweep (shared phases):

    argmax +0.64 | parabola +0.66 | centroid>95% +0.73 | centroid>90% +0.75 |
    centroid>80% +0.86 | centroid_all +0.95

**centroid>80% is the pick** (+0.86, keeps the full signal SD 4.5 fr and the MMC clinical effect:
+0.049 -> +0.050s). `centroid_all` has the best r (0.95) but a shrunken signal SD (3.5) -- more
reliable about a different, less variable quantity. NOTE the parabola sub-frame fix does NOT help
(+0.66): it refines within +/-1 frame while the error is a 4-frame slide across a plateau -- the
obvious precision fix attacks the wrong scale.

**THE LESSON OF THE ENTIRE SESSION:**
    signal quality was never the problem. The measures' ESTIMATORS were.
    * max()    on a noisy signal   -> selects for NOISE by construction
    * argmax() on a flat plateau   -> selects arbitrarily
Days of geometry (bone-lock v1/v2, PBD, HYBRID, KF, KF+RTS, direction-smoothing, skeleton-aware
smoother) produced ONE improvement across all measures. Two one-line estimator changes produced
**+0.78** (peak_elbow_ang_vel: 0.07->0.85) and **+0.22** (time_to_peak: 0.64->0.86).
**Fix the estimator before touching the pipeline.**

RECOMMENDATION: in score.py / murphy_measures.py, replace `max(angular_velocity)` with p90, and
`argmax(speed)` with the centroid of the >80%-of-max region. Validate on a 2nd participant.

### RETRACTION: the "estimator fix rescues peak_elbow_ang_vel (0.07->0.85)" headline was POOLED = WRONG
Asked "does a good correlation mean constant bias and little noise?" -- which exposed two errors.

**1. I published a POOLED correlation.** Pooling affected+unaffected manufactures r out of the
GROUP DIFFERENCE. I identified this exact trap earlier in the session, warned about it in writing,
then reproduced it in the estimator sweep. Honest PER-SIDE numbers:

    measure / estimator          POOLED    RIGHT     LEFT   mean(per-side)
    peak_elbow_ang_vel  max       +0.07    -0.26    -0.42       -0.34
    peak_elbow_ang_vel  p90       +0.85    +0.69    -0.39       +0.15   <- NOT rescued
    time_to_peak     argmax       +0.64    +0.68    +0.22       +0.45
    time_to_peak   centroid       +0.86    +0.91    +0.56       +0.74   <- REAL fix
    peak_velocity       max       +0.65    +0.87    +0.71       +0.79
    peak_velocity       p90       +0.87    +0.92    +0.84       +0.88   <- real, modest

So the estimator lesson is REAL but SMALLER than claimed: **one solid fix (time_to_peak +0.29), one
modest (peak_velocity +0.09), one FAILURE (peak_elbow_ang_vel: -0.34 -> +0.15, still useless; the
left arm is NEGATIVE).** peak_elbow_ang_vel is the derivative of a jittery angle; nothing rescues
it. Also note peak_velocity was ALREADY fine per-side (0.79) -- the earlier "0.59" was the
own-phases run. Docs (README + memory) corrected.

**2. A HIGH r DOES NOT MEAN A CONSTANT BIAS** -- r is invariant to BOTH offset and scale, so it only
says the scatter about SOME line is small. The SLOPE distinguishes them, and real SCALE errors exist:

    measure                    slope (R/L)   meaning
    max_trunk_displacement      0.32 / 0.36  we capture only 1/3 of the real trunk range
    shoulder_flexion_reaching   1.39 / 1.26  we EXAGGERATE the range 26-39%
    elbow_extension_reaching    0.86 / 0.81  we compress ~15-20%

**max_trunk_displacement has r=0.71 AND slope 0.32** -- it ranks trials correctly while compressing
magnitude 3x. Clinically serious: trunk displacement IS the compensation measure. An offset needs
one subtraction; a slope needs a GAIN, and compression means extremes stay under-reported even
after calibration. (My "slope~1 -> pure offset" verdicts were junk: with n~10 the slope stderr is
huge, so "not significantly different from 1" is a statement about NO POWER. A slope of 1.39 got
labelled "pure offset". Report the CI, not a significance verdict.)

### peak_elbow_ang_vel: SETTLED -- NOT RECOVERABLE (and the "mean" is a different measure, not a fix)
Full per-side breakdown + the proxy checks that decide it (P14, n=11 R / 9 L, all WITHIN side):

    statistic     POOLED   RIGHT    LEFT   mean/side
    max            +0.07   -0.26   -0.42     -0.34    <- as shipped: selects jitter spikes
    p90            +0.85   +0.69   -0.39     +0.15    <- my "fix": fails on the AFFECTED arm
    mean           +0.98   +0.96   +0.64     +0.80    <- tracks... but see below
    median         +0.86   +0.37   +0.63     +0.50
    (elbow ANGLE's own range)  +0.83  +0.88  +0.86    <- the SIGNAL is fine

    r(OMC max , OMC p90 )  = +0.87 / +0.87   -> p90 IS a legitimate proxy for the max
    r(OMC max , OMC mean)  = +0.32 / +0.29   -> the MEAN is a DIFFERENT CONSTRUCT
    r(MMC mean, OMC max )  = +0.23 / -0.10   -> our mean does NOT recover mocap's max

**Verdict: the measure is well-posed (p90 proxies the max at 0.87 on both sides) but WE CANNOT
MEASURE IT (+0.69/-0.39).** The mean we CAN measure (0.96/0.64) is nearly INDEPENDENT of the max
(r~0.3) -- adopting it would SILENTLY SUBSTITUTE a different measure, not fix this one.

**The nastiest part: the AFFECTED arm fails worst.** It moves slower (mean eav 48.3 vs 64.5), so
the signal shrinks while our noise stays constant -> **the measure degrades most exactly where the
impairment is greatest.** That is the opposite of what a clinical measure must do, and it means
peak_elbow_ang_vel cannot be rescued by any estimator on this hardware.

OPPORTUNITY (needs clinical buy-in, not just statistics): **mean elbow angular velocity** is
well-tracked (r=0.80 per-side) AND highly discriminative (MMC d=-5.73, stronger than mocap's own
peak at d=-3.87). A plausible NEW mocap-free-native measure -- but it must be justified as its own
construct, not rebadged as "peak".

### REFRAMING (and un-retraction): WITHIN-HAND r IS THE WRONG CRITERION FOR LOW-VARIANCE MEASURES
Prompted by: "the inter-hand correlation can also be small just because the measure doesn't change
much trial to trial". Correct, and it overturns my previous entry. A low within-hand r has TWO
possible causes -- our noise, OR **the measure barely varying within a hand** (nothing to
correlate). The diagnostic is mocap's OWN within-hand SD and the implied CEILING on r:

    measure                      OMCsig  ourErr  maxR   gotR |  OMC d   MMC d   verdict
    total_movement_time            0.39    0.00  1.00  +1.00 |  +1.59  +1.59   works (degenerate)
    elbow_extension_reaching       2.53    1.56  0.85  +0.81 |  -3.35  -1.86   WORKS
    peak_elbow_ang_vel  p90        5.34    9.52  0.49  +0.15 |  -5.02  -3.29   **WORKS**
    peak_elbow_ang_vel  max        7.24   22.17  0.31  -0.34 |  -3.87  -0.46   useless
    peak_velocity       p90       69.60   35.25  0.89  +0.88 |  -1.23  -0.62   weak
    shoulder_flexion_reaching      1.50    1.91  0.62  +0.73 |  +1.79  +0.20   we LOSE the signal
    max_trunk_displacement         7.78    6.05  0.79  +0.63 |  -1.56  +0.12   SIGN FLIP
    number_of_movement_units       0.64    1.50  0.39  +0.11 |  +0.44  +0.55   no signal in MOCAP
    time_to_peak__centroid         0.07    0.05  0.83  +0.74 |  +0.45  +0.57   no signal in MOCAP

**UN-RETRACTION: the p90 fix for peak_elbow_ang_vel IS REAL.** Its within-hand r is only +0.15 --
but the CEILING is 0.49, because mocap's own within-hand SD is just 5.34. Judging it by within-hand
r was asking a low-variance measure a question it cannot answer. By the criterion that matters --
**does it detect the impairment** -- p90 gives **d=-3.29 vs mocap's -5.02**, while `max` gives
**-0.46**. So `max`->p90 turns a useless measure into a working one. (My earlier retraction was
right about the POOLING error and wrong about the conclusion.)

**AND THE STING: `time_to_peak` has NO CLINICAL SIGNAL EVEN IN MOCAP (OMC d=+0.45).** My proudest
result of the day -- "centroid fix, r 0.45->0.74" -- improved the reliability of a measure that has
nothing to detect. `number_of_movement_units` likewise (OMC d=+0.44), now confirmed dead for the
RIGHT reason.

**METHOD RULE: report BOTH (a) within-hand r AGAINST ITS SNR CEILING, and (b) between-hand d in
BOTH mocap and MMC.** They answer different questions: (a) can we track trial-to-trial variation,
(b) can we detect the impairment. A measure can fail (a) and ace (b). And ALWAYS check OMC's own d
first -- if mocap cannot discriminate with the measure, our error is irrelevant.

================================================================================
## 2026-07-17 — DELTA camera-quality: desync vs miscalibration, and the cohort
================================================================================

Big picture: what looked like "the pose/measures don't transfer to DELTA" was almost
entirely BAD CAMERAS poisoning triangulation, not the pipeline. Corrected the whole
diagnosis, built (and mostly retired) sync-repair tooling, and characterized the cohort.

### Corrections to earlier claims (all retracted/fixed in
### object_tracking/docs/context/project_delta_cohort_transfer.md)
1. "P14 didn't replicate / pose broken on P15/P17/P19" — WRONG. It was bad cameras.
   robust_triangulate seeds from an all-cam DLT then ejects the worst reprojector
   (breakdown point 50%). With ~5/10 cams bad (P15), the seed is corrupted and it ejects
   the GOOD cams -> bimodal 0-or-10 consensus. Restrict to good cams: P15 wrist coverage
   25->96-100%, reproj 10.8->7.5px, elbow-angle err 24->4deg; concurrent validity
   elbow_extension P15 0.32->0.95 (matches P14 0.87). REPLICATION RECOVERED.
2. The "L-R bias" framework was CIRCULAR (algebraic identity a-b+b=a). Retracted.
3. "cup detector doesn't transfer" was P14-only; COCO teacher yolo26x-seg gets 77%
   >=3-cam consensus on P14 (our BRIO finetune 8.2%).

### Camera failure modes: DESYNC vs MISCALIBRATION (two different problems)
- DESYNC = camera shows a different INSTANT. 2D looks fine; disagrees in 3D during MOTION.
- MISCALIB = 2D right, geometry wrong. High reproj EVEN WHEN STILL.
- The RELIABLE classifier is REPROJECTION vs a RANSAC consensus, STRATIFIED BY WRIST SPEED
  (still<3mm/fr vs moving). NOT wrist-speed correlation -- that is VIEWPOINT-CONFOUNDED
  (P10/P19 cam2 read low-corr but reproject at 16-18px = FINE; a fine camera whose wrist
  moves toward/away has low apparent 2D speed).
- Discriminator: desync => low still-reproj + high move-reproj + a CONSISTENT lag (low SD,
  high r) that explains it. miscalib => high still-reproj, no lag helps (small lag cannot
  cause large still-reproj).

### RANSAC reference (should have done from the start)
reaudit_cam_quality.py: per-frame 3D from the LARGEST mutually-agreeing camera subset
(pairwise-seeded, max-inlier, refit). Survives >50% bad cams where all-cam iterative
ejection fails. Consensus reached jumped P19 7%->91%, P15 45%->100%, P17 47%->88%.
VALIDATED against ground truth: P11 came out 0% consensus, matching the study's own
2024_10_23 calibration_errors.csv (P11 = 57.82px, catastrophic).

### COHORT (10 participants with calib; P24/P25 have NO calib on share) -- usable cams in 1-5
  P07 5/5, P08 5/5, P15 5/5  (clean now)
  P13 5/5 after cam2 re-cut (the ONLY true re-cuttable desync in 1-5; +139fr, verified)
  P10 4/5 (cam4 miscal), P12 3/5 (cam4,5), P14 3/5 (cam4,5; its GOOD cams are 6,7,9,10),
  P19 3/5 (cam2,5), P17 2/5, P11 0/5 (calib broken).
=> CLEAN 5-camera cup cohort = P07, P08, P15 now + P13 after one re-cut = 4 participants.
=> SYNC REPAIR IS DONE. Every other bad camera is MISCALIBRATION -> the lever to grow the
   cohort is RECALIBRATION (05_Calib_before/ Charuco footage, or bundle adjustment), not sync.

### Sync-repair tooling (built, verified, barely needed)
- sync_fix_delta.py: per-camera constant-lag estimate + reindex (superseded by re-cut).
- delta_recut.py --align: MOTION-ENERGY alignment (ported from ~/Downloads/fix_slicing.py
  align_trial), run at FULL fps (1-frame precision vs the original's 10fps ~6-frame).
  Downloads uncut LOCALLY first (sequential read ~32MB/s -> ~33s/GB, vs 8MB/s many-small),
  caches coarse thumbs to cache/delta/_thumbs/*.npy. Verified on P13 cam2: r@lag0 -0.08->0.86,
  cup consensus (COCO teacher) 4cams 48% -> +fixed-cam2 59% (vs +broken-cam2 32%).
  KEY: for the CUP, a desynced camera HURTS (48->32) but once synced HELPS (48->59) -- cam2
  is a high-recall cup viewpoint (teacher 96%). So 5 synced cams > 4 for the cup, as the user
  argued. But this only mattered for P13 cam2 across the whole cohort.
- reproj_render_delta.py: diagnostic grid, GREEN=detected wrist vs RED=reprojected RANSAC
  consensus per cam; gap constant-when-still => miscalib, gap opens-in-motion => desync.
  (Note: verdict labels read reaudit_cam_quality.json which the last run overwrote to only
  P07/08/11/12; gap numbers are live/correct regardless. Re-run reaudit --parts <all> to
  restore full labels.)

### Cup label pool (ready for finetune)
build_cup_labels_delta.py --usable-cams: reject-then-fill (object_tracking pipeline) with
COCO teacher, RESTRICTED to sync+calib-clean cams (else fill/refill poison the bad cams).
P14 pool: 14,728 labels over 7 usable cams (11,725 real / 2,940 fill / 63 refill = 0.4%).
The 0.4% refill confirms the retained cams genuinely agree. Ready to train yolo26s (user's
stated student) -- labels are SEG polygons (yolo26n-seg/yolo26s-seg compatible; a merged
pose+cup detect head would need box labels + a rebuild).

### NEXT (open)
- Decide: cup finetune on the clean cohort NOW (recommended; deliverable, P14 pool ready)
  vs recalibrate the miscalibrated cameras first (grows cohort past 4).
- Recalibration is the real remaining lever (P10/P12/P14/P19 have miscalib cams in 1-5).
- P24/P25 blocked on missing calibration.
- data kept: uncut downloads in cache/delta/<P>/uncut/, re-cuts in .../recut/, thumbs in
  cache/delta/_thumbs/. NO deletes (bigrun.sh's rm was the lesson).

## 2026-07-20  DESYNC/MISCALIB RE-INVESTIGATION -> it was SHUFFLED CUTS (three-signal fingerprint)

The 2026-07-17 "recut vs miscalib" labels were made with too-narrow search windows and a
reprojection-only test. The user (watching renders) kept correctly flagging cameras I'd
mislabeled. Full re-investigation converged on a THREE-SIGNAL fingerprint per camera, and a
NEW root cause the earlier pass never named: the study's CUT CLIPS are placed at the wrong
time (often a DIFFERENT repetition of the drink task), which masquerades as miscalibration in
every geometric test.

### Why every single-signal test failed (each confounded a different way)
- reaudit / lag_probe (reprojection, shift WITHIN the cut clip): a mis-cut clip's correct
  content is OUTSIDE its window, so no within-clip shift reaches it -> "MISCALIB". Also capped
  at +-200 fr.
- uncut_offset_probe (reprojection, shift within +-15s of the uncut): when a camera is
  miscalibrated, reproj stays high at EVERY offset -> geometry masks any timing dip. And a
  weak aggregate dip (P12 cam4 +467) was an ARTIFACT of inconsistent per-trial offsets.
- motion_sync_delta / delta_recut --align (motion energy on 7s clips): repetitive drink motion
  gives many near-equal xcorr peaks -> scattered, low-q, useless for a constant offset.
- MEASURED whole-session (74min) motion xcorr is the ONLY periodicity-immune sync test
  (cam4<->cam5 same-PC lag 0.0 confirmed the user: they're a synced pair).

### The winning tool: cut_placement_audit.py (calibration-free, periodicity-free)
1. locate2(): each cut clip is a re-encode of its OWN uncut, so slide it over that uncut and
   take the PIXEL-EXACT match (fine 10fps sequence NCC ~0.999 at the true source vs ~0.986 at
   any other repetition). The v1 coarse-2fps argmax picked wrong repetitions (put two P13
   trials at one spot -- impossible); the fine top-k disambiguation fixed it, RE-VALIDATED on
   P13 (reproduces +1.8s constant on 4/6 trials, exposes 2 trials mis-cut +10/+17s).
2. whole-session motion xcorr gives the inter-PC recording offset (drift-free).
3. misplacement = (t_suspect - t_ref) - session_offset. Zero => correct cut.
   verdict: |mis|<=1s all trials CUTS-OK; constant>1s CONST-OFFSET; varying SHUFFLED.

### Multi-joint + consistency fingerprint (multijoint_reproj.py + scratchpad probes)
For CUTS-OK cams, decide real-miscalib vs detection-quirk by:
- multi-joint: geometry error is GLOBAL (all 9 COCO joints, incl static nose/shoulders/hips);
  a wrist-only error = detection quirk, NOT miscalib (rescued P14 cam2).
- per-trial: real miscalib repeats the SAME error MAGNITUDE and OFFSET VECTOR every trial.
- spatial: 4x3 grid, per-cell mean offset vector + COHERENCE = |mean vec|/mean|vec|. ~1.0 =
  smooth systematic field (geometry); ~0 = random (detection noise). P10 cam4 0.96 (field even
  sign-flips across image -> "left side fine" was a zero-crossing, not calibration being ok).

### FINAL COHORT (cams 1-5), classified
              fine        mis-cut(RE-CUT fixes)      real-miscalib(recalib)   now -> after re-cut
  P07  5/5    1,2,3,4,5   -                          -                        5/5
  P08  5/5    1,2,3,4,5   -                          -                        5/5
  P15  5/5    1,2,3,4,5   -                          -                        5/5
  P13         1,3,4,5     cam2 (+1.8s const, 2 trials shuffled)  -            4/5 -> 5/5
  P14         1,2,3,4     -                          cam5 (coh .93, 90px)     4/5 (cam2 RESCUED: wrist-only quirk)
  P17         1,2,3,4     -                          cam5 (coh dominant .88, +also 0.5s clock-desync)  4/5
  P10         1,2,3,5     -                          cam4 (coh .96)           4/5
  P12         1,2,3       cam4(shuffled ~+144s+/-), cam5(+144.3s const)  ?(unknown until re-cut)  3/5 -> up to 5/5
  P19         1,3,4       -                          cam2(mild 24px), cam5(severe 147px)  3/5
  P11         -           -                          all 5                    0/5
Cause census of 13 broken cams: 3 mis-cut, 9 miscalib, 1 both (P17 cam5). ZERO classic desync
in the original sense -- the study's cutter mostly DID compensate clock offsets (P10 +8.5s,
P11 -8s handled fine); it failed only by SHUFFLING whole repetitions (P12 pc3, P13 cam2) and a
0.5s residual (P17 cam5).

### STUDY'S OWN DOCS (cache/delta/study_docs/, copied off the share)
2024_10_23_1138_Calibration_errors.csv = per-participant calibration reprojection error (px):
  P08 0.85  P07 1.55  P13 2.69  P12 4.90  P15 4.31  P10 5.61  P11 57.82  (5-cam parts + P07/08)
  P14 3.58  P19 6.34  P17 13.32  (10-cam RMS -- NOT directly comparable to our cam1-5)
CROSS-CHECK (5-cam participants, directly comparable):
  * P12 (4.90) + P13 (2.69) have GOOD calibration -> their cam4/5/cam2 errors CANNOT be geometry
    -> INDEPENDENTLY CONFIRMS the mis-cut diagnosis. Strong external validation.
  * P11 (57.82) confirms genuinely-broken calibration (all 5, 0/5). 
  * P10 (5.61) modest 5-cam mean, one bad cam (cam4) diluted.
IMPLICATION -- recalibration may NOT fix "miscalib" cams: P14 calib was GOOD (3.58 over 10 cams
  -> cam5 could NOT have been 90px at calib time) yet cam5 reprojects 90px in the task with a
  coherent frozen field => the CAMERA MOVED between calib and task. There is ONLY 05_Calib_before
  (no calib_after), so re-running calibration reproduces the SAME extrinsics -- a moved camera is
  UNRECOVERABLE for 3D. Recalibration only rescues source-bad calibration (P11). Moved cams
  (likely P14 cam5, P19 cam5, maybe P10 cam4) are stuck for 3D, though their 2D/cup dets may
  still be usable single-view.
P13 Notes .txt: "Sync Box bei RPS korrigiert" -- study WAS aware of sync + had a sync box; P13
  is exactly where we found the cam2 cut shuffle. Only P13 has a notes file.

### NEW SCRIPTS (all in scripts/, cache-only unless noted)
- cut_placement_audit.py  -- THE classifier (locate2 fine-NCC + whole-session motion offset).
- multijoint_reproj.py    -- global-vs-wrist-only geometry check (rescued P14 cam2).
- uncut_offset_probe.py   -- reproj offset sweep (SUPERSEDED for classification; kept: shows
  the reproj-masking failure mode).
- motion_sync_delta.py    -- per-clip motion xcorr (SUPERSEDED; periodicity-cursed, documents it).
- lag_probe_delta.py      -- within-clip reproj sweep (SUPERSEDED).
Uncuts downloaded (kept): P12 cam3/4/5, P17 cam5(+cam9), P10 cam4, P14 cam2/3/5, P19 cam2/3/5,
  P11 cam1-5, P13 cam2. Thumbs cached cache/delta/_thumbs/. NO deletes.

### NEXT (revised)
1. RE-CUT P12 cam4/5 + P13 cam2 from local uncuts at t_ref+session_offset (positions measured &
   pixel-verified) -> re-detect -> P12 up to 5/5, P13 5/5. (P12 cam5 = single +144.3s shift;
   cam4 = per-trial; P13 cam2 = +1.8s const w/ 2 trials needing individual placement.)
2. CUP FINETUNE can start now: P07/P08/P15 (5-cam) + P13/P14/P17/P10 (4-cam) = 7 parts >=4 cams.
3. Recalibration workstream (05_Calib_before per participant) -- but EXPECT it to fail for moved
   cams; only P11 is a true recalibration candidate. Verify moved-vs-source by comparing our
   shipped TOML to a fresh solve from 05_Calib_before.

## 2026-07-20 (cont.)  5-GOOD-CAMERA COHORT WORK SETS BUILT (10L+10R x 5cam)

User: work with participants that have all 5 of cams 1-5 geometrically good (not miscalibrated),
10 left + 10 right trials each, RE-CUT the mis-cut cams first. Cohort = P07,P08,P15 (already
clean) + P12,P13 (after re-cut). Excluded: P10/P11/P14/P17/P19 (genuine miscalib cams).

TOOLING (new):
- recut_from_audit.py: per trial, correct source pos = locate2(ref clip in ref uncut) +
  session_offset, refined by NARROW-window (+-2s) motion align (periodicity-safe). Writes
  cache/delta/<P>/recut/. VALIDATED P13 cam2: 167/174/169/170px (broken) -> 9/11/21/8px (recut),
  incl the 2 shuffled trials. 0 LOW-Q across 20 trials.
- build_work_set.py: assembles cache/delta/<P>/work/{clips,dets} = 10L+10R x 5cam, correctly
  timed. Good cams staged from study cut clips SCALED to 1920x1080 (calib res); mis-cut cams via
  recut_from_audit; detect all 5 per trial (detect_rep_batched). COHORT dict maps good/recut cams.

RESULT (leave-one-out wrist reproj, median px, L-trials/R-trials, cams1-5):
  P07  10/14  7/10  9/12  8/7  10/18   clean
  P08  22/11  11/5  13/10 12/7 18/15   clean
  P13  15/16  12/17 11/13 8/11 20/32   cam2 RE-CUT (validated both hands)
  P15  16/12  12/9  10/16 12/9 17/22   cam1 720p->1080p fix
  P12  16/30  19/35 29/24 48/34 26/43  cams4/5 RE-CUT; SOFTER calib (~2-3x, matches CSV 4.90 vs
                                       P13 2.69); R>L uniformly = AFFECTED arm detection noise
  proof re-cut works (not just noise): P12 cam4 92->33px, cam5 136->27px on overlapping trial.

TWO DATA BUGS FIXED (both now handled in build_work_set.py):
  1. SHUFFLED study cut clips (P12 cam4/5, P13 cam2) -> re-cut from uncut.
  2. 720p cam1 on 10-cam rigs (P14/P15/P17/P19): study cut clip is 1280x720 but calib is 1080p;
     plain copy -> 347px reproj. FIX: ffmpeg scale=1920:1080 when staging. (P15 rebuilt; bad build
     preserved at cache/delta/P15/work_bad720cam1/, not deleted.)

STATUS: 5-good-camera cohort READY = P07,P08,P13,P15 (tight ~5-22px) + P12 (soft but usable).
Each cache/delta/<P>/work/ is self-contained (clips + pose+cup dets), correctly synced.
NEXT: the actual work on this cohort (cup label pool / pose validation) can start here.

## 2026-07-20 (cont.)  RE-CUT VALIDATION BUG -> corrected cohort (user caught it in the render)

MISTAKE: I validated the per-trial re-cut by the motion-align QUALITY SCORE (peakedness), not by
REPROJECTION. Bad aligns passed silently. The user spotted cam2 desync in the P13 trial_1 render.
Per-trial reproj scan (wrist, leave-cam-out RANSAC consensus) revealed:
  P13 cam2: 19/20 trials good (9-16px), trial_1 = 125px (a single failed align) -> DROP/re-align.
  P12 cam4: 0/20 good (all 36-53px);  cam5: 5/20 good (rest 27-134px, R-trials worst).
    -> P12 cam4/5 are NOT re-cuttable. Motion-stratified: still 44/51px == move 65/54px (high WHEN
    STILL) = MISCALIBRATION (confirmed visually by user via render_wrist_consensus.py). P12 = 3/5
    (cam1,2,3), matching the ORIGINAL RANSAC audit. My "92->33px re-cut worked" was a
    rationalization -- 33px is not synced quality (good cams ~10px).
I also wrongly reported P12 as a 5-good-camera participant and built mixed-5 labels/eval that
included P12 cam4/5 + P13 trial_1 cam2 -> those are CONTAMINATED and must be rebuilt.

MANDATORY GATE from now on: every re-cut (and every work-set camera-trial) must be REPROJECTION-
validated -- wrist reproj vs leave-it-out >=3-cam RANSAC consensus, DROP the cam-trial if >~20px.
Never trust align-q. Tool: the per-trial scan in scratchpad + render_wrist_consensus.py (green=det,
red=consensus reproj; gap high WHEN STILL => miscalib, high only in MOTION => desync/bad-cut).

CORRECTED 5-good-camera cohort: P07 5/5, P08 5/5, P15 5/5, P13 5/5 (minus trial_1 cam2), P12 3/5.
Cup-detector findings still stand (domain gap = room+cup; mixed/cross-person works, cross-domain
fails; per-participant models sharper) but the NUMBERS that used P12 cam4/5 need re-running clean.

## 2026-07-20 (cont.)  REPROJ GATE on work sets + reject classification (desync vs miscalib)

Added a MANDATORY reprojection gate to build_work_set.py (reproj_gate / --gate-only): per
camera-trial, median wrist reproj vs the leave-it-out >=3-cam RANSAC consensus; >20px ->
QUARANTINE the clip+dets into work/rejected/ (kept, not deleted) + gate_report.json. This is the
check I should have run on every re-cut instead of trusting motion-align quality-score.

GATE RESULT (20px), trials with >=3 good cams:
  P07 20/20 (3 dropped, cam5 borderline)   P15 20/20 (10 dropped, mostly cam5)
  P08 20/20 (12 dropped, cam1/cam5 affected-side)   P13 19/20 (21 dropped; trial_1 lost cam2)
  P12 4/20 (78 dropped -- cam1/2/3 also >20px on most trials -> broadly soft, effectively out)
=> CLEAN 5-good-camera cohort = 4 PARTICIPANTS: P07, P08, P15, P13(-trial_1cam2). P12 = 3/5 at
best and really only 4 usable trials.

REJECT CLASSIFICATION (classify_rejects.py -- lag-sweep each reject vs leave-it-out consensus):
  * P13 trial_1 cam2 = DESYNC -73fr -> 8px == RECOVERABLE (the re-cut align landed 73 frames off;
    correct content IS in the clip). Also trial_41 cam2 +15fr, trial_45 cam2 +33fr. Re-cut fixes.
  * cam5 across P07/P13/P15 (20-35px, NO lag helps) = miscalib/COMPRESSION (2.5 Mbps camera),
    not desync -- just over the strict 20px gate. cam5 is a systematic weak camera on 10-cam rigs.
  * P08 cam1 (20-26px) borderline miscalib.  P12 = collapsed (no-ref) + cam4 genuine miscalib.

*** LESSON (user, emphatically): DESYNC and MISCALIBRATION are INDEPENDENT, CO-OCCURRING failure
    modes -- NEVER either/or. A camera can be BOTH. "MISCALIB" does not mean "synced". A binary
    DESYNC-vs-MISCALIB label (my classify_rejects rule) hides the desync on a both-broken camera:
    the lag-sweep finds the timing offset but reproj floors at the miscalib residual, so it gets
    stamped MISCALIB. FIX: report BOTH axes -- best-lag frame offset (timing) AND residual reproj
    at that lag (geometry). Fix desync first, THEN re-check residual miscalib. Saved to memory:
    docs/context/feedback_desync_miscalib_independent.md. (Same trap as P17 cam5 earlier.)

TODO next: fold lag-recovery INTO the gate (try the lag correction before dropping; re-cut &
keep if it reaches <15px, only drop if geometry floors high) and relax gate 20->25px (DELTA
baseline ~10-15px vs BRIO ~3px; most non-desync rejects are cam5 at 20-24px = marginal-usable).
Then rebuild cup labels/eval clean on the gated 4-participant cohort. Minor: classify_rejects
prints 1e9 for no-consensus frames -> label them "no-ref" not MISCALIB.

## 2026-07-20  >>> CONTINUATION HANDOFF (compact point) <<<

VERIFIED STATE (do not re-derive):
* CLEAN 5-good-camera cohort = 4 PARTICIPANTS: P07, P08, P15, P13. P12 is OUT (cam4/5
  miscalibrated + cam1/2/3 broadly soft >20px; user confirmed miscalib via wrist render).
* Each of the 4 has cache/delta/<P>/work/ = self-contained 10L+10R x5cam, CORRECTLY SYNCED,
  ALREADY GATED: work/clips (synced <=20px only) + work/dets (pose+cup json) + work/rejected/
  (quarantined bad cam-clips, kept not deleted) + gate_report.json.
  Synced clips kept: P07 97, P08 88, P15 90, P13 79.  Trials with >=3 good cams: P07/P08/P15 20/20,
  P13 19/20 (trial_1 lost its desynced cam2).
* SYNC IS VERIFIED by wrist reproj (5-22px) + motion-stratified (still≈move). The gate
  (reproj vs leave-it-out RANSAC consensus, >20px -> quarantine) is the acceptance test.

NEXT STEP the user asked for: rebuild the cup pipeline on the 4 good participants, SYNCED CLIPS
ONLY (work/clips is already gated so this is automatic). Exact commands:
  # 1. labels (reject-then-fill, COCO yolo26x-seg teacher) from the GATED work/clips:
  for P in P07 P08 P15 P13; do
    python scripts/build_cup_labels_delta.py --part $P --cams 1 2 3 4 5 \
      --clips-dir cache/delta/$P/work/clips --out data/delta_cup_4good --vid-stride 15 --stride 1
  done
  # 2. trial-stratified split (all 4 rooms/cups in train, held-out TRIALS for val -- NOT
  #    participant-holdout, which is cross-domain and fails):
  python scripts/prep_cup_dataset.py --pool data/delta_cup_4good --val-frac 0.15
  # 3. train (imgsz 640 MANDATORY, batch 8 for the 7.6GB 3060 Ti):
  python scripts/train_cup_seg.py --data data/delta_cup_4good/cup.yaml --epochs 3 --batch 8 --name cup_seg_4good
  # 4. 3D cup-tracking eval (the metrics that matter, NOT mAP-vs-teacher):
  python scripts/eval_cup_3d_delta.py --model runs/segment/runs/cup_seg/cup_seg_4good/weights/best.pt \
    --parts P07 P08 P15 P13 --max-trials 4 --fstride 4

KEY SCRIPTS (all in cup-task/scripts/):
  build_work_set.py      -- assembles work/ (stage good cams SCALED to 1920x1080 + re-cut mis-cut
                            cams) + reproj_gate() [--gate-only re-runs the gate].
  build_cup_labels_delta.py -- reject-then-fill labels; --clips-dir, --cams, --vid-stride, --only-trials.
  prep_cup_dataset.py    -- split; --train-parts (participant-holdout), --train-trials, --train-count.
  train_cup_seg.py       -- yolo26s-seg finetune, imgsz 640.
  eval_cup_3d_delta.py   -- precision_3d / good-frame% / median-px / per-cam det-rate (ported from
                            object_tracking run_clean3d_fill). --max-trials --fstride sample.
  classify_rejects.py    -- lag-sweep each reject: desync (recoverable) vs miscalib.
  render_wrist_consensus.py / render_cup_consensus.py -- 5-cam grid, GREEN=det RED=consensus-reproj
                            (the metric-matching render; use to eyeball miscalib-vs-desync).
  cut_placement_audit.py -- pixel-exact-NCC clip placement + whole-session motion offset.
  recut_from_audit.py    -- re-cut from uncut at t_ref+offset (BUG: no reproj gate -> use build_work_set gate).

WHAT THE CUP DETECTOR STUDY SHOWED (stands):
  * pipeline works; 3 epochs enough WHEN training spans the domains. DELTA is multi-domain
    (per-participant ROOM + CUP COLOR); BRIO was single-rig so 1 participant sufficed.
  * cross-PERSON generalizes (leave-P08-out: box mAP50 0.52) IF the domain is covered by another
    participant; cross-DOMAIN fails (P07/P08 model -> P13 green-screen room 0%).
  * mixed-all-participants (held-out trials) = box mAP50 ~0.49; 3D metrics precision_3d ~86%,
    median 6-9px clean parts, good-frame ~46% (weak spot = under-detection, esp hard cam3).
  * per-participant single-trial models are SHARPER on their own participant (precision 94-98%,
    recover cam3) -- BUT must train on the FULL trial (~1835 imgs), not a stride-5 subsample.
  * NOTE: mixed-5 + all earlier eval INCLUDED contaminated P12 cam4/5 + P13 trial_1 cam2 -> those
    numbers need RE-RUNNING clean on the 4-good gated cohort (that's the NEXT STEP above).

OPEN TODO (deferred):
  * fold lag-recovery INTO the gate: try the lag shift before dropping; P13 trial_1 cam2 is a
    RECOVERABLE desync (-73fr -> 8px) that a corrected re-cut would restore (+ trial_41 +15, trial_45 +33).
  * relax gate 20->25px (DELTA baseline ~10-15px; most non-desync rejects are cam5 compression at 20-24px).
  * classify_rejects: report BOTH axes (timing lag + geometry residual), fix 1e9 -> "no-ref" label.
  * then: cup-3D / phase / Murphy on the clean cohort (the original deliverable direction).

LESSONS THIS SESSION (also in object_tracking/docs/context/):
  * DESYNC and MISCALIB are INDEPENDENT co-occurring failures -- never either/or; report timing AND
    geometry. (feedback_desync_miscalib_independent.md)
  * VALIDATE re-cuts by REPROJECTION, never by motion-align quality-score (that bug shipped
    trial_1 cam2 at 125px and mislabeled P12 as 5/5).
  * study cut clips are SHUFFLED (wrong repetition) on some cams; 10-cam rigs have a 720p cam1 +
    a 2.5Mbps cam5 (compression -> cam5 is a systematic weak camera).

## 2026-07-20  GATE CORRECTION (user caught it): don't trial-drop NOISY cams, only BROKEN ones
The 20px reproj gate conflated two different things and over-dropped:
  * BROKEN (drop): systematically wrong on EVERY frame -> desync/mis-cut/miscalib. P13 trial_1
    cam2 (125px), P12 cam4 (40px+), P12 broadly. These must be quarantined.
  * NOISY (KEEP): merely higher-variance detection, ~20-25px. cam5 across P07/P13/P15 = the
    2.5 Mbps COMPRESSED camera -> noisier, not broken. Dropping the whole camera-trial is wrong:
    downstream triangulation ALREADY uses per-frame RANSAC consensus that rejects bad frames
    individually, so a noisy cam5 costs nothing (bad frames auto-rejected, good frames add a view).
FIX for continuation: raise the trial-gate to ~30-35px (broken-vs-noisy), OR don't trial-drop on
median at all -- keep all cams and rely on the per-frame RANSAC consensus gate (THR=30px) that the
3D pipeline already applies. Only hard-quarantine cams that are clearly broken (desync/miscalib,
median >>30px). So the earlier "P07/P08/P15 lost cam5 on N trials" is an ARTIFACT of the too-strict
gate, NOT lost data -- re-run build_work_set --gate-only after raising the threshold to restore cam5.
Net: the 4 good participants are even cleaner than the 20px-gate table implied; cam5 stays.

## 2026-07-20  CORRECTION: cam5 is NOT compressed (I generalized from P19 -- verify, don't assume)
Claimed cam5 = "2.5 Mbps compressed camera" -> WRONG. That bitrate cap was measured on P19 only
(P19 cam1 & cam5 = 2.5 Mbps). ffprobe of the SOURCE cut clips for the 4-good cohort shows cam5 is
actually the HIGHEST bitrate cam: cam5 10-14 Mbps vs cam2 6.5-11.9 (P07 10.0, P08 10.1, P13 14.2,
P15 14.0). So cam5's slightly-higher wrist reproj (~20-25px) is NOT compression -- likely VIEWPOINT
(oblique angle/distance -> harder wrist localization) or a marginally looser cam5 calibration. The
KEEP-cam5 correction still stands (it's noise not broken), but the "compression" reason was
fabricated by generalizing one P19 measurement. (NB the work/clips are re-encoded crf18 by me, so
check SOURCE cut clips / uncuts for real bitrate, not work/clips.)

---

## 2026-07-20 >>> 4-GOOD-PARTICIPANT REBUILD: gate, constant-offset recut, 3 experiments

### 1. Gate at 40px + cross-correlation desync detection
Re-gated P07/P08/P15/P13 at `--gate-px 40` (20px was dropping merely-NOISY cam5; 40 drops only
broken cam-trials). P07/P08/P15 lose ZERO cam-trials. P13 lost 2, both cam_2 — and the
cross-correlation lag sweep classified BOTH as recoverable DESYNC, not miscalib
(trial_1_L −73fr → 9px, trial_45_R +34fr → 14px).

### 2. THE SESSION OFFSET IS A CONSTANT (user's insight, confirmed)
> "if the uncut is correctly aligned with another uncut, then you just need the timestamp"

MEASURED P13 cam2 vs cam3, n=20 trials over 13 min: **offset = −3.0597s, sd 0.041s (2.5 fr),
range −3.153…−2.983, NO drift trend.** Spread fully explained by measurement resolution
(`locate2` resolves t_ref on a 10fps grid = ±3fr, + best-lag noise on a flat reproj curve).

**The per-trial `align_trial` refinement was the bug.** Drink motion is quasi-periodic, so
motion-energy correlation has near-equal peaks at neighbouring repetitions; the align wanders and
its self-score `q` ENDORSES the wrong answer confidently (trial_45: 1.15s wrong at q=3.75;
trial_48: 26fr wrong at q=1.09). The pipeline ALREADY had a constant-offset model
(`guess = t_ref + off`) — it was just 0.57s wrong, because `off` came from a coarse 2fps motion
xcorr (r=0.464). Per-trial align was then handed that error and scattered ±26fr "fixing" it.

New: `scripts/constant_offset_recut.py` — estimate the constant by REPROJECTION, average, apply to
every trial, verify by reprojection. Result: **max |lag| 26fr → 5fr; trial_48 32.4px → 14.6px;
trial_41 24.1px → 15.5px.** Median barely moved (13.6 → 13.0) and SHOULD NOT — 18/20 were already
fine. The win is entirely in the tail. Residual ±5fr remains (the 0.1s locate2 grid), harmless
because reproj is flat over ±7fr. Originals preserved in `work/rejected_preRecut/`.

### 3. ROOM STRUCTURE — CORRECTED BY LOOKING AT THE FOOTAGE
I twice theorised from pooled numbers and was wrong both times:
* "P13's cup is 2.8× larger" — pooled median; per-camera it's **cam2 only** (0.131 vs ~0.037),
  cams 1/3/4 normal and cam5 SMALLER. Not a participant-wide scale shift.
* "P15 is a different room" — **it is not.** P07/P08/P15 all share ONE room (same dark table,
  truss, whiteboard, floor). Only **P13** is a different site (white table, green screen, curtains,
  brighter). Established by rendering a comparison sheet (`p13_vs_others.png`), not by statistics.

So leave-one-out has exactly ONE true new-site test: P13.

### 4. THE THREE EXPERIMENTS (pool 2845 imgs: P07 860 / P08 950 / P13 575 / P15 460)
All yolo26s-seg, imgsz 640, batch 8, 3 epochs, trial-level splits (0 group overlap verified).

**mAP50 (vs teacher, box):** mixed4 0.717 | leaveP07 0.636 | leaveP08 0.620 | leaveP15 0.596 |
leaveP13 **0.123** | onetrial P07 0.531 / P08 0.556 / P15 0.728 / P13 0.505

**3D metrics (precision_3d / good_frame% / median_px) — THE ONES THAT DECIDE:**
| part | leave-one-out | one trial | mixed4 (all) |
|------|---------------|-----------|--------------|
| P07  | 81.2% / 40.4% / 3.6px | **97.8% / 61.5% / 6.3px** | 97.8% / 55.6% / 4.6px |
| P08  | 97.1% / 65.7% / 7.1px | **95.4% / 71.7% / 8.6px** | 98.6% / 66.7% / 7.5px |
| P15  | 85.2% / 18.3% / 6.2px | **89.6% / 36.0% / 8.7px** | 94.5% / 31.1% / 7.8px |
| P13  | **0.0% / 0.0% / nan**  | **94.1% / 54.1% / 13.3px** | 94.0% / 50.6% / 27.3px |

**FINDINGS**
1. **A new ROOM is fatal without data from it: leaveP13 = 0.0% good frames.** Zero. mAP 0.123 was
   flattering; in 3D the model never gets 3 cameras to agree. A new CUP COLOUR is nearly free
   (leaveP15 0.596 mAP, same room).
2. **ONE TRIAL beats the full 4-participant model for EVERY participant** on good_frame%
   (+6/+5/+5/+3.5 pts) despite 7× less data. For P13 it also HALVES 3D error (27.3 → 13.3px) —
   so P13's sloppy triangulation was substantially DETECTOR, not the cam2 calibration floor I
   had blamed.
3. **mAP RANKED IT BACKWARDS.** mAP put leave-one-out above one-trial for P07 (0.636 vs 0.531) and
   P08 (0.620 vs 0.556); 3D reverses both. mAP-vs-teacher rewards imitating the teacher on easy
   frames; good_frame% asks whether ≥3 cams agree. Same models, opposite conclusion.
4. Trade-off: one-trial models find MORE frames but are slightly LESS precise where things already
   worked (P07 6.3 vs 4.6px). And **cam3 detects 0% in every one-trial model** while mixed4 gets
   20-32% — only broad training cracks the hardest camera (cup is 0.007-0.013% of frame there).
5. mixed4 was **still improving at epoch 3** (0.568 → 0.589 → 0.717). All numbers are a floor.

**PRODUCT ANSWER:** a new clinic needs ~ONE labelled trial — not zero, not a cohort.

### 5. NEXT
* **mixed4 FINE-TUNED on one trial** (untested; should combine mixed4's cam3 coverage with the
  one-trial frame yield) — the obvious win.
* Longer training (3 epochs is undertrained).
* Scale/colour augmentation for cam3 + P15's cup.
* **RE-TEST P12 with constant_offset_recut.py** — its "miscalib" verdict came partly from renders
  of per-trial-align cuts, which we now know scatter ±26fr. Its study calib error was 4.90 (good).
  May rescue it from 3/5 cams.

Artifacts: models in `runs/segment/runs/cup_seg/{cup_seg_mixed4,cup_seg_leave*,cup_1trialfull4_*}`;
pools `data/delta_cup_4good`, `data/delta_cup_1trial_*`, `data/delta_cup_1trialmix_*`;
logs in scratchpad (`train_all.log`, `train_1trial.log`, `eval3d.log`, `p13_offsets.log`).

---

## 2026-07-20 >>> RETRACTION: cam3 "cup too small" was measuring label conventions, not the cup

User pushed back on "drink_study cups are larger / DELTA cam3 cup is below the detection floor."
Investigating, I was WRONG at the measurement level three times over:

1. First "cam3 = 4.8px" came from LABEL files. But `build_cup_labels_delta.teacher_dets` keeps only
   the CENTROID (line ~104: discards x1,y1,x2,y2) and the label is a SYNTHETIC SQUARE sized by
   `apparent_radius_px` = a fixed 35mm world sphere (CUP_R) reprojected per camera. So EVERY label,
   every camera, real-or-fill, is that sphere-convention size — NOT the cup's pixel extent. All my
   per-cam label sizes (cam1 51 / cam3 14 / cam5 61 px) are the convention, useless for a size claim.
2. Then "cam3 real cup = 37px" came from `work/dets/*.cup.json` — but that is `cup_clean3d_refill.pt`
   (the drink_study model used for the REPROJ GATE), NOT the teacher. Run cross-domain on DELTA it
   FALSE-POSITIVES on arms / wrist markers / torso stripes at the wide cam3/cam5 views (crop sheet
   `cam3_cup_crops.png` shows the blue box on a forearm, not a cup). Its 25k cam3 "detections" are
   mostly not cups.
3. The teacher (yolo26x-seg COCO) WAS used for labels (confirmed line 42) — that part of the
   worklog is right. But its box dims were discarded, so we have no teacher-measured cup size.

**RETRACTED:** the earlier claim "cam3 cup is 0.007-0.013% of frame / below the YOLO floor / only
broad training cracks it" is NOT supported — it measured the sphere convention. cam3's low
detection rate is real (it shows up in the 3D eval), but the CAUSE is unknown; "too small" is not
established. Likewise the drink_study-vs-DELTA cup-SIZE comparison is UNSUPPORTED (both sides were
label-derived / FP-contaminated). The camera-COUNT argument for the good_frame gap (N=10 vs 5,
binomial ≥3-of-N) still stands — that used detection RATES, not sizes.

**Bigger latent issue surfaced:** the student is trained on 35mm-SPHERE SQUARES, not real cup boxes.
Cosmetic for 3D (pipeline consumes only the centroid) but NOT for training — the student learns to
find sphere-squares, which may mismatch the real cup and could be part of the cam3/cam5 weakness.
Same root as the drink_study "fill box too small" parked note.

**TODO before trusting any size number:** re-run the teacher KEEPING x1..y2 (it already computes
them), rebuild labels with real boxes, and (a) measure real cup px per cam on BOTH datasets same
method, (b) test whether real-box labels fix cam3/cam5. Until then, no size claims.

---

## 2026-07-20 >>> ROOT CAUSE of cam3: apparent_radius_px offsets in WORLD-X (bug), not the cup

User: "why would cam3 have a different reprojection size since its not any further from the cup?"
Correct instinct. Grounded it in the calibration geometry (P08), NOT label files:

* Focals near-identical (~1130-1175px all cams) -> apparent size is pure distance.
* cam3 distance-to-cup = 1631mm, basically tied with cam1 (1575mm). Predicted apparent size ~24px
  (2nd largest). It is NOT far and NOT small.
* `apparent_radius_px` does `Xo[0] += CUP_R` — offsets the cup ONLY along WORLD-X, then measures
  projected px. NOT rotation-invariant. cam3's optical axis · world-X = **1.00** (it looks straight
  down world-X), so the offset is fully foreshortened -> projects to **1.9px** (floored to 6). True
  perp radius = 24.6px. cam1/cam5 (axis·X = 0.01) are unaffected -> code == truth there.

So cam3's 14px "tiny cup" was an ARTIFACT of this axis-aligned offset, not the data. At imgsz 640
the 6px-floored square is ~2px = below the stride-8 grid -> plausibly why cam3 barely trains and
detects ~0%. **Not resolution, not distance, not a bad camera — a one-line label-geometry bug**
affecting every near-world-X camera (cam3 worst; cam2/cam4 partial at axis·X ~0.75-0.79).

**FIX (one line):** offset perpendicular to each camera's viewing ray, or just USE the teacher's
real box (build_cup_labels_delta already computes x1..y2 at ~line 102 and discards it). Latter also
answers the size question and gives real-shaped labels.

**Meta:** this whole cam3 thread (4.8px floor / crop idea / "cups larger in drink_study") was me
reading label numbers without grounding them in geometry. Every size claim from the label files is
now retracted. The pipeline is a REIMPLEMENTATION of drink_study's run_clean3d_fill (not the same
code) and — per the user — is performing fine once you account for 5 vs 10 cameras (binomial
>=3-of-N: P08 one-trial 71.7% good frames ~= a 0.75/cam detector on ~4-5 cams). The cam3 bug is the
one concrete defect found.

---

## 2026-07-20 >>> cam3 FIX VERIFIED (retraction resolved) + NVDEC wired into labeler

Fixed `apparent_radius_px` in BOTH codebases (build_cup_labels_delta.py + drink_study
run_clean3d_fill.py): offset CUP_R along the camera IMAGE-X axis `R[0]` (perpendicular to the
viewing ray, orientation-independent) instead of `Xo[0] += CUP_R` (world-X, foreshortens to ~0 for
a camera whose optical axis // world-X). This restores the drink_study docstring's stated intent
("offset along image-x ... fronto-parallel") — it was an implementation bug latent since drink_study,
harmless there (no BRIO cam looked down world-X) but fatal for DELTA cam3 (axis.X=1.00).

Also wired NVDEC (`scripts/gpu_decode.py`, copied from drink_study lib) into the labeler's frame
read loop — 213 fps decode, ~2.5min/participant vs CPU timing out. Teacher dets still cached (no
GPU there).

Rebuilt labels -> data/delta_cup_4good_fix (same 2845 imgs, sizes fixed):
  per-cam label width @640: cam3 4.8px -> 19.1px (was below the stride-8 grid; now trainable),
  cam2 9.2->12.6, cam4 8.4->13.7, cam1/cam5 unchanged (were correct).

**3D eval, OLD -> FIX (cam3 det-rate | good_frame%):**
| model / part | cam3 OLD | cam3 FIX | good% OLD | good% FIX |
|--------------|----------|----------|-----------|-----------|
| mixed4 P07   | 26% | 60% | 55.6 | 60.5 |
| mixed4 P08   | 20% | 69% | 66.7 | 71.2 |
| mixed4 P15   | 22% | 37% | 31.1 | 35.6 |
| mixed4 P13   | 32% | 61% | 50.6 | 59.6 |
| 1-trial P08  | **0%** | **86%** | 71.7 | 78.7 |
mixed4 mAP50 also 0.717->0.882 box, 0.510->0.820 mask.

**VERDICT: the ~0% cam3 was the label-geometry bug, CONFIRMED and FIXED.** Not resolution, not
distance, not a bad camera, not too little data (all earlier theories retracted). cam3 is now
in-family (57-69% mixed4, 86% one-trial). good_frame% +4-9pts everywhere. Trade: median_px slightly
looser on easy parts (P07 4.6->6.6) since cam3's near-axis weak geometry now joins consensus — good
deal for the coverage. P13 improved on BOTH (27.3->21.7px, +9% good).

User was right on every push this session: reuse the pipeline (it was faithful, bug was upstream),
results fine for camera count, and cam3 apparent size shouldn't differ by distance (it doesn't —
the bug foreshortened it). The whole "cups larger in drink_study / cam3 too small" detour was me
trusting label numbers over geometry; every size claim from label files was retracted.

---

## 2026-07-20 >>> 1000-img controlled comparison: YOLO-seg vs RF-DETR (+ cup-loss gap analysis)

### Setup (fair by construction)
Per participant: dense (vid-stride 1) labels for 4 trials spanning BOTH conditions (2 unaffected +
2 affected — no reason to bias to unaffected; the cup is the same object either way), then
`--train-count 1000` so every participant trains on EXACTLY 1000 images. Val = held-out trials.
Same images for both architectures; `yolo_to_coco_cup.py` only re-expresses the YOLO-TXT labels as
COCO-JSON (RF-DETR can't read YOLO format; both are COCO-*pretrained*, different label *format*).
NVDEC (`scripts/gpu_decode.py`) wired into the labeler — 213fps decode, ~2.5min/participant.

### 3D metrics (precision_3d / good_frame% / median_px)
| part | YOLO-seg @640 (12M) | RF-DETR Nano @384 (30M) |
|------|---------------------|--------------------------|
| P07  | 98.6 / 72.1 / 5.6px | 94.6 / 73.5 / 6.0px |
| P08  | 97.1 / 80.5 / 7.7px | 96.0 / 87.8 / 9.1px |
| P15  | 95.2 / 82.9 / 7.8px | 97.9 / **100.0** / 8.3px |
| P13  | 92.3 / 63.6 / 20.9px | 94.9 / 65.0 / 19.8px |

⚠ NOT resolution-matched: RF-DETR Nano's config is 384 (must be div by 56, so 640 unavailable).
It is HANDICAPPED on accuracy and FLATTERED on speed. User chose to accept this and report it.

### Per-frame agreement (agree_yolo_rfdetr.py) — the decisive test
| part | BOTH | YOLO only | RFDETR only | NEITHER | RF-DETR gain |
|------|------|-----------|-------------|---------|--------------|
| P15  | 82.9%| 0.0%      | **17.1%**   | **0.0%**| +17.1 |
| P08  | 80.3%| 0.2%      | 7.5%        | 12.0%   | +7.3 |
| P13  | 63.6%| 0.0%      | 1.3%        | **35.0%**| +1.3 |
| P07  | 70.8%| 1.2%      | 2.7%        | **25.2%**| +1.4 |

1. **RF-DETR STRICTLY DOMINATES** — `YOLO only` is 0-1.2%, i.e. it essentially never catches
   anything RF-DETR misses. NESTED, not complementary -> an ensemble buys ~1pt max, not worth it.
2. **Same object**: where both agree, 3D cup points are median 3.2-8.7mm apart. Small real-
   disagreement tail: P07 3 frames / P13 6 frames (~1-2%) >50mm (max 77mm). NOT confident-wrong.
3. **The win size is set by how much is RECOVERABLE, not by the model.** P15 had 17.1% recoverable
   and RF-DETR took all of it. P07/P13 only have 1-3% left because 25-35% is unrecoverable by
   EITHER -> RF-DETR can't help there.

### Cup-loss GAP analysis (gap_analysis_cup.py) — good_frame% hides the shape
YOLO 1k, gaps = runs of consecutive frames with no >=3-cam consensus:
| part | good% | #gaps | median gap | max gap | lost in <=100ms | lost in >500ms |
|------|-------|-------|------------|---------|-----------------|----------------|
| P07  | 72.1% | 5     | 1267ms     | 2267ms  | 0%              | **100%** |
| P08  | 80.5% | 7     | 1333ms     | 2067ms  | 2%              | 96% |
| P15  | 82.9% | 6     | 1600ms     | 2267ms  | 0%              | 98% |
| P13  | 63.6% | 5     | **2667ms** | 3000ms  | 0%              | 99% |
**The loss is NOT scattered blips — it is 5-7 BLACKOUTS of 1.3-2.7s each**, 96-100% of lost time in
gaps >500ms. Nothing interpolatable. And they sit at **30-60% through the trial = the DRINK APEX**
(cup at lips, hand wrapping it) — the exact window the Murphy measures need. So chasing
good_frame% from 80->85 is worthless; the question is why these specific multi-second windows fail.

### Speed (bench_yolo_vs_rfdetr.py; same protocol as bench_streams.py/realtime.md:
### frames PRELOADED so decode excluded, WARMUP, cuda.synchronize before stopping the clock)
| cams | YOLO @640 | RF-DETR @384 | ratio |
|------|-----------|--------------|-------|
| 1    | 5.7ms / 177fps | 17.4ms / 57fps | 3.1x |
| 4    | 17.4ms / 58fps | 46.2ms / 22fps | 2.7x |
| 10   | **41.5ms / 24fps** | **99.5ms / 10fps** | 2.4x |
RF-DETR is 2.4-3x SLOWER *while running at 2.8x fewer pixels*. At matched resolution, worse still.
realtime.md's budget: 10 cams, ~12-15fps suffices (score.py lowpasses at 4Hz). RF-DETR's 10fps at
10 cams is the CUP NET ALONE — add pose and it's under the floor. YOLO's 24fps leaves room for pose.

### VERDICT
* **KEEP YOLO for the live rig** — 2.4x faster, only one that fits 10 cams + pose.
* **RF-DETR is the better DETECTOR** (strictly dominates, +17pts on P15) — worth it OFFLINE if an
  analysis needs P15-grade coverage and can afford 10fps.
* **The real bottleneck on P07/P13 is not the model.** 25-35% of frames are unrecoverable by either
  detector because no 3rd camera sees the cup at the drink apex. That is a RIG/CAMERA-PLACEMENT
  problem. Detector work cannot fix it; moving a camera can.

### Corrections made this session (all mine, all caught by the user)
* "cups larger in drink_study / cam3 cup too small" — RETRACTED, was measuring the 35mm-sphere
  LABEL convention and then a cross-domain model's FALSE POSITIVES on arms. Real cause = the
  apparent_radius world-X foreshortening bug (see previous entry).
* "physical occlusion, no detector fixes it" — I retracted this on P15 evidence alone (NEITHER=0%),
  but P07 25.2% / P13 35.0% / P08 12.0% show the ORIGINAL claim was right for 3 of 4. The
  retraction was itself the overreach. P15 is the exception.
* Used RFDETRBase (29M) before checking size; no RF-DETR variant is size-matched to yolo26s (~12M)
  — smallest (Nano) is 30.5M, so RF-DETR always has a capacity advantage in this comparison.
* pgrep-based wait loop deadlocked on ITSELF (pattern matched the wrapper's own cmdline).
* Negative-stride crash from `img[:,:,::-1]` — fixed in eval, then had to fix again in the bench.

---

## 2026-07-20 >>> DETECT-ONCE + SIAMESE TRACK beats detect-every-frame on all 4 participants

**User's idea, and I initially built the wrong thing.** Asked for "SiamFC with YOLO detecting once
at the start", I built per-camera GAP BRIDGING instead (YOLO every frame, tracker patches misses)
and reported +0.0 as if it were a verdict on the idea. It wasn't -- my gate only ran
`if Xb is not None`, i.e. only where consensus ALREADY existed, so bridging could never rescue a
lost frame by construction. Chicken-and-egg: the safety gate required the thing it was meant to
restore. The +0.0 measured my design flaw, not SiamFC.

**The actual experiment** (`siam_from_first_detection.py`, `siam_percam_drift.py`): per CAMERA, run
YOLO until it first finds the cup, seed a DaSiamRPN tracker with that box, then the tracker alone
produces the point for the rest of the trial. No re-detection. Feed into the SAME >=3-cam consensus.

| part | YOLO every frame | detect-once + track | gain |
|------|------------------|---------------------|------|
| P08  | 80.5% | **100.0%** | **+19.5** |
| P07  | 72.1% | **87.0%**  | **+15.0** |
| P15  | 82.9% | **93.5%**  | **+10.6** |
| P13  | 63.6% | **70.3%**  | **+6.7**  |

Better AND ~60x cheaper on detection (YOLO once per camera vs every frame). Also beats both
detectors from the earlier comparison: RF-DETR hit 100% on P15 but at 10fps@10cams; this reaches
93.5% on P15 with ~1 YOLO call per camera.

**PER-CAMERA: failure is always a MINORITY of cameras, and which ones varies by participant.**
| part | clean cams (median / %>30px) | failing cams |
|------|------------------------------|--------------|
| P08  | c1 7px/0%, c2 4px/0%, c3 4px/0% | c4 7px/**35%**, c5 10px/**24%** |
| P07  | c3 5px/0%, c5 3px/0%, c2 8px/6% | **c1 176px/64%**, c4 7px/32% |
| P15  | c1 9px/2%, c2 5px/0% | c3 38px/69%, **c4 117px/74%**, c5 23px/33% |
| P13  | (none clean) c1 11px/20%, c2 12px/32%, c3 23px/11% | c4 15px/32%, **c5 41px/69%** |
No fixed "bad camera" -- c1 is flawless on P08 and the WORST on P07. 100% on P08 works because
>=3-of-5 consensus outvotes whichever cameras are failing; it is NOT "all 5 track correctly".

**DRIFT vs TIME SINCE SEED — it is NOT gradual staleness, it is a DISCRETE CAPTURE EVENT.**
Profiles (median px, binned by frames since seed):
  t10 c4: 6 -> 5 -> 5 -> 8 -> 6      (flat, fine all trial)
  t10 c5: 7 -> 6 -> 22 -> 28 -> **5** (transient bump, RECOVERS)
  t1  c4: 15 -> 11 -> 26 -> **215 -> 221** (permanent)
  t2  c4: 7 -> 6 -> 6 -> **124 -> 144**    (permanent)
  t3  c5: 6 -> 5 -> 6 -> **487 -> 540**    (permanent)
Flat at 5-7px for HUNDREDS of frames, then a 2-orders-of-magnitude jump in one bin. Some recover,
some never do. A single unrecovered camera then contributes bad frames for the whole remainder,
which is what inflates the %>30px numbers.

**=> FIX IS EVENT-TRIGGERED RE-ANCHOR, NOT PERIODIC.** Failures announce themselves (5px -> 487px)
and the consensus already identifies WHICH camera disagrees per frame. Re-seed that camera from
YOLO when it exceeds the gate. Periodic re-detection would waste calls during the long flat
stretches and could still miss the jump. NOT YET BUILT.

**Caching added** (`cache/yolo_dets/<model-hash>__<part>__<trial>__fs<N>.json`): the YOLO-every-
frame baseline is the expensive part and does not change unless the model does. User caught that I
was recomputing it every run (same mistake as the earlier 5-cam redetect). Keyed on model hash.

**Also flagged, not done:** DaSiamRPN currently runs 5 sequential single-image ONNX forwards per
frame (one per camera) -- launch-bound, same regime realtime.md documented for the detectors.
Batching the 5 cameras' search crops requires driving the ONNX graphs directly (cv2's
TrackerDaSiamRPN is a closed wrapper with no batch entry point) and reimplementing the
crop/scale-pyramid logic. Deferred until the accuracy design settles.

**Corrections this session:** I mis-read my own render (claimed the tracker "stalled while the cup
moved to her mouth"; the consensus marker had not moved either -- it was tracking correctly), and
argued from that misreading for several turns. User's reading -- "the render worked well, the
numbers suggest ONE camera drifted" -- was right on both counts.

---

## 2026-07-20 >>> ⚠ RETRACTION: detect-once+track does NOT beat detect-every-frame (OMC says otherwise)

The earlier entry reported detect-once+DaSiamRPN beating YOLO-every-frame by +6.7 to +19.5 pts of
good_frame%. **That was scored against YOLO's OWN consensus as reference** — it measures AGREEMENT,
not accuracy. User asked to compare against OMC instead. DELTA c3d files contain a real
**`cluster_cup_1..4` marker cluster** = genuine cup ground truth (7 of 16 trials have c3d).

Compared trajectory displacement-from-a-COMMON-origin (rotation/translation-free, so no
mocap-lab↔rig alignment needed), with MATCHED frame sets:

| | YOLO vs OMC | TRACKER vs OMC |
|---|---|---|
| **frames BOTH provide** (n=845) | **5.7mm** | **17.4mm** |
| frames ONLY the tracker provides (n=195) | — | **37.5mm** |

**The tracker is ~3x worse than YOLO even on the easy shared frames**, and the extra coverage it
buys carries ~37mm error (~6x YOLO's typical). Per-trial the only tie is P08 t10 (3.6 vs 3.5mm);
elsewhere the tracker is 2-9x worse.

By trial phase, the damage concentrates exactly where the coverage is gained:
| trial | EARLY 0-0.3 (Y/T) | **DRINK 0.3-0.6 (Y/T)** | LATE 0.6-1 (Y/T) |
|-------|-------------------|--------------------------|------------------|
| P08 t10 | 0.6 / 4.0 | **10.1 / 31.9** | 3.6 / 2.8 |
| P07 t10 | 0.2 / 2.1 | **5.9 / 62.5** | 2.6 / 6.2 |
| P15 t10 | 1.7 / 5.6 | **16.5 / 28.3** | 3.3 / 16.1 |
Both are excellent early (0.2-5mm); the apex occlusion is what wrecks the tracker.

**CORRECTED CONCLUSION: detect-once+track is NOT a replacement for detect-every-frame.** It is
defensible only where YOLO has NOTHING (e.g. P13 t10 drink phase: YOLO no consensus at all, tracker
14.3mm) and only if downstream tolerates ~15-37mm. Dwell timing might; peak velocity will not. The
decision should be made on whether the Murphy measures degrade, NOT on frame counts.

**METHOD LESSON (the actual takeaway):** never score a candidate against the incumbent's own output
and call it accuracy. Good_frame% measured vs YOLO-consensus rewarded a tracker for AGREEING with
YOLO where YOLO was confident, and gave it free credit for frames nothing could check. Two bugs in
my first OMC pass compounded it and were only caught on the user's push-back: (a) each series used
its OWN first valid frame as displacement origin (different origins = spurious offset), (b) the
medians were over DIFFERENT frame sets (YOLO's excluded exactly the frames it failed on). Both
fixed above. See [[feedback_dont_claim_definitive]].

---

## 2026-07-21 >>> TRACKER THREAD COMPLETE: detect-once + UETrack-B + greedy>=2 consensus

**Goal:** replace detect-every-frame with detect-once + a modern tracker so the cup is covered
through the drink apex (where YOLO loses >=3-cam consensus). Full arc, many corrections (all
user-caught):

**Tracker progression:** cv2 DaSiamRPN (2018, CPU-only, retracted) -> LiteTrack-B4 (ICRA2024, GPU,
256fps/1cam) -> **UETrack-B (CVPR2026, GPU, 232fps/1cam, best)**. Setup gotchas in
[[project_litetrack_setup]] + [[project_tracker_shootout_uetrack]] (Fast-iTPN backbone dl,
weights_only=False, stub lib/train/admin/local.py, ONE shared model + per-cam state swap to avoid
5x1.3GB OOM). Wrappers scripts/{litetrack,uetrack}_wrap.py.

**KEY RESULTS (n=18 c3d trials, P07/P08/P13 x trial_10-15):**
1. Detect-once + UETrack traces the cup incl the apex; YOLO detect-every-frame has a HOLE there.
2. vs OMC: raw residual is a CONSTANT ~60mm (cup-centroid vs rim-markers) + ~10% DISPLACEMENT-SCALE
   (a reconstruction/calibration error, affects YOLO too -- worth chasing separately). The tracker
   itself is ~sub-mm at rest, r=0.999 in shape. Confidence-by-displacement stays high at MAX reach
   => tracks the cup at all motion levels.
3. **Per-trial correlation (PLAIN tracker + greedy>=2 consensus): median 0.9995, 17/18 >=0.998,
   worst P13 trial_15 = 0.931.** Every trial works.

**CORRECTIONS (each flipped a wrong claim of mine):**
- RE-ANCHOR IS HARMFUL, not helpful: the trigger uses a leave-out consensus that INCLUDES bad cams;
  one bad cam drags the reference -> flags+re-seeds a GOOD cam onto the wrong obj -> cascade until
  all cams agree on the hand. Plain keeps cam errors INDEPENDENT (the reason >=3 consensus works).
  The "confident-wrong all-cams-on-hand" failure I diagnosed at length was CREATED by re-init; plain
  those trials are 0.9997. (position-only vs re-init irrelevant -- trigger is poisoned either way.)
- "100% COVERAGE" WAS A BUG: counted >=3 cams producing a BOX, not >=3 AGREEING post-gate. Real
  coverage on P13 trial_14/15 is 42-57% at >=3 (only c3+c4 hold the cup at that viewpoint).
- FINAL CONSENSUS = greedy biggest-subset, min 2, but a size-2 point only accepted if within 150mm
  of the previous point (temporal continuity) -> rejects spurious frozen+noise pairs. trial_15:
  0.94@42%(>=3) / 0.82@100%(naive>=2) / **0.93@94%(greedy>=2)** -- strictly better. scripts/consensus_greedy.py.
- P13 cam2 recut for trial_11-15 (local uncut, const offset -3.0597s, ncc 0.999) but cam2 is ALSO
  weak at P13's viewpoint -> didn't add a 3rd good cam. trial_15 is a genuine 2-good-cam RIG limit.
- confidence: absolute (sigmoid 0-1), corr -0.4 with per-frame loss (usable flag) but does NOT drop
  on a confident wrong-obj lock -> can't catch the failure that matters; moot since plain needs none.

**NEXT (this session):** SmoothNet / STGFormer to improve POSE tracking (the pose keypoints, used
for the head-frame + as the OMC-alignment reference, are noisier than the cup).

---

## 2026-07-21 >>> SMOOTHNET ON POSE — jitter −93%, peak-velocity fixed, beats a low-pass

**Ask:** "run smoothnet on pose tracking to see if it improves jitter, speed correlation with omc,
and reproduction. test it on 3d and on 2d" — on the SAME n=18 as the tracker thread (P07/P08/P13
trial_10-15, each has a c3d = OMC ground truth). Tracked arm: L for P07/P13, R for P08.

**Setup.** SmoothNet (ECCV'22, `github cure-lab/SmoothNet` → `external/SmoothNet`) is a plug-and-play
temporal-only refiner: a tiny MLP (encoder `Linear(window,512)` → 5 residual blocks → decoder) that
acts on the TIME axis only, so it's joint-count-agnostic — the pretrained h36m checkpoint loads clean
on our 11-keypoint pose (0 missing/0 unexpected). Used PRETRAINED window=32, NO retraining. Weights:
`scratchpad/smoothnet_ckpts/checkpoints/` (windows 8/16/32/64). Runner `scripts/smoothnet_pose_delta.py`
reuses the proven `compare_pose_omc_delta.py` loaders / sync / Kabsch so triangulation+alignment are
identical to the incumbent harness.

**Two experiments.** 3D = smooth the triangulated pose (T,33) in metres, per joint. 2D = smooth each
camera's raw keypoints (T,22) in normalised px, THEN triangulate the smoothed 2D.

**RESULT (median, n=18, wrist):**

| method | jitter mm/s² | OMC speed-corr | reprod mm | peak-vel err |
|---|---|---|---|---|
| raw | 14417 | +0.923 | 38.0 | **+31%** |
| **3D SmoothNet** | **1018** (−93%) | **+0.984** | 38.4 | **+4%** |
| 3D Butterworth 6Hz | 1450 | +0.979 | 38.0 | +11% |
| 2D SmoothNet | 1078 | +0.986 | 39.3 | +5% |
| 2D Butterworth | 1996 | +0.974 | 38.2 | +10% |

**Peak-velocity error is the load-bearing check** (guards against a low-pass faking a win by killing
signal). Raw wrist OVERSHOOTS OMC's true peak speed by +14..+79% per trial — that overshoot *is* the
jitter. SmoothNet pulls all 18 trials to within ±9% (mostly ±5%). A plain Butterworth cuts jitter too
but ROUNDS OFF the peak (+11%) because a linear low-pass attenuates + phase-lags. SmoothNet cuts MORE
jitter (1018 < 1450) AND preserves the peak — so it is NOT a rebranded low-pass. Reproduction (median
Kabsch mm) stays flat (+0.4mm): position denoised, not distorted. This rescues the Murphy peak-velocity
measure.

**Pick = 3D SmoothNet** (best jitter+peak, reprod unchanged, simpler — no re-triangulation). The ~38mm
reproduction floor is untouched by smoothing: it's the SAME rig/body-fit alignment floor the cup tracker
hit (pose residual 32-46mm), not a jitter problem.

**Render** `out/smoothnet_wrist_speed.png` (P08/P07/P13 wrist speed, raw/Butter/SmoothNet/OMC) confirms
the numbers by eye: green (SmoothNet) sits on the black OMC peaks where blue (raw) spikes past them and
orange (Butter) rounds them off. Caveat visible on P13: BOTH filters share a small phase-lag vs OMC =
P13's known rig desync (raw speed-corr already lowest 0.78), upstream of smoothing.

**Bugs caught + retracted mid-run:** first 3D pass root-subtracted by left_hip, whose gaps propagated
NaN to every joint → 95% of frames blanked → looked like "3D smoothing fails" (speed-corr 0.92→0.36).
Fix = smooth PER JOINT (each joint over its own coverage) + drop root subtraction (the filter is
offset-covariant per channel). After the fix, 3D became the winner.

**Not yet done:** LOPO-train SmoothNet on our own 18 trials (we have OMC truth) — pretrained already
wins but a data-specific filter might close P13's residual gap; wire 3D SmoothNet into the pose stage
that feeds head_distance / OMC alignment. Memory: `project_smoothnet_pose`.

---

## 2026-07-21 (cont.) >>> MODEL HUNT: no ST 3D→3D refiner has weights; VideoPose3D lifter adds nothing

**Ask:** try STGFormer/SOTFormer/"stformer", then "look for a 2D→2D or 3D→3D that uses all of the pose
at once", then "see whether the best lifter does something interesting" (framing corrected mid-thread:
NO apex occlusion for pose — the occlusion problem was the CUP; this is just "is tracking better with
the lifter", a straight quality comparison).

**Field survey (verified each model's actual input/output from its method text, not its name):**
| model | spatial+temporal | 3D→3D refiner? | weights |
|---|---|---|---|
| SmoothNet | ✗ temporal-only | ✓ refiner | ✓ (the ONLY refiner with weights) |
| STGFormer / STFormer / PoseFormer / STCFormer | ✓ | ✗ **2D→3D lifter** | some |
| SOTFormer | — (box tracker, CVPR'26) | ✗ | ✗ 134B LFS stubs |
| DDHPose, FinePOSE (diffusion) | ✓ | ✗ **2D→3D lifter** (diffuse 3D conditioned on 2D) | ✓ but lifter |
| VideoPose3D | temporal conv | ✗ **2D→3D lifter** | ✓ (AWS, runnable) |
| GCN-Pose-Refinement (CVPR'24 wkshp) | ✓ | ✓ refiner | ✗ no weights (BlendMimic-trained, train-your-own) |
| D3PRefiner, HPR-Net, StarPose, attention-refiner | ✓ | ✓ refiner | ✗ no repo / README-only |

**Conclusion:** the field has a real GAP — the spatial+temporal "uses all joints at once" architecture
exists only as LIFTERS with weights, or as REFINERS without weights. SmoothNet is the sole 3D→3D refiner
that ships weights, and it's deliberately temporal-only (its paper argues cross-joint correlations are
noisy and hurt transfer — which is why it's the one that generalizes). MotionBERT (best lifter) weights
are OneDrive-only → 401 in headless env, not fetchable. SmoothNet's 2D checkpoint == its 3D checkpoint
(same file), so our earlier 2D experiment already used the correct weights — no separate 2D refiner exists.

**VideoPose3D lifter probe (runnable — AWS weights).** Ran it per-camera as a monocular 2D→3D lifter on
all 18 c3d trials. ⚠ Compromised by design: it needs the FULL 17-joint H36M skeleton; we capture 11 COCO
joints (upper body + hips + head). Derived Spine/Thorax/Neck/Hip; FABRICATED both legs (static neutral)
— off-distribution for its all-joint temporal conv. `scripts/videopose3d_probe.py`.

| | OMC speed-corr | jitter mm/s² |
|---|---|---|
| our triangulation | +0.921 | 14417 |
| VP3D best single cam (ORACLE, picks winning cam via OMC) | +0.915 | 8001 |
| VP3D mean-of-cams (honest) | +0.802 | — |
| SmoothNet-refined triangulation | **+0.984** | **1018** |

**Answer: the lifter adds NOTHING.** Its best-cam shape only TIES raw triangulation (+0.915 vs +0.921),
and that's with an OMC-oracle picking the best camera; the honest mean-of-cams is clearly WORSE (+0.802)
because per-cam lifts disagree and averaging degrades. Jitter beats raw triangulation (temporal conv
low-pass) but is 8× noisier than SmoothNet. Render `out/vp3d_probe_P07_trial_10_L_unaffected.png`: lifts
track the 4 drink reaches but wander at rest where OMC+triangulation sit near zero. Expected — a
monocular learned prior (fabricated legs) can't beat real multi-view geometry, nor a purpose-built
refiner on smoothness.

**Pose thread verdict: SmoothNet-refined triangulation is the winner and the answer.** No runnable
ST-refiner exists to try; the SOTA lifters don't fit our upper-body multi-view data. Memory:
[[project_smoothnet_pose]]. Open if wanted: LOPO-train GCN-Pose-Refinement on our OMC truth (the only
untried ST-refiner path, requires training from scratch).

---

## 2026-07-21 (cont.) >>> v2 PIPELINE BUILT + MEASURED (branch pipeline-v2-uetrack-smoothnet)

Built the v2 stages, wired behind flags (both-off = v1), got results on new + old datasets, benchmarked.

**New modules:** `cup_task/pose_smooth.py` (SmoothNet 3D refine), `cup_task/cup_track.py` (detect-once
UETrack + greedy consensus, with a cache-consuming path `track_cup_3d_from_cache` for the OMC results),
`cup_task/consensus.py` (greedy ≥2-cam). Flags `--smooth-pose` / `--cup-track` on `pipeline.py`.
Results: `scripts/results_v2_delta.py`. Benchmark: `scripts/bench_realtime_v2.py`. SmoothNet ckpt copied
to `models/smoothnet_h36m_fcn_ckpt_32.pth.tar`.

**NEW dataset (DELTA n=18) — v2 wins both:**
- Segmentation drink dwell vs OMC: v1 (every-frame YOLO) 767ms → v2 (detect-once UETrack) **67ms** (~11x).
  v1 over-estimates dwell (+500..1200ms, noisier cup lingers), breaks on P13_11. ⚠ first pass used a
  made-up speed-proxy dwell that showed v2 WORSE; the REAL segment.segment_cup_only reversed it.
- Murphy: peak_velocity |err| 45.2→26.3 mm/s (−42%), movement_units 1.0→0.5. total_movement_time (phase-
  driven) + trunk unchanged. The SmoothNet win landing on the clinical measures.

**OLD dataset (drink_study n=139) — pose-smooth = no change:** SAME-segmenter dwell |err| 183ms both
v1/v2 (per-phase onsets byte-identical); e2e 433→467ms (noise). Correct: dwell is CUP-driven and only the
POSE changed here (cup = shared `_refill_cup`). Confirms the segmentation win is a CUP-tracker win, not a
pose-smoothing one. `scripts/compare_phases_omc.py --smooth-pose`.

**Real-time (YOLO-pose, real 1080p frames, warm, ONE shared UETrack):**
- Live loop fps @1/5/10 cam: pose (batched YOLO) 258/85/47, UETrack 193/41/20.
- Whole offline pipeline/trial (8.9s video): consensus 38ms + SmoothNet 768ms (3 joints warm; +950ms load
  once) + segment 0.6ms = 806ms.
- Verdict: batched pose real-time to ~5 cams; UETrack is the scale bottleneck (sequential per-cam update)
  → 5-10cam live needs BATCHED tracker inference next. Offline stages trivially fast.

**Bugs fixed:** (1) bench OOM'd making 1 UETrack model per cam → ONE shared model. (2) SmoothNet vs UETrack
both ship top-level `lib/` → load SmoothNet's model by ABSOLUTE PATH (spec_from_file_location) so the name
collision is irrelevant regardless of load order.

**Follow-up (user asked, planned):** benchmark RTMPose + BlazePose alongside YOLO-pose (neither installed;
rtmlib + mediapipe pip-installable; speed-only first — both use non-COCO-11 layouts).

## 2026-07-21 (cont.) >>> BATCHED UETrack + simultaneous pose+cup

User asked: is UETrack batched, and simultaneous pose+cup fps. Before, UETrack ran one backbone forward
PER camera (sequential) = the scale bottleneck; pose was already batched.

**Built `UETrackBatch`** (scripts/uetrack_wrap.py) + `cup_track.track_cup_3d_batched`: N per-camera
states, ONE batched forward_encoder per rig-frame (batching search/template/anno/text_src). Correctness:
N=1 == sequential to 0.000px over 533 frames; N>=2 <=2px (bounded GPU matmul-order numerics, << 30px
gate). Speedup end-to-end: 1cam 1.0x, 5cam 1.8x (36->64fps), 10cam 2.1x (19->41fps). Backbone alone is
4-5x but the per-camera crop stays sequential CPU-side -> ~2x end-to-end (honest deployable number).

**Simultaneous pose+cup** (both batched, 2 CUDA streams): 1/5/10cam = 72/37/22 fps. ~half the min of
the two individual rates -- both compute-bound on the 7.6GB GPU so streams contend, don't overlap.
Real-time (>=60fps): comfortable at 1-2 cams; 5 cams borderline (37fps); 10 cams needs a bigger/second
GPU. Benchmark: scripts/bench_realtime_v2.py --what live. (Corrected an inverted speedup label mid-run.)

## 2026-07-21 (cont.) >>> batched the crop too + profiled the real bottleneck

Batched the per-camera crop+upload (stack N crops, ONE cuda upload+normalize vs N separate transfers).
Speed barely moved (5cam 64->65, 10cam 41->41). Profiled update() -- crop 0.56ms + upload 0.46ms are
NEGLIGIBLE vs encoder 16.36ms at N=10. The ENCODER (Fast-iTPN) is the bottleneck and does NOT amortize
(N=1->10: 3.24->16.36ms = 5x) -- it already saturates the 7.6GB GPU's compute. So ~2x is the batching
CEILING on this GPU; the earlier "4-5x backbone" was vs a cold-warmup baseline (misleading, corrected).
Crop-batching KEPT anyway = correctness win: N>=2 batched now matches sequential to 0.004px (was 2.1px
from mixed per-cam upload); N=1 still 0.000px. More tracker throughput needs bigger GPU / lighter
backbone / cameras-across-devices, not more batching.

## 2026-07-21 (cont.) >>> pose-model speed: YOLO vs RTMPose vs BlazePose

User asked to compare RTMPose + BlazePose vs YOLO-pose. Installed rtmlib + mediapipe.
scripts/bench_pose_models_multi.py. fps = rig-frames/sec (all cams, one frame):

| model | device | 1cam | 5cam | 10cam |
|---|---|---|---|---|
| YOLO-pose (batched) | GPU | 243 | 80 | 44 |
| RTMPose (top-down)  | CPU | 26 | 5 | 3 |
| BlazePose (1-person)| CPU | 29 | 6 | 3 |

**⚠ NOT apples-to-apples, and the caveat IS the finding:**
- RTMPose is CPU-ONLY here: onnxruntime-gpu needs CUDA 12, env is CUDA 11.8 (torch's). Tried
  ort-gpu 1.16.3/1.17.1 -> execstack + librt.so.1 load errors; reverted to CPU ort 1.23.2 (env intact,
  torch/YOLO/UETrack/SmoothNet all verified working). RTMPose's 5fps@5cam is an ENV limit, not the
  model -- on a CUDA-12 box RTMPose-GPU would be competitive.
- BlazePose CANNOT batch by design: MediaPipe is a stateful single-person graph (det on frame1 -> ROI
  tracker predicts next frame's crop = frame-to-frame dependency), single-person, and the .task runtime
  is a fixed C++ graph with no exposed batch dimension. So 10 cams = 10 sequential detect() calls.
- YOLO-pose is a stateless multi-person detector with native predict([N imgs]) = true GPU batch, so it
  wins decisively on THIS rig (80fps@5cam real-time). Keep YOLO for the live rig; RTMPose only worth
  revisiting on a CUDA-12 env for the offline/accuracy question.

## 2026-07-21 (cont.) >>> pose-model ACCURACY: YOLO wins that too

Followed the speed comparison with accuracy, using the metrics we already have (jitter / OMC speed-corr /
reproduction). Cached RTMPose (90/90) + BlazePose (partial, stopped at 46/90 once the trend was clear)
per-camera on DELTA clips -> our .pose.json schema (keypoints mapped to COCO-11, full coverage verified)
-> SAME triangulation + OMC comparison as YOLO. Scripts: cache_pose_altmodels.py,
accuracy_pose_models_delta.py (--matched for the fair shared subset). compare_pose_omc_delta gained a
DETS_SUBDIR override so alt models reuse the whole harness.

MATCHED subset (10 trials all 3 models cached = P07 all + P08 t10-12), median:

| model | jitter | OMC corr | reprod mm | wrist cov |
|---|---|---|---|---|
| yolo | 13299 | +0.934 | 32.2 | 100% |
| rtmpose | 35944 | +0.717 | 41.0 | 75% |
| blazepose | 37069 | +0.742 | 34.4 | 99% |

**YOLO-pose wins accuracy decisively AND consistently** -- ~2.7x lower jitter, +0.93 vs +0.72-0.74 OMC
corr, best reproduction, full coverage. Matches the earlier detector lesson: multi-view CONSISTENCY
matters more than single-image benchmark accuracy, and YOLO's stateless multi-person detection suits the
rig. RTMPose's 75% coverage = per-camera keypoints disagree across views (verified NOT low conf: median
0.837; triangulation rejects them).

⚠ Caveat kept for the record: RTMPose fed a FULL-FRAME person bbox (top-down models prefer a tight crop),
so its numbers are a LOWER BOUND -- a proper person-crop version might improve. BlazePose self-detects
(fair as-is). Even granting RTMPose the caveat, YOLO's margin is large enough that the conclusion holds.
VERDICT: keep YOLO-pose for the rig; no reason to switch. (Speed already favored YOLO 80 vs 5-6 fps@5cam.)

## 2026-07-21 (cont.) >>> optical flow for wrist-SPEED accuracy (metric: absolute mm/s, not correlation)

User idea: SmoothNet's speed isn't perfect; can optical flow help? Our speed = differentiate triangulated
position (amplifies jitter). Flow measures pixel VELOCITY directly (no differentiation). Per cam: PyrLK
the YOLO wrist pixel t->t+1 -> 2D pixel velocity; triangulate {p} and {p+flow} -> 3D velocity from the
FLOW FIELD, never differencing positions across time. scripts/flow_velocity_probe.py.

⚠ METRIC CORRECTED (user): speed is rotation/translation-invariant SCALAR, so ABSOLUTE |Δspeed| mm/s is
the honest metric; correlation only measures shape + HID the magnitude error. Also added CACHING
(cache/flow_vel/<clip>__<method>.npy) after the user flagged I was re-decoding video every run.

RESULT (median |Δspeed| vs OMC, moving frames, n=18):
| signal | |Δspeed| mm/s | peak err |
|---|---|---|
| pos-diff | 48.6 | 31% |
| smoothnet (was best) | 15.0 | 4% |
| flow (PyrLK) | 9.8 | 12% |
| FUSE (PyrLK + SmoothNet, speed-weighted) | 8.5 | 4% |

**Fused flow ~HALVES SmoothNet's wrist-speed error: 15.0 -> 8.5 mm/s.** Much clearer than correlation
implied (0.984->0.991 looked marginal). Fusion = trust flow at low/med speed (best shape), SmoothNet's
magnitude at the fast peak (flow OVER-estimates the peak +10-20% = motion blur, verified by SIGNED error;
my earlier "PyrLK window too small -> underestimate" was WRONG). PyrLK cost 0.5ms/wrist/cam = negligible.

FLOW-METHOD SHOOTOUT (user: tuned-LK, DIS, deep FastFlowNet/EdgeFlowNet). **Plain PyrLK WINS:**
| method | flow mm/s | fuse mm/s |
|---|---|---|
| PyrLK plain | 9.8 | 8.5 |
| tuned-LK (win11/6lvl/eig-gate) | 9.5 | 9.4 |
| RAFT-small (deep, 256px wrist crop) | 18.1 | 12.3 |
| DIS (dense variational) | 9.3 | 8.4 |

RAFT (a STRONGER deep model than FastFlowNet) LOSES on absolute error (18.1 vs 9.8) despite a marginally
better peak (10% vs 12%) -- correlation-era "RAFT looks great" was misleading. Dense deep flow on a small
crop averages the neighborhood, loses precise point velocity; full-frame RAFT OOMs the 7.6GB GPU. Tuning
LK didn't help. ⇒ FastFlowNet skipped (deep flow doesn't help here; not a capacity problem). Plain PyrLK
is cheapest AND best. NOT yet wired into cup_task (probe only).

## 2026-07-21 (cont.) >>> flow-speed: peak vs off-peak, and the WINNING fusion (metrics: docs/SPEED_METRICS.md)

Deep dive on wrist-SPEED accuracy (user: SmoothNet's speed isn't perfect). Full metrics table lives in
docs/SPEED_METRICS.md; the narrative + corrections here.

**Per-frame vs Murphy metrics DISAGREE — and that flipped the winner.** Per-frame |Δspeed|: pure flow
6.8 wins (halves SmoothNet 13.5). But the pipeline outputs PEAK velocity + time-to-peak, and on the
PEAK, pure flow is the WORST (61mm/s over-shoot from motion blur) while SmoothNet is best (20mm/s).
Decomposition: flow clean off-peak (+0.4 bias) / blurred at peak (+61); SmoothNet phantom-speed at rest
(+8) / accurate at peak (+13.6). Complementary, non-overlapping failure regimes.

**The WINNING fusion = speed-weighted BLEND** (which I'd earlier WRONGLY dismissed by scoring it on
per-frame instead of peak): `wb=sigmoid((flow_speed-350)/120); blend=(1-wb)*flow+wb*smoothnet`. Slow →
flow, fast → SmoothNet. Gets peak 19.7 + off-peak 4.7 + best worst-case timing (max 133ms vs 233-333ms
for the others = HALF). Beats/ties every method on every Murphy metric. ⚠ 350/120 hand-set, need tuning.
Other fusions FAIL: KF-prior (over-smooths velocity, 16.6/peak 68), flow-integrate→SmoothNet (10.9/peak
45). Lesson (again): the METRIC determines the winner; I kept defaulting to per-frame when the pipeline
needs peak+timing.

**Flow method: PyrLK wins** (cheapest 0.5ms + effectively best). DIS ties per-frame but 5x slower; RAFT
(deep, wrist-crop) LOSES; tuned-LK no help. ALL flows over-shoot the peak → fundamental to flow-tri, not
algorithm. FastFlowNet skipped.

**KEY CORRECTIONS this thread (user drove all of them):**
1. Metric: use ABSOLUTE mm/s speed error, not correlation (speed is frame-invariant; corr hid magnitude).
2. Caching: cache the per-clip flow (cache/flow_vel/) — was re-decoding video every run.
3. P13 = LINEAR CLOCK DRIFT (−8→+3fr, 3.8% rate mismatch), NOT desync/miscalib — EXCLUDED as bad GT.
   User's eye caught it from the shape traces (out/p13_speed_traces.png). Uniform-across-cams shift can
   only be OMC-vs-video lag, not inter-cam desync.
4. time-to-peak medians are frame-quantized (33/50ms = 2/3 frames) — report MEAN + MAX, not median.
5. Re-scored earlier fusions on PEAK not per-frame → post-hoc blend un-dismissed (it's the winner).

**NEXT: try POINT TRACKING as a speed method** (e.g. CoTracker/PIPs/TAPIR — track the wrist point
directly through time as an alternative to per-frame optical flow; may avoid flow's blur over-shoot at
the peak while keeping the direct-motion advantage).

## 2026-07-21 (cont.) >>> point-tracking (CoTracker/TAPNext) as a speed method — negative result

User idea: could a point tracker (track the wrist through time) beat YOLO+flow for pose speed? Full
results in docs/SPEED_METRICS.md (point-tracking section). Summary:

CoTracker3 single-seed DRIFTS after ~2s (good 0-1.3s @ 0-4px, then 12px@2s, 28px@3.3s). Re-seeding from
YOLO every 30fr keeps it accurate. But CT-reseed(30) is a 2D SMOOTHER not a better detector: same
position as YOLO (4mm displacement, re-seed anchors it), 4x smoother (jitter 0.65 vs 2.5mm), SmoothNet-
level speed (16mm/s, loses to flow 8). Every combination fails: as speed method (=SmoothNet), 2D->
SmoothNet/blend (double-smoothing, worse), flow-seeded-from-CT (median peak 56<64 BUT p90 297>127, MAX
665>162 = catastrophic tail when CT drifts within a window). User's p90/p95 check caught the tail-risk
the median hid. TAPNext++ = 28fps bf16 (6fps 5-cam) = too slow. VERDICT: point-tracking doesn't beat
YOLO+flow — a strong per-frame detector beats tracking-from-seed. Weights kept for future revisit.

Also this session: TAPNext speed was mis-measured at 13fps (fp32) -> 28fps with bf16 autocast (user
caught it). And a methods lesson reinforced repeatedly: CHECK THE TAIL (p90/p95/max) and use ABSOLUTE
error not correlation, or you get fooled (correlation hid speed magnitude; median hid CT drift-risk).

## 2026-07-21/22 >>> v3 PIPELINE: online/offline split, built + benchmarked + measured end-to-end

**Ask:** "design the new pipeline, choose what will run online and what will run offline (online only
if ultimately faster or if it looks better on screen), benchmark the speed, get official metrics using
displacement and speed errors for the relevant joints and the cup, compute segmentation errors and the
Murphy measures." Then, across the session: score ALL phases not just drinking, use every Murphy
measure, don't hold phases fixed, and render the graphs. Full writeup: **docs/PIPELINE_V3.md**
(schema + every table). This entry records the reasoning and the corrections.

### The split, and why

Drawn at THE POINT A STAGE STOPS NEEDING RAW PIXELS. Online: decode, YOLO-pose, cup detect-once +
UETrack, PyrLK flow (wrist AND cup). Offline: triangulation, consensus, flow 2D->3D lift, SmoothNet,
speed blend, segmentation, Murphy.

FLOW IS THE ONLY REAL DECISION and the user's criterion settles it: online the frame pair is already
in hand, so its MARGINAL cost is 0.4-3.4ms/rig-frame (CPU thread pool, PyrLK releases the GIL, GPU
pass hides the rest); offline it forces a SECOND full decode = 5182ms/trial = 6x the entire offline
budget (887ms). Same numbers, ~6x cheaper online.
SMOOTHNET MUST BE OFFLINE: symmetric +-16-frame window = 0.27s of FUTURE. Not a preference.

### Speed (RTX 3060 Ti, 8GB, 38 SMs)

Online rig-fps 1/5/10 cam: pose 237/75/42, cup 170/84/46, BOTH+flow 100/38.5/21.6. Offline 887ms/trial
(10% of realtime), 90% of it SmoothNet.

**"Saturated" DIAGNOSED, not assumed.** NOT memory (1.1GB of 8.2GB). CUDA-EVENT timing at 5 cams:
pose 16.06 + cup 11.37 = 27.43ms device work in a 29.7ms loop = **92% GPU busy**, ceiling ~36fps even
with perfect overlap. That is why threading the two nets is WORSE than serial (0.81/0.87/0.91x) and
batching gives no economy past 1 cam (raw pre-resized batched forward == full predict()).
⚠ TRAP: predict() RETURNS BEFORE THE GPU FINISHES — wall-clock around it times LAUNCHES and made an
earlier pass conclude "99% CPU-bound", the exact opposite. Use torch.cuda.Event.
Two benchmark artifacts fixed: charging a per-frame 1080p BGR->RGB copy (6.96ms > UETrack's own
4.04ms step) to the tracker read 91fps instead of 170+; and reporting flow's SERIAL cost (ncam x
per-wrist) instead of its marginal overlapped cost, overstating it ~5-30x.

### Accuracy (n=12, P07+P08 — see "cohort" below)

DISPLACEMENT = origin-relative ||X(t)-X(t0)|| per the user's definition: rotation-invariant like
speed, needs no rigid fit, so it does NOT inherit the ~38mm rig<->mocap calibration floor.
Pose: wrist 4.9mm / elbow 18.5 / shoulder 10.3 / nose 2.6; SmoothNet is position-neutral (+-0.7mm)
but cuts speed error ~4x (wrist 42.7 -> 10.6 mm/s). Wrist speed BLEND 6.9mm/s per-frame, 20.9 at the
peak (v1 pos-diff: 42.7 / 142.3).
Cup: d-corr 0.9996 (reproduces the v2 shootout), displ 2.3mm median, 100% coverage.

### Corrections (each reversed a stated conclusion)

1. **COHORT.** P13 was excluded from speed but kept elsewhere — inconsistent. Its OMC clock drifts
   3.8%, which corrupts POSITION too (cup displ 10-12mm vs 2-3mm; d-corr 0.974 vs 0.998; it owned the
   entire 504mm tail). Now excluded EVERYWHERE. n=18 -> 12.
2. **"USE v1 FOR CUP SPEED" — RETRACTED.** Pure selection bias: v1's 78% coverage is almost entirely
   the STATIONARY cup (median OMC cup speed 0.6mm/s on frames v1 HAS vs 139.3mm/s on frames it
   MISSES), so its all-frames median scores "is the still cup still?". On MOVING frames v1 is 1.8x
   worse and sees half of them. v3 also wins displacement where both exist (4.1 vs 5.0mm).
3. **CUP SPEED NOISE IS NOT TRACKER ERROR.** ~1mm per-frame positional wobble = 0.15% of a 700mm
   trajectory, invisible on a displacement plot, becomes ~60mm/s differentiated. During real motion
   OMC and v3 agree (p90 step 11.6 vs 10.2mm); the whole gap is at rest. So "positions match but
   speeds differ" was never a bug.
4. **SEGMENTATION WAS NOT SOLVED.** Reporting the dwell alone (67ms) hid that returning/rest_post were
   NEVER PRODUCED in 11/12 trials and total_movement_time over-ran by 1.50s.

### The segmentation fix (user's diagnosis)

Cause: grasp_end = "last frame with speed > BACK_OFF(10mm/s)". After the cup lands there is a burst of
small noise right at that gate, so the boundary was set by whichever signal twitched last; a
detect-once tracker never falls silent (rest 40mm/s vs OMC 11) so it never fired.
User: *"the position stops moving at the same time relative to origin... its just that theres some
noise in the speeds when the cup reaches the table"*. Exactly right — READ THE BOUNDARY OFF THE
DISPLACEMENT CURVE. Both tracks flatten at the same instant even when their speeds disagree. First
sustained flat run after the cup returns near rest, anchored inside the transport window.
RESULT: missing phases 11/12 trials -> **0/83**; back_transport offset -> 167ms; returning onset ->
200ms; every other boundary 17-125ms; total_movement_time 1.50s -> **0.03s** (now BEATS v1's 0.04).
Two failed attempts recorded in the doc so they are not retried (sign-only rule: 21/52 misses;
stability+direction: 3/83). ⚠ I broke a working version twice while "improving" it — the user had to
say stop.

### Murphy — full set, END-TO-END

8/8 position measures + 7 raw-point angle measures (NOT the ported set: the container needs a MuJoCo
qpos IK fit and refuses raw-point angles; both sides use the identical formula so the comparison is
fair, but they are computable-to-SEE only). Phases NO LONGER held fixed — each arm segments with its
own cup, which is what the pipeline actually delivers now that segmentation is good.
peak_velocity -42%, time_to_peak_% -52%, movement_units -50%, peak_elbow_ang_vel -32%,
total_movement_time -20%. Static angles barely move (correctly). time_to_first_peak_velocity is NaN
in BOTH arms (n=8) — unresolved in the measure itself, not a v3 regression.

### Cup flow — the wrist method transfers (user's idea)

cup_task.flow_speed is target-agnostic; pointed at the cup pixel from the UETrack cache
(scripts/cup_flow_probe.py, n=12): MOVING-frame speed err pos-diff 77.4 -> **flow 25.3** mm/s, rest
speed 42.5 -> 4.8 (clears the 10mm/s gate). Unlike the wrist, the BLEND does not win (SmoothNet
already handles the cup's slower peaks) so plain flow is the pick for reporting cup speed.
NOT adopted for the segmenter, two measured reasons: in the return region flow ties SmoothNet (11.8
vs 12.0) with a worse tail (p90 25.5 vs 16.0), and the segmenter needs a POSITION track that flow
cannot provide. Also exposed the 3D velocity VECTOR (direction is free from the flow triangulation) —
but flow-projected radial velocity lost to d(disp)/dt on the SmoothNet track (5.8 vs 4.6), because
smoothing had already removed the noise flow exists to avoid.

### Bugs fixed

- **SmoothNet was TRANSLATING every track ~84mm** — absolute world coords (~1.5m out) fed to a
  root-relative-trained h36m net, so its learned bias landed as a constant offset. Fixed by centring
  each window on its own mean: 84mm -> 1.9mm, jitter (-92%) and peak-velocity preserved, speed
  IMPROVED. Invisible to jitter/speed-corr/peak — all blind to a constant offset. Only origin-relative
  DISPLACEMENT caught it. See project_smoothnet_offset_bug.
- **UETrackBatch.update called .tolist() PER CAMERA** = N GPU->CPU syncs re-serialising the batch
  right after the batched forward (~50% of its CPU time). One transfer: 178 -> 217fps, identical
  output (0.0001px).
- **UETrack weights lived in a deleted session scratchpad**, silently breaking every tracker run.
  Re-fetched to models/trackers/uetrack/ (repo-persistent); wrapper overrides the config path.
- **Per-axis displacement** measured the MMC<->OMC frame ROTATION, not tracking error (cup read
  299mm). Origin-relative fixes it.
- **The cup was never being smoothed** (smooth_tracks excludes it by default).

Commits: b9a4ab5 (v3 split + flow/blend + SmoothNet fix), 928b521 (cohort + cup-speed + GPU
diagnosis), ae67bc5 (segmentation displacement boundary + all-phase metric + full Murphy).
Scripts: bench_v3.py, results_v3_delta.py, cup_flow_probe.py. Modules: cup_task/{flow_speed,
speed_blend}.py.

## 2026-07-22 >>> the NaN Murphy measures, and WHY flow is not used for segmentation

Two follow-ups the user asked for after the v3 writeup.

### The NaNs were TWO real bugs in score.py, not a missing measure

`time_to_first_peak_velocity` (+_percent) reported NaN for BOTH arms on 8/12 trials, and
peak_velocity on 10/12. Root causes, both mine-adjacent and both fixed:

1. **`_butter_lowpass` NaN-poisons the whole trial.** scipy's filtfilt propagates a single NaN across
   its ENTIRE output — so **2 missing frames in a 596-frame trial nulled every measure for that
   trial**. (The 2 NaNs came from my own `_shift` end-padding, so a harness detail was silently
   deleting measures.) Now: interpolate gaps -> filter -> restore NaN only where the input was
   actually missing. medfilt was checked and is already NaN-safe, so the lowpass was the sole culprit.
2. **`find_peaks` is structurally blind to edge peaks.** It returns only INTERIOR local maxima, so a
   single-peaked reach whose speed peaks at the window edge (measured: argmax at frame 0 on P07_12,
   frame 7 + prominence-rejected on P07_11) returned nothing at all. A single-peaked reach is the
   NORMAL healthy case and must not report "no first peak". Falls back to argmax, which for that
   profile IS the first peak.

RESULT: all 8/8 position measures now report on all 12 trials. peak_velocity n=10->12 and improves to
-46%; time_to_first_peak_velocity -20%, its _percent -41%.
⚠ time_to_peak_velocity_percent was -52% and is now +2% — the old figure was computed on the
NaN-poisoned 8-trial subset. **A measure that silently drops trials also biases the ones it keeps.**

### Why the segmenter uses SmoothNet and NOT flow (asked directly; my earlier answer was thin)

I had justified it only as "flow gives no position". The real reason is stronger and measured: flow is
the better REPORTING signal but the worse GATING signal.

rest_pre speed (before the cup moves at all), vs the FWD_ON=15mm/s onset gate:
  OMC        median  0.4  p95   3.6 mm/s
  SmoothNet  median  3.3  p95  14.2      <- p95 just UNDER the gate
  flow       median  4.9  p95  81.3      <- p95 5x ABOVE the gate

So flow spuriously trips the onset gate before the cup moves. Its excellent MEDIAN (4.9) hides the
tail — the same median-vs-tail trap as the point-tracking thread. Consequence, first crossing of
FWD_ON vs OMC's crossing (n=12): **SmoothNet 308ms median / 460 p90, flow 600 / 748** — flow is 2x
WORSE at the gate the segmenter actually depends on. Plus the segmenter needs a POSITION track for
displacement-from-rest, which flow cannot provide.

VERDICT: flow for reporting cup speed (25.3 vs 77.4 mm/s on moving frames), SmoothNet for driving the
segmenter. Same division as the wrist, same underlying reason: a direct velocity measurement is
accurate in the mean but noisy frame-to-frame, which is exactly wrong for a threshold crossing.

## 2026-07-22 (cont.) >>> the release boundary was ALREADY SOLVED — I re-derived a worse fix

User: *"remember that we had adjusted all these segmentation before already and found a good method
and idk what happened here but Im not sure why it got way worse"*. They were right, and this is the
correction.

**What was already there.** Commits b7eeb0f + 7b16f20 (13 Jul) fixed EXACTLY the bug I "discovered":
`returning` never appearing, back_transport swallowing the tail, caused by an unsigned cup-speed gate
sitting inside the cup's ~30-50mm/s jitter floor. The fix was the **wrist->cup PLATEAU** in
`refine_grasp_with_pose`: hand and cup are ONE RIGID BODY between grasp and release, so the distance
CLOSES (reach), goes FLAT (holding), then OPENS (release). Scale-free, no cup-speed threshold at all,
and it was validated BY WATCHING THE RENDER at both boundaries.

**Why it looked broken again — my harness never called it.** `results_v3_delta.py` called
`segment_cup_only` directly; `refine_grasp_with_pose` is only called by `pipeline.py`. So every
segmentation number I reported came from a stripped-down segmenter with the fix absent, and I
re-derived a weaker cup-only rule for a solved problem.

**Two bugs in the refinement were ALSO blocking it** (so it would not have fully applied anyway):
  * it BAILED OUT ENTIRELY when the onset already agreed (`if onset <= gs or onset >= ge: return
    seg`), discarding the release fix with it — 10/12 trials;
  * it clamped the release with `min(offset, ge)`, so it could only ever move the boundary EARLIER —
    12/12 trials — but every cup-only rule ends transport EARLY, because the cup goes still while the
    hand is still holding it. The two ends are now independent fixes.

**The right question, asked by the user:** not "do the two rules agree" but "does OMC agree with the
TRACKER". Same rule on BOTH sides, n=12:
    wrist->cup plateau   median  17ms   p90  50ms   <- 4 trials agree EXACTLY
    cup-only rule        median 167ms   p90 643ms
Restoring the plateau: back_transport offset 167 -> **25ms**, returning onset 200 -> **33ms**, still
0/83 misses. **Every phase boundary is now 0-67ms (<=4 frames).**

**WHY the cup cannot see the release** (user asked to test the jitter just before vs just after the
hand lets go). On RAW un-low-passed tracks, 0.5s windows, jitter = residual about a line fit + 2nd
difference:
                     OMC before  OMC after | v3 before  v3 after
    net drift           3.00mm      0.17mm |   4.79mm     3.96mm
    jitter (line fit)   0.51mm      0.03mm |   1.03mm     0.97mm
    jitter (2nd diff)   0.09mm      0.01mm |   2.21mm     1.59mm
In the MOCAP the pre-release wobble IS the hand — everything collapses ~17x the instant it lets go.
But v3 barely changes: its post-release motion is TRACKER NOISE (1.59mm HF, 50-150x OMC's floor),
comparable to the real pre-release hand wobble. **For the tracker there is no contrast at the release
to detect**, even though the physical event is sharp. The plateau works because it keys on a ~300mm
ramp, far above any noise floor. (⚠ my first jitter measure was `median|Ddisp|*fps` on the 6Hz
LOW-PASSED disp — that is a radial SPEED, not jitter, and it reported an implausible 82x. Measure
jitter on the RAW track about a local trend.)

**Boundary definitions verified against the source protocol** (van Andel et al., PMC5933268 Table 1,
user supplied). Paper: reaching ENDS at glass velocity > 15mm/s, back_transport ENDS at glass < 10mm/s
— i.e. **reaching INCLUDES grasping** and **back_transport INCLUDES release of grasp**. Our boundaries
land at +0ms (exact, 12/12) and +50ms of those criteria respectively; reach_end sits +100ms AFTER the
grasp, so the grasp is inside reaching exactly as specified. No change needed — and this retires my
earlier worry that reaching ended at the grasp and was shifting peak_velocity's window (it does not;
I had read `seg["grasp"]` without checking what to_murphy_phases does with it downstream).

**LESSON: check git log + memory for prior work on a boundary BEFORE diagnosing it as new**, and never
let a harness call a sub-stage directly when the pipeline calls a refined path. I broke a working
version twice while "improving" it.

## 2026-07-22 (cont.) >>> FLOW DIAGNOSTICS: where the 2D speed error really comes from

Long diagnostic thread, driven almost entirely by user questions that each caught an error in my
reasoning. **No shipping-pipeline change came out of it** beyond the flow camera gates -- it is a
map of what the error IS and which fixes are dead ends. Numbers in docs/RESULTS.md §10.

### Camera gating (the one change that shipped)

User: "does flow use all cameras or only the consensus ones, and the reprojected or detected point?"
It used ALL cameras and the RAW DETECTED pixel. Points are right (flow must be measured where the
image evidence is); ignoring the consensus was not. gate_consensus=True is now default:
cup MOVING err 25.3 -> 19.8 mm/s, wrist unchanged (its consensus rejects ~nothing).

User then suggested a consensus on the FLOW VECTORS themselves rather than a geometric occlusion
test -- better idea, and now default (gate_flow=True): cup 25.3 -> 19.4, wrist 21.7 -> 18.7.
It is NOT an occlusion detector (33% recall) -- it makes the fusion ROBUST, since triangulating flow
is least-squares with breakdown point 0.

### The occlusion finding (user: "check if the wrist is between the cup and the camera")

Right, and 2D proximity does NOT capture it (medians identical). The ANGULAR test does, 50x:
    wrist IN FRONT, <10deg   median flow 0.957 px
    wrist BEHIND             median flow 0.019 px
PyrLK tracks the HAND'S texture where the hand covers the cup. Dropping occluded cameras: rest p95
81.3 -> 11.9 mm/s. CUP-ONLY -- cup-as-occluder-of-wrist fires on 72.5% of camera-frames (the cup is
HELD by the wrist) and wrecks it (21.7 -> 92.3).

### Six methods that lost to PyrLK

    best-RMS integer shift   0.92 px  vs PyrLK 0.71   (searching ANY translation)
    ECC affine (6 param)     0.91     vs ECC-translation 0.75
    LightGlue (DISK) best    0.89     (r25 crop; 1.47 with all keypoints)
    RAFT-large               0.95     RAFT-small 0.97
    CLAHE / Wiener / minEig  all neutral-to-worse
**Fitting the patch better != estimating the motion better**, six times over.

### What the error actually is

* NOT a texture limit. LK information floor sigma/sqrt(lambda_min) = 0.025px vs 0.708 observed (28x).
* It IS model error: shifting frame t+1 back by the measured flow (INTEGER shifts, no interpolation)
  leaves 10.51 grey vs a 0.87 static floor; the BEST possible translation still leaves 7.09 (8.1x).
  No translation can match these patches.
* But that gap is NOT recoverable -- chasing it backfires (best-RMS shift is FURTHER from truth).
* ~51% of the 3D speed error is PROJECTION-IRREDUCIBLE: feeding the pipeline the TRUE projected flow
  (oracle) still gives 11.13 of the 21.71 mm/s. 49% is PyrLK. Pixel quantisation alone is 15.10.

### Corrections the user forced (each reversed a claim of mine)

1. "Dominated by cross-camera disagreement" -- WRONG. 94% of the 6.35px reprojection residual is
   COMMON-MODE and cancels between Xp and Xv; only 0.40px survives against a 2px signal.
2. "Perpendicularity dominates (-0.228)" -- ARTIFACT of dividing by displacement. Against ABSOLUTE
   error it is +0.014. Displacement is the only real driver (+0.285).
3. "Blur is the weakest factor / no correlation" -- measured with Laplacian variance, which tracks
   CONTENT not sharpness (it read HIGHER on visibly blurrier crops). The correct measure is
   directional ANISOTROPY (along-motion vs across-motion gradient energy), validated on synthetic
   blur (0.88 -> 0.08 over 0-20px). Blur IS present: aniso 0.99 at rest -> 0.52 at 6-12px.
4. User: "if a correlates with b and c, b should correlate with c" -- correct, and Pearson was
   hiding it: Spearman disp-blur -0.388 (vs -0.138), blur-error -0.115 (vs -0.002). The relationship
   is THRESHOLDED (blur only starts above ~2px), so Pearson understates it. The blur->error curve is
   U-SHAPED: error 0.64 at the blurriest, min 0.26 near aniso 1.3, 0.59 at the sharpest.
5. My first affine test was invalidated by a SIGN BUG (ECC-translation scored 7.05px, as bad as
   affine -- the tell). A synthetic known-shift check catches it in seconds. ALWAYS sanity-check an
   estimator on synthetic ground truth BEFORE comparing it.

### RAFT: rejected, and WHY it loses is the interesting part

RAFT-large has near-identical per-camera marginals (mean 1.307 vs PyrLK 1.306, median 0.73 vs 0.79)
and a 3.5x SMALLER max (15.4 vs 54.8 px), plus 5-10x less BIAS (+0.05 vs +0.28; at 6-12px PyrLK
over-reads +2% while RAFT is flat). User asked the right question: does that survive to 3D? NO --
PyrLK wins on every fuser (20.5-22.6 vs 22.9-25.3 mm/s).
WHY: decomposing the per-frame errors into the part all cameras SHARE and the part that differs:
    |frame mean| (does NOT cancel)   pyrlk 0.515   raft 0.581
    within-frame spread (averages)   pyrlk 1.012   raft 1.148
RAFT's errors are more CORRELATED ACROSS CAMERAS, and correlated error is exactly what multi-view
fusion cannot remove. **For a multi-view fuser, error correlation across views matters more than
per-view accuracy** -- invisible in any per-camera statistic. Also ~20x PyrLK's cost.

### The L1 fuser -- ⚠ HEADLINE RETRACTED at n=12, NOT adopted

User: "the consensus thing doesn't seem very robust, I'd prefer other fast methods with similar
results." Correct in principle. Comparing fusers on the Jacobian formulation (u_dot = J v, which
models what each camera can actually see). The n=6 probe run said:
    fuser      moving err    PEAK err
    plain        21.91        64.53
    loo (now)    20.52        80.06   <- WORST at the peak, worse than plain
    huber        21.12        59.99
    l1           20.88        35.34   <- "less than HALF of loo's peak error"
    trimmed      22.64        60.49

**⚠ That 2x claim did NOT survive the full cohort.** fuser_validate.py, n=12, both targets, paired on
the same OMC peak events (cached PyrLK only -- the fuser question never needed RAFT, so 25s not
minutes):

    WRIST (48 matched peaks)          CUP (24 matched peaks)
    fuser     moving  peak  p90       fuser     moving  peak  p90
    plain      20.53  64.78 150.5     plain      25.49  59.48 243.3
    loo (now)  18.95  55.37 150.1     loo (now)  19.98  26.85  90.0  <- BEST
    huber      19.29  59.52 125.1     huber      23.75  48.27 130.2
    l1         18.21  34.16 117.9     l1         22.32  38.48 102.9
    trimmed    19.92  60.57 143.2     trimmed    21.30  30.22 123.3

WHAT ACTUALLY MOVED: `l1`'s own number was stable (35.34 -> 34.16). **It was `loo`'s BASELINE that
was unstable** -- 80.06 at n=6, 118.95 on a 2-trial subset, 55.37 across 12. The "2x win" was a
bad-luck baseline draw, not an l1 gain. The real wrist win is 55.4 -> 34.2 median (better on 64.6%
of peaks, max tail 270 -> 185): real, but ~4x smaller than advertised.

AND IT REVERSES ON THE CUP, which the n=6 probe never tested: `loo` is best there and beats `l1` on
66.7% of cup peaks. This is NOT a surprise -- it is exactly what flow_consensus_cams' own docstring
already documents. The two targets fail differently:
  * CUP   = ONE camera, SUSTAINED occlusion -> hard-dropping is the correct response; soft
            down-weighting still lets the bad camera contribute.
  * WRIST = TRANSIENT blur across SEVERAL cameras -> a soft, median-like estimator survives it,
            and a hard gate is on the wrong side of its threshold too often.

DECISION: **not adopted, fuser unchanged.** No single fuser is uniformly better, so a global swap
trades a wrist gain for a cup regression. Per-target fusers (l1 wrist / loo cup) are defensible by
mechanism but are two code paths and a per-target choice fit on n=12 -- that is the overfitting the
"general mechanisms only" rule exists to prevent. `trimmed` is the quiet runner-up (zero knobs, one
refit instead of eight, second-best on the cup, better wrist tails than loo at 143/165 vs 150/270)
and is where to look first if this is revisited.

The durable lesson is methodological, and it is the same one as the detect-once/Siamese retraction:
**a headline computed against a noisy baseline is a claim about the baseline, not the candidate.**
n=6 was too thin to pin `loo`, and the entire "2x" rested on that.

Scripts: flow_gating_matrix.py, flow_model_shootout.py, flow_3d_survival.py, fuser_validate.py
(per-frame arrays saved to out/figures/*.npz so tail/threshold questions need no GPU recompute).

### ...then ADOPTED anyway, after asking the right question (fuser_noninferior.py)

User: "even if l1 was not better but just not worse I'd still prefer it because it's an actual
mechanistic thing that we can explain." That reframes the test from SUPERIORITY to NON-INFERIORITY,
and the reframe changed the answer. Also asked "is loo the consensus mechanism? why is it only one
that can be removed" -- which caught TWO errors in my write-up above:

  * LOO is ITERATIVE (a while-loop, up to 2 drops from 5), not "drop one". I had described it
    correctly in prose and then compared it as if `trimmed` were its near-equivalent.
  * My harness called flow_consensus_cams with NO `max_drop`, while the docstring's own measured
    optimum is PER TARGET (wrist 2, cup 3). So the head-to-head was unfair in opposite directions
    on the two targets. (Production also ships uncapped -- the per-target optima were measured but
    never wired in, which is its own finding.)

Re-run with both caps, and a paired bootstrap resampling TRIALS (not peaks -- peaks within a trial
share a geometry and a participant, so 24 peaks are not 24 independent samples):

  * `max_drop=3` is UNREACHABLE: 5 cameras, min 3 kept => at most 2 drops. loo_cap3 is
    bit-identical to uncapped. The docstring's "CUP 3:19.4 <- keeps improving" describes a setting
    that cannot differ from the default.
  * Against loo at its BEST setting (cap2) the cup gap VANISHES:
        wrist  l1 - loo_cap2 = -4.03 mm/s  CI [-14.32, +3.25]
        cup    l1 - loo_cap2 = +1.46 mm/s  CI [ -4.27, +9.17]   <- straddles 0
    NOT DISTINGUISHABLE. My "loo wins the cup" rested on comparing against the uncapped variant.

ADOPTED: l1 is the default fuser, gate_flow now OFF. Tie on numbers => the tie-break is that l1 has
NO tuned constants while the LOO gate has `tol=20.0` AND a per-target `max_drop`.

AND IT IS BETTER ON THE SHIPPING PATH, on BOTH targets: wrist 18.41 -> 18.19, cup 19.61 -> 18.66.
The cup IMPROVES even though the isolated probe showed a regression -- because l1 composes with the
GEOMETRIC consensus (gate_consensus, still on) better than the LOO gate did, and the probe had that
pathway disabled. A probe that isolates a component can invert the sign of its effect in situ.

End-to-end (results_v3_delta.py) the diff is SIX LINES -- only the flow and blend rows move:
    flow   per-frame 8.4->7.2  off-peak 5.2->4.4  peak 49.3->34.2  max 257.3->184.8  time 83->65ms
    BLEND  per-frame 6.8->6.7  off-peak 4.5->4.0  peak 15.0->17.0  max  85.7-> 73.4
Segmentation boundaries and all 8 Murphy position measures are BYTE-IDENTICAL.

Code moved into cup_task/flow_speed.py (projection_jacobian + solve_velocity), verified
bit-identical to the probe on 200 random cases before switching the default. flow_consensus_cams
pins itself to fuser="dlt2" internally -- its `tol` was tuned against that fuser, so letting it
inherit the new default would silently change what the threshold means.

## 2026-07-22 (cont.) >>> MULTI-CAMERA SURFACE CLOUD (branch feat/multicam-cloud-velocity)

User asked for a cloud-based 3D tracker (Shi-Tomasi sub-features -> epipolar NCC matching ->
triangulate -> Kabsch between frames) and later clarified it was "a direction more than precise
instructions". Built, measured, found the spec's architecture unworkable for a specific and
measurable reason, and replaced the broken stage.

### The spec's architecture cannot work here, and it is not a tuning problem

First build (cloud_velocity.py) scored 13667 mm/s against OMC on the DELTA cup vs the shipping
flow path's 17.3 -- ~780x worse. The user asked the question that cracked it: "I don't get why it
doesn't work on one timepoint". Correct instinct -- a SINGLE timepoint involves no correspondence-
across-time at all, so if it fails there the across-time story is irrelevant. It does fail there:
1-2 matches out of ~15 candidates per camera pair.

Two wrong diagnoses of mine, both killed by measurement:
  * "re-detection breaks correspondence across frames" -- true but NOT the blocker; the single
    timepoint is already broken.
  * "wide baselines break appearance matching" -- REFUTED: corr(baseline angle, n matches) = +0.44,
    i.e. wide pairs match slightly BETTER. cam_1-cam_5 at 142deg gets 2 matches, cam_3-cam_4 at
    29deg gets 0.

THE ACTUAL CEILING: take TRUE correspondences (project the same 3D point into two cameras) and ask
what NCC the patches get. Median **0.01**, and only **5%** clear the 0.65 gate. The gate is not too
strict -- the patches genuinely do not match. A curved specular cup lit differently per view does
not produce cross-view-matchable texture at 11x11. NO detector, threshold or descriptor fixes a
signal that is not there. (Ruled out first: the cup spans 43-69px, LARGER than the 35px ROI, so
the ROI is entirely on the object and background contamination is not the cause.)

### The fix: correspondence by CONSTRUCTION (cloud_track.py)

Never compare appearance across cameras. Seed points ON a cup-sized cylinder in 3D (position from
the consensus tracker, radius known) and project them into every camera -- point k in cam A and
point k in cam B are the same physical point BECAUSE THEY WERE THE SAME 3D POINT. Then PyrLK each
track forward within its own camera, which is the user's point: PyrLK already supplies t->t+1
correspondence for free, and within-camera is the only place correspondence is obtainable at all.
Forward-backward check drops drifting tracks. Track identity survives the 3D lift, so consecutive
clouds are corresponded with nothing matched anywhere.

### The lever-arm bug -- a real geometry error, worth remembering

After the seeding fix the cloud still read 2.10x the true speed on moving frames. Isolating it:
    OMC truth            556.8 mm/s
    Kabsch t_vec        1146.7 mm/s   ratio 2.10
    raw centroid delta   413.3 mm/s   ratio 0.77
The TRACKING was fine; the REFERENCE POINT was wrong. Kabsch returns t = centroid_B - R@centroid_A,
which is the translation of the **world origin**, not of the object. With the cloud ~1700mm from
that origin, a small rotation error swings R@centroid_A through that LEVER ARM and appears as a
huge phantom translation. Report the CENTROID's motion instead. Regression test pins it via a
stationary spinning object (mutation reproduces 5.099 m/s of phantom speed).

### Where it landed (n=12, DELTA cup, vs OMC)

    cloud  29.7 mm/s  93.2% coverage      (13667 -> 883 -> 411 -> 29.7 over the session)
    flow   18.7 mm/s  100%                 SHIPPING, still better overall
Cloud wins on 3 of 12 trials. NOT a replacement -- flow is better and cheaper (no video decode).

ANGULAR velocity is the one thing a single tracked point CANNOT do, and it now has real truth: the
C3D carries several `cluster_cup*` markers (results_v3_delta._omc_cup averages them away, which is
right for position and throws away exactly this). Kabsch on the separated markers = true cup
rotation (cloud_rotation_truth.py):
    correlation +0.639,  at rest 0.047 vs OMC 0.020 rad/s
    ⚠ magnitude ratio 1.67, per-trial 0.52-3.69 -- the SHAPE tracks, the SCALE does not.
So: a real signal, independently validated, NOT yet a usable measurement. If cup tilt is ever
wanted (drinking is a rotation), this is the thread; the open problem is the scale, not the shape.

An inlier gate (min_inliers=8) refuses to answer on thin evidence rather than emitting a confident
wrong number -- measured 5-7 inliers -> 531 mm/s error vs 113 at 8-11, with blow-ups to 17 m/s.
⚠ AND a harness trap worth remembering: H._lp INTERPOLATES ACROSS NaN, so coverage measured after
low-passing always reads 100% and a refusal gate looks like it does nothing. Measure coverage on
the RAW signal.

Kept: cloud_velocity.py stays as the recorded negative result (it is why we know cross-view NCC is
a dead end). Branch NOT merged.

### Trying to push the cloud below 29.7 -- what the error actually is (and a tuning trap)

User: "I feel like you could make that error better even without matching without the texture."
Right that there is headroom; wrong about where. Profiling P07 trial_10 (median err 11.5 mm/s, so
the 29.7 trial-level figure is driven by a minority of frames):

    band (OMC mm/s)   n    med err   ratio(cloud/omc)
    0-50            305      4.0       4.60   <- OVER-reads: a noise floor
    50-200          113     29.2       0.69   <- UNDER-reads
    200-500          38     93.6       0.76
    500+             47    133.3       0.86

TWO OPPOSITE BIASES, so they are two different problems. Hypotheses tested and KILLED:

  1. "PyrLK loses the fast tracks, survivors biased slow" -- REFUTED. Track survival is 99.5-102.8%
     at EVERY speed band. Nothing is being lost.
  2. "Reseeding re-pins the cloud to a biased reference; coast longer" -- REFUTED. reseed_below
     4 or 2 collapses coverage (95% -> 34%) because tracks die and are never replenished.

WHAT IT ACTUALLY IS: **the cloud inherits its seed source's bias**. The smoothed consensus cup3
that seeds it has the SAME signature -- ratio 5.34 at rest, 0.83-0.97 moving, vs the cloud's 4.60
and 0.69-0.86. The cloud is not creating this bias, it is anchored to something that already has
it. Note the cloud is already BETTER than its own seed on moving frames (median 11.5 vs 35.7), so
the tracking is adding real value on top of a flawed anchor.

The REST noise is also not what it looks like: per-point 3D jitter is only 0.07mm (=4 mm/s), and
~20 independent points should average to ~0.9 mm/s at the centroid -- but the reported rest speed
is 3.7 mm/s, 4x that. So **the jitter is CORRELATED across points, not independent**, and the
centroid cannot average it away. Same lesson as the RAFT-vs-PyrLK finding: correlated error
survives multi-view/multi-point fusion, and only the correlated part matters.

⚠ TUNING TRAP, and I nearly shipped it. A single-trial sweep on P07_10 said nseed=240 was a big
win (24.2 vs 48's 40.6, at BETTER coverage). Across 6 trials it REVERSES:
    nseed= 48  median 31.0  mean 29.6  cov 96.6%
    nseed=240  median 36.6  mean 35.6  cov 98.2%
240 won only on the trial it was tuned on and lost on 4 of the other 5 (P08 39-44 vs 19-32). The
non-monotonic sweep (240 good, 360/480 bad, 720 middling) was the tell -- a real optimum does not
zigzag. Kept nseed=48. Same failure the l1 retraction was about: a headline from too little data.

CONCLUSION: the cloud's remaining error is NOT in the cloud. It is inherited from the consensus
seed track, and the rest-noise is correlated so more points cannot fix it. Tuning this pipeline
further is not the lever; the seed track is. Left at nseed=48 / min_inliers=8 / reseed_below=8.

### The fix that worked: use the keypoint and the consensus gate we already had

User: "keep in mind we have a keypoint at each frame and we already have a mechanism to filter the
bad tracks." Both were being ignored by the cloud path. That was the entire remaining error.

WHAT WAS WRONG -- the cloud was NOT RIGID. Points that should sit at a fixed radius on a ~40mm cup
wandered with **29mm std**, and pairwise distances swelled 1.6%/frame at speed (spread
0.948-1.047). Kabsch ASSUMES rigidity, so that deformation was silently absorbed into the motion
estimate: under-read 15-30% while moving, over-read 4.6x at rest. Two hypotheses killed first:
PyrLK is NOT losing the fast tracks (survival 99.5-102.8% at EVERY speed band), and reseeding less
is worse (coverage 95% -> 34%).

CAUSE: `_lift()` triangulated each track from whatever cameras happened to have it, with NO
consensus gate. One camera's drifting track drags the 3D point and nothing catches it -- which is
precisely what `consensus3` exists for. It was being applied to the OBJECT but never PER TRACK.
Plus: there is a detection every frame, so a track that has slid off the object is detectable
geometrically (too far from this frame's keypoint), with no appearance model at all.

    config                cup MOVING err   coverage    cloud radius wander
    before (neither)          29.7          93.2%           29 mm
    per-track consensus       22.2          97.7%           11.7 mm
    + anchor 40px             18.0          97.7%
    + anchor 30px             16.0          96.6%    <- DEFAULT
    + anchor 25px             15.2          93.5%           5.9 mm
    flow (SHIPPING)           18.7         100%
    (15px anchor kills every track -> 0% coverage)

**The cloud now BEATS the shipping flow path** (16.0 vs 18.7 median, 15.4 vs 18.6 mean at 25px).
This one REPLICATES -- monotonic across the sweep and 30px wins on 11/12 trials -- unlike the
n_seed sweep that looked great on the tuning trial and reversed on the cohort.

ROTATION also improved with the rigid cloud: corr vs OMC-marker truth **+0.639 -> +0.772**.
⚠ But the SCALE got WORSE (ratio 1.67 -> 1.99) and is consistently ~2x. Checked and NOT a
stride/half-angle artifact (OMC truth at stride1 0.139 vs stride2 0.138 rad/s). Caveat on the
truth itself: only 4 cup markers spanning 25-48mm, and OMC's own median angular speed is 0.139
rad/s -- a tight cluster moving slowly, so the reference is not itself precise, and the cloud's
residual jitter plausibly inflates the median ratio. ANGULAR IS STILL NOT A USABLE MEASUREMENT;
the shape is now good, the scale is not.

Also tested and REJECTED: fitting every frame to ONE rigid reference model built from the tracks'
median centroid-relative offsets (rigid_model_probe.py). Worse than chaining (44.4 vs 36.1),
because no single rigid shape fits the sliding tracks -- median residual to the best-fit model was
55mm on a 40mm object. And a common-subset centroid (average only points present in BOTH frames)
was slightly worse than the current intersect-then-Kabsch (16.9 vs 14.2).

The centroid question: rotation IS about the cloud's own centroid (both clouds are centered before
the SVD, which is what decouples R from t). That centroid is NOT the cup's physical centre -- it
sits 34.6mm off the detected cup position -- but the offset is STABLE (std 4.7-5.5mm/axis), and a
constant offset cancels in a difference, so it is harmless.

### ⚠ RETRACTION: "the cloud beats the flow path" was a MIX EFFECT

User: "when the wrist is on the cup you're not necessarily going to get the same results, also
check if the distribution of the errors, also based on the speed is similar." Both checks were
right to demand, and together they overturn the headline.

STRATIFYING BY HAND PROXIMITY (wrist-cup 3D distance; note the wrist JOINT never gets closer than
~123mm to the cup CENTRE even in a firm grasp, so the bands are quartiles of the real
distribution, not absolute "touching" thresholds). Restricted to MOVING frames so speed does not
confound:

    band                  n     cloud    flow
    Q1 closest (held)   1090     17.6    16.3   <- FLOW wins where the hand holds the cup
    Q2                   911     16.7    17.4
    Q3 (freer)           258     12.5    13.9   <- cloud wins only when the cup is unobstructed

Cloud coverage also degrades with the hand present (93.6% Q1 vs 98.8% Q4). ⚠ The UNSTRATIFIED
quartile table looks like a huge cloud win (0.7 vs 1.6 in Q4) but that is the SPEED CONFOUND --
far-from-hand frames are also the stationary ones.

ERROR DISTRIBUTION on moving frames -- identical in the body, DIVERGENT IN THE TAIL:
    method   p25    MED    p75    p90    p95    p99     max
    cloud    7.0   16.0   34.9   75.5  113.3  214.8   447.7
    flow     7.5   16.5   33.4   57.9   77.5  133.3   203.0
    frames over threshold:  >100mm/s  cloud 6.4% vs flow 2.5%
                            >200mm/s  cloud 1.3% vs flow 0.0%
By speed band the cloud is competitive to 400mm/s then COLLAPSES: 800+mm/s gives 102.1 vs 59.0.
Both UNDER-read at speed (ratio 0.90-0.95) -- that bias is shared, not cloud-specific.

VERDICT: the cloud TIES on the median (16.0 vs 16.5) and LOSES where it matters -- held cup, high
speed, and the whole upper tail. By the repo's own rule (judge by the fraction of frames crossing
a usable threshold, not the median) FLOW IS STILL THE BETTER ESTIMATOR. The earlier "cloud beats
flow at 16.0 vs 18.7" is retracted: that pooled median is dominated by easy slow unoccluded
frames, which is exactly the selection trap the detect-once/Siamese retraction was about.

WHAT REMAINS TRUE: the per-track consensus gate + keypoint anchor are a real, replicating
improvement to the CLOUD (29.7 -> 16.0, radius wander 29mm -> 5.9mm, monotonic, 11/12 trials).
The cloud is now a viable estimator rather than a broken one -- it is just not a better one, and
its unique output (rotation) is still scale-unreliable.

### Chasing the shared ~6% under-read: FOUR hypotheses, all dead

Both estimators under-read speed by a flat ~0.94 ratio across every speed band (flat = a SCALE, not
a saturation), and it is SHARED, so it lives upstream of both. Tried, in order:

  1. **OMC resample 100Hz -> 60Hz shortens the path.** NO: native 167.8 vs resampled 167.2 mm/s,
     a 0.3% effect, and in the wrong direction to explain a 6% under-read.
  2. **Chording -- a 1/60s difference cuts the corner on a curved path.** NO: widening the stride
     INCREASES measured speed (stride1 167.2, stride4 176.2), the opposite of curvature loss. The
     trajectory is not curved enough at 60Hz to matter.
  3. **A universal f-vs-f+1 indexing offset.** Found a REAL convention mismatch in the code --
     flow_track_from_clip stores flow(f-1 -> f) at index f-1 (the START of the interval) while
     H._speed assigns its diff to index f (the END). Shifting flow +1 frame cut cohort error
     18.7 -> 14.7 (-21%). ⚠ BUT IT IS NOT UNIVERSAL: P08 improves 5/6 (19.0 -> 12.1) while P07
     does not move at all (18.6 -> 18.9, 3/6). The same code runs on both, so a real convention
     bug would have to be universal. What it actually reflects: _find_lag returns median -1 for
     P07 and 0 for P08, so "+1 everywhere" is just supplying P08's missing lag and overshooting
     P07 by a frame. NOT ADOPTED -- it would be fitting one half of the cohort.
  4. **Estimate the lag on the CUP instead of the wrist** (the harness aligns on the wrist but we
     compare the cup). WORSE: 20.5 vs 18.7.

And the follow-up that would have made a tidy story -- "the lags are spurious, use 0" -- is ALSO
refuted: corr(delta|lag|, delta err) = -0.289, shrinking |lag| made error WORSE on average (+3.0),
and trials with a nonzero lag have LOWER error (14.7) than trials with lag 0 (22.3).

CONCLUSION, and the reason to stop: every one of these looked convincing on partial evidence and
died on the full cohort. At n=12 trials, with effects that live at +-1 frame, this harness cannot
separate a real timing bug from noise -- the +1 shift is a 21% "improvement" that is almost
certainly half-cohort overfitting. The honest next step is NOT another candidate fix; it is either
more trials (P13's linear time-warp would add 6) or a sync measurement that does not depend on the
speed signal being compared. ⚠ Also worth remembering: all of this is a MEASUREMENT-HARNESS
question. A better lag estimate makes the EVALUATION more accurate; it does not make the tracker
better in production, where there is no OMC to align to.

### The idea I should have tried first: FUSE the two estimators (one-sided error)

User: "you don't have any other ideas? surely you haven't tried every mathematical idea." Fair --
I had exhausted ideas about the TIMING OFFSET and then stopped as though that were the whole
space. The estimator itself had an obvious untried move: cloud and flow are two INDEPENDENT
measurements of the same quantity, and their errors correlate only **0.473**, so over half the
error is independent and combinable. Both signals already exist; the fusion is free.

THE MECHANISM, and why it is `max` rather than an average. Every failure mode in either pipeline
-- a lost track, a smeared patch, a camera dropping out, a deforming cloud -- makes a measured
displacement SHORTER. Neither can invent motion. So the error is ONE-SIDED:
    cloud under-reads on 76.4% of frames (median signed -12.66 mm/s)
    flow  under-reads on 71.2% of frames (median signed -10.60 mm/s)
With one-sided error the LARGER estimate is the better one, and averaging two under-estimates just
gives a smaller under-estimate. Confirmed directly: on frames where the two DISAGREE the larger is
closer **74%** of the time, while "which method is better" is a coin flip (cloud closer 47%). And
it does not inflate the easy frames -- where the two agree within 5% the fused ratio is 0.958,
still slightly under rather than over.

    method   median   mean    p90   >100mm/s   ratio     (n=12 trials, cup, moving frames)
    cloud     16.49  17.99  68.86      5.0%    0.943
    flow      18.66  18.59  56.68      1.8%    0.952    <- previous shipping path
    mean      14.51  15.66  59.39      3.2%    0.947
    max       15.20  14.47  49.80      1.5%    0.984    <- ADOPTED

**max beats flow on 11/12 trials**, and unlike everything else tried tonight it improves the TAIL
rather than the median: p90 56.7 -> 49.8, >50mm/s frames 13.7% -> 9.6%, worst frame 203 -> 165.
The ratio going 0.95 -> 0.98 is the shared under-read being partly corrected, which is the
mechanism doing exactly what it predicts.

⚠ IT SURVIVES THE STRATIFICATION THAT KILLED THE PREVIOUS CLAIM. Where the HAND HOLDS THE CUP
(Q1, the hard case and most of the task) max gives 13.5 vs flow's 16.3; it wins every speed band
including 800+ mm/s (50.0 vs 59.0). This is not a mix effect.

SCOPE, and the guard rail: the rule is a claim about the error's SIGN, not a general fuser.
`signed_error_split()` exists to check the assumption before applying it elsewhere -- on a
symmetric-error signal `max` is biased HIGH and the mean is correct, which is pinned by a test.
Validated on the CUP only; the wrist needs its own signed-error measurement first.

### Umeyama scale: a REAL quality signal (adopted) -- but the deformation is NOT correctable

Added `umeyama()` (Kabsch + scale) alongside `kabsch()`. A rigid cup cannot change size, so a
fitted s != 1 is a direct, GROUND-TRUTH-FREE measurement of the cloud deforming. Verified exact on
synthetic data (recovers s to 1e-9; Kabsch cannot fit a 10% scale change, resid > 1mm).

**IT PREDICTS ERROR, and it is the only thing here that does.** n=12, cup, moving frames, error by
|s-1| quartile:
    Q1 most rigid 11.3 | Q2 13.1 | Q3 18.9 | Q4 most deformed 36.3 mm/s     (3.2x spread)
    Spearman rho(|s-1|, err) = +0.360
    ⚠ compare rho(n_inliers, err) = **-0.009** -- the existing min_inliers gate has NO predictive
      power whatsoever, and RAISING it actively hurts (>=12 keeps 27% of frames and makes the
      median WORSE, 19.96 vs 16.20). Scale finds exactly what inlier count misses.

As a gate at |s-1| < 0.01 (n=12, cloud only):
    no gate  15.95 median / p90 63.40 / >100mm/s 3.4% / cov 92.7%
    gate     13.58 median / p90 54.73 / >100mm/s 2.3% / cov 72.5%   <- ADOPTED (max_scale_dev)
Monotonic in the threshold (0.05->0.002 gives 16.24, 14.99, 14.05, 12.55, 11.45), which is what a
real signal looks like.

**BUT THE DEFORMATION CANNOT BE CORRECTED -- three attempts, all worse than doing nothing:**
  1. **Scalar powers of s on the speed** (/s, *s, /s^2): 36.81 -> 36.64 at best on deformed
     frames. Because deformed frames carry the SAME 0.938 ratio as rigid ones -- the deformation
     DETECTS bad frames without biasing them in a recoverable direction.
  2. **Global running rigid shape** (accumulate each track's mean offset, fit every frame to it):
     **60.62 vs 14.82 raw** -- catastrophic. Same failure as the earlier rigid_model_probe: a mean
     shape built from sliding tracks is a BAD reference, and then every frame is fitted to it.
  3. **Local similarity undo** (measure s on the consecutive pair only, rescale B about its own
     centroid before differencing -- nothing accumulated): **17.64 vs 14.82**. Iterating the
     correction changes nothing at all (17.64 both).

Also tried and REJECTED: **retiring the tracks RANSAC rejects on a deformed frame** (14.70 vs the
gate's 13.58 -- it disturbs the cloud more than it cleans it), and **retiring by track AGE**, which
the data says would not help anyway: per-track radius drift jumps to ~4mm in the first 30 frames
and then PLATEAUS (0.25, 4.32, 3.72, 3.49, 4.29, 2.61 ... mm). Sliding is per-frame NOISE, not
accumulated drift.

CONCLUSION: the scale is a DETECTOR, not a corrector. Refusing the frame is the right response,
and it is worth 15.95 -> 13.58 at the cost of 20 points of coverage.

### ⚠⚠ THE METRIC IS AMBIGUOUS BY MORE THAN THE ERROR WE ARE OPTIMISING

User: "consider that there's 4 cup points" -- and, on my first attempt to use them, "don't actually
use the omc to make the cloud, I just meant that for the truth." Correct on both counts: feeding
markers into the tracker would be circular and would produce a pipeline that cannot run without a
mocap lab. Used as TRUTH ONLY, the 4 markers expose a flaw in how everything tonight was scored.

THE PROBLEM. Every number so far compares the speed of the CLOUD'S CENTROID against the speed of
the OMC MARKER CENTROID. Those are two DIFFERENT PHYSICAL POINTS on the cup -- measured 34mm apart,
because the cloud only ever sees the near hemisphere. For pure translation that is harmless (all
points move alike), but under ROTATION v_P = v_C + omega x (P - C), and the cup rotates throughout
drinking.

HOW BIG IS IT? With >=3 markers the cup's full 6-DoF pose is known, so we can evaluate the truth at
ANY body point. Sampling 24 directions at the measured 34mm lever:

    **speed spread across body points 34mm apart: 42.1 mm/s**
    (per trial 35.7-47.0, against median truths of 157-358 mm/s)

The error being chased all evening is ~14 mm/s. **The metric ambiguity is 3x LARGER than the
signal.** "Cup speed" is not well-defined without specifying WHICH POINT on the cup.

Consequently, scoring at a fitted body point instead of the centroid:
    truth               median   mean    p90   ratio
    centroid-truth       14.07  14.75  51.23  0.934
    bodypoint-truth       9.07   9.18  38.98  0.989
⚠ bodypoint-truth is an ORACLE (lever direction picked per trial as the best of 24, scored against
the error it minimises) so the 9.07 is a CEILING, not an adoptable result. But the RATIO is the
tell: 0.934 -> 0.989 means **the shared under-read I chased through four dead hypotheses
(resampling, chording, indexing, cup-lag) is largely a LEVER-ARM ARTEFACT**, not a tracker defect.
A fitted direction should not systematically remove a bias unless the bias is real.

WHAT THIS INVALIDATES: not the RANKINGS (both estimators were scored the same way, so cloud-vs-flow
comparisons stand) but the PRECISION. Differences smaller than ~40 mm/s between methods are inside
the metric's own ambiguity. That includes most of tonight's tuning deltas.

WHAT TO DO INSTEAD, if this thread continues: report the speed of a DEFINED body point (e.g. the
cup centroid recovered from the tracked cloud's own geometry, not its visible-hemisphere mean), or
report a rotation-invariant quantity. Fixing the estimator further is not the lever; defining the
measurand is.

⚠ Also fixed a bug of mine along the way: expressing the cloud centroid in the marker frame via
`Rs.T @ (cen - Cs)` is MEANINGLESS -- MMC world and mocap world are different coordinate frames,
and it produced a 1842mm "body offset" (room scale). The lever must be measured within one frame.

### ⚠ CORRECTION to the previous entry: the ambiguity is ~6 mm/s, NOT 42

User: "how is the metric ambiguous by 42 mm/s, is that noise or rotation things?" Both halves of
that question were worth asking, and answering them corrects my own headline.

**IT IS ROTATION, NOT NOISE** -- three independent confirmations:
  * observed spread 42.1 vs the rigid-body prediction 2*omega*r = 51.9 mm/s; observed sits BELOW
    the theoretical max (24 random directions rarely include an antipodal pair). Noise would
    exceed the prediction, not undershoot it.
  * corr(observed, predicted) = **+0.719** across trials.
  * spread SCALES WITH omega (12.6 -> 21.7 mm/s from the 0.3-0.8 to the 0.8-1.5 rad/s band). A
    noise floor would be flat.
The cup really does rotate at ~0.75 rad/s, so body points genuinely move at different speeds. That
part of the previous entry stands.

**BUT THE MAGNITUDE WAS WRONG.** I quoted max-min over 24 sampled directions at a 34mm lever --
the gap between the two most OPPOSED points, i.e. a worst case, not a typical one. The relevant
quantity is how far ONE body point's speed sits from the CENTROID's, since that is the actual
comparison being made. Measured on the 4 REAL markers (physical points, no sampling, no model,
lever arms 23.2-24.4mm), n=12 trials:

    |marker speed - centroid speed|, median :  4.2 mm/s
    max-min across the 4 markers            : 11.5 mm/s
    scaled to the cloud's ~34mm lever       : ~5.9 mm/s

**So the metric ambiguity is ~6 mm/s, not 42.** That is roughly 40% of the ~14 mm/s error being
optimised -- material, worth stating alongside any result, but NOT larger than the signal.

REVISED VERDICT on the previous entry: "differences smaller than ~40 mm/s are inside the metric's
ambiguity" is RETRACTED. The correct statement is ~6 mm/s. Tonight's larger deltas (the rigidity
gate's 15.95 -> 13.58, the fusion's 18.66 -> 15.20) are OUTSIDE that and survive; the small ones
(a 0.8 mm/s difference between anchor thresholds) are inside it and were never meaningful.

The lever-arm point itself is unchanged and still worth acting on: the oracle body-point scoring
moved the ratio 0.934 -> 0.989, so a real part of the residual bias is a lever artefact. Defining
the measurand is still the right next move -- it is just worth ~6 mm/s, not the 3x-the-signal I
claimed.

### How much error is the LEVER ARM and how much is genuine? ~50/50

User: "how much of the speed difference is sitting outside of this marker speed spread, also what
is this lever arm thing."

THE LEVER ARM, plainly: the distance from the point being measured to the body's centre of
rotation. On a rigid body v_point = v_centroid + omega x r, so a point at distance r picks up
|omega|*r of extra tangential speed. The cloud's centroid sits ~34mm from the cup's centroid
(it only ever sees the near hemisphere), so at the measured omega ~0.75 rad/s it legitimately
carries ~26 mm/s that the marker centroid does not. That is not error -- it is two different
points on the same rigid body genuinely moving at different speeds.

THE SPLIT. For each frame, build the ENVELOPE of speeds that ANY body point 34mm out could
legitimately have (48 sampled directions, using the markers' 6-DoF pose). Error INSIDE the
envelope is explainable by point choice; the EXCESS beyond it is genuine tracker error that no
choice of measurand can excuse. n=6 trials, moving frames:

    method   TOTAL err (med/p75/p90)   EXCESS beyond envelope (med/p75/p90)   frames w/ excess
    cloud      18.4 / 39.5 / 80.2            0.0 / 13.2 / 41.3                     50%
    flow       19.9 / 37.6 / 61.8            0.0 / 12.3 / 28.3                     48%

**Roughly half the error is the lever arm and half is real.** On ~50% of frames the cloud's speed
falls INSIDE the legitimate envelope -- zero excess, the tracker is not wrong at all, it is
reporting a different point. On the other half there is genuine error, and the tail is where it
lives (p90 41.3 cloud / 28.3 flow).

NOTE the asymmetry: flow's total p90 (61.8) and the cloud's (80.2) differ by 18, but their GENUINE
p90s differ by 13 (28.3 vs 41.3). So the cloud's worse tail is disproportionately REAL error, not
point-choice -- which is consistent with the earlier stratified finding that the cloud degrades
where the hand occludes it.

IMPLICATION: the ceiling on measurand-fixing alone is roughly half the current error, and it is
concentrated in the frames where the cup is rotating. The other half needs actual tracker work,
and for the cloud specifically that means the occluded-cup case.

### ⚠ CORRECTION: nobody here measures the cup centroid, and the "34mm lever" is not what I said

User: "how do you measure the cup centroid?" Answer: I do not. THREE different points were being
called the cup position tonight, and none of them is the geometric centre:

    A  mean of the 4 `cluster_cup_*` markers   (TRUTH side, mocap world)
    B  UETrack's triangulated point            (MMC world) -- seeds the cloud
    C  tracked-cloud centroid                  (MMC world) -- what the cloud reports

  * **A is a marker CLUSTER stuck on one patch of the cup**, not a symmetric arrangement around it:
    pairwise marker distances 25.3-48.2mm, each sitting ~24mm from their own mean, on a cup of
    radius ~40mm and height ~95mm. So A is a consistent RIGID REFERENCE, offset from the true
    centre by an unknown amount.
  * B is wherever the tracker's box centre happens to land, triangulated. Also not the centre.
  * C sees only the near hemisphere, so it is biased toward the camera-facing surface.

**The "34mm lever arm" is |C - B| = 34.5mm (sd 4.3), i.e. cloud-centroid to UETRACK-POINT.** It is
NOT the distance to the cup's centre, and I used it as though it were a lever from the centre of
rotation. |C - A| is not even computable -- different world frames (the same mistake that produced
the 1842mm nonsense).

WHAT SURVIVES: the rotation is real (spread scales with omega, corr +0.719 with 2*omega*r), and the
lever-vs-genuine envelope split still holds, because it only uses the MARKERS' OWN 6-DoF pose plus
a 34mm MAGNITUDE -- two rigid points 34mm apart really do differ by that much wherever the centre
is. WHAT DOES NOT: calling A or B "the cup centroid", and any statement about where the cloud sits
relative to the cup's actual centre.

TO ACTUALLY MEASURE IT you need the cup's GEOMETRY fitted to observed surface points, so the centre
is inferred from the surface rather than read off whatever point a detector emitted. The cloud is
the only signal here that can support that -- it has surface points, not a box centre.
