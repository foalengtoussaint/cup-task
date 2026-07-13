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
