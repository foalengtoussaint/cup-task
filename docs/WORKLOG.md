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
holds. Decode is only 1.9ms/fr; batching buys ~0 (GPU already saturated per-frame).
Caveat: `model.predict(source=<path>)` adds ~2x overhead -- feed it cv2-decoded frames.
**An offline batch job's wall-clock says nothing about the live rate.** Don't conflate.

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
