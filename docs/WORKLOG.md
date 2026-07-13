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
