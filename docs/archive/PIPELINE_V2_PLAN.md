# cup-task v2 — pipeline plan & results

**Validated on** P07 · P08 · P13 · trial_10–15 (n=18, OMC ground truth) · env `object_tracking` (torch 2.7) · 2026-07-21

**Bottom line.** Replace the cup's detect-every-frame + consensus-gate stage with **detect-once +
UETrack + greedy ≥2-cam consensus**, and add a **SmoothNet 3D refinement pass** on the triangulated
pose. Everything else in the DAG stays. Both changes are cache-first, reversible (behind a flag), and
measured against mocap.

Interactive version: `out/pipeline_plan.html` (published artifact).

---

## 1. Where we are

Current pipeline (`cup_task.pipeline`), linear cache-first DAG:

```
clips → per-cam 2D (cup + pose) → triangulation + consensus gate → cup-only phase seg (pose-refined onset) → scoring (stub)
```

Two stages leave measurable accuracy on the table:

- **Cup 3D** runs YOLO every frame of every camera, then a ≥3-cam / 30px consensus gate — expensive, and
  the gate discards frames rather than resolving disagreement.
- **Pose 3D** is raw triangulation. The wrist carries a heavy jitter floor and **overshoots** OMC peak
  velocity by a median +31%, corrupting the most clinically relevant Murphy measure (peak hand velocity).

## 2. The v2 DAG (two stages change, rest untouched)

| # | Stage | Module | Change |
|---|-------|--------|--------|
| 1 | Per-cam 2D pose | `cup_task.pose_keypoints` | keep |
| 2 | **Cup 3D — detect-once tracking** | `cup_task.cup_track` · `uetrack_wrap.py` · `consensus_greedy.py` | **new** |
| 3 | Pose 3D triangulation | `cup_task.triangulate` | keep |
| 4 | **Pose refinement — SmoothNet** | `cup_task.pose_smooth` · `external/SmoothNet` | **new** |
| 5 | Phase segmentation | `cup_task.segment` | keep (now fed cleaner inputs) |
| 6 | iMOVE scoring | `cup_task.score` | todo (gated on spec) |

Every stage still caches to JSON; both new stages sit behind flags, so both-off = byte-identical to v1.

## 3. Cup tracking — detect-once + UETrack + greedy consensus

Seed a UETrack-B tracker per camera from YOLO's first cup detection; track the rest of the trial with
**no re-detection**. Feed per-camera points into a greedy biggest-agreeing-subset consensus (min 2 cams,
150mm temporal-continuity gate). **Re-anchoring is off — it is harmful** (re-seeding from a shared
leave-out consensus lets one bad camera poison the good ones).

| Metric | Result | Read |
|---|---|---|
| Median trajectory corr vs OMC | **0.9995** | 17/18 trials ≥ 0.998 |
| Worst trial (P13 t15) | 0.931 | genuine 2-good-camera rig-geometry floor |
| Consensus rule | ≥2 greedy | biggest agreeing subset + 150mm continuity gate |
| Re-anchoring | off | poisons via shared consensus |

**Caveat.** Absolute cup-vs-OMC distance is 84–108mm, but that's a body-fit alignment error at the hand
+ ~40mm rig↔mocap calibration floor — NOT the tracker (matches YOLO's own trajectory to 1.4–5.9mm). The
right question was trajectory-match, not absolute mm. Eval is n=18 / 3 participants.

## 4. Pose refinement — SmoothNet (pretrained H36M, no retraining)

Sliding-window temporal refinement of the triangulated 3D pose, per joint, absolute metres.

| Method | Jitter mm/s² | OMC speed-corr | Reprod. mm | Peak-vel err |
|---|---|---|---|---|
| raw triangulation | 14417 | +0.923 | 38.0 | +31% |
| **SmoothNet 3D** | **1018** | **+0.984** | 38.4 | **+4%** |
| Butterworth 6 Hz (baseline) | 1450 | +0.979 | 38.0 | +11% |
| SmoothNet 2D → triangulate | 1078 | +0.986 | 39.3 | +5% |

**Peak-velocity is the load-bearing number.** Any low-pass cuts jitter; the test that it isn't just
killing signal is peak velocity. Raw overshoots OMC's true peak by +31% (the overshoot *is* the noise);
SmoothNet pulls every trial to within ±9%; Butterworth only reaches +11% because a linear filter
attenuates the peak it smooths. SmoothNet cuts more jitter **and** keeps the peak. Reproduction holds
flat (38mm = rig/body-fit floor) → position not moved, only denoised.

**Pick 3D over 2D:** wins jitter + peak-vel, reproduction unchanged, simpler (no re-triangulation). Two
rules baked into the runner: smooth **per joint** (a gap in one joint must not blank the frame) and **no
root subtraction** (offset-covariant per channel).

**Known limit:** P13 keeps a small phase-lag vs OMC that *both* filters share = P13's rig desync,
upstream of smoothing, not a SmoothNet artifact.

## 5. What we ruled out (model survey)

Verified each model's real input/output, not its name:

| Model(s) | Category | Weights | Verdict |
|---|---|---|---|
| **SmoothNet** | 3D→3D refiner (temporal) | yes | **adopted** |
| STGFormer · STFormer · PoseFormer · STCFormer | 2D→3D lifter | some | wrong category |
| DDHPose · FinePOSE (diffusion) | 2D→3D lifter | yes | wrong category |
| VideoPose3D | 2D→3D lifter | yes | probed — adds nothing |
| MotionBERT | 2D→3D lifter (SOTA) | OneDrive (unreachable, 401) | not fetchable |
| GCN-Pose-Refinement | 3D→3D refiner (spatial+temporal) | none | only untried path |
| SOTFormer | box tracker | 134 B stubs | no weights |
| D3PRefiner · HPR-Net · StarPose | 3D→3D refiner | none / no repo | unrunnable |

**VideoPose3D probe:** run per-camera as a monocular lifter — best single-camera wrist only *ties* raw
triangulation on shape (+0.915 vs +0.921, and that's oracle-picking the best camera per trial via OMC);
honest mean-of-cameras is worse (+0.802); 8× noisier than SmoothNet. A monocular learned prior can't beat
real multi-view geometry — expected, now measured.

**Field gap:** the spatial+temporal architecture exists only as lifters *with* weights, or refiners
*without* weights. SmoothNet is temporal-only by design (authors: cross-joint correlations are noisy, hurt
transfer — which is why it's the one that generalizes).

## 6. Build plan

1. **Stage 2 — `cup_task.cup_track`.** Lift `uetrack_wrap.py` + `consensus_greedy.py` from `scripts/`
   into the package. In `fuse_3d`, branch the cup target: if `--cup-track`, seed-once per camera + greedy
   consensus; else fall back to current `triangulate_target`. Cache per-camera tracked points. **Risk:
   med** (new stage, UETrack import chain to package).
2. **Stage 4 — `cup_task.pose_smooth`.** Wrap `smoothnet_pose_delta.smooth_sequence` into a stage taking
   triangulated mouth/wrist tracks → refined tracks. Insert between `fuse_3d` and `phases_from_3d` behind
   `--smooth-pose` (default on). **Risk: low** (pure post-process, validated).
3. **Regression gate.** Both flags off = byte-identical to v1. Test per fold: refined wrist OMC speed-corr
   ≥ raw on the 18 c3d trials. **Risk: low.**
4. **Then scoring.** With the wrist de-jittered, peak hand velocity becomes trustworthy — the first Murphy
   measure blocked on jitter rather than on the missing spec.

**Open, if wanted:** LOPO-train SmoothNet (or the GCN-Pose-Refinement graph refiner) on our 18 OMC-truth
trials — pretrained already wins, but a data-specific one might close P13's residual gap. Only
spatial+temporal refiner path left untried; needs training from scratch.

---

*Full narrative: `docs/WORKLOG.md`. Memory: `project_smoothnet_pose`, `project_tracker_shootout_uetrack`.
Renders: `out/smoothnet_wrist_speed.png`, `out/vp3d_probe_*.png`. Runners: `scripts/smoothnet_pose_delta.py`,
`scripts/videopose3d_probe.py`, `scripts/uetrack_wrap.py`, `scripts/consensus_greedy.py`.*

---

# RESULTS (2026-07-21, branch pipeline-v2-uetrack-smoothnet)

Built the two stages, wired them behind flags (both-off = v1 byte-identical), and measured. Modules:
`cup_task/cup_track.py`, `cup_task/consensus.py`, `cup_task/pose_smooth.py`; results harness
`scripts/results_v2_delta.py`; benchmark `scripts/bench_realtime_v2.py`.

## NEW dataset (DELTA, n=18 OMC-truth trials) — v2 WINS both

**Segmentation — drink dwell vs OMC** (real `segment.segment_cup_only`, same segmenter on MMC + OMC cup):

| cup source | median dwell |err| vs OMC |
|---|---|
| v1 every-frame YOLO triangulation | **767 ms** |
| v2 detect-once UETrack + greedy consensus | **67 ms** |

v1 systematically OVER-estimates dwell (+500..+1200ms, its noisier cup lingers in the slow phase) and
breaks on P13_11 (lost the drink entirely). v2 is within ±100ms on 14/18. **~11x better.** ⚠ First
attempt used a hand-rolled speed-proxy dwell that showed v2 WORSE — a made-up metric; the REAL segmenter
reversed it. Lesson re-learned (feedback_dont_claim_definitive).

**Murphy position measures — v1 raw pose vs v2 SmoothNet pose, error vs OMC** (shared OMC-cup phases):

| measure | v1 |err| | v2 |err| | n |
|---|---|---|---|
| total_movement_time | 0.000 | 0.000 | 18 |
| peak_velocity | 45.2 mm/s | **26.3 mm/s** | 10 |
| number_of_movement_units | 1.0 | **0.5** | 18 |
| max_trunk_displacement | 7.01 | 7.01 | 5 |

peak_velocity −42% and movement_units halved = the SmoothNet win landing on the clinical measures
(jitter inflates both). total_movement_time (phase-driven) + trunk (displacement extremum) unchanged, as
expected.

## OLD dataset (drink_study, n=139 comparable stems) — pose smoothing = NO CHANGE

| variant | v1 dwell |err| | v2 (SmoothNet pose) |err| |
|---|---|---|
| SAME segmenter (isolates pose) | 183 ms | **183 ms** (identical, per-phase onsets byte-equal) |
| END-TO-END | 433 ms | 467 ms (noise) |

**Honest read:** segmentation dwell is CUP-driven, and on the old dataset only the POSE was changed (the
cup is the shared `_refill_cup` both times). So pose smoothing alone can't move dwell — and doesn't
regress it. The new-dataset dwell win came from the CUP-tracking change, which isn't applied here. Not a
contradiction; a confirmation that the segmentation win is a cup-tracker win.

## Real-time speed (YOLO-pose)

Live loop, real 1080p frames, warm models, ONE shared UETrack (per-cam = sequential update):

| cams | pose fps (batched YOLO) | UETrack fps |
|---|---|---|
| 1 | 258 | 193 |
| 5 | 85 | 41 |
| 10 | 47 | 20 |

Whole offline pipeline per trial (8.9s video): consensus 38ms + SmoothNet 768ms (3 joints, warm; +950ms
one-time model load) + segment 0.6ms = **806 ms/trial**.

**Verdict:** batched YOLO-pose is real-time (>60fps) to ~5 cams; UETrack is the scale bottleneck (41fps
@5cam, 20fps @10cam) because cameras update sequentially — real-time 5-10cam tracking needs BATCHED
tracker inference (all cameras' search regions in one forward), the next optimization. Offline stages are
trivially fast.

**Follow-up planned:** benchmark RTMPose + BlazePose alongside YOLO-pose (neither installed yet; rtmlib +
mediapipe are pip-installable, speed-only first since both use non-COCO-11 keypoint layouts).

---

# ADDENDUM — batched UETrack + simultaneous pose+cup (2026-07-21)

Two follow-up questions: is UETrack batched, and can pose+cup run simultaneously.

**Batched UETrack (new `UETrackBatch` in `scripts/uetrack_wrap.py`, `cup_track.track_cup_3d_batched`).**
Before: the tracker ran ONE backbone forward PER camera (sequential) — the scale bottleneck. Now N
camera states are held and all cameras' search crops go through ONE batched `forward_encoder` per
rig-frame. Batching 4 internal tensors (search / template / anno / text_src).

Correctness: N=1 batched is BYTE-IDENTICAL to sequential (0.000px over all 533 frames of a real trial).
N>=2 differs <=2px (bounded, no drift) — pure cross-batch matmul reduction-order numerics tipping the
argmax to an adjacent grid cell; far under the tracker's own ~5px noise and the 30px consensus gate.

| cams | trk sequential | trk batched | speedup |
|---|---|---|---|
| 1 | 100 fps | 104 fps | 1.0x |
| 5 | 36 fps | 64 fps | 1.8x |
| 10 | 19 fps | 41 fps | 2.1x |

⚠ The BACKBONE alone batches 4-5x, but the per-camera crop/preprocess (`sample_target`) stays
sequential CPU-side, so END-TO-END it's ~2x. The ~2x is the honest deployable number; batching the
crop too would recover more.

**Simultaneous pose + cup (both batched, separate CUDA streams):**

| cams | pose only | cup only | SIMULTANEOUS |
|---|---|---|---|
| 1 | 250 | 104 | 72 fps |
| 5 | 87 | 65 | 37 fps |
| 10 | 47 | 41 | 22 fps |

Simultaneous ≈ HALF the min of the two individual rates: on this 7.6GB GPU both models are
compute-bound, so the streams contend for SMs rather than truly overlapping (stream overlap needs one
memory-bound + one compute-bound).

**Real-time verdict (this GPU):** 1-2 cams comfortably real-time (72fps both models). 5 cams = 37fps
simultaneous (just under 60fps capture — real-time-ish; hit 60 by batching the crop, a bigger GPU, or
cup+pose on separate devices). 10 cams = 22fps, not real-time here → separate GPUs or a bigger card.

---

# ADDENDUM 2 — batched the crop too; profiled where the time actually goes (2026-07-21)

Batched the per-camera crop+upload as well (crop all N on CPU, stack, ONE cuda upload + normalize for
the whole batch instead of N separate `.cuda()` transfers). Result: **speed barely moved** (5cam 64→65,
10cam 41→41 fps). Profiled the `update()` to find out why — the real per-phase split at N=10:

| phase | N=1 | N=5 | N=10 |
|---|---|---|---|
| crop | 0.07 | 0.29 | 0.56 ms |
| upload+norm | 0.08 | 0.23 | 0.46 ms |
| **encoder** | **3.24** | **9.23** | **16.36 ms** |
| decoder | 0.73 | 1.12 | 2.36 ms |

**The crop+upload were never the bottleneck** (~1ms of ~19ms at N=10). The ENCODER dominates, and it does
NOT amortize with batch — N=1→10 the encoder goes 3.24→16.36ms (5x), because the Fast-iTPN backbone at
batch=10 already saturates this 7.6GB GPU's compute. So **~2x is the ceiling on this GPU** (batched 16ms
vs sequential ~32ms for 10 cams); the earlier "4-5x backbone" figure compared against a cold-warmup
sequential baseline and was misleading.

The crop-batching is kept anyway — it's a CORRECTNESS win: with all cameras' crops going through one
stacked-tensor path, N>=2 batched now matches sequential to **0.004px** (was 2.1px from the mixed
per-camera upload path). N=1 stays byte-identical (0.000px/532fr).

**Takeaway:** on this GPU the tracker is encoder-compute-bound; more throughput needs a bigger GPU, a
lighter tracker backbone, or splitting cameras across devices — not more batching.
