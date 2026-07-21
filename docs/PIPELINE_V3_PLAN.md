# cup-task pipeline plan (v3) — consolidated 2026-07-21

Supersedes the v2 plan. Adds the **wrist-speed** thread (flow + blend) and records what was **ruled
out** (RTMPose/BlazePose, the fusion variants, point-tracking). Validated on P07/P08/P13 trial_10–15
against OMC (P13 excluded from speed = clock drift). Branch: `pipeline-v2-uetrack-smoothnet`.

## The DAG (what runs, in order)

```
per-cam clips
  ├─► YOLO cup detect ──► [Stage 2] detect-once UETrack + greedy ≥2-cam consensus ──► cup 3D
  └─► YOLO-pose ─────────► triangulate ──► [Stage 4] SmoothNet 3D refine ──┐
                                            + [Stage 4b] PyrLK flow speed ──┴─► [Stage 5] speed BLEND
                                                                                     │
                                        ──► [Stage 6] segment (cup phases + pose onset)
                                        ──► [Stage 7] Murphy scoring (peak-vel from the blend)
```

| # | stage | module | status |
|---|-------|--------|--------|
| 1 | per-cam 2D (cup + pose) | `cup_detect`, `pose_keypoints` | keep |
| 2 | **cup 3D — detect-once UETrack + greedy consensus** | `cup_track`, `consensus` | built (v2) |
| 3 | pose 3D triangulation | `triangulate` | keep |
| 4 | **pose refine — SmoothNet 3D** | `pose_smooth` | built (v2) |
| 4b | **wrist speed — PyrLK optical flow** | (new: `flow_speed`) | validated, not yet a module |
| 5 | **speed BLEND (flow off-peak + SmoothNet at-peak)** | (new: `speed_blend`) | validated, not yet a module |
| 6 | phase segmentation | `segment` | keep |
| 7 | Murphy scoring | `score` | position measures done; peak-vel now trustworthy via the blend |

## What's decided (with evidence — see docs/SPEED_METRICS.md, PIPELINE_V2_PLAN.md)

**Cup tracking (v2):** detect-once UETrack-B + greedy ≥2-cam consensus, batched (~2x). Traces OMC to
0.9995 median corr. Re-anchoring OFF (poisons). → keep.

**Pose refine (v2):** SmoothNet 3D (pretrained h36m, window 32) — wrist jitter −93%, peak-vel restored.
→ keep.

**Wrist SPEED (the day's main result):** the pipeline's Δposition/Δt is velocity-noisy (differentiation
amplifies YOLO's ~2.5mm jitter). Two better signals, FUSED:
- **PyrLK optical flow** = direct velocity (no differentiation). Best OFF-peak (4mm) but OVER-shoots the
  peak +61mm/s (motion blur).
- **SmoothNet** = accurate at the peak (works on unblurred position) but noisy off-peak.
- **BLEND** (`wb=sigmoid((flow_speed-350)/120); blend=(1-wb)*flow+wb*smoothnet`) gets BOTH: peak 20mm/s,
  off-peak 4.7, best worst-case timing (max 133ms vs 233-333 for either alone). → adopt for the speed
  path. PyrLK is the flow (cheap 0.5ms, beats DIS/RAFT/tuned). ⚠ 350/120 hand-set — LOPO-tune first.

**Pose model:** YOLO-pose stays. It BEATS RTMPose/BlazePose on both speed (GPU-batched) AND accuracy
(multi-view consistency); see project_pose_model_speed. RTMPose CPU-only here (CUDA 11.8 vs onnxruntime's
CUDA 12).

## What was RULED OUT (tested, don't revisit without a reason)

- **Fusion variants for speed:** KF-prior (16.6, over-smooths), flow-integrate→SmoothNet (10.9). The
  speed-weighted blend wins; others lose. (Judged on PEAK, not per-frame — the metric flips the winner.)
- **Point-tracking (CoTracker3, TAPNext++):** doesn't beat YOLO+flow. CoTracker drifts (single-seed) or
  = a 2D smoother (re-seeded, SmoothNet-level, loses to flow); flow-seeded-from-CT has a catastrophic
  p90/max tail (665 vs 162) when it drifts. TAPNext++ too slow (28fps bf16). A strong per-frame detector
  beats tracking-from-a-seed. Weights kept for a future revisit. See project_pointtrack_negative.
- **Deep optical flow (RAFT/FastFlowNet):** loses to PyrLK on absolute error; not a capacity problem.

## Build order (what to wire next)

1. **`cup_task.flow_speed`** — PyrLK the wrist pixel per cam, triangulate {p} and {p+flow} → 3D velocity.
   Cache per-clip (cache/flow_vel/). Cheap (0.5ms/wrist/cam). **Risk: low** (validated).
2. **`cup_task.speed_blend`** — sigmoid gate between flow-speed and SmoothNet-speed. Feed `score.py`'s
   peak_velocity. **First: LOPO-tune the 350/120 gate** on P07/P08 (hand-set now). **Risk: low.**
3. Wire the v2 cup_track + pose_smooth flags (already built) into the default path.
4. Then scoring: peak hand velocity is now trustworthy (blend peak err 20mm/s vs raw pos-diff 144mm/s).

## Open threads (future)

- LOPO-tune the blend gate (350/120) + validate on a held-out cohort.
- P13's clock drift: a linear time-warp of OMC would recover P13 as usable ground truth (currently
  excluded); worth doing to expand the validation cohort.
- Point-tracking revisit if a fast (>60fps) non-drifting tracker appears, or for objects with weak
  per-frame detection (the cup, not the wrist).
