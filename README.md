# cup-task

**Mocap-free drink-task scoring from multi-camera video.** Detect the cup and the person's
keypoints, fuse to 3D with the existing camera calibration, segment the movement phases, and emit the
Murphy clinical measures — no motion capture required.

Standalone productization of the `object_tracking/experiments/drink_study` research pipeline.

## Documentation

| | |
|---|---|
| **[SPEC.md](archive/docs_20260820/SPEC.md)** | **how the pipeline works** — stages, signals, phase definitions, conventions |
| **[RESULTS.md](archive/docs_20260820/RESULTS.md)** | **every measured number** — accuracy vs OMC, speed, segmentation, Murphy |
| [WORKLOG.md](archive/docs_20260820/WORKLOG.md) | chronological record, including what was tried and rejected |
| [SPEED_METRICS.md](archive/docs_20260820/SPEED_METRICS.md) | the wrist-speed method comparison in detail |

## Run it

```bash
# whole pipeline on one rep's clips
python -m pipeline.pipeline CLIPDIR --calib calibration.toml -o out/ \
       --smooth-pose --cup-track

# validation against OMC ground truth (cache-only, no GPU)
python scripts/results_v3_delta.py --what all

# speed benchmark (online fps + offline post-processing)
python scripts/bench_v3.py --cams 1 5 10
```

## Pipeline at a glance

```
ONLINE   capture ─┬─ YOLO-pose ──────────────┐
                  ├─ YOLO cup once → UETrack ┤   (2D per camera)
                  └─ PyrLK flow @ wrist, cup ┘
                     ─────────────────────────────────────────
OFFLINE           triangulate → consensus → SmoothNet → blend
                  → segment (7 phases) → Murphy measures
```

Split at **the point a stage stops needing raw pixels**. Flow is online because it needs the frame
*pair* (offline it would force a second full decode); SmoothNet is offline because it is a
non-causal ±16-frame window. Full reasoning in [SPEC.md](archive/docs_20260820/SPEC.md).

| stage | module | what it does |
|---|---|---|
| cup detection | `pipeline.cup_detect` | YOLO, one-shot seed |
| cup tracking | `pipeline.cup_track` + `consensus` | detect-once UETrack, ≥2-cam greedy consensus |
| body keypoints | `pipeline.pose_keypoints` | YOLO-pose, COCO-17 upper body |
| 3D triangulation | `pipeline.triangulate` | multi-view DLT |
| temporal refinement | `pipeline.pose_smooth` | SmoothNet (pose **and** cup) |
| wrist/cup velocity | `pipeline.flow_speed`, `speed_blend` | PyrLK flow → 3D velocity; speed-gated blend |
| phase segmentation | `pipeline.segment` | 7 Murphy phases, van Andel definitions |
| clinical measures | `pipeline.score` | 8/8 position measures |

## Headline results

Validated on DELTA, **n = 12** (P07+P08 × trial_10–15) against Qualisys OMC:

| | |
|---|---|
| cup trajectory | displacement corr **0.9996**, median error **2.3 mm**, 100 % coverage |
| wrist speed | **6.9 mm/s** per-frame, **20.9 mm/s** at the peak (6–7× better than the v1 baseline) |
| segmentation | **0/83 missed phases**, every boundary within **0–67 ms** (≤4 frames) |
| Murphy | peak_velocity **−46 %** error, movement_units **−50 %**, all 8 measures on all 12 trials |
| realtime | 100 fps at 1 camera, 38.5 at 5 (GPU-bound); offline post-processing ~887 ms/trial |

Full tables, caveats and the cohort exclusion in [RESULTS.md](archive/docs_20260820/RESULTS.md).

## Keypoints

YOLO-pose (COCO-17), upper body — head (mouth proxy), shoulders, elbows, wrists, hips. Knees/ankles
dropped (no signal for a seated drinking task). The mouth-proxy point (nose → eye/ear fallback)
replaces the QTM head marker the research pipeline depended on, which is what makes scoring
mocap-free.

## Layout

```
pipeline/        the pipeline modules
scripts/         active harnesses (results_v3_delta, bench_v3, cup_flow_probe) + shared libs
scripts/archive/ settled investigations, kept — see its README
models/          pose + SmoothNet + UETrack weights (repo-persistent on purpose)
cache/           cached detections/tracks/flow — the offline path runs from here with no GPU
runs/segment/    cup YOLO weights (still referenced by cache_tracks.py)
out/figures/     current figures cited by the docs
archive/         settled outputs and checkpoints, kept — see its README
```

## Setup

```bash
conda activate object_tracking        # torch 2.7 + cu118, ultralytics 8.4.49
pip install ultralytics opencv-python ezc3d huggingface_hub
```

Weights and data (`models/`, `cache/`, `runs/`, `data/`, `external/`) are **not versioned** — they are
large and regenerable, but nothing is ever auto-deleted. If the UETrack weights go missing, re-fetch
them from HuggingFace `kangben258/UETrack` into `models/trackers/uetrack/`.
