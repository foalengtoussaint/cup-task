# cup-task

**Markerless scoring of the standardized drinking task, and its validation against optical mocap.**

Five synchronized webcams in, the twelve drinking-task movement-quality measures out — with the
movement phases found from the video itself rather than from the mocap, no inverse kinematics and no
fitted biomechanical model.

This repository is the **research and validation record** behind the paper in [`paper/`](paper/): the
scripts that produce every table and figure, and a claim-by-claim account of what each number was
measured from. To *use* the method rather than reproduce the paper, see the pipeline-only repository
(link in `paper/main.tex`), which carries the same stages without the research clutter.

## What is validated

On **750 trials** from **eleven recording units** (ten participants, one recorded twice under
different calibrations) of the DELTA cohort, recorded simultaneously with Qualisys optical mocap:

| | |
|---|---|
| measures, markerless pose **and** markerless phase windows | 10 of 12 at `r_s ≥ 0.84`, `r_av ≥ 0.93` |
| measures, both systems given the same optical windows | 11 of 12 |
| speed | a 9.2 s trial from five cameras scored in **16.8 s** on one consumer GPU |
| of which not overlappable with capture | **1.3 s** — the rest is per-frame work a front end absorbs |
| segmentation | the segmenter **declines 39 of 790** trials and says so, from the cup track alone |

Every one of these is traced to the code that produced it in [`paper/VERIFY.md`](paper/VERIFY.md).
Numbers quoted anywhere else in this repo that disagree with the paper are superseded by the paper.

## Layout

```
pipeline/    the stages, imported by everything below
scripts/     the harnesses that build the caches and score the cohort
paper/       the paper, its scripts, its tables and its provenance record
archive/     settled investigations and superseded outputs, kept deliberately
```

`pipeline/` holds the eight modules the current chain uses — `cup_track`, `consensus`,
`cup_ba_refine`, `triangulate`, `kalman_3d`, `pose_smooth`, `segment`, `score`. The per-frame front
end that feeds them (detection, 2D pose, optical flow) predates the reorg and still sits in
`pipeline/_archive_20260820/`; `scripts/latency_bench.py` binds it back under its original name on
purpose, so the timings it reports are of the code that actually built the caches.

## Reproduce

Full chain, in order, with the environment variables that select the shipped caches:
**[`paper/README.md`](paper/README.md)**. Stage-by-stage derivation and the reasoning behind each
choice: [`paper/results/PIPELINE.md`](paper/results/PIPELINE.md).

```bash
conda activate object_tracking        # torch 2.7 + cu118, ultralytics 8.4.49
export OT_SEG_INPUTS_DIR=seg_inputs_ship OT_TRACKS_DIR=tracks_uetrack_26x OT_NCAMS_DIR=cup_ncams_26x
```

Those three variables are not optional — they select the caches the published numbers were computed
from. `scripts/seg_sequential.py` additionally reads `OT_SEG_ONSET` / `OT_SEG_SETTLE` /
`OT_SEG_PEAK_FRAC`, which default to the shipped rules, so leaving them unset reproduces the paper.

## What is not in the repository

`models/`, `cache/`, `external/`, `data/` and the archived run outputs are large and regenerable, so
they are gitignored — but nothing here deletes them. The offline scoring path runs from `cache/`
with no GPU; only rebuilding the caches needs one. If the UETrack weights go missing, re-fetch them
from HuggingFace `kangben258/UETrack` into `models/trackers/uetrack/`.

The DELTA recordings themselves are not ours to distribute. Access is via the study of Unger et al.

## Archive

`archive/` and `scripts/archive/` hold work that is settled rather than wrong — the investigations
that were run, and the ones that failed. `paper/VERIFY.md` records why each was cut and what it
found; the reach-onset and settle rule sweep under `paper/scripts/seg_rules/` is worth reading before
anyone touches a phase boundary, because the boundary-agreement metrics there actively mislead.
