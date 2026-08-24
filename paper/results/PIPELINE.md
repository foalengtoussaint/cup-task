# How every number in `results/` was produced — and why each choice was made

Read this with `README.md` (which holds the tables). This file explains the pipeline stage by stage,
what each script does, the exact commands, and the decisions — including the things that did **not**
work, so they are not re-tried.

Everything is cache-first: each stage writes a cache, and later stages never recompute it. Three
environment variables redirect the caches so an alternative can be built without overwriting the
default one:

| variable | selects | values used here |
|---|---|---|
| `OT_TRACKS_DIR` | UETrack cup tracks | `tracks_uetrack` (old), `tracks_uetrack_26x` (final) |
| `OT_SEG_INPUTS_DIR` | segmenter input tracks | `seg_inputs`, `seg_inputs_26x` |
| `OT_NCAMS_DIR` | per-frame cup consensus size | `cup_ncams`, `cup_ncams_26x` |

---

## Stage 1 — the cup detector (this is what was wrong)

`cache/delta/*/dets/*.cup.json` records the model that produced it. For this cohort it said
`models/cup_clean3d_refill.pt` — the BRIO fine-tune that `archive/docs_20260820/WORKLOG.md:927` had **already**
recorded as the wrong model here: *"COCO teacher yolo26x-seg gets 77% >=3-cam consensus on P14, our
BRIO finetune 8.2%"*. Measured cohort-wide, that fine-tune detects the cup on 6–54 % of frames and is
near-blind in most views (P08 cam4 0.02, P10 cam5 0.00) while a single camera carries 0.85–0.98 —
the viewpoint-transfer failure, not a data-volume one.

Why that is fatal specifically under **detect-once**: seeding needs one box anywhere in ~437 frames,
so a camera detecting at 0.02 still gets seeded, off whatever that lone frame fired on. UETrack then
tracks that confidently for the whole trial, and the cameras never agree. Symptom: a tracker point on
~380–408 of 408 frames in every camera, and 0 or 2 agreeing cameras on every frame.

**Fix — `scripts/cache_cup_seed26x.py`.** Stock COCO `yolo26x-seg`, scanning forward from frame 0
(stride 3) and stopping at the **first frame whose cup-like detections actually triangulate** (≥3
cameras within 30 px — the pipeline's own gate). Cameras that did not detect at that frame get a seed
box reprojected from the consensus 3D, sized along `calib.R[0]` (offsetting along a *world* axis
foreshortens to ~0 for a camera looking down it — the old apparent-radius bug). Trials that never
reach consensus are written `seed: null` rather than falling back to a lone detection.

    python scripts/cache_cup_seed26x.py            # -> cache/cup_seed26x/, ~9 min, 636/636 seeded

Cost saving is the point of the forward scan: the detector runs on a **median of 1 frame per trial**
(p90 19), not the whole video.

## Stage 2 — re-track

    python scripts/cache_uetrack_tracks.py --parts P07 P08 P10 P12 P13 P14 P15 P17 P19 P251 P252 \
        --seeds26x                                 # -> cache/tracks_uetrack_26x/, ~2 h GPU

`--seeds26x` reads the consensus seeds instead of each camera's first `clean3d_refill` box, and
writes to a separate directory so the old cache stays intact for comparison.

Result: frames with ≥3 agreeing cameras **46 % → 87 %**; trials with no ≥3-cam frame anywhere
**139 → 0**; median longest dropped run 2.40 s → 0.00 s. Per participant in
`table_consensus_coverage.csv` (P08 and P13 went 0.00 → 1.00 and 0.60).

## Stage 3 — inputs and consensus size

    OT_TRACKS_DIR=tracks_uetrack_26x OT_SEG_INPUTS_DIR=seg_inputs_26x \
        python scripts/cache_seg_inputs.py         # cup/wrist/nose from OMC and MMC + AutoMQ windows
    OT_TRACKS_DIR=tracks_uetrack_26x OT_NCAMS_DIR=cup_ncams_26x \
        python scripts/cache_cup_ncams.py          # cup 3D + n_cams per frame

`cache_seg_inputs.py` exists because every segmenter experiment was re-triangulating the cup and
re-parsing C3Ds (~4–5 min per source per run); with it, a new segmentation rule scores in 30 s.
`cache_cup_ncams.py` stores the **consensus size per frame**, which is what makes the ≥3 floor
applicable after the fact.

## Stage 4 — the cup the measures actually use

Three steps, in this order, all in `pipeline/triangulate.py`:

1. **Strict ≥3 floor.** Frames where fewer than 3 cameras agree are set to NaN. A 2-camera cup is not
   weak evidence — the robustness study puts 2-camera triangulation in the >1 m error regime. On the
   old cup this was 30 % of all frames; on the new one it is far less but still the failure mode
   during the drink.
2. **`kf_fill_gaps`** — KF+RTS, but only where filling is defensible: interior gaps ≤ 0.75 s, never
   extrapolating past the first or last real observation, and nothing at all below 30 % coverage.
   Necessary because an unguarded `kf_rts_smooth` produces a *smooth* curve across a whole trial that
   simply never goes where the cup went (measured: a trial 87 % without consensus coasts 300–600 mm
   from the mouth through the entire drink, and looks fine on a plot).
3. **`fill_cup_from_wrist`** — what the KF cannot justify is filled from the wrist, which still
   carries the cup: `cup ≈ wrist + median(cup − wrist)`. Two details that matter:
   - the offset is estimated **only over the hold** (frames where wrist→cup is within 60 mm of its own
     minimum, using observed frames only). Over the whole trial the estimate blends holding with
     not-holding, because before grasp and after release the cup sits on the table.
   - it is applied to the **cup→mouth channel only** (`segment_sequential(..., cup_mouth_xyz=…)`).
     Filling the cup position makes wrist→cup a constant, and grasp/release are read from exactly
     that channel — applied to both channels it cost release p90 16.5 → 33.5 frames and peak
     velocity 0.85 → 0.81.

## Stage 5 — phases and measures

    OT_TRACKS_DIR=tracks_uetrack_26x OT_SEG_INPUTS_DIR=seg_inputs_26x OT_NCAMS_DIR=cup_ncams_26x \
        python scripts/score_seg_boundaries.py     # -> seg_boundaries.csv (boundary agreement)
    OT_TRACKS_DIR=tracks_uetrack_26x OT_NCAMS_DIR=cup_ncams_26x \
      OT_CUP_STRICT_KF=1 OT_CUP_WRIST_PROXY=1 OT_E2E_OUT=score_e2e_final.csv \
        python scripts/score_e2e_seq.py            # -> score_e2e_final.csv (11 measures x 4 phase srcs)

`score_e2e_seq.py` holds the pose fixed (BA + SmoothNet) and varies **only** the phase windows:
`automq` (ground-truth phases), `seq_omc` (our rule on the OMC cup), `seq_mmc` (our rule on the MMC
cup), `omc_full` (OMC pose + OMC phases). `seq_omc` vs `seq_mmc` is therefore the isolated cost of
markerless segmentation — that is the `delta` column in `table_measures_final.csv`.

`score_seg_boundaries.py` scores the same rule on OMC vs MMC input **paired per trial**. Note the
metric: vs-AutoMQ error is *not* the target, because a boundary bias shared by both sides cancels when
MMC measures are compared against OMC measures. What survives is the OMC-vs-MMC disagreement, so that
is what the table reports.

## Stage 6 — ground truth for the angles

    python scripts/omc_matched_angles.py           # -> omc_matched_angles.csv

AutoMQ's stored angles use **world reference axes** (flexion = arm projected on the world Y–Z plane vs
global Z; verified in their notebook source — no IK, no OpenSim), while ours are body/trunk
referenced. Scoring against them mixes a definition difference into what reads as pose error. This
script recomputes the OMC side from the OMC markers with **our own operator** (`_planar_body_angles`,
`_elbow_series`, same low-pass, same windows), so the residual is pose error. The elbow measure is the
built-in control: the same three-point construction on both sides, so its matched and unmatched values
must agree — an earlier hand-rolled version was caught exactly there (0.808 vs 0.863).

Effect: flexion 0.34 → 0.63, abduction 0.75 → 0.79. All tables in `README.md` use this truth.

---

## Things that did not work (do not re-try)

- **Anchor-then-expand segmenter** (`scripts/seg_anchored.py`). Replacing "first run clearing a
  threshold" with "the run at `argmin`" fails on markerless input: the MMC wrist→cup minimum often
  falls *after* release, when the hand passes near the set-down cup, so grasp jumps ~400 frames.
  `grasp_blowup.png` shows it. Reach onset +90 frames, 498/636 trials with all boundaries.
- **Isolating just the run-selection change** (`scripts/seg_anchor_min.py`) — the two-line version of
  the above. Grasp p90 62 → 189 frames. So first-run selection was **not** the cause of the boundary
  tail; the cup track was.
- **Strict ≥3 without a fill.** Dropping sub-floor frames leaves holes that the segmenter's
  `_interp_nan_xyz` spans with a chord, which is worse than the 2-camera points it removed: drink
  offset p90 21 → 91 frames, and ~195 trials lose the drink phase entirely.
- **Wrist proxy on both channels.** See stage 4.3.
- **The anat12 landmark correction for interjoint.** It lifts the three angle measures to 0.92–0.96
  but *lowers* interjoint every time (0.45 → 0.21 here). Interjoint is a correlation, so it is blind
  to the bias the correction removes and sensitive to the shape it changes.
- **`predict(half=True)`** for fp16 — a no-op on `.pt` weights; the AutoBackend must be cast
  explicitly. And a GPU letterbox was a measured loss.

## Two open items

- **Movement units** is the one measure markerless segmentation clearly costs (0.43 vs 0.43 here after
  the guarded fill, but 0.31 before it) and it is the weakest measure in absolute terms in every
  variant.
- **QC without mocap.** `hold_corr.csv` holds corr(cup→mouth, wrist→mouth) inside each trial's own
  hold — computable from MMC alone. On OMC it never drops below 0.99; on MMC 5 % of trials go
  negative, and those trials carry the boundary errors (median |grasp error| 56 frames at r ≤ 0 vs 7
  at r ≥ 0.95). `qc_kept_trials.csv` is the 482/636 that pass at 0.95 **on the old cup** — it has not
  been recomputed on the new one.
