# Final results - drink-task Murphy measures, markerless vs OMC

Pipeline as scored here: pose = BA + SmoothNet (fixed). Cup = stock COCO **yolo26x-seg** consensus
seed (scanned forward to the first frame that triangulates >=3 cameras within 30 px) + detect-once
UETrack; then the >=3-agreeing-camera floor applied strictly, remaining interior gaps <=0.75 s filled
by KF+RTS, and what the KF cannot justify filled from the wrist (cup ~= wrist + median(cup-wrist) over
the hold) on the cup->mouth channel ONLY -- the wrist proxy makes wrist->cup constant, and grasp and
release are read from exactly that channel. Phases = `segment_sequential` on that cup.
Ground truth for the three angle measures and interjoint is the **matched-definition** OMC (our own
operator applied to the OMC markers), not AutoMQ's world-axis angles.

## Measures: cost of markerless phase segmentation
Pose and ground truth identical in both columns; only the phase windows differ.

| measure | rs_omc_seg | rs_mmc_seg | delta | r_mmc_seg | med_abs_err_mmc_seg | n |
|---|---|---|---|---|---|---|
| max_trunk_displacement | 0.933 | 0.927 | -0.005 | 0.91 | 12.242 | 595 |
| total_movement_time | 0.942 | 0.91 | -0.033 | 0.779 | 0.68 | 577 |
| time_to_peak_velocity | 0.938 | 0.897 | -0.04 | 0.683 | 0.063 | 577 |
| time_to_first_peak_velocity | 0.902 | 0.89 | -0.012 | 0.87 | 0.063 | 577 |
| max_elbow_angle | 0.862 | 0.866 | 0.004 | 0.891 | 10.462 | 595 |
| peak_velocity | 0.858 | 0.814 | -0.043 | 0.824 | 49.043 | 597 |
| max_shoulder_abduction | 0.795 | 0.799 | 0.003 | 0.811 | 5.096 | 595 |
| peak_elbow_angular_velocity | 0.783 | 0.708 | -0.075 | 0.643 | 12.643 | 595 |
| max_shoulder_flexion | 0.634 | 0.626 | -0.008 | 0.678 | 4.323 | 595 |
| interjoint_coordination | 0.449 | 0.444 | -0.005 | 0.256 | 0.013 | 575 |
| number_of_movement_units | 0.426 | 0.433 | 0.007 | 0.55 | 1.0 | 577 |

## Phase-boundary agreement, MMC input vs OMC input (same rule, frames @60 Hz)

| boundary | n | median_f | abs_median_f | p90_abs_f | frac_gt5f | frac_gt15f |
|---|---|---|---|---|---|---|
| reach onset | 636 | 0.0 | 2.0 | 5.0 | 0.094 | 0.025 |
| grasp | 636 | 3.0 | 6.5 | 24.0 | 0.564 | 0.165 |
| drink onset | 597 | 4.0 | 8.0 | 31.4 | 0.637 | 0.208 |
| drink offset | 597 | -7.0 | 8.0 | 18.0 | 0.638 | 0.161 |
| release | 636 | -2.0 | 3.0 | 16.5 | 0.281 | 0.104 |
| settle | 636 | -1.0 | 3.0 | 14.0 | 0.259 | 0.094 |

## Cup 3D consensus coverage per participant (median fraction of frames with >=3 agreeing cameras)

| part | clean3d_refill | yolo26x_seg | n_trials | zero_old | zero_new |
|---|---|---|---|---|---|
| P07 | 0.968 | 1.0 | 63 | 0 | 0 |
| P08 | 0.0 | 1.0 | 47 | 36 | 0 |
| P10 | 0.109 | 1.0 | 73 | 15 | 0 |
| P12 | 0.047 | 0.557 | 77 | 35 | 0 |
| P13 | 0.0 | 0.598 | 73 | 40 | 0 |
| P14 | 0.765 | 1.0 | 83 | 4 | 0 |
| P15 | 0.563 | 1.0 | 56 | 0 | 0 |
| P17 | 0.369 | 0.592 | 55 | 7 | 0 |
| P19 | 0.576 | 0.85 | 30 | 2 | 0 |
| P251 | 0.864 | 1.0 | 39 | 0 | 0 |
| P252 | 0.784 | 1.0 | 40 | 0 | 0 |

## Figures

- `fig4_final.png` - the 11-measure scatter grid, MMC's own phases.
- `seg_mmc_vs_omc_kf.png` - two failure trials: cup->mouth, MMC vs OMC, with sub-floor frames marked.
- `seq_outliers.png` - worst boundary disagreements on the old cup (broken cup tracks).
- `grasp_blowup.png` - why an argmin-anchored grasp rule fails on markerless input.
- `proximal_vc.png` - vector-coding alternative to Pearson interjoint coordination.

## Backing per-trial data (this folder)

- `score_e2e_final.csv` - every measure, every trial, all four phase sources.
- `seg_boundaries.csv` / `_flags.csv` - per-trial boundaries per segmenter variant and cup input.
- `drink_approach.csv` - cup's closest approach to the mouth inside the true drink window.
- `hold_corr.csv` - corr(cup->mouth, wrist->mouth) inside the hold: the MMC-only QC signal.
- `qc_kept_trials.csv` - trials passing that QC at r >= 0.95 (482 of 636 on the old cup).
- `omc_matched_angles.csv` - matched-definition OMC angles.
- `anat12w0_omc_values.csv`, `recon_offsets_r*.csv`, `all_measures_w.csv` - landmark-offset analysis.
- `vector_coding.csv` - coupling-angle bins per trial.
