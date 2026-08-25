# Paper artifacts — Fast MMC vs OMC (DELTA drink task)

Validation of our markerless motion-capture (MMC) drink-task pipeline against the DELTA study's
optical-mocap ground truth (AutoMQ), modeled on **Unger et al., "Differentiable Biomechanics for
Markerless Motion Capture in Upper Limb Stroke Rehabilitation" (ICORR 2025, arXiv:2411.14992)**.

Pipeline under test: **BA + SmoothNet** (robust-reprojection bundle adjustment, no bone prior →
SmoothNet temporal smoother). Cohort: 11 participants (P07 P08 P10 P12 P13 P14 P15 P17 P19 P251
P252). The optical reference is the DELTA study's C3D markers put through the same measure operator
and the same segmenter as the markerless pose; the study's own stored measures and phase windows are
not used (see Reproduce). All markerless measures are computed from our pose keypoints only.

## Contents

| file | what | mirrors |
|---|---|---|
| `main.tex` | the paper itself; `\input{}`s everything in `tables/` | — |
| `fig4a_omcseg_anat12.png` `fig4b_mmcseg_anat12.png` | 12-panel correlation scatter, MMC vs OMC, per participant/arm — optical phase windows and markerless ones, same trials and axes, landmark-matched. `main.tex` uses these; the `_anat12`-less pair is the uncorrected version | **Fig 4** |
| `fig3_trajectories.png` | exemplary per-frame trajectories (6 signals × 2 arms) with phases + MU marks | **Fig 3** |
| `tables/table1_trajectories.tex` | per-trajectory RMSE / r / bias / lag, median[IQR], split by arm | **Table I** |
| `tables/table2_bias_iqr.tex` | mean IQR of bias across patients (within-patient variability) | **Table II** |
| `tables/table3_measures.tex` | discrete-measure r_s (single trials) + r_av (per-arm averaged), under both window sources | **Table III** |
| `tables/table4_boundaries.tex` | phase-boundary agreement, MMC cup vs OMC cup | **Table IV** |
| `table1_trajectories.md` `table2_bias_iqr.md` `table3_measures.md` | markdown mirrors of the above | — |
| `trajectory_agreement.csv` | per-trial per-trajectory raw agreement (source for Tables I/II) | — |
| `table3_measures.csv` | Table III as data (source for `table3_measures.tex`) | — |
| `fig4_pair.csv` | the per-measure r under each window source, and their difference | — |
| `VERIFY.md` | claim-by-claim provenance: every number traced to the code that made it | — |
| `results/` | the CSVs behind the numbers, plus `PIPELINE.md` (stage-by-stage derivation) | — |
| `SESSION_2026-08-19.md` | session note; records the Table III chain recovery | — |

Definitions for every measure (operator, window, reduction) now live in the Methods section of
`main.tex`. The older standalone `measures_methods.md` was superseded by it and is in
`archive_20260820/`.

## Layout

```
paper/
  main.tex  refs.bib  README.md  VERIFY.md  SESSION_2026-08-19.md
  fig3_trajectories.png  fig4a_omcseg.png  fig4b_mmcseg.png
  table1_trajectories.md  table2_bias_iqr.md  table3_measures.md  table3_measures.csv
  trajectory_agreement.csv
  tables/    table1..table4 .tex     <- what main.tex \input{}s
  results/   PIPELINE.md + the backing CSVs
  scripts/   fig4_pair.py  measures_table.py  paper_trajectories.py
             make_tables_tex.py  make_seg_table.py  plot_*.py
             anat12/   anatomical-frame work behind the angle measures
```

The scripts here import the repo's core modules (`compare_pose_omc_delta`, `gnn_train`,
`results_v3_delta`, `pipeline.score`) from `../scripts` and `..` via a path shim, and read the pose
caches under `../cache/` — so this folder is not standalone, it rides on the cup-task repo. Outputs
are written back into `paper/`.

## Reproduce

Everything the paper reports is scored with **our** measure operator and **our** segmenter on both
sides. The DELTA study's stored measures and phase windows are not used: its phases are classified
from end-effector velocity on the optical wrist and cup markers (Unger et al. II-D-2), a different
rule from the two distance channels here, so scoring against them would fold a segmentation
difference into what should read as pose error. The optical reference is the C3D markers put through
the same code as the markerless pose.

```bash
conda activate object_tracking
cd ~/Documents/cup-task
export OT_SEG_INPUTS_DIR=seg_inputs_ship OT_NCAMS_DIR=cup_ncams_26x

# --- measures: one scorer run gives all three columns (omc/omc, mmc/omc, mmc/mmc) ---
python scripts/score_own_phases.py            # ~70 s -> out/scoring/score_own_phases.{csv,npz}
# landmark offsets (anat12), then the tables/figures the paper uses
python paper/scripts/anat12/prep_cache2.py build          # cache/omc_prep2, once
python paper/scripts/anat12/anat12_lopo.py --wv 1 --wa 1 --no-lopo \
    --out out/scoring/anat12_wv1wa1.csv                   # -> ..._theta.csv
python scripts/score_own_phases.py --anat12 out/scoring/anat12_wv1wa1_theta.csv \
    --out out/scoring/score_own_phases_anat12.csv

python paper/scripts/measures_table.py                    # uncorrected -> table3_measures.{md,csv}
python paper/scripts/measures_table.py --own out/scoring/score_own_phases_anat12.csv \
    --suffix _anat12                                      # what Table III uses
python paper/scripts/fig4_pair.py                         # uncorrected pair
python paper/scripts/fig4_pair.py --own out/scoring/score_own_phases_anat12.csv \
    --suffix _anat12                                      # what Fig 4 uses

# --- phase-boundary table ---
python scripts/score_seg_boundaries.py        # -> out/scoring/seg_boundaries.csv
cp out/scoring/seg_boundaries.csv paper/results/
python paper/scripts/make_seg_table.py        # -> tables/table4_boundaries.tex

# --- trajectories (Fig 3 + Tables I/II), independent of the above ---
python paper/scripts/paper_trajectories.py    # ~4 min, aligns each trial
                                              # -> fig3, table1/2 .md, trajectory_agreement.csv

# --- LaTeX tables last: reads table3_measures.csv + trajectory_agreement.csv ---
python paper/scripts/make_tables_tex.py       # -> tables/table1..table3 .tex
```

`make_tables_tex.py` must run **after** `measures_table.py` and `paper_trajectories.py`, since it
reads the CSVs they write. Nothing in the PDF is typed by hand: `main.tex` `\input{}`s only what
these scripts emit, so the paper cannot drift from the caches.

Table III and Fig 4 both exclude the trials the segmenter declines, the same exclusion Table IV
makes, so all three are scored on the same 750 trials of 790. `--keep-declined` puts them back.

`make_tables_tex.py` reads `table3_measures_anat12.csv` by default; `M3_CSV=table3_measures.csv`
renders the uncorrected table instead. Both CSVs are kept.

Two exclusions live in `scripts/cache_seg_inputs.py` and are **different in kind**: `_MISPAIRED`
(3 trials whose clip matches a neighbouring C3D better) and `_BAD_OMC` (4 trials with a marker
fault `_destep` cannot repair). Neither is the segmenter's 4.9% decline rate, which is detected
from the markerless cup track and is a property of the method rather than of the reference.

To rebuild the upstream pose caches from scratch (relaxed reprojection gate), see
`paper/scripts/rebuild_relaxed_cache.sh` — that is the four-stage cache rebuild, not part of the
per-figure loop above.

## Retired 2026-08-21

`paper_table3.py`, `fig4_correlation.py`, `scripts/omc_matched_angles.py` and
`scripts/merge_matched_angles.py` built Table III and Fig 4 against the DELTA study's own processing
on its own phase windows, which needed a separate step to reconcile its world-axis angle definitions
against our body-frame ones. Scoring our operator against itself on both sides removes that step
entirely, so the chain and its `SCORE_CSV=…matchedangles.csv` warning are gone. `score_vs_automq.py`
stays — the paper imports its operators (`_planar_body_angles`, `_elbow_series`,
`peak_velocity_reduce`, `angle_measures_automq`), and the ablation grid still runs through it.

## What differs from Unger

- **11 panels, not 12** — the optical reference has no drinking-phase shoulder-flexion column, and
  the two percent-timing variants Unger also omits are excluded. See the Methods section of `main.tex`.
- **Phases from the markerless recording** — Unger classify phases from optical data for both
  systems and name markerless phase classification as unvalidated in their Limitations. Table III's
  right-hand pair and Fig. 4b are that validation.
- **Geometry, not IK** — AutoMQ (and our scorer) compute angles geometrically from 3D points; Unger's
  OMC/MMC use full OpenSim/MuJoCo inverse-kinematics biomech models. Our trajectory correlations are
  therefore a notch below theirs (e.g. elbow extension r≈0.96 vs their 0.99) — same ordering, right
  ballpark, on consumer webcams at 60 Hz.

## Current state (2026-08-25)

Paper is 7 pages of content plus references — **one page over the 6-page target**. Measured: deleting
Fig. 1 restores 6 exactly and nothing else does (not shrinking it, not any float placement). Awaiting
a decision.

Table II is now full width (`table*`) and carries 95% bootstrap intervals on all five r columns,
because interjoint coordination's r_s spans 0.27–0.98 across random subsamples of the same trials and
a bare point estimate cannot be read. Bibliography is 10 entries, every field publisher-checked, with
PDFs of 7 in `refs/` and provenance at the end of `refs.bib`.

Regenerating everything:

```bash
export OT_SEG_INPUTS_DIR=seg_inputs_ship OT_TRACKS_DIR=tracks_uetrack_26x OT_NCAMS_DIR=cup_ncams_26x
python paper/scripts/anat12/anat12_lopo.py --wv 1 --wa 1 --no-lopo \
    --out out/scoring/anat12_wv1wa1.csv          # -> ..._theta.csv (arm/shoulder offsets)
python paper/scripts/anat12/trunk_theta.py       # -> out/scoring/trunk_theta.csv (sternum)
python scripts/score_own_phases.py --anat12 out/scoring/anat12_wv1wa1_theta.csv \
    --trunk-theta out/scoring/trunk_theta.csv \
    --out out/scoring/score_own_phases_anat12.csv
python scripts/score_seg_boundaries.py && cp out/scoring/seg_boundaries.csv paper/results/
python paper/scripts/paper_trajectories.py       # Table I + Fig 1
python paper/scripts/measures_table.py --own out/scoring/score_own_phases_anat12.csv --suffix _anat12
python paper/scripts/make_seg_table.py           # Table III
python paper/scripts/measure_cis.py              # -> table3_cis.csv, the intervals in Table II
python paper/scripts/make_tables_tex.py          # all .tex tables (reads table3_cis.csv)
python paper/scripts/fig4_pair.py --own out/scoring/score_own_phases_anat12.csv --suffix _anat12
cd paper && tectonic main.tex
```

`fps_full.py` (30 Hz, three shards, ~1 h) feeds Table II's last column via `paper/table3_fps30.csv`
and does not need re-running unless the 60 Hz numbers change.

Analyses that are NOT in the paper but whose scripts and outputs are kept: `fps_cup.py` (cup
re-tracked at both rates), `alt_measures.py` (vector coding, LDLJ, the travel-share coordination
index), `scripts/latency_bench.py` and `scripts/latency_opt.py`, and `scripts/seg_rules/` (the
reach-onset and settle rule sweep — `seg_rule_sweep.csv`, `seg_rule_measures.csv`; conclusion was to
keep the shipped rules, and the reason boundary metrics mislead here is worth reading before touching
a boundary). See VERIFY.md for why each was cut and what it found.

`scripts/seg_sequential.py` reads `OT_SEG_ONSET` / `OT_SEG_SETTLE` / `OT_SEG_PEAK_FRAC`, defaulting to
the shipped `pos` / `end` — so every published number reproduces unless they are set.
