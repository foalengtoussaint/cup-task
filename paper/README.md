# Paper artifacts — Fast MMC vs OMC (DELTA drink task)

Validation of our markerless motion-capture (MMC) drink-task pipeline against the DELTA study's
optical-mocap ground truth (AutoMQ), modeled on **Unger et al., "Differentiable Biomechanics for
Markerless Motion Capture in Upper Limb Stroke Rehabilitation"**, IEEE T-MRB 8(1):90–97, 2026
(doi 10.1109/TMRB.2025.3605962). Every number quoted from them in the paper was read off arXiv v1;
the published version is not open access and those numbers are still to be re-checked against it.

Pipeline under test: **BA + SmoothNet** (robust-reprojection bundle adjustment, no bone prior →
SmoothNet temporal smoother). Cohort: **ten participants in eleven recording units** — P07 P08 P10
P12 P13 P14 P15 P17 P19 P251 P252, where P251 and P252 are the same participant recorded twice under
different calibrations and treated separately throughout. The optical reference is the DELTA study's C3D markers put through the same measure operator
and the same segmenter as the markerless pose; the study's own stored measures and phase windows are
not used (see Reproduce). All markerless measures are computed from our pose keypoints only.

## Contents

| file | what | mirrors |
|---|---|---|
| `main.tex` | the paper itself; `\input{}`s everything in `tables/` | — |
| `fig3_trajectories.png` | exemplary per-frame trajectories (6 signals × 2 arms) with phases + MU marks | **Fig. 1** |
| `fig4b_mmcseg_anat12.png` | 12-panel correlation scatter, MMC vs OMC, per participant/arm, markerless phase windows, landmark-matched | **Fig. 2** |
| `fig4a_omcseg_anat12.png` | the same scatter under optical windows. Generated as the pair of Fig. 2 and used to compute `fig4_pair.csv`, but **not** input by `main.tex` — the paper reports that comparison in Table II instead | — |
| `tables/table1_trajectories.tex` | per-trajectory RMSE / r / bias / lag, median[IQR], split by arm; the former Table II is folded in as its Bias IQR row | **Table I** |
| `tables/table3_measures.tex` | discrete-measure r_s + r_av under both window sources and at 30 Hz, with bootstrap intervals (`table*`, full width) | **Table II** |
| `tables/table4_boundaries.tex` | phase-boundary agreement, MMC cup vs OMC cup | **Table III** |
| `tables/table2_bias_iqr.tex` `table5_latency.tex` `table6_fps.tex` `table7_alt.tex` | **not input by `main.tex`.** Outputs of analyses that were folded in elsewhere or cut; see VERIFY.md for which and why | — |
| `table1_trajectories.md` `table2_bias_iqr.md` `table3_measures.md` | markdown mirrors of the above | — |
| `trajectory_agreement.csv` | per-trial per-trajectory raw agreement (source for Tables I/II) | — |
| `table3_measures.csv` | Table II as data (source for `table3_measures.tex`) | — |
| `fig4_pair.csv` | the per-measure r under each window source, and their difference | — |
| `VERIFY.md` | claim-by-claim provenance: every number traced to the code that made it | — |
| `results/` | the CSVs behind the numbers, plus `PIPELINE.md` (stage-by-stage derivation) | — |
| `SESSION_2026-08-19.md` | session note; records the measures-table chain recovery | — |

**The `.tex` file names predate the paper's numbering and no longer match it.**
`table1_trajectories` is Table I, `table3_measures` is Table II, `table4_boundaries` is
Table III. The names are kept because VERIFY.md traces numbers to them throughout.

Definitions for every measure (operator, window, reduction) now live in the Methods section of
`main.tex`. The older standalone `measures_methods.md` was superseded by it and is in
`archive_20260820/`.

## Layout

```
paper/
  main.tex  refs.bib  README.md  VERIFY.md  SESSION_2026-08-19.md
  fig3_trajectories.png  fig4a_omcseg_anat12.png  fig4b_mmcseg_anat12.png
  refs/     the reference PDFs, 7 of the 10 bib entries
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

`make_tables_tex.py` must run **after** `measures_table.py` and `paper_trajectories.py`, since it
reads the CSVs they write. Nothing in the PDF is typed by hand: `main.tex` `\input{}`s only what
these scripts emit, so the paper cannot drift from the caches.

Table II and Fig. 2 both exclude the trials the segmenter declines, the same exclusion Table III
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
`scripts/merge_matched_angles.py` built the measures table and correlation figure against the DELTA study's own processing
on its own phase windows, which needed a separate step to reconcile its world-axis angle definitions
against our body-frame ones. Scoring our operator against itself on both sides removes that step
entirely, so the chain and its `SCORE_CSV=…matchedangles.csv` warning are gone. `score_vs_automq.py`
stays — the paper imports its operators (`_planar_body_angles`, `_elbow_series`,
`peak_velocity_reduce`, `angle_measures_automq`), and the ablation grid still runs through it.

## What differs from Unger

- **11 panels, not 12** — the optical reference has no drinking-phase shoulder-flexion column, and
  the two percent-timing variants Unger also omits are excluded. See the Methods section of `main.tex`.
- **Phases from the markerless recording** — Unger classify phases from optical data for both
  systems and name markerless phase classification as unvalidated in their Limitations. Table II's
  markerless-window pair and Fig. 2 are that validation.
- **Geometry, not IK** — AutoMQ (and our scorer) compute angles geometrically from 3D points; Unger's
  OMC/MMC use full OpenSim/MuJoCo inverse-kinematics biomech models. Our trajectory correlations are
  therefore a notch below theirs (e.g. elbow extension r≈0.96 vs their 0.99) — same ordering, right
  ballpark, on consumer webcams at 60 Hz.

## Current state (2026-08-25)

Builds clean: 8 pages (7 of content plus references), 0 overfull boxes, 0 undefined references.
Whether 7 content pages is an overrun depends on the venue, which is not yet fixed; deleting Fig. 1
is the only measured way to reach 6 and has **not** been done.

Table II is full width (`table*`) and carries 95% bootstrap intervals on all five r columns, because
interjoint coordination's r_s spans 0.27–0.98 across random subsamples of the same trials and a bare
point estimate cannot be read. Bibliography is 10 entries, all cited, every field publisher-checked,
with PDFs of 7 in `refs/` and provenance at the end of `refs.bib`.

Eight `\todo`s remain and none is a writing task. Author input: title, group name, co-author list and
affiliations, funding. Blocked on data that is not on this machine: the study population (FMA-UE
totals and the Record ID → P-label key — see the todo in §IV-A). Blocked on a paywall: re-checking
the numbers quoted from Unger et al. against the published version. Resolves on publication: the two
repository URLs.

The chain that regenerates all of it is under [Reproduce](#reproduce) above; it is
stated once, there.


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
