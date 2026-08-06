# Paper artifacts — Fast MMC vs OMC (DELTA drink task)

Validation of our markerless motion-capture (MMC) drink-task pipeline against the DELTA study's
optical-mocap ground truth (AutoMQ), modeled on **Unger et al., "Differentiable Biomechanics for
Markerless Motion Capture in Upper Limb Stroke Rehabilitation" (ICORR 2025, arXiv:2411.14992)**.

Pipeline under test: **BA + SmoothNet** (robust-reprojection bundle adjustment, no bone prior →
SmoothNet temporal smoother). Cohort: 11 participants, ~687 scorable trials. All measures are
computed from **our pose keypoints only**; AutoMQ is read as ground truth, never used in our
computation.

## Contents

| file | what | mirrors |
|---|---|---|
| `measures_methods.md` | exact operator, window, and reduction for every measure (+ our-vs-AutoMQ differences) | Methods §D |
| `fig4_mmc_vs_omc.png` | 11-panel correlation scatter, MMC vs OMC, per participant/arm | **Fig 4** |
| `fig3_trajectories.png` | exemplary per-frame trajectories (6 signals × 2 arms) with phases + MU marks | **Fig 3** |
| `table1_trajectories.md` | per-trajectory RMSE / r / bias / lag, median[IQR], split by arm | **Table I** |
| `table2_bias_iqr.md` | mean IQR of bias across patients (within-patient variability) | **Table II** |
| `table3_measures.md` | discrete-measure r_s (all trials) + r_av (per-arm averaged) | **Table III** |
| `trajectory_agreement.csv` | per-trial per-trajectory raw agreement (source for Tables I/II) | — |
| `table3_measures.csv` | Table III as data | — |

## Layout

Everything paper-related is in this one folder:

```
paper/
  README.md  measures_methods.md
  fig3_trajectories.png  fig4_mmc_vs_omc.png
  table1_trajectories.md  table2_bias_iqr.md  table3_measures.md  table3_measures.csv
  trajectory_agreement.csv
  scripts/   fig4_correlation.py  paper_table3.py  paper_trajectories.py
```

The three scripts here generate everything above. They import the repo's core modules
(`compare_pose_omc_delta`, `gnn_train`, `results_v3_delta`, `cup_task.score`) from `../scripts` and
`..` via a path shim, and read the pose caches under `../cache/` — so this folder is not standalone,
it rides on the cup-task repo. Outputs are written back into `paper/`.

## Reproduce

```bash
conda activate object_tracking
cd ~/Documents/cup-task
python paper/scripts/fig4_correlation.py     # Fig 4      (reads out/automq/score_vs_automq.csv)
python paper/scripts/paper_table3.py         # Table III  (same CSV)
python paper/scripts/paper_trajectories.py   # Fig 3 + Table I + Table II  (~4 min, aligns each trial)
```

Prerequisite: `out/automq/score_vs_automq.csv` (the discrete-measure scores) is produced by
`scripts/score_vs_automq.py` in the repo (the scorer itself is not duplicated here — it's shared
pipeline code). See `measures_methods.md` for the definitions behind it.

## What differs from Unger

- **11 panels, not 12** — AutoMQ has no drinking-phase shoulder-flexion column, and the two
  percent-timing variants Unger also omits are excluded. See `measures_methods.md`.
- **Geometry, not IK** — AutoMQ (and our scorer) compute angles geometrically from 3D points; Unger's
  OMC/MMC use full OpenSim/MuJoCo inverse-kinematics biomech models. Our trajectory correlations are
  therefore a notch below theirs (e.g. elbow extension r≈0.96 vs their 0.99) — same ordering, right
  ballpark, on consumer webcams at 60 Hz.
