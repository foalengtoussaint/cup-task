#!/usr/bin/env bash
# Rebuild the pose caches with the RELAXED reprojection gate (cup_task/triangulate.py, 2026-08) so the
# live scorer matches the paper. Gated cache backed up to cache/_gated_backup_20260810/ first.
# Stages: (1) gnn_build_dataset (re-triangulate mmc + reproj sidecars, relaxed gate)
#         (2) ba_cache_traj_all (BA trajectories from the new mmc)
#         (3) cache_smoothed_pose (SmoothNet on new mmc + BA)
#         (4) score_vs_automq (re-score with planar default) -> out/automq/score_vs_automq.csv
set -e
cd /home/imove/Documents/cup-task
source ~/miniconda3/etc/profile.d/conda.sh && conda activate object_tracking
PARTS="P07 P08 P10 P12 P13 P14 P15 P17 P19 P251 P252"

echo "=== [1/4] gnn_build_dataset (relaxed-gate re-triangulation) $(date) ==="
python -u scripts/gnn_build_dataset.py --force --audit-clean --parts $PARTS

echo "=== [2/4] ba_cache_traj_all (BA on new mmc) $(date) ==="
python -u scripts/ba_cache_traj_all.py

echo "=== [3/4] cache_smoothed_pose (SmoothNet, overwrite) $(date) ==="
python -u scripts/cache_smoothed_pose.py --overwrite

echo "=== [4/4] score_vs_automq (planar default) $(date) ==="
python -u scripts/score_vs_automq.py --out out/automq/score_vs_automq.csv --parts $PARTS

echo "=== REBUILD COMPLETE $(date) ==="
