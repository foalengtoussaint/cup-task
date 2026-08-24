#!/usr/bin/env bash
set -u
PY=~/miniconda3/envs/object_tracking/bin/python
cd /home/imove/Documents/cup-task
export OT_NCAMS_DIR=cup_ncams_26x OT_TRACKS_DIR=tracks_uetrack_26x
echo "=== 1/4 SmoothNet on the freebone BA ($(date +%H:%M:%S)) ==="
$PY -u scripts/cache_smoothed_pose.py --overwrite 2>&1 | grep -a "PROCESSING CHECK"
echo "=== 2/4 measures ($(date +%H:%M:%S)) ==="
$PY -u scripts/score_vs_automq.py 2>&1 | grep -a "PROCESSING CHECK\|tidy rows"
echo "=== 3/4 seg inputs ($(date +%H:%M:%S)) ==="
OT_SEG_INPUTS_DIR=seg_inputs_ship $PY -u scripts/cache_seg_inputs.py 2>&1 | grep -a "PROCESSING CHECK"
echo "=== 4/4 seg boundaries ($(date +%H:%M:%S)) ==="
OT_SEG_INPUTS_DIR=seg_inputs_ship $PY -u scripts/score_seg_boundaries.py 2>&1 | grep -a "PROCESSING CHECK\|wrote"
echo "=== SHIP CHAIN COMPLETE ($(date +%H:%M:%S)) ==="
