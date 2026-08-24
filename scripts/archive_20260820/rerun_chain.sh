#!/usr/bin/env bash
# Full downstream rebuild after the trial<->C3D repair moved into H._load_omc / R._omc_cup.
# Every stage logs to its own file; this driver echoes a banner + timestamp per stage.
set -u
PY=~/miniconda3/envs/object_tracking/bin/python
cd /home/imove/Documents/cup-task
export OT_NCAMS_DIR=cup_ncams_26x OT_TRACKS_DIR=tracks_uetrack_26x
banner () { echo; echo "=================== $1  ($(date +%H:%M:%S)) ==================="; }

banner "1/6 gnn_pairs (P10 P251 P252: cup sync signal was from the wrong rep)"
$PY -u scripts/gnn_build_dataset.py --parts P10 P251 P252 --force 2>&1 | tail -5

banner "2/6 BA re-solve (cohort)"
$PY -u scripts/ba_cache_traj_all.py 2>&1 | grep -a "PROCESSING CHECK\|GUARD\|wrote\|DONE"

banner "3/6 SmoothNet cache"
$PY -u scripts/cache_smoothed_pose.py --overwrite 2>&1 | grep -a "PROCESSING CHECK\|DONE"

banner "4/6 measures rescore"
$PY -u scripts/score_vs_automq.py 2>&1 | grep -a "PROCESSING CHECK\|tidy rows\|DONE"

banner "5/6 segmenter inputs (OMC channels now from the right rep)"
OT_SEG_INPUTS_DIR=seg_inputs_final $PY -u scripts/cache_seg_inputs.py 2>&1 | grep -a "PROCESSING CHECK\|DONE"

banner "6/6 segmentation boundaries"
OT_SEG_INPUTS_DIR=seg_inputs_final $PY -u scripts/score_seg_boundaries.py 2>&1 | grep -a "PROCESSING CHECK\|wrote\|DONE"

banner "CHAIN COMPLETE"
