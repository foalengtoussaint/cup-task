#!/usr/bin/env bash
set -u
PY=~/miniconda3/envs/object_tracking/bin/python
cd /home/imove/Documents/cup-task
echo "=== 1/3 solve freebone_0.05 on the CURRENT caches ($(date +%H:%M:%S)) ==="
$PY -u scripts/ba_free_bone.py --limit 900 --lams 0.05 2>&1 | grep -a "trials,\|PROCESSING CHECK\|wrote\|fail"
echo "=== 2/3 SmoothNet ($(date +%H:%M:%S)) ==="
$PY -u scripts/apply_smoothnet_variants.py --tags freebone_0.05 2>&1 | grep -a "PROCESSING CHECK\|wrote"
echo "=== 3/3 score both variants ($(date +%H:%M:%S)) ==="
$PY -u scripts/score_variant_measures.py --tags fix__sn freebone_0.05__sn 2>&1 | grep -a "^=== \|^tag \|^fix__\|^freebone\|PROCESSING CHECK:"
echo "=== FREEBONE CHAIN COMPLETE ($(date +%H:%M:%S)) ==="
