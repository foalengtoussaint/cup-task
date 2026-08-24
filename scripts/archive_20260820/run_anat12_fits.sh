#!/usr/bin/env bash
set -u
PY=~/miniconda3/envs/object_tracking/bin/python
cd /home/imove/Documents/cup-task/paper/scripts/anat12
for V in fix__sn freebone_0.05__sn; do
  echo; echo "=================== anat12 FIT :: $V  ($(date +%H:%M:%S)) ==================="
  OT_OMC_PREP_DIR="omc_prep_${V}" CFG="anat12:0" TAG="_${V}" \
    PYTHONPATH=/home/imove/Documents/cup-task:/home/imove/Documents/cup-task/scripts \
    $PY -u anat_frame.py 2>&1 | grep -va "^  \[" 
done
echo; echo "=================== ANAT12 FITS DONE ($(date +%H:%M:%S)) ==================="
