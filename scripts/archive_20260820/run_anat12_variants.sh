#!/usr/bin/env bash
# Build the anat12 input cache and run the landmark fit for BOTH BA variants.
# The fit reads cache/pose_smoothed's ba_sn, so each variant is materialised into its own
# pose cache first (score_variant_measures.materialise does exactly this) and _SMCACHE pointed at it.
set -u
PY=~/miniconda3/envs/object_tracking/bin/python
cd /home/imove/Documents/cup-task
for V in fix__sn freebone_0.05__sn; do
  echo; echo "=================== anat12 :: $V  ($(date +%H:%M:%S)) ==================="
  OT_OMC_PREP_DIR="omc_prep_${V}" $PY -u - <<PYEOF
import sys, os
sys.path.insert(0,'paper/scripts/anat12'); sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import score_variant_measures as SVM, score_vs_automq as S
n,tot = SVM.materialise("$V")
print(f"materialised {n}/{tot} trials for $V", flush=True)
S._SMCACHE = SVM.TMP
import prep_cache
print("omc_prep dir:", prep_cache.CACHE, flush=True)
prep_cache.build()
PYEOF
done
echo; echo "=================== ANAT12 CACHES BUILT ($(date +%H:%M:%S)) ==================="
