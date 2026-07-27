#!/bin/bash
# Autonomous driver: SynBody pretrain -> DELTA finetune -> summary. Runs unattended.
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate object_tracking 2>/dev/null
cd /home/imove/Documents/cup-task
SCR=/tmp/claude-1000/-home-imove-Documents-object-tracking/25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad

echo "===== [$(date +%T)] STAGE A: SynBody pretrain ====="
stdbuf -oL -eL python -u scripts/pretrain_synbody.py --epochs 10 --batch 6 --workers 6 --frame_stride 10 --lr0 1e-3 \
    2>&1 | stdbuf -oL grep -vE "UserWarning|warnings.warn" | tee "$SCR/pretrain_full.log"

CKPT=/home/imove/Documents/cup-task/models/cdrnet_synbody_pretrained.pt
if [ ! -f "$CKPT" ]; then echo "PRETRAIN PRODUCED NO CKPT — aborting finetune"; exit 1; fi
echo "===== [$(date +%T)] STAGE A done, ckpt saved ====="

echo "===== [$(date +%T)] STAGE B: DELTA finetune from pretrained ====="
stdbuf -oL -eL python -u scripts/finetune_cdrnet_delta.py --pretrained "$CKPT" --epochs 30 \
    2>&1 | stdbuf -oL grep -vE "UserWarning|warnings.warn" | tee "$SCR/finetune_full.log"

echo "===== [$(date +%T)] ALL DONE ====="
