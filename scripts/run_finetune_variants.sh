#!/bin/bash
# Wait for the running full-backbone finetune to finish, then run the FROZEN-backbone variant
# (plan's recommended: DELTA too small to finetune the whole backbone). Produces both models +
# a head-to-head so we keep whichever generalizes best. Runs unattended.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate object_tracking 2>/dev/null
cd /home/imove/Documents/cup-task
SCR=/tmp/claude-1000/-home-imove-Documents-object-tracking/25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad
CKPT=models/cdrnet_synbody_pretrained.pt

# 1) wait out the currently-running full-backbone finetune (finetune2.log)
echo "[$(date +%T)] waiting for full-backbone finetune to finish..."
while pgrep -f 'finetune_cdrnet_delta.*--epochs 30' | grep -qv "$$"; do
    # stop waiting once its log shows FINAL or CRASHED
    grep -qE "FINAL HELDOUT|CRASHED" "$SCR/finetune2.log" 2>/dev/null && break
    sleep 20
done
echo "[$(date +%T)] full-backbone done. full-backbone result:"
grep -E "FINAL HELDOUT|saved" "$SCR/finetune2.log" 2>/dev/null | tail -2

# 2) FROZEN-backbone finetune (fusion+decoder only) — the plan's recommended variant
echo "[$(date +%T)] STARTING frozen-backbone finetune"
stdbuf -oL -eL python -u scripts/finetune_cdrnet_delta.py --pretrained "$CKPT" --epochs 30 \
    --freeze_backbone --lr0 3e-4 --out models/cdrnet_delta_finetuned_frozen.pt \
    2>&1 | stdbuf -oL grep -vE "UserWarning|warnings.warn" | tee "$SCR/finetune_frozen.log"

# 3) head-to-head summary
echo "===== [$(date +%T)] FINETUNE VARIANTS SUMMARY ====="
echo "zero-shot pretrained (no DELTA training):  179.9 mm"
echo "full-backbone finetune:  $(grep 'FINAL HELDOUT' "$SCR/finetune2.log" 2>/dev/null | tail -1)"
echo "frozen-backbone finetune: $(grep 'FINAL HELDOUT' "$SCR/finetune_frozen.log" 2>/dev/null | tail -1)"
echo "from-scratch DELTA baseline (no pretrain): overfit 221->195->221 mm"
echo "===== DONE ====="
