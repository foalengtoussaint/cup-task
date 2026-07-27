# CDRNet-on-CSPDarknet multi-view 3D lifter — results

Your idea: strip YOLO's detection head, use the **CSPDarknet neck feature maps** as CDRNet's
encoder (replacing ResNet152), and train the canonical-fusion + differentiable-DLT lifter
end-to-end so the backbone becomes a **view-invariant 3D-triangulation** feature extractor.

Branch: `feat/cdrnet-cspdarknet-lifter`. Package: `cup_task/mv3d/`.

## Verdict: THE IDEA WORKS (pretraining), DELTA is data-starved (finetuning)

### 1. SynBody pretraining — clean success ✅
Faithful CDRNet (canonical fusion FTL/FTL_inv → soft-argmax → SII DLT) on a CSPDarknet encoder,
trained with YOLO's exact recipe (AMP, EMA, cosine, warmup, accumulate-to-nbs) + random 3–6 view
sampling, on 100 synthetic seqs × 8 synced cams (perfect in-frame GT → real MPJPE).

**Held-out MPJPE: 649 → 350 → 274 → 187 → 167 → 153 → 154 mm** (10 epochs, monotonic, no overfit).

→ The CSPDarknet-encoder canonical-fusion model genuinely learns to lift multi-view 3D.
**Deliverable model: `models/cdrnet_synbody_pretrained.pt`** (EMA weights).

### 2. Transfer to the real DELTA rig — pretraining transfers ✅
The pretrained backbone, **zero-shot on the DELTA rig it never saw = 179.9 mm** held-out wrist
displacement — already better than a from-scratch DELTA model (which overfits to 221 mm).
The random-view pretraining made our 5-cam rig "just another subset", as planned.

### 3. DELTA finetuning — degrades (data wall, as the plan predicted) ⚠
| model | held-out (mm) |
|---|---|
| from-scratch DELTA (no pretrain) | 221 (overfit) |
| **SynBody-pretrained, ZERO-SHOT on DELTA** | **179.9  ← best** |
| pretrained → full-backbone finetune | 219.6 (regressed epoch 0) |
| pretrained → frozen-backbone finetune | ~200+ (regressed) |

With only **4 training trials**, any DELTA finetuning pulls the model away from the pretrained
optimum. **Keep the zero-shot pretrained model.** To improve on DELTA, the lever is MORE DELTA
data (more participants/trials) or heavier pretraining — not more finetuning on 4 trials.

## Where to go next
- More SynBody epochs / more synthetic corpora (Syn4D, AIST++) to push MPJPE below 154 mm.
- Score on the drink APEX / occluded frames specifically (that's where this should beat plain
  triangulation + SmoothNet — position is already ~5mm on clean frames).
- If finetuning DELTA: get more trials first; freeze the backbone; tiny LR + early stop on the
  zero-shot number.
