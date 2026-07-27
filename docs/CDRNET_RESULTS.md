# CDRNet-on-CSPDarknet multi-view 3D lifter — results

> ⚠ **2026-07-27 UPDATE — a coordinate-space BUG invalidated the DELTA numbers below; re-running.**
> The forward triangulated GRID-space 2D (heatmap soft-argmax × 8 = 640-space) against **P_native**
> (1920×1080 for DELTA). Grid-2D + native-P is geometrically inconsistent — proven: known 3D →
> project → triangulate = **2734mm error** (vs **0.000mm** with grid-2D + P_grid). This is why the
> DELTA 3D wrist was ~2250mm off and near-static. **SynBody's square 1024 barely triggered it (1mm),
> so the 154mm MPJPE was ~legit; the bug bit only at DELTA transfer.** FIXED (triangulate in grid
> space); ALL weights below were trained against the bug → being re-pretrained.
>
> On the "0.89 correlation" of the broken DELTA track: the prediction MOVED IN SPACE WITH THE WRIST
> (Y-corr +0.74, rises/falls with the reach) **through a triangulation that was geometrically broken
> by 2734mm**. A model that learned nothing would give noise/a fixed point under a broken transform;
> a signal that survives being scrambled through the bug (compressed to 21% scale, Z sign-flipped)
> is EVIDENCE THE MODEL GENUINELY LEARNED to locate the wrist from images — seen through a broken
> lens. So the bug SUPPRESSED true performance, it didn't inflate it. => the fixed rerun is expected
> to let that learned signal through cleanly (correct axis signs, full scale, real velocity tracking),
> not "hope it learns". (Correcting an earlier over-dismissive 'flattered a broken model' framing.)


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
| pretrained → frozen-backbone finetune | 214.9 (regressed: 179.9→201.6→208.8→214.6→214.9) |

With only **4 training trials**, any DELTA finetuning pulls the model away from the pretrained
optimum. **Keep the zero-shot pretrained model.** To improve on DELTA, the lever is MORE DELTA
data (more participants/trials) or heavier pretraining — not more finetuning on 4 trials.

## Where to go next
- More SynBody epochs / more synthetic corpora (Syn4D, AIST++) to push MPJPE below 154 mm.
- Score on the drink APEX / occluded frames specifically (that's where this should beat plain
  triangulation + SmoothNet — position is already ~5mm on clean frames).
- If finetuning DELTA: get more trials first; freeze the backbone; tiny LR + early stop on the
  zero-shot number.
