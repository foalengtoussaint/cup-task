# CDRNet-on-CSPDarknet multi-view 3D lifter — results

> ⚠⚠⚠ **2026-07-27 LATEST — the "domain gap" conclusion below was CONTAMINATED; must retrain.**
> The non-square P-rescale fix (native_size, per-axis sx/sy) changed the SynBody grid scale from the
> trained 1/8 (0.125) to (640/1024)/8 = 0.078. So every test AFTER that fix ran the OLD 141mm weights
> under a NEW geometry they never trained on → garbage 2D (SynBody err 271px) and the "DELTA decoder
> still frozen" result. Weight↔forward MISMATCH, not a domain gap. Also: the "(79,78) corner-peak"
> was a red herring — the model uses center-of-mass (soft-argmax), which one hot corner pixel barely
> moves. **Do NOT trust the domain-gap / transfer conclusions until a RETRAIN with the fixed forward.**
> Open honestly: we don't yet know if it transfers. Retraining next.


> ⚠ **2026-07-27 UPDATE — a coordinate-space BUG invalidated the DELTA numbers below; re-running.**
> The forward triangulated GRID-space 2D (heatmap soft-argmax × 8 = 640-space) against **P_native**
> (1920×1080 for DELTA). Grid-2D + native-P is geometrically inconsistent — proven: known 3D →
> project → triangulate = **2734mm error** (vs **0.000mm** with grid-2D + P_grid). This is why the
> DELTA 3D wrist was ~2250mm off and near-static. **SynBody's square 1024 barely triggered it (1mm),
> so the 154mm MPJPE was ~legit; the bug bit only at DELTA transfer.** FIXED (triangulate in grid
> space); ALL weights below were trained against the bug → being re-pretrained.
>
> ⚠⚠ **2026-07-27 FINAL root cause (after the fixed rerun): it's a synthetic→real DOMAIN GAP, not
> the geometry bug.** Re-pretrained with fixed geometry (SynBody MPJPE 649→141, unchanged — confirming
> the bug barely touched square-1024 SynBody). But fixed-geometry DELTA transfer is WORSE/near-frozen:
> pred wrist motion range **14mm** (OMC 276mm), velocity-corr 0.08, slope 0.02. DIRECT CHECK: the
> model's 2D wrist decoder on DELTA barely moves (cam_1 X-std **6px**, Y-std 2px) while YOLO's real 2D
> wrist moves 77–103px std. => **the SynBody-trained heatmap decoder does not fire on real DELTA
> photographs** (it emits a near-constant prior location). The earlier "moves with the wrist" was the
> triangulation BUG scrambling that constant 2D into apparent 3D motion — NOT transfer. Fixing the bug
> removed the illusion and exposed the real wall. So: SynBody pretraining works (decoder fires on
> SYNTHETIC images, 33–104px); the SYNTHETIC→REAL gap is what breaks DELTA — not data size, not geometry.
> NEXT: either (a) bypass the learned decoder — feed YOLO's own Pose26 2D into the canonical-fusion/DLT
> (tests if the FUSION adds anything over plain DLT), or (b) train the decoder on real images (needs
> real 2D labels / more DELTA). Corrected twice — trust this block over the older ones below.

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
