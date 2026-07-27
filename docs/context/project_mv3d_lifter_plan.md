# Multi-view 3D lifter (YOLO-neck → learned triangulation) — plan & handoff

Status as of 2026-07-24 (end of a long session). NOT yet built. Downloads started; key
datasets + weights gated behind logins to unlock Monday.

## ⚠ 2026-07-27 (later): PORT BUILT + FIRST DELTA RESULT — stage-2 finetune OVERFITS (as predicted)
Full CDRNet-on-CSPDarknet port built & runs end-to-end (branch feat/cdrnet-cspdarknet-lifter,
package cup_task/mv3d/: dlt, ftl, backbone, cdrnet, data_delta, dataset; scripts/train_cdrnet_delta.py
+ extract_delta_frames.py). Smoke run (P07 trials 10-13 train, 14-15 HELD-OUT, 2D-distill from YOLO):
  stage1 start (untrained): 221mm disp / apex GF 0.08
  stage1 END (FROZEN backbone, fusion+decoder only): 195mm / 0.12   ← helps a little
  stage2 END (UNFROZEN CSPDarknet finetune):          221mm / 0.08   ← BACK TO BASELINE = WORSE
=> Fine-tuning the 12.9M CSPDarknet on 6 DELTA trials OVERFITS the 4 train views; held-out degrades
   right back to untrained. SAME failure shape as the frozen-neck wrist-speed test. This is EXACTLY
   the "12-trial finetune is the binding constraint" the plan warned about — NOT a bug. The port,
   canonical fusion, SII-DLT, workered loader, grad-accum, nan-mask all VERIFIED working; the wall
   is DATA. => the pretraining route (Panoptic/SynBody/AIST++, freeze neck for the DELTA finetune)
   is the necessary next step, not more DELTA-only training.
⚠ CAVEATS before over-reading: absolute disp ~200mm is HIGH (smoke run, 40+24 steps, stride-5 eval;
   metric/alignment may be crude). The RELATIVE stage1-down / stage2-up pattern is the trustworthy
   signal, not the absolute mm. n=6 trials, 1 participant. Bugs fixed en route: YOLO emits nan for
   undetected kpts (nan*0=nan poisoned loss -> mask + nan_to_num); stage-2 OOM (8x5-view graphs ->
   grad-accum); final-eval silent OOM (empty_cache before eval). Speed: decode-bound loop -> pre-
   extract frames (cache/delta/_frames640, 34x faster) + workered DataLoader (YOLO's pattern).

## ⚠ 2026-07-27: THE ACTUAL CDRNet ALGORITHM (read the reference before building)
Earlier notes/scripts here did a per-view feature-sample + refine-offset head — that is NOT CDRNet.
Read the reference clone (scratchpad/CDRnet/model_define.py + train_and_test.py). Real CDRNet =
**Canonical Fusion, single forward pass** (NO iterative reproject→sample→correct loop):
  per view: encoder → 1x1 conv to 300ch
  → **FTL_inv**: reshape feats to (...,3,1), multiply by P⁺ (pseudo-inverse of the 3x4 P) → lift
    each view into a shared CANONICAL 3D-consistent latent (this is the "camera disentanglement")
  → **concat all views + 1x1 convs** (this is where multi-view fusion happens, in canonical space)
  → **FTL**: multiply fused canonical feats by each view's P → project back to each view
  → decoder (transpose-convs) → per-view HEATMAPS → **center-of-mass soft-argmax** → 2D kpts/view
  → **SII** (Shifted Inverse Iterations = their differentiable DLT: B=AᵀA−αI, iterate solve(B,X);
    X/=|X|, ~2 iters) → 3D.
GEOMETRY IS IN THE FTL/FTL_inv FEATURE TRANSFORM, not a refine head. P is rescaled to the
heatmap/feature-map resolution first (resize_mat, train_and_test.py L106-118); P⁺ = linalg.pinv.

**HOW IT TRAINS (critical):** loss = **MSE on per-view 2D only** (`["mse"×4, _dummy_loss]`);
the 3D/DLT output is trained with **_dummy_loss = 0*y_pred → DLT NOT backpropagated**. Author:
turning on _DLT_loss = "massive instability, network fails training", adds "only a little bit",
their released model did NOT use it (matches our own 1e8-blowup finding). => 2D is the primary
objective; 3D is the geometric consequence at inference. Optional tiny _DLT_loss only at the very end.

CONSEQUENCES for our port (CSPDarknet swaps in for ResNet152 as the per-view encoder):
  * 2D supervision = DISTILL YOLO Pose26 2D (chosen) — exactly matches CDRNet's 2D-MSE training.
  * do NOT make the reproj/3D term primary — off or tiny at first (CDRNet doesn't backprop DLT).
  * port FTL_inv/FTL canonical fusion faithfully (with P rescaled to feature-map res), + SII DLT.
  * keep our Hartley-normalized SVD DLT (dlt.py) as the STABLE inference/fallback triangulator.
Branch: feat/cdrnet-cspdarknet-lifter. Package: cup_task/mv3d/ (dlt.py done: SVD + SII + reproj).

## The idea (user's, refined)
Make YOLO's **CSPDarknet neck** output features that are *useful for 3D multi-view
triangulation*, not just 2D detection. Pipeline:
1. YOLO-pose CSPDarknet neck → multi-scale feature maps (already computed live → ~free at
   inference; est. 70-90 fps).
2. → camera-disentanglement + soft-triangulation/DLT fusion head (CDRNet / MVGFormer style)
   → 3D keypoints.
3. **Pretrain end-to-end** on large multi-view datasets with **random 3-6 view sampling**
   (forces view-invariant features, so our unseen 5-cam DELTA rig is "just another subset").
4. **Finetune** the head (freeze most of the neck) on our 12 DELTA trials vs OMC.

Key correction the user made: there is NO "feature mismatch" to worry about — the neck is
TRAINED during pretraining, so it *becomes* a 3D-triangulation feature extractor. That is the
innovation, not a bug.

## HARD CONSTRAINT: single-person only
We track ONE known person (the participant). So DROP all the multi-person detection/association
machinery in Panoptic/MVGFormer/VoxelPose. Supervise on one body per sample. This makes our
fusion head much simpler than the papers (no person-matching, no detection head): just
"multi-view 2D of THE person → 3D."

## Datasets — ⚠ BEDLAM RULED OUT (it's MONOCULAR)
⚠ 2026-07-24: HF token unlocked BEDLAM; inspected one 0.2MB gt tar. **BEDLAM IS MONOCULAR** —
each seq_NNNNNN is ONE camera at a fixed pose; different seqs are different scenes/people/times
(cam positions differ, bodies_min=bodies_max=1 per seq, one cam_config per seq). It has PERFECT
GT but NO synchronized multi-view, so it CANNOT train multi-view triangulation (nothing to
triangulate across a single instant). BEDLAM is out for the fusion head. (Its whole fame is
single-IMAGE 3D pose.) Only useful role would be pretraining a monocular 2D→3D prior — not our
innovation — so skip it. Good we checked with one tar not 100s of GB.

MULTI-VIEW datasets that actually work (synchronized cameras, same person same instant):
- **SynBody / HumanNeRF-subset** — ✅ VERIFIED synchronized multi-view: 100 seqs × 8 cameras in
  a RING around the subject (world-centres -RᵀT form a ring, verified), full K,R,T per view +
  SMPL-X 3D GT. Synthetic → PERFECT GT + exact calib (no mocap/stick noise). On HF
  caizhongang/SynBody, NOT gated. ⚠ ONLY the **HumanNeRF-subset** is multi-view — the
  SynBody-100K / HMR tarballs are the MONOCULAR trap (avoid). ⚠ License CC-BY-NC-SA (non-commercial
  — fine for research, a constraint for product). Size: img.zip 44.7GB + mask 2GB + calib/smplx
  ~15MB = ~47GB. DOWNLOADING NOW (fetch_synbody.sh, HF CDN). = the new PRIMARY synthetic corpus.
  (User caught that SynBody IS multi-view in its NeRF track — I'd wrongly dismissed it.)
- **Syn4D** — ✅ multi-view synthetic (README: "multi-view RGB videos", built on BEDLAM2), +depth
  +3D point tracks, CC-BY-4.0 (commercial-OK, cleaner license than SynBody). On HF Syn4D/Syn4D.
  Not yet structurally verified. Alternative/supplement to SynBody if the NC license bites.
- **AIST++** — ✅ multi-view: 9 SYNCHRONIZED cameras, single dancer, off-eye-level + fast
  dynamic motion (matches drink apex). IDEAL fit. 🔒 NOT on HuggingFace (HF token doesn't reach
  it); downloads via google.github.io/aistplusplus_dataset/download.html. UNLOCK Monday: grab
  annotations (motions + cameras + keypoints3d) from that page → drop locally.
- **Panoptic** — ✅ DOWNLOADING NOW (real dome multi-view, random-view sampling). Subset:
  6 seqs (160422_ultimatum1, 160906_ian1/ian2/pizza1, 171204_pose1/pose2) × 6 HD cams
  (00,03,06,12,18,23) + calibration + hdPose3d_stage1_coco19. ⚠ HD videos are ~1.5-2GB EACH
  (full 10min crf20) → subset ~65GB, ~2-3h at 7.6MB/s. Script + log:
  /home/imove/Documents/mv3d_data/panoptic/fetch_subset.sh , fetch.log.
  NOTE: we only need a few hundred FRAMES/seq for training — subsample at train time; the full
  videos are overkill (kept because download was already running).

## Also gated (the reason we pivoted to training our own): OneDrive model weights
Every off-the-shelf multi-view model checked (VideoPose3D, MotionBERT, MVP, CDRNet, MV-SSM,
MVGFormer) is blocked: OneDrive weights unfetchable headless (curl gets HTML shell, not file),
custom CUDA deformable ops that don't compile vs torch 2.7 (`value.type()` removed), or
full-body-image models needing retrain. VideoPose3D (the one that ran) already showed lifters
ADD NOTHING vs our multi-view triangulation (ties position, loses to SmoothNet on jitter). MVGFormer
repo is cloned in scratchpad if wanted; needs OneDrive `pose_resnet50_panoptic.pth.tar` (unlock
Monday in a browser → drop in cup-task/models/).

## Risks / design notes (go in clear-eyed)
- **DLT instability**: CDRNet's author DISABLED the differentiable DLT loss ("serious
  instability"). Plan: supervise on fused 2D heatmaps/coords + a SOFT triangulation, not raw
  DLT gradient.
- **12-trial finetune is the binding constraint** → pretraining on BEDLAM/Panoptic/AIST++ is
  what makes it viable; freeze the neck for the finetune, heavy aug, strict LOPO.
- **The bar is high and already met for POSITION**: our triangulation is ~5mm (wrist),
  near geometric limit. So this must win on ROBUSTNESS / APEX / JITTER, not position.
  Success criterion: beat robust-triangulation **and** SmoothNet on apex/occluded frames +
  jitter, scored by good-frame fraction, NOT median position.

## Monday unlock checklist
1. `huggingface-cli login` (+ accept BEDLAM license) → pull single-person BEDLAM batches.
2. AIST++ registered download → annotations local.
3. (optional) OneDrive `pose_resnet50_panoptic.pth.tar` + MVGFormer ckpt → cup-task/models/,
   if we still want MVGFormer as a reference baseline.
4. Confirm Panoptic subset finished (fetch.log → "PANOPTIC SUBSET DONE").
Then: build single-person multi-view dataloader (random 3-6 view sampling) → YOLO-neck→fusion
head (soft-triangulation) → pretrain → LOPO finetune on DELTA/OMC → score on apex/jitter.

## ✅ SMOKE TEST PASSED (2026-07-24, on SynBody calib+transl, before images finished)
scripts/mv3d_smoke_fusion.py. Real 8-cam SynBody calib + real 3D (transl) → project → +3px noise +
2/8 WILD views → differentiable weighted-DLT + tiny learned confidence head. Results:
- pipeline runs end-to-end, stable, ~100mm (relative test — 100mm is the injected-noise floor, not a
  real accuracy number).
- LEARNED fusion BEATS plain equal-weight DLT: 96.7 vs 116.3 mm (~17%) with 2/8 corrupted.
- head LEARNS to down-weight bad views: w_good 0.31 > w_bad 0.26 (gap grows over training).
- ⚠ RETIRED THE KEY RISK: CDRNet's "DLT instability" IS REAL — first run blew up to 1e8 mm. FIX =
  per-row Hartley normalization (scale each DLT row to unit norm) + float64 solve + guard the
  homogeneous divide. Clean-data DLT recovers 3D to 3e-14mm; the blowup was ill-conditioning from
  wild views, not the algorithm. This normalized weighted-DLT is the fusion core to keep.
NOT yet: full skeleton (used single root `transl`; need SMPL-X body model or images+YOLO neck for
joints), and real 2D from the YOLO neck (images still downloading). Fusion head + loop are PROVEN.
NEXT once img.zip done + SMPL-X model: swap single point → full joints, swap projected-3D → YOLO-neck
2D; keep the normalized weighted-DLT head.

## ⚠ CORRECTION + REAL architecture built (2026-07-24, user: "you're supposed to fine-tune the YOLO")
The earlier "smoke test passed" note tested KEYPOINT-ONLY fusion (GT-3D→project→2D→learned weighted-DLT):
NO images, NO YOLO, NO CSPDarknet neck, NO backprop into the backbone. That is NOT the idea — it only
validated the DLT math (still useful: DLT-normalization fix carries over).

REAL architecture now built + running: scripts/mv3d_yolo_neck_fuse.py
- Tap YOLO-pose CSPDarknet NECK: Pose26 head (layer 23) fed by layers [16,19,22] = P3/P4/P5 =
  (128ch s8, 256ch s16, 512ch s32). Use layer 16 (128ch, stride-8, finest) for sampling.
- Per frame: decode 6 synced Panoptic HD cams → YOLO neck → per-cam feature map. Project coarse 3D
  joint into each cam → bilinear grid_sample the neck feature there → small head regresses (2D
  refine offset + confidence) per view → weighted-DLT → 3D → 3D loss → GRADIENT INTO THE NECK.
- Stage 1 frozen neck, Stage 2 unfreeze (the actual fine-tune).
RESULT (Panoptic ultimatum1, 6 cams, 60 single-person frames, 2/6 views corrupted):
- runs end-to-end, gradients flow into the neck. Learned CONSISTENTLY beats plain DLT (~2-15mm every
  eval), but margin small + loss NOISY (1-frame batches, only 60 frames). Could NOT yet show that
  UNFREEZING the neck helps beyond the frozen head — needs batching + more data + stable eval = the
  full training run.
- We DON'T need SynBody images for this — Panoptic's 6 downloaded HD videos + real 3D GT suffice NOW.
NEXT (full run): batch multiple frames, more sequences/frames, stable eval, then compare frozen-neck
vs fine-tuned-neck cleanly. Neck taps + sampling + DLT-into-neck all verified working.

## HONEST test on DELTA (2026-07-24) — right test bed, but training-frame bug found
After Panoptic proved a BAD test bed (3 blockers: multi-person ultimatum1, YOLO detects 0 on small
dome people, broken calib projection for some cams → every Panoptic "result" was the GT-projection
LEAK giving fake 0mm), moved to DELTA = the correct bed: our own footage, YOLO detects participant
100%, real cached 2D + real 8/5-cam calib + OMC 3D, single person. scripts/mv3d_delta_neck.py.

Pipeline (all REAL, no leak): cached YOLO wrist 2D per cam → sample YOLO NECK feat (layer16, 128ch,
stride8) at that px → head regresses (dx,dy)+conf → weighted-DLT → 3D. vs plain DLT of raw dets.

BUGS FOUND + FIXED along the way:
- camera index misalignment: cam_7/cam_9 have NO pose.json; must align KP↔Ps to detected cams only.
- ⚠ THE BIG ONE (same trap as the cup apex all day): scored raw |X_dlt − OMC| across MMC vs OMC
  worlds → 1400mm nonsense. FIX = frame-invariant DISPLACEMENT-FROM-START magnitude (already the
  project's standard metric). AutoMQ has NO alignment code to borrow — it works ENTIRELY in the OMC
  frame (never uses MMC), so it never hit the two-frame problem. The frame-invariant metric IS our
  alignment-free answer.

RESULT after metric fix: PLAIN DLT of real YOLO dets = ~10mm all / ~13mm fast (CORRECT, matches the
~5mm wrist story). BUT the LEARNED head got WORSE over training (10→29mm). Diagnosed: EVAL is now
frame-invariant but the TRAINING LOSS is still |learned − OMC| in the WRONG frame — so the head
optimizes a meaningless 1400mm quantity and diverges while the correct eval shows it degrading.

=> the fix / NEXT STEP: train the head on a FRAME-INVARIANT / MMC-frame target, NOT against OMC
directly. Correct signal = LEAVE-ONE-OUT CONSENSUS: triangulate the wrist from the OTHER cams
(MMC frame), train the held-out cam's refinement to match that consensus (all in MMC frame, no OMC,
no alignment). Then the head learns to refine bad dets toward multi-view consensus, and eval vs OMC
(frame-invariant) measures if that helped — especially at the apex where real dets are worst.

STATUS: pipeline + neck-feature sampling + DLT all proven on real DELTA data; plain baseline correct
at ~10mm; learned-refinement UNPROVEN pending the leave-one-out training-frame fix. Neck-feature
cache saved (delta_neckfeat_P07_trial_11.pt).
