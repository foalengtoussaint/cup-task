# Multi-view 3D lifter (YOLO-neck → learned triangulation) — plan & handoff

Status as of 2026-07-24 (end of a long session). NOT yet built. Downloads started; key
datasets + weights gated behind logins to unlock Monday.

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

## Datasets (BEDLAM majority, single-person)
- **BEDLAM** — synthetic, PERFECT 3D GT (SMPL-X) + exact calib, arbitrary cameras, huge
  diversity. Best supervision (no mocap/stick-lever noise). 🔒 GATED on HuggingFace
  (Intelligent-Systems/BEDLAM, gated:auto → 401 GatedRepo). UNLOCK: accept license on HF +
  `huggingface-cli login` with token in this env. Then pull single-person batches.
- **AIST++** — single-dancer, real, off-eye-level cameras + fast dynamic motion (matches drink
  apex). 🔒 downloads via google.github.io/aistplusplus_dataset/download.html (GCS direct
  paths 403). UNLOCK: registered download link → drop annotations locally.
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
