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
