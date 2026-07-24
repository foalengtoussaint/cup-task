# cup-task context index

One-line index of project findings. Open the individual files for detail.

- [Repo + architecture](project_repo.md) — one merged YOLO-pose model (person+17 kpts+cup, one pass); keypoint branch FROZEN → cup training kept keypoints Δ=0; held-out val picked the shared frozen head over a separate head that overfit; detection LIVE, 3D/phase/scoring OFFLINE.
- [Real-time proof](realtime_proof.md) — merged model 94.6 fps / 8.7 ms p50 on 1080p@60 rig footage = 1.6× faster than source = real-time. Deliverable: `docs/realtime.md` + `scripts/realtime_demo.py` + `out/realtime_p07.{json,mp4}`.
- [Multi-view 3D lifter plan](project_mv3d_lifter_plan.md) — YOLO CSPDarknet neck → learned triangulation head; pretrain Panoptic(downloading)+AIST++(gated,ideal); ⚠BEDLAM RULED OUT=monocular w/ random 3-6 view sampling, finetune 12 DELTA trials; SINGLE-PERSON only; must beat triangulation+SmoothNet on apex/jitter not position; Monday unlock: HF login + AIST++ + OneDrive weights
