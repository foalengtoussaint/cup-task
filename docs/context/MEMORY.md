# cup-task context index

One-line index of project findings. Open the individual files for detail.

- [Repo + architecture](project_repo.md) — one merged YOLO-pose model (person+17 kpts+cup, one pass); keypoint branch FROZEN → cup training kept keypoints Δ=0; held-out val picked the shared frozen head over a separate head that overfit; detection LIVE, 3D/phase/scoring OFFLINE.
- [Real-time proof](realtime_proof.md) — merged model 94.6 fps / 8.7 ms p50 on 1080p@60 rig footage = 1.6× faster than source = real-time. Deliverable: `docs/realtime.md` + `scripts/realtime_demo.py` + `out/realtime_p07.{json,mp4}`.
