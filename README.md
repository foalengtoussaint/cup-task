# cup-task

Offline drink-task scoring from multi-camera video. Standalone productization of
the `object-tracking-master/experiments/drink_study` research pipeline: detect the
cup and the person's body/hand keypoints, fuse to 3D with the existing camera
calibration, segment the drink phases, and emit the iMOVE task-scoring metrics.

Inference (cup + pose) is real-time-capable; the 3D fusion → phase → scoring
pipeline runs offline (per trial) for now.

## Pipeline

| Stage | Module | Status |
|-------|--------|--------|
| Cup detection | `cup_task.cup_detect` | TODO (port existing seg model) |
| Body/hand keypoints | `cup_task.pose_keypoints` | ✅ YOLO-pose, upper body |
| 3D triangulation | `cup_task.triangulate` | TODO (reuse drink_study calib) |
| Phase segmentation | `cup_task.segment` | TODO (port segment_cup_only) |
| iMOVE metrics | `cup_task.score` | TODO (spec on iMOVE Docker) |

## Keypoints

YOLO-pose (COCO-17), full upper body — head (mouth proxy), shoulders, elbows,
wrists, hips. Knees/ankles dropped (no signal for a seated drinking task). The
mouth-proxy point (nose → eye/ear fallback) replaces the QTM head marker the old
pipeline depended on, making scoring mocap-free.

## Quick start

```bash
pip install ultralytics onnxruntime opencv-python   # torch already present
cp ../object-tracking-master/yolo11n-pose.pt .

# keypoints for one clip -> JSON
python -m cup_task.pose_keypoints CLIP.mp4 -o out.pose.json

# overlay video to eyeball it
python scripts/visualize_pose.py CLIP.mp4 -o overlay.mp4
```

## Notes

- Cup model weights (`yolo11n-pose.pt` and the cup seg model) are copied in, not
  versioned (see `.gitignore`).
- Camera calibration lives in the source repo at `data/calib/PXX/calibration.toml`.
- Metric definitions come from iMOVE and are not yet available on this machine;
  `cup_task.score` is a placeholder until that spec lands.
