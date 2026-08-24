# Repo + architecture

Standalone repo for real-time / near-real-time cup + body-keypoint detection →
phase detection → scoring-quality data for the drink task. Grew out of the
object-tracking drink_study (reuses its no-GT `agreement.py` metric, the
`cup_clean3d_refill.pt` cup model, same rig/participants).

## Architecture split (deliberate)

Detection is **live** per-camera; 3D fusion + phase + scoring is **offline**
(needs all frames, not latency-bound). See [realtime_proof.md](realtime_proof.md).

```
  LIVE (per camera, per frame, ~95 fps)      OFFLINE (after capture)
  frame ─► ONE model pass ─►                 per-cam 2D tracks
    • person box                               │ multi-view triangulation
    • 17 body keypoints    ──────────────────► ▼ + consensus gate
    • cup box                                 3D cup + keypoints
                                               │ phase detection
                                               ▼ reach/lift/drink/place
                                              task scoring
```

## Model = ONE merged YOLO-pose network

Single forward pass → person box + 17 upper-body keypoints
(nose/eyes/ears/shoulders/elbows/wrists/hips — NO knees/ankles) + cup box.
Keypoints are the mocap-free replacement for the QTM head marker
(mouth = nose→eye/ear proxy). `cup_task/` holds the lib: `pose_keypoints.py`,
`cup_detect.py`, `triangulate.py`, `kalman_3d.py`, `dual_head.py`.

## Key finding — frozen head beats separate head

Trained cup detection with the backbone/neck **and** the keypoint branch (cv4)
FROZEN → keypoints stayed bit-identical to the base pose model (Δ=0). Held-out
validation (participants never trained on) picked the **shared frozen head**
(3.6–9.3 px reprojection, `runs/pose/runs/cup_head_frozen_p3/weights/best.pt`)
over a fully SEPARATE cup head that overfit the single training participant
(22–44 px). The same-participant val numbers had ranked the separate head best
(0.923 precision) — the held-out test reversed it. **Ship the frozen head.**

## Layout / data policy

`cup_task/` lib · `scripts/` (train_*, realtime_demo, visualize_*, animate_3d) ·
`docs/` (realtime.md + this context/). `.gitignore` excludes `*.pt` (weights),
`data/` (datasets), `runs/` (training), `out/` (videos) — all kept **on disk**,
not in git (never auto-delete experiment data; new metrics often need the
originals).
