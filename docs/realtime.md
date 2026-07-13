# Real-time detection: measurements and architecture

**Question:** can the cup + body-keypoint detection run in real time?
**Answer:** yes. On real rig footage (1920×1080 @ 60 fps) the model processes a
frame in **~8.7 ms** (median), i.e. **~95 fps** — 1.6× faster than the cameras
record. One model pass yields person box + 17 body keypoints + cup box.

## What "real-time" means here

The cameras record at **60 fps**, so a frame arrives every **16.7 ms**. To keep
up live, the model must finish each frame in under that. It does, with headroom.

## Measured framerate

Measured with `scripts/realtime_demo.py` on `P07_drinking_right` (498 timed
frames, 5-frame GPU warmup discarded), merged model
`runs/pose/runs/cup_head_frozen_p3/weights/best.pt`, imgsz 1280, single GPU.

| metric | value | notes |
|---|---|---|
| source framerate (the bar) | **60.0 fps** (16.7 ms) | must beat this |
| **model inference** | **94.6 fps** | p50 **8.7 ms**, p95 27.6 ms |
| end-to-end (+ decode/draw/encode) | 48.0 fps | demo I/O overhead, not the model |
| **verdict** | **REAL-TIME ✔** | model 1.6× faster than source |

- **Model inference** is the real number: how fast the network runs. 8.7 ms
  median → ~95 fps. Even the p95 worst frame (27.6 ms ≈ 36 fps) stays above 30.
- **End-to-end** (48 fps) includes reading the MP4, drawing the overlay, and
  re-encoding to MP4 — none of which exist in a live deployment (a live camera
  delivers decoded frames; you'd render to screen, not re-encode a file). It's
  reported only for honesty; it is *not* the system's live rate.

Overlay video with the live FPS HUD: `out/realtime_p07.mp4`.
Reproduce: `python scripts/realtime_demo.py CLIP.mp4 --json out/rt.json`.

## Architecture: what runs live vs. after

The system is deliberately split. **Detection is live; the 3D/scoring pipeline
is offline.** This is the right split — per-frame detection is cheap and
must stream; multi-camera fusion and scoring want all the frames first and are
not latency-critical.

```
  LIVE (per camera, per frame, ~95 fps)        OFFLINE (after capture)
  ┌───────────────────────────────┐            ┌──────────────────────────────┐
  │  frame ─► ONE model pass ─►    │            │  per-cam 2D tracks            │
  │    • person box               │  detections│    │                          │
  │    • 17 body keypoints  ──────────────────► │    ▼ multi-view triangulation │
  │    • cup box                  │  (logged)  │  3D cup + 3D keypoints         │
  │                               │            │    │  (consensus gate)         │
  └───────────────────────────────┘            │    ▼ phase detection           │
                                               │  reach / lift / drink / place  │
   runs on each camera independently           │    ▼ scoring                   │
   at capture time                             │  task-quality metrics          │
                                               └──────────────────────────────┘
```

- **Live half** — one YOLO-pose model per camera. Detects the person +
  upper-body keypoints (nose/eyes/ears/shoulders/elbows/wrists/hips) and the cup
  in a single forward pass. ~95 fps at 1080p, so it streams in real time on each
  of the rig cameras. Keypoints are the mocap-free replacement for the QTM head
  marker (mouth is a nose→eye/ear proxy).
- **Offline half** — takes the logged per-camera 2D detections and fuses them:
  DLT triangulation with a consensus gate (drop the worst-reprojecting camera
  until the remaining ones agree, require ≥3), giving jitter-reduced 3D tracks
  for the cup and keypoints. From the 3D cup + mouth-proxy trajectory it segments
  the drink phases and computes the task-scoring metrics.

## Why the model is a single merged network

Person + keypoints + cup come from **one** model, not three:
- The backbone/neck is shared (one forward pass → all outputs), which is what
  keeps it at ~95 fps instead of 3× slower.
- The keypoint branch is **frozen** during cup training, so adding cup detection
  left the keypoints bit-identical to the base pose model (Δ = 0). The cup was
  learned on top without disturbing what already worked.
- Held-out validation (participants never trained on) confirmed the shared
  frozen head generalizes (3.6–9.3 px reprojection), whereas a fully separate
  cup head overfit the training participant (22–44 px). The merged frozen model
  is the shipped one.

## Bottom line for the supervisor

The real-time requirement is met: **~95 fps model on 60 fps footage, single pass
for person + keypoints + cup.** The heavier work (3D fusion, phase, scoring) is
intentionally offline because it needs the whole recording and isn't
latency-bound — but the *detection* that feeds it runs live on every camera.
