# Real-time proof

The real-time requirement is met. Detection was never the hard part (it already
cleared framerate); this packages it as a measured deliverable.

## Numbers

`scripts/realtime_demo.py` on a real rig clip (P07_drinking_right, 1920×1080 @
60 fps), merged frozen model, imgsz 1280, single GPU, 498 timed frames:

| metric | value | notes |
|---|---|---|
| source framerate (the bar) | 60.0 fps (16.7 ms) | must beat this |
| **model inference** | **94.6 fps** | p50 **8.7 ms**, p95 27.6 ms → real-time ✔ |
| end-to-end (+ decode/draw/encode) | 48.0 fps | demo I/O overhead, NOT the live system |

Model is **1.6× faster than source**. End-to-end includes reading/encoding MP4,
which does not exist in a live deployment.

## Deliverables

- `docs/realtime.md` — writeup: framerate table + live-vs-offline architecture
  diagram + supervisor bottom-line (the thing to show).
- `out/realtime_p07.json` — raw numbers; `out/realtime_p07.mp4` — overlay with a
  live FPS HUD.
- `scripts/realtime_demo.py` — reproduce on any clip; trivial swap to
  `VideoCapture(0)` for a live webcam.

## Not-yet-built real-time extensions

- Live sliding-window phase/score demo (offline pipeline running on a rolling
  window as frames arrive).
- On-device ONNX/TensorRT export, if "real-time" must mean "no workstation."
