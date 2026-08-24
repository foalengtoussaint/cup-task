# Mocap-free drink tracking: real-time, and validated against optical mocap

The cup-task pipeline replaces the Qualisys/QTM optical-mocap rig with two camera detectors
(a finetuned cup detector + `yolo26`-pose). This document establishes two things about that
replacement:

1. **It runs in real time** on commodity hardware.
2. **It reproduces the mocap's drink-phase signals** — to millimetres — across a cohort.

All numbers were measured on an **RTX 3060 Ti (8 GB)**, 1920×1080 source, up to 10 cameras.
None is quoted from a spec sheet.

---

## 1. Real time

**cup detector + yolo-pose, imgsz 640, batched across cameras, both nets on separate CUDA
streams:**

| cameras | 1 | 2 | 4 | 6 | 8 | **10** |
|---|---|---|---|---|---|---|
| **fps (yolo26n-pose)** | 173 | 119 | 65 | 42 | 32 | **26** |
| **fps (yolo26s-pose)** | — | — | — | — | — | **19** |
| ms/frame | 5.8 | 8.4 | 15.5 | 23.6 | 31.7 | 37.8 |

**26 fps at the full 10-camera rig is real time for this task, and 60 fps was never
required.** Downstream scoring low-passes position at 4 Hz before differentiating it, so by
Nyquist ~12–15 fps already fully samples every measure — frames beyond that are smoothed away
before anything reads them. (The research pipeline's robustness study independently found the
whole thing survives 15 fps.)

Three findings underpin these numbers (full detail in [realtime.md](realtime.md)):

- **imgsz must be 640 (the training size).** Running the detectors at 1280 was 2× slower
  *and* dropped cup recall from 64 % to 8 % — upsampling past the training size is
  off-distribution, not free detail.
- **The GPU is launch-bound below imgsz ≈ 640**, so *batching across cameras* is the main
  lever (10 cameras cost 4.6× one camera, not 10×), and crop-tracking buys nothing.
- **Separate CUDA streams** let the two nets' kernels interleave (1.4× at 1 camera), with
  output verified bit-identical.

**Pose model choice: yolo26s.** Jitter — not any distance-to-a-reference — is the criterion,
because scoring differentiates position, so noise is amplified while constant offsets cancel:

| model | 10-cam fps | wrist 3D jitter |
|---|---|---|
| yolo26n | 26 | 4.37 mm |
| **yolo26s** | **19** | **3.55 mm** |
| yolo26m | 11 | 3.49 mm |
| yolo26x | ~6 | 3.27 mm |

`n → s` is a real 19 % noise cut; `s → m → x` is not worth 2–4× the cost.

> Decode is excluded from the inference budget: a live camera hands you a decoded frame in a
> capture thread, in parallel. Whether 10 simultaneous USB streams keep up is a separate
> capture question, not measured here.

---

## 2. Validated against optical mocap

**The question.** Does the camera pipeline produce the same *drink-phase signals* the
sub-millimetre mocap does? The drink phase rests on exactly two signals, and we compare each
directly against mocap:

- **cup → head distance** — the "cup is at the mouth" trigger.
- **cup speed** — the transport-window gate (80 mm/s) and the "slow enough to be drinking"
  gate (150 mm/s).

**The two sides.**

| | cup | head |
|---|---|---|
| **OMC (ground truth)** | QTM mocap, 4 markers, 100 Hz, sub-mm | QTM mocap, 5-marker head cluster |
| **MMC (cup-task)** | finetuned cup detector (`cup_clean3d_refill.pt`), multi-view triangulated | `yolo26s`-pose head-proxy (nose / eye centroid) |

Both MMC inputs are cup-task's own detectors — the cup model is md5-identical to the one that
produced the research cup cache. Mocap and video are time-synced per rep (validated 0.98
cup-speed correlation) from the existing `qtm_align` mapping; no re-processing was needed.

### Result (n = 189 reps, 7 participants; cohort still growing)

**Signal agreement — the camera signals track mocap to millimetres:**

| signal | median correlation | agreement |
|---|---|---|
| **cup → head distance** | **1.000** (p10 0.999, min 0.980) | offset-removed \|Δ\| median **2.5 mm** (p90 4.2 mm) |
| **cup speed** | **0.985** (p10 0.971) | the 80 & 150 mm/s gates fire the **same as mocap on 95 % / 97 %** of frames |

The cup → head curve carries a constant **−90 mm** offset (the pose nose-proxy sits below the
mocap head-marker cluster); it is a fixed anatomical offset, not error, and is removed before
the \|Δ\|. What matters for segmentation is the *shape* and the *gate crossings*, and both
match.

**Phase agreement — the same rule on each side's signal lands the drink boundaries within
1–3 frames:**

| | median bias | median \|error\| |
|---|---|---|
| drink onset | 0 ms | 33 ms (2 frames) |
| drink offset | −33 ms | 33 ms |
| drink dwell duration | −50 ms | 50 ms |

86 % of reps are within 100 ms of the mocap dwell; the residual sits at the 60 fps
quantization floor (50 ms = 3 frames).

### What this means

The signals the drink segmentation is built on are **not the limitation** — the camera
pipeline reproduces the sub-millimetre mocap cup → head distance to 2.5 mm and its speed gates
to 95–97 % of frames. Any larger phase disagreement seen elsewhere comes from a *segmenter
choosing a different rule* on these same good signals (e.g. triggering drink on hand → mouth
vs cup → head), **not** from the mocap-free inputs being noisy.

---

## Reproduce

```
conda activate object_tracking          # ultralytics 8.4.x (loads yolo26 Pose26)

# real-time benchmarks
python scripts/bench_streams.py          # the fps table (serial / threads / streams)

# segmentation-vs-mocap
python scripts/compare_cup_head.py       # signal + phase agreement; writes cache/cup_head_compare.json
```

- OMC cup + head per rep: `cache/qtm_omc/<stem>.json` (see its README for sync + provenance).
- MMC cup: research refill track (`…/track3d_clean3d_refill`), made by
  `models/cup_clean3d_refill.pt`. MMC head: `cache/pose_models/<stem>/yolo26s.2d.json`.

**Cohort caveat.** n grows as the background pose-cache job completes (currently ~215/697
reps); the mocap-truth ceiling is ~150 reps that have both pose and a matched C3D. Re-running
`compare_cup_head.py` after it finishes updates the numbers with no new GPU work — it only
re-joins the caches. The medians above have been stable from n = 14 to n = 189.
