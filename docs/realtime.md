# Real-time: what the rig actually does, and why

Measured **2026-07-13** on an **RTX 3060 Ti (8GB)**, 1920x1080 @ 60fps source, up to 10
cameras. Every number here was measured on this machine; none is quoted from a spec sheet.

> **This supersedes the earlier version of this file**, which claimed "~95fps, real-time ✔"
> from a single-camera test at `imgsz=1280`. That was wrong twice over: it never asked how
> many cameras the GPU must serve *at once*, and 1280 turns out to be a resolution at which
> the cup detector barely works at all (see below). A per-camera fps with no camera count is
> not a real-time claim.

**Decode is excluded from the budget throughout.** Reading an mp4 costs ~2ms/frame, but a
live camera hands you the frame already decoded -- you pay that in a capture thread, in
parallel, not in the inference budget. (Whether ten simultaneous USB streams *can* keep up
is a capture question we have NOT measured. It may well be the real bottleneck.)

---

## The answer

**cup detector + yolo26n-pose, imgsz 640, batched across cameras, both nets on separate CUDA
streams:**

| cameras | 1 | 2 | 4 | 6 | 8 | **10** |
|---|---|---|---|---|---|---|
| **fps** | 173 | 119 | 65 | 42 | 32 | **26** |
| ms/frame | 5.8 | 8.4 | 15.5 | 23.6 | 31.7 | 37.8 |

With **yolo26s-pose** (better keypoints, see the end): 10 cameras -> **19 fps**.

### 26fps at 10 cameras is ENOUGH. 60fps was never a real requirement.

`score.py` lowpasses hand position at **4 Hz** before differentiating it. By Nyquist,
~12-15fps already fully samples everything downstream -- frames beyond that are *smoothed
away before any measure sees them*. Running at 60fps would buy data the filter then
discards. (The research pipeline's robustness study reached the same conclusion from the
other side: the whole thing survives 15fps.)

---

## The three findings behind those numbers

### 1. `imgsz=1280` was a bug: 2x slower AND 8x worse

The detectors defaulted to `imgsz=1280`. The research pipeline (`cache_dets_model.py:49`)
passes **no imgsz at all** -- ultralytics' default **640**, which is also the **training**
size. Measured on P07 cam_4 against the research cache:

| imgsz | cup found | agrees w/ research <10px | ms |
|-------|-----------|--------------------------|-----|
| **640** | **64%** | **100%** | **4.3** |
| 960   | 62%       | 94%                      | 5.5 |
| 1280  | **8%**    | 47%                      | 9.0 |
| 1920  | **0%**    | --                       | 16.1|

**Upsampling past the training size is OFF-DISTRIBUTION, not free detail.** The detector
learned its size priors at 640; at 1280 the cup arrives at twice the scale it ever saw and
the model stops finding it. There was no speed/accuracy trade being made -- 1280 was **worse
on both axes at once**.

End-to-end: **cup 3D coverage 76% -> 91%.** Much of what we had been calling "the cup is
occluded 24% of the time" was *this bug*. The real occlusion is the remaining 9% -- her hand
wraps the cup at her lips, visible in the render, and no detector fixes that.

### 2. The GPU is LAUNCH-bound, not compute-bound

Inference time vs imgsz, batch=1, either net:

    imgsz   128   512   640  1024  1280  1920
    ms      2.8   2.9   3.2   4.3   6.6  13.3
            ^------- FLAT -------^   ^-- compute-bound --^

**A 25x cut in pixel count (128 vs 640) changes inference by 4%.** The cost is per-*call*
overhead -- ~200 CUDA kernel launches, the Python round-trip, the sync -- not arithmetic. A
640x640 image briefly occupies a few hundred of the card's 4864 cores; the rest is ceremony.
The knee is ~640-1024.

Two consequences, both tested rather than assumed:

- **BATCHING across cameras is the main lever**, because it is the only thing that amortizes
  a fixed per-call cost across more images. 10 cameras cost **4.6x** one camera, not 10x.
- **CROP-TRACKING IS A NULL. Do not propose it again.** The idea (crop to the subject so the
  cup is bigger on the model's canvas) is mechanically sound, was tested twice, works exactly
  as theorized, and buys nothing:

  | | cup found | cup size on canvas |
  |---|---|---|
  | full frame @640 | 354/485 (73%) | 15.6px |
  | task-region crop @640 | 358/485 (**74%**) | **20.1px** |

  A **29% bigger cup finds 4 more frames out of 485**, because the misses are *occlusion*,
  not resolution. Cropping *below* 640 is worse than useless: you shrink the cup for zero
  speed gain, since inference there is flat.

### 3. Use separate CUDA STREAMS, not just threads

Two Python threads calling `predict()` both submit into the **same default CUDA stream**, so
their kernels still serialize *on the device*; all threading overlaps is one net's CPU work
(letterbox, H2D copy) with the other's GPU work. A `torch.cuda.Stream` **each** lets them
interleave:

| cams | serial | threads | **streams** | streams gain |
|---|---|---|---|---|
| 1  | 8.3ms  | 6.6  | **5.8**  | **1.42x** |
| 4  | 20.2   | 17.8 | **15.5** | 1.30x |
| 10 | 45.2   | 41.2 | **38.6** | 1.17x |

**The gain decays with batch size (1.42x -> 1.17x), and that decay CONFIRMS the launch-bound
diagnosis by its shape:** at batch=1 the GPU idles between small kernels and a second stream
fills the gaps; at batch=10 it is genuinely computing and there are no gaps left to fill.

Output verified **bit-identical** (same detection md5s, same phase intervals). Concurrency
that moved a number would be a bug, not a speedup.

**Gotcha:** `torch.cuda.stream()` is **thread-local**. The context must be entered INSIDE the
worker thread. Set it on the main thread, submit to a pool, and you silently get the default
stream back -- i.e. you measure your own bug and conclude streams don't work.

---

## Levers not taken

| lever | expected | status |
|---|---|---|
| **TensorRT** | 2-3x (fuses kernels -> attacks the launch bound *directly*) | **untried.** The one real lever left |
| merged net | ~1.5x over streams | not worth it *for speed*: streams already recover most of the "one launch sequence instead of two" argument. Its appeal is **architectural** (shared backbone, bit-identical frozen keypoint head), not throughput |
| `half=True` on a `.pt` | nothing | **measured no-op** (4.0 vs 4.1ms). Ultralytics still runs the fp32 graph; real fp16 needs a TRT export |
| lower imgsz | nothing | flat below 640 (launch-bound), and 640 is the training size |
| crop-tracking | nothing | **measured null** (see above) |

TensorRT is worth pulling only if 26fps stops being enough. It currently is enough.

---

## Which pose model (n / s / m / x)

| | 10-cam fps | wrist 3D jitter |
|---|---|---|
| **yolo26n** | **26** | 4.37mm |
| **yolo26s** | **19** | **3.55mm** |
| yolo26m | 11 | 3.49mm |
| yolo26x | ~6 | 3.27mm |

**Jitter is the only defensible criterion**, and the reason is specific: `score.py`
*differentiates* hand position, so a constant offset in a keypoint **cancels**, while noise
gets **amplified** into bogus `peak_velocity` and phantom movement units. `n -> s` is a real
**19% noise cut**. `s -> m -> x` is not (3.55 -> 3.49 -> 3.27 -- gaps too small to defend on
one rep).

**Every other metric we tried turned out to measure the RIG, not the model:**

- **Reprojection residual is not model quality.** It is dominated by camera *distance*
  (r = **-0.905** between distance and px error: far cameras look good because each pixel
  covers more mm). The honest unit is **mm at the joint's depth**, where the pooled wrist
  residual is **11.4mm**, not "6.2px".
- **There is no objective wrist.** COCO's "wrist" is an annotation convention, not a physical
  landmark. Different nets learn slightly different conventions, so "distance to yolo26x" is
  **disagreement, not error** -- and x, being the same architecture on the same data, shares
  the family's biases anyway. It is a sharper ruler of the same make, not ground truth.
- **Per-camera confidence must NEVER weight the DLT.** Two distinct effects, easily conflated:
  - *Within* a camera, confidence is a valid per-frame quality signal (8/10 cameras show
    r = -0.3 to -0.64 between confidence and 2D jitter -- less sure really is shakier).
  - *Across* cameras it is **actively misleading**: corr(median confidence, median jitter) =
    **+0.56** (s) / **+0.79** (x). The more confident a camera is on average, the *shakier*
    it is. Weighting cameras by confidence would up-weight the worst ones.

  A per-camera signal cannot see a *cross-camera* disagreement -- the same structural lesson
  as the consensus gate being redundant for pose.

### OPEN: cam_4, and a metric that may be lying

cam_4 is the worst camera on the task wrist (**20.9mm** vs cam_8's 7.5mm) and stays worst
after the distance correction, identically across **all four** pose models. Its
confidence<->error correlation is *inverted* (**+0.669** within-camera, where every other
camera is negative), and so is its confidence<->jitter (+0.46 vs everyone else's -0.4).

**But on inspection, cam_4's own 2D prediction looks BETTER than the reprojected consensus.**
If that is right, cam_4 is *correct* and the other nine are dragging the triangulated point
off the true wrist -- and the reprojection residual is *penalising the one camera that can
see*. Every consistency metric in this document silently assumes the consensus is truth.

Settling this needs an **independent** reference (MeTRAbs multicam 3D -- different
architecture, different training), not another statistic derived from the cameras under test.
