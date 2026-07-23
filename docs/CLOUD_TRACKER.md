# The surface-cloud velocity tracker

`cup_task/cloud_track.py` — a multi-camera tracker that measures the 3D **speed** (and, more
weakly, the **rotation**) of an object or pose keypoint, from a small cloud of tracked surface
points. It needs no object model, no CAD, no known dimensions — only a per-frame 2D detection of
the thing (which you already have) and calibrated cameras.

It is the successor to the single-point flow-speed path (`flow_speed.py`) and beats it on the
wrist; it is at the OMC geometric floor on the cup.

---

## The problem it solves

To measure how an object moved between two frames with a rigid fit (Kabsch), you need
**corresponded** 3D points: you must know that point *k* in frame A is the same physical bit of
surface as point *k* in frame B. That requires solving correspondence **across cameras** (which
pixel in cam 1 is the same surface point as which pixel in cam 5?) and **across time**.

The natural approach — detect features per camera and match them by appearance — is a **dead end**
here, and not for a tunable reason. Measured on the DELTA cup: at *known-true* correspondences,
normalized cross-correlation between two cameras has median **0.01** and clears a 0.65 gate only
**5 %** of the time. A plain, curved, specular cup viewed from 27–166° apart simply does not
present the same appearance twice. Even a corner-free dense matcher fails identically — the true
match beats decoys along the epipolar line at barely above chance. The wall is the imagery, not
the method.

## The trick: never match appearance

Correspondence is established **by construction**, then maintained by tracking:

1. **Seed geometrically.** Take the detected 3D position of the object. Generate ~24 random points
   on a coarse surface model around it (a cup-sized cylinder, or a small sphere for a keypoint —
   *the exact shape does not matter*, see below). Project each 3D point into every camera.
   Point *k* in cam 1 and point *k* in cam 5 are **the same physical point because they are
   projections of the same 3D point** — no matching, no NCC, no epipolar search.

2. **Track in 2D.** From then on, PyrLK optical flow follows each point **within its own camera**,
   independently, with a forward–backward consistency check. This is the only place correspondence
   across time is obtainable, and it is obtainable for free — PyrLK *follows* a point, it does not
   re-find it.

3. **Lift to 3D.** Each track has a pixel in ≥2 cameras, so triangulate it. Track identity is
   preserved, so the 3D cloud at *t* and *t+1* is the same set of points in the same order — which
   is exactly what Kabsch needs. **The 3D is rebuilt every frame; only the 2D tracks persist.**

4. **Fit.** Intersect track ids with the previous frame, robust-fit the rigid motion (RANSAC),
   report speed.

The tracking *is* the method. Reseeding every frame reduces it to differentiating the detection
(measured: identical to the raw detection, ~2× worse). The value comes entirely from PyrLK
following real image texture over time.

---

## Why the details are the way they are (all measured, n=12 vs OMC)

| Component | What / why |
|---|---|
| **Seed shape doesn't matter** | A generic 20 mm blob beats the true 40×95 cylinder (12.8 vs 15.6 mm/s). It only has to land points *on* the object; too big and seeds fly off (80×160 → 19 % coverage). This is why the tracker is object-agnostic and runs on a bare keypoint. |
| **Anchor** (`anchor_px=30`) | Drop a track whose pixel drifts >30 px from the frame's detection — it has slid off the object. Uses the detection you already have; no appearance model. |
| **Consensus gate** (per track) | Each track is triangulated only from the cameras that agree about where it is (`consensus3`), applied *per track* — a track drifting in one camera can't drag its 3D point. |
| **Rigidity gate** (`max_scale_dev=0.01`) | Fit a *similarity* (Umeyama) too; a rigid object can't change size, so a fitted scale ≠ 1 means the cloud deformed (tracks sliding). Refuse those frames. The **only** quality signal that predicts error (ρ +0.36); inlier count predicts nothing (ρ −0.01). ⚠ It is partly a speed filter — do **not** compute peak_velocity from gated output. |
| **RANSAC** | Removes tracks inconsistent with the rigid motion. It is *also* an implicit maturity filter: it rejects young (jittery) tracks 20× more than old ones, so an explicit age gate is redundant. |
| **Warmup** (`warmup=5`) | A freshly-seeded cohort is all-young-at-once and aperture-limited (its PyrLK windows haven't settled onto real texture). Skip the first 5 frames after each reseed. 14.7 → 13.4 mm/s, 12/12 trials. |
| **Median-step speed** (`median_step=True`) | Report the **median of each point's own step**, not the centroid step. The centroid averages positions then differences, which cancels the per-point rotation component and under-reads rotating motion. 15.6 → 13.9 mm/s, 10/12. No fitted constant — it's the geometrically-correct estimator. |

### Track maturation (why settling matters)

A fresh seed is a synthetic 3D point; PyrLK often lands it on an **aperture-limited** window
(texture in one direction only), where the shift is under-determined and the step jitters. Over
~10 frames PyrLK walks it onto a real 2D-textured feature and the step settles (per-frame step
scatter 150 → 8 mm/s). Validated against OMC, this maturation is real but **modest** — a ~10-frame
warmup, then flat. It is *not* exploitable per-track (RANSAC already drops the jittery young ones);
the one useful lever is the cohort-level `warmup`.

---

## What it can and cannot do

**Linear speed** is at the geometric floor. The residual ~4 % under-read is a **speed-dependent
per-frame tracking-noise floor** — jitter grows with speed and is spread across all tracks, not
concentrated in fixable "bad" tracks. It is irreducible per frame (a small real motion and jitter
produce the *same* point positions in one frame; only time separates them).

**Rotation** is the one signal a single tracked point cannot produce at all, but it is
scale-imperfect. Per-frame `|ω|` **over-reads** (~1.10×) because jitter rectifies into positive
rotation. The fix — validated but not yet wired in — is **temporal-informed**: average the rotation
*vectors* over a short window to find the real (coherent) axis, then keep each frame's magnitude
*along that axis* and discard the perpendicular (jitter) component. Real rotation is directionally
coherent across frames (axis coherence 0.94); jitter is not (0.15). Best angular estimator found:
0.172 → 0.155 rad/s error.

**Generalization:** works with no object dimensions, on the cup and on every pose keypoint tested
(wrist / elbow / shoulder / nose), ~100 % coverage. On the wrist it beats the shipping flow path
(paired 15.5 vs 17.7 mm/s).

---

## Usage

```python
from cup_task.cloud_track import CloudTracker
trk = CloudTracker(calib, units_per_metre=1000.0)     # calib = {cam: CamCalib}, mm world
for frame in stream:
    res = trk.update(gray_by_cam, obj_xyz_3d, dt=1/60, kp_by_cam=detections_2d)
    if res is not None and res.linear_speed is not None:
        print(res.linear_speed, res.angular_speed, res.active_3d_points_count)
```

`obj_xyz_3d` is the detected 3D position (seeding only — this is *not* mocap; it is your own
detector's consensus point). `kp_by_cam` is the per-camera 2D detection (enables the anchor).

**Visual:** `scripts/render_cloud_track.py --part P07 --trial trial_11_L_unaffected --cam cam_3`
renders an annotated video — tracks colored by age, RANSAC inliers, centroid, spin axis, and live
speed/rotation vs OMC. See it to relate the numbers to what the cloud is doing.

## Staggered variant (coverage, not accuracy)

Seeding fresh generations every ~10 frames *without* full reset (so ages coexist) and fitting on
tracks aged ≥10 gives ~100 % coverage at essentially the same accuracy (paired 13.5 vs 13.3). It is
the coverage-maximizing option; the default reseed-when-collapsed policy is the accuracy-maximizing
one. The staggered path lives in the probe scripts, not the shipping class — its gaps were only
1-frame holes, so coverage was never the bottleneck for a speed measurement.
