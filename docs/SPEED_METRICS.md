# Wrist-speed accuracy — method comparison (persistent metrics)

Cohort: **P07 + P08, 12 trials** (P13 EXCLUDED — linear clock drift, see below). Metric: wrist speed
vs OMC ground truth. Flow = PyrLK at YOLO wrist pixel → triangulate {p} and {p+flow} → 3D velocity
(no position-differencing). All measured 2026-07-21. Scripts: `scripts/flow_velocity_probe.py`,
flow cached in `cache/flow_vel/<clip>__<method>.npy`.

## Per-frame speed error (whole trace)

Metric = median |Δspeed| mm/s vs OMC. **Speed is frame-invariant, so ABSOLUTE mm/s is the honest
metric — NOT correlation** (correlation hides magnitude error).

| method | \|Δspeed\| mm/s |
|---|---|
| pos-diff (original pipeline) | ~43 |
| SmoothNet (current) | 13.5 |
| **pure flow (PyrLK)** | **6.8** ← best per-frame |
| post-hoc blend | 8.5 |
| flow→KF-prior | 16.6 |
| flow-integrate→SmoothNet | 10.9 |

## Peak-velocity + time-to-peak (the MURPHY measures — per reach)

⚠ THE PER-FRAME WINNER (flow) IS THE WORST AT THE PEAK. Metric choice flips the answer. Peaks
detected independently per signal, matched to OMC peaks (straight peak-to-peak, NO windowed argmax).

| method | peak VALUE (med/p95/max) mm | peak TIME (med/mean/p95/max) ms |
|---|---|---|
| pos-diff | 144 / 432 / 562 | 50 / 77 / 228 / 333 |
| SmoothNet | 20 / 65 / 87 | 33 / 49 / 133 / 233 |
| pure flow | 61 / 147 / 162 | 50 / 65 / 133 / 267 |
| **speed-weighted blend** | **20 / 64 / 90** | **33 / 42 / 117 / 133** |

**The blend wins/ties every peak statistic**, and its standout is WORST-CASE timing: max 133ms vs
233-333ms for the others (~half). It eliminates the catastrophic misses both pure methods suffer.
Time medians (33/50ms) are frame-quantized (2/3 frames @60fps) — the MEAN and MAX are the honest
differentiators. Flow has a −28ms early-peak bias (blur inflates velocity on the way up).

## Off-peak vs at-peak decomposition (WHY the blend works)

| method | OFF-peak err | PEAK signed err | off-peak bias | off-peak noise |
|---|---|---|---|---|
| SmoothNet | 10.6 | +13.6 | +8.1 | 25.2 |
| flow | 4.1 | **+61.3** | +0.4 | 14.8 |

- **Flow off-peak**: clean (+0.4 bias, low noise) — direct velocity, no differentiation.
- **Flow at-peak**: over-shoots +61 — motion BLUR inflates the flow displacement at the fast wrist.
- **SmoothNet off-peak**: +8 phantom speed at rest + more noise — differentiation of smoothed-but-
  imperfect position.
- **SmoothNet at-peak**: accurate (+13.6) — works on position (unblurred), preserves peak magnitude.

They fail in COMPLEMENTARY, non-overlapping regimes → a speed gate captures the best of each.

## THE BLEND (recommended for the pipeline speed path)

```python
wb    = 1 / (1 + exp(-(flow_speed - 350) / 120))   # sigmoid gate on the CURRENT speed
blend = (1 - wb) * flow + wb * smoothnet
```
Slow (<350 mm/s, between reaches) → flow. Fast (>350, at peak) → SmoothNet. ⚠ The 350/120 constants
are HAND-SET on P07+P08, not tuned — validate/LOPO-tune before trusting on new cohorts.

## Flow-method shootout (which flow to use in the blend)

| flow method | per-frame mm/s | peak err | off-peak | cost |
|---|---|---|---|---|
| **PyrLK (pyramidal LK)** | 6.8 | 61 | 4.1 | 0.5ms/wrist/cam ← pick |
| DIS (dense) | 6.8 | 57 | 3.5 | ~5x slower |
| tuned-LK | 9.5 | 76 | 5.1 | — |
| RAFT-small (deep, wrist crop) | 18.1 | 52 | 6.6 | GPU, OOM-prone |

ALL flows over-shoot the peak (best RAFT 52 still ≫ SmoothNet 20) → peak over-shoot is FUNDAMENTAL to
flow-triangulation, not an algorithm choice. PyrLK = cheapest and effectively best. Deep flow NOT worth
it (RAFT loses; FastFlowNet skipped — not a capacity problem).

## P13 EXCLUDED — bad ground truth (linear clock drift)

P13's OMC↔video lag DRIFTS linearly through each trial: −8 → +3 frames over ~6s, slope +2.3 fr/s ≈
**3.8% clock-rate mismatch** (mocap vs video clocks run at different rates). Consistent across all 6
P13 trials; P07/P08 are stable (lag ±1). This is a ground-truth alignment defect, not a pipeline error,
so P13 can't be used to judge speed methods. (Confusingly first mis-labeled as desync then miscalib;
the drifting-lag test — best-local-lag in a sliding window — is what nailed it. A uniform-across-cameras
shift can only be an OMC-vs-video global lag, not inter-camera desync.)

---

## Point-tracking as a speed method — TESTED, doesn't help (2026-07-21)

Tried CoTracker3 + TAPNext++ to see if temporal point-tracking beats YOLO+flow. Cohort P07+P08.

**CoTracker3 (drift-vs-time):** perfect for the first ~1.3s (0-4px vs YOLO), then drifts — 12px@2s,
28px@3.3s. So single-seed tracking a point through a 10s repetitive drink task drifts badly. RE-SEEDING
from YOLO every 30 frames (before drift sets in) keeps it in the accurate regime.

**CoTracker-reseed(30) = a 2D temporal smoother, NOT a better detector:**
- DISPLACEMENT (raw, no low-pass): 4-6mm vs OMC = SAME as YOLO (re-seed anchors position to YOLO).
- JITTER (raw 3D 2nd-diff): 0.65mm vs YOLO's 2.5mm = ~4x SMOOTHER.
- SPEED |err|: 16 mm/s = SmoothNet-level (14), LOSES to flow (8).
So it cleans velocity via smoothing (same as SmoothNet) but doesn't track a more accurate position.

**Every way to combine CoTracker with the pipeline — none helps:**
| combination | result |
|---|---|
| CoTracker-reseed speed alone | 16 mm/s (= SmoothNet, loses to flow 8) |
| SmoothNet/blend ON CoTracker-2D input | WORSE (SN 17 vs 14, blend 7.9 vs 6.8) = double-smoothing |
| flow SEEDED from CoTracker pixel | median peak 56 vs 64 (better) BUT p90 297 vs 127, MAX 665 vs 162 |

The flow-seed case is the trap: median LOOKS better, but p90/p95/max are catastrophic — when CoTracker
drifts within a window, the flow seed lands off-wrist and velocity blows up (665 mm/s). YOLO's fresh
per-frame detection is jittery but NEVER drifts, so no catastrophic tail. **Always check the tail (p90/
p95/max), not just the median** — the median hid CoTracker's drift risk.

**TAPNext++ (the anti-drift model):** loads clean (PL checkpoint → extract `state_dict` + strip
`tapnext.` prefix; 194M params). Speed: fp32 = 13fps, **bf16 autocast = 28fps/point** (→ ~6fps for a
5-cam rig). Locked to 256x256 (pos-emb). Too slow for real-time; base TAPNext is same architecture so no
faster. Not evaluated for accuracy (disqualified on speed + CoTracker already showed point-tracking
doesn't beat YOLO+flow).

**VERDICT: point-tracking doesn't beat YOLO+flow here.** A strong per-frame detector (YOLO) beats
tracking-from-a-seed when you have one — the tracker only wins where there's no detector. Weights kept
(TAPNext++ 2.5GB, CoTracker cache) for future revisit. scripts/pointtrack_probe.py, ctreseed_batch.py.
