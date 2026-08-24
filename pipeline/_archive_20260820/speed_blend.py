"""Speed-weighted blend of flow-speed and SmoothNet-speed — the pipeline's wrist-speed signal.

The two speed sources fail in COMPLEMENTARY, non-overlapping regimes (measured, P07+P08, 12 trials,
full table in docs/SPEED_METRICS.md):

    regime      | flow (PyrLK)            | SmoothNet (d position/dt)
    ------------|-------------------------|--------------------------------
    off-peak    | 4.1 mm/s  (+0.4 bias)   | 10.6 mm/s (+8.1 phantom speed at rest)
    at the peak | +61.3 mm/s (motion BLUR)| +13.6 mm/s (position is unblurred)

Flow is a direct velocity measurement, so at rest it correctly reads ~zero while a differentiated
position keeps twitching. But at the fast peak the wrist patch blurs and PyrLK matches the smeared
patch too far, inflating the displacement. SmoothNet has the opposite profile: it works on POSITION,
which blur does not displace, so it holds the peak magnitude — at the cost of differentiation noise
when the wrist is nearly still.

So gate on the CURRENT speed: trust flow when slow, SmoothNet when fast.

    wb    = sigmoid((flow_speed - GATE_MID) / GATE_SOFT)
    blend = (1 - wb) * flow + wb * smoothnet

The gate is soft on purpose. A hard threshold would inject a discontinuity into the speed trace right
where peak-detection looks, creating spurious local maxima; the sigmoid crossfades over ~2*GATE_SOFT.

Result: the blend wins or ties EVERY peak statistic (peak value 20mm/s, off-peak 4.7mm/s) and its
standout is worst-case timing — max peak-time error 133ms vs 233-333ms for either source alone. It
removes the catastrophic misses that both pure methods suffer.

⚠ GATE_MID/GATE_SOFT are hand-set on P07+P08. They are a documented tuning debt: LOPO-tune before
trusting them on a new cohort. `fit_gate` is here for exactly that.
"""
from __future__ import annotations

import numpy as np

# Hand-set on P07+P08 (n=12). 350mm/s sits in the valley between the at-rest noise floor and the
# reach peaks (peaks run 700-1500mm/s); 120 makes the crossfade span roughly 350+-240.
GATE_MID = 350.0
GATE_SOFT = 120.0


def blend_weight(flow_speed, gate_mid: float = GATE_MID, gate_soft: float = GATE_SOFT):
    """Sigmoid weight on SmoothNet: 0 = all flow (slow), 1 = all SmoothNet (fast)."""
    fs = np.asarray(flow_speed, dtype=float)
    return 1.0 / (1.0 + np.exp(-(fs - gate_mid) / gate_soft))


def blend(flow_speed, smoothnet_speed, gate_mid: float = GATE_MID,
          gate_soft: float = GATE_SOFT) -> np.ndarray:
    """Blend the two speed traces. Falls back to whichever source is finite where one is missing.

    The gate is driven by FLOW speed (not the blend's own output) so the weighting cannot feed back
    on itself. Where flow is absent (too few cameras saw the wrist), the weight would be undefined,
    so we hand those frames to SmoothNet outright — it is the source that survives sparse views.
    """
    fl = np.asarray(flow_speed, dtype=float)
    sn = np.asarray(smoothnet_speed, dtype=float)
    n = max(len(fl), len(sn))
    fl = np.pad(fl, (0, n - len(fl)), constant_values=np.nan)
    sn = np.pad(sn, (0, n - len(sn)), constant_values=np.nan)

    wb = blend_weight(fl, gate_mid, gate_soft)
    have_fl, have_sn = np.isfinite(fl), np.isfinite(sn)
    out = np.full(n, np.nan)

    both = have_fl & have_sn
    out[both] = (1 - wb[both]) * fl[both] + wb[both] * sn[both]
    out[have_fl & ~have_sn] = fl[have_fl & ~have_sn]
    out[~have_fl & have_sn] = sn[~have_fl & have_sn]
    return out


def fit_gate(traces, grid_mid=None, grid_soft=None):
    """LOPO/grid-tune (GATE_MID, GATE_SOFT) against ground-truth speed.

    traces = [(flow_speed, smoothnet_speed, truth_speed), ...] — one tuple per trial. Scored on the
    PEAK error, because that is what the Murphy measures read; per-frame error would pick a different
    (worse) optimum, which is exactly the metric trap documented in SPEED_METRICS.md.
    """
    from scipy.signal import find_peaks
    grid_mid = np.arange(200.0, 601.0, 25.0) if grid_mid is None else np.asarray(grid_mid)
    grid_soft = np.arange(40.0, 241.0, 20.0) if grid_soft is None else np.asarray(grid_soft)

    best, best_score = (GATE_MID, GATE_SOFT), np.inf
    for m in grid_mid:
        for s in grid_soft:
            errs = []
            for fl, sn, truth in traces:
                b = blend(fl, sn, m, s)
                for p in find_peaks(truth, height=300, distance=30, prominence=150)[0]:
                    spk, _ = find_peaks(b, height=200, distance=25, prominence=100)
                    if len(spk):
                        j = spk[np.argmin(np.abs(spk - p))]
                        if abs(j - p) <= 20:
                            errs.append(abs(b[j] - truth[p]))
            if errs:
                score = float(np.median(errs))
                if score < best_score:
                    best, best_score = (float(m), float(s)), score
    return best, best_score
