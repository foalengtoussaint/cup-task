"""Combine two independent 3D speed estimates by exploiting the SIGN of their error.

WHY `max` AND NOT AN AVERAGE. Both estimators under-read, and they under-read for the same
structural reason: every failure mode that afflicts them -- a lost track, a smeared patch, a
camera that stops contributing, a cloud that deforms -- makes a measured displacement SHORTER than
the true one. Nothing in either pipeline invents motion that did not happen. Measured on DELTA
(2259 moving camera-frames, cup):

    cloud under-reads on 76.4% of frames    median signed error -12.66 mm/s
    flow  under-reads on 71.2% of frames    median signed error -10.60 mm/s

When an error is ONE-SIDED, the larger of two estimates is the better one, and the usual
argument for averaging (independent zero-mean noise cancels) does not apply -- averaging two
under-estimates gives a smaller under-estimate. The numbers agree: on frames where the two
DISAGREE, the larger is closer to truth **74%** of the time, while "which method is better" is a
coin flip (cloud closer on 47%).

The remaining worry with a max rule is that it must not inflate frames where both are already
right. It does not: where the two agree within 5% (38% of frames) the fused ratio is 0.958, i.e.
still slightly under, not over. The bias correction only fires where there is real disagreement.

MEASURED, n=12 trials, cup speed vs OMC (moving frames):

    method   median   mean    p90   >100mm/s   ratio
    cloud     16.49  17.99  68.86      5.0%    0.943
    flow      18.66  18.59  56.68      1.8%    0.952     <- previous shipping path
    mean      14.51  15.66  59.39      3.2%    0.947
    max       15.20  14.47  49.80      1.5%    0.984     <- this

`max` beats flow on 11/12 trials and improves the TAIL, not just the median (p90 56.7 -> 49.8,
worst frame 203 -> 165). It also survives the stratification that retracted an earlier claim:
it wins where the HAND HOLDS THE CUP (13.5 vs 16.3), which is the hard case and most of the task,
and at every speed band including 800+ mm/s (50.0 vs 59.0).

⚠ SCOPE. This is validated on the CUP, where both inputs exist and both under-read. Do not assume
it transfers to the wrist without re-measuring the signed-error split -- the rule is only valid
while the errors are genuinely one-sided.
"""
from __future__ import annotations

import numpy as np


def fuse_speed(a: np.ndarray, b: np.ndarray, how: str = "max") -> np.ndarray:
    """Fuse two aligned (n,) speed tracks -> (n,). NaN-tolerant: one valid input passes through.

    `how`:
        max   DEFAULT. Correct when both inputs under-read (see module docstring).
        mean  the conventional choice; better than either input but keeps ~half the shared bias.
        min   included only to make the asymmetry visible in tests -- it is the WRONG direction
              and measurably worse (median 20.19 vs max's 12.95).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if how == "max":
        return np.fmax(a, b)          # fmax/fmin ignore NaN, so one live input still answers
    if how == "min":
        return np.fmin(a, b)
    if how == "mean":
        both_nan = ~np.isfinite(a) & ~np.isfinite(b)
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():      # an all-NaN column is expected, not an error
                warnings.simplefilter("ignore", RuntimeWarning)
                out = np.nanmean(np.stack([a, b]), axis=0)
        out[both_nan] = np.nan
        return out
    raise ValueError(f"unknown fuser {how!r}")


def signed_error_split(est: np.ndarray, truth: np.ndarray,
                       moving: float = 50.0) -> tuple[float, float]:
    """(fraction under-reading, median signed error) on moving frames.

    The VALIDITY CHECK for `fuse_speed("max")`: the rule is justified only while the error is
    one-sided. Run this before applying the fuser to a new target or a new rig -- if the fraction
    is near 0.5 the errors are symmetric and `max` would inflate rather than correct.
    """
    est = np.asarray(est, float); truth = np.asarray(truth, float)
    m = np.isfinite(est) & np.isfinite(truth) & (truth > moving)
    if m.sum() < 20:
        return float("nan"), float("nan")
    d = est[m] - truth[m]
    return float((d < 0).mean()), float(np.median(d))
