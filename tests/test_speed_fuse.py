"""Tests for cup_task.speed_fuse -- the one-sided-error fusion rule.

The rule is only correct while both inputs UNDER-read. These tests pin that reasoning, so that if
someone later applies it to a target whose errors are symmetric, the assumption is visible.

    python tests/test_speed_fuse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cup_task.speed_fuse import fuse_speed, signed_error_split   # noqa: E402


def test_max_beats_mean_when_both_under_read():
    """The core claim: with one-sided (under-reading) errors, max beats mean AND either input."""
    rng = np.random.default_rng(0)
    truth = rng.uniform(100, 800, 4000)
    # each estimator loses an independent, strictly non-negative amount of motion
    a = truth * (1 - rng.uniform(0, 0.20, truth.size))
    b = truth * (1 - rng.uniform(0, 0.20, truth.size))
    err = lambda e: np.median(np.abs(e - truth))
    assert err(fuse_speed(a, b, "max")) < err(a)
    assert err(fuse_speed(a, b, "max")) < err(b)
    assert err(fuse_speed(a, b, "max")) < err(fuse_speed(a, b, "mean"))
    # and min is the wrong direction -- worse than either input
    assert err(fuse_speed(a, b, "min")) > err(a)


def test_max_is_wrong_when_errors_are_symmetric():
    """⚠ The guard rail. With SYMMETRIC noise, max is biased high and mean is correct.

    This is why signed_error_split exists: the rule is a claim about the error's SIGN, not a
    general-purpose fuser, and applying it to a symmetric-error signal makes things worse.
    """
    rng = np.random.default_rng(1)
    truth = rng.uniform(100, 800, 4000)
    a = truth + rng.normal(0, 40, truth.size)
    b = truth + rng.normal(0, 40, truth.size)
    err = lambda e: np.median(np.abs(e - truth))
    assert err(fuse_speed(a, b, "mean")) < err(fuse_speed(a, b, "max")), \
        "with symmetric errors the mean must win -- if not, the test data is wrong"
    assert np.median(fuse_speed(a, b, "max") - truth) > 0, "max should be biased HIGH here"


def test_nan_tolerance_one_live_input_still_answers():
    a = np.array([10.0, np.nan, 30.0, np.nan])
    b = np.array([np.nan, 20.0, 25.0, np.nan])
    out = fuse_speed(a, b, "max")
    assert out[0] == 10.0 and out[1] == 20.0 and out[2] == 30.0
    assert np.isnan(out[3]), "both inputs NaN must stay NaN, not become a number"
    m = fuse_speed(a, b, "mean")
    assert m[0] == 10.0 and np.isnan(m[3])


def test_signed_error_split_detects_one_sidedness():
    rng = np.random.default_rng(2)
    truth = rng.uniform(100, 800, 2000)
    under = truth * (1 - rng.uniform(0, 0.2, truth.size))
    sym = truth + rng.normal(0, 40, truth.size)
    f_under, med_under = signed_error_split(under, truth)
    f_sym, _ = signed_error_split(sym, truth)
    assert f_under > 0.95 and med_under < 0, "should flag a one-sided under-read"
    assert 0.4 < f_sym < 0.6, "symmetric noise should come out near 0.5"


def test_shape_mismatch_raises():
    try:
        fuse_speed(np.zeros(5), np.zeros(6))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched shapes")


def _main() -> int:
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1; print(f"  FAIL  {name}: {e}")
        except Exception as e:                              # noqa: BLE001
            bad += 1; print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
