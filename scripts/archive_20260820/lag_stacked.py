"""Stacked cross-correlation lag estimate: combine the SIGNAL CURVES, not the per-signal lags.

_find_lag_multi takes an argmax over 8 candidate signals and keeps the single best-correlating
one, discarding the rest. With 8 candidates, the maximum is exactly the statistic that can be
inflated by noise -- which is how a trial ends up shifted by 151 frames while five other signals
agree on ~0.

Instead, evaluate every signal's FULL correlation curve c_i(tau), Fisher-transform to make them
additive, stack with weights, and take one argmax of the sum:

    z_i(tau) = arctanh(clip(c_i(tau)))
    Z(tau)   = sum_i w_i z_i(tau) / sum_i w_i
    tau*     = argmax Z(tau)

Signals that agree reinforce at the same tau; a lone spurious peak is outvoted. The stacked
curve also yields a confidence: peak height, and margin over the best competing peak outside a
small exclusion window.

    from lag_stacked import find_lag_stacked
    lag, conf = find_lag_stacked(mmc, omc, side, mcup, ocup)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H      # noqa: E402


def _curve(a, b, max_lag):
    """Correlation of a vs lag-shifted b at every lag. Returns (lags, corr)."""
    a = H._lp(a); bl = H._lp(b)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        bs = np.roll(bl, lag)
        m = np.isfinite(a) & np.isfinite(bs)
        if m.sum() < 40:
            continue
        c = np.corrcoef(a[m], bs[m])[0, 1]
        if np.isfinite(c):
            out[i] = c
    return lags, out


def find_lag_stacked(mmc, omc, side, mmc_cup=None, omc_cup=None, max_lag=180,
                     min_peak=0.3, excl=10):
    """Returns (lag, dict of diagnostics). Signals identical to _find_lag_multi's candidate set."""
    cands = []
    for j in ("wrist", "elbow", "shoulder"):
        jn = f"{side}_{j}"
        if jn in mmc and jn in omc:
            cands.append((f"{j}_speed", H._speed(mmc[jn]), H._speed(omc[jn])))
            cands.append((f"{j}_disp", H._disp_from_start(mmc[jn]), H._disp_from_start(omc[jn])))
    if mmc_cup is not None and omc_cup is not None:
        cands.append(("cup_speed", H._speed(mmc_cup), H._speed(omc_cup)))
        cands.append(("cup_disp", H._disp_from_start(mmc_cup), H._disp_from_start(omc_cup)))
    if not cands:
        return 0, {"n_sig": 0, "peak": np.nan, "margin": np.nan}

    lags, Z, W = None, None, 0.0
    per = {}
    for name, a, b in cands:
        lg, c = _curve(a, b, max_lag)
        if not np.isfinite(c).any():
            continue
        peak = np.nanmax(c)
        if peak < min_peak:                       # a signal with no usable peak abstains
            continue
        z = np.arctanh(np.clip(np.nan_to_num(c, nan=0.0), -0.999, 0.999))
        w = max(peak, 0.0) ** 2                   # weight by how informative the signal is
        Z = z * w if Z is None else Z + z * w
        W += w; lags = lg
        per[name] = int(lg[int(np.nanargmax(c))])
    if Z is None:
        return 0, {"n_sig": 0, "peak": np.nan, "margin": np.nan}
    Z = Z / max(W, 1e-9)
    i = int(np.argmax(Z))
    lag = int(lags[i])
    # margin: how much better the winning peak is than the best peak outside +-excl frames
    mask = np.abs(lags - lag) > excl
    margin = float(Z[i] - (np.nanmax(Z[mask]) if mask.any() else -np.inf))
    return lag, {"n_sig": len(per), "peak": float(np.tanh(Z[i])), "margin": float(margin),
                 "per_signal": per}
