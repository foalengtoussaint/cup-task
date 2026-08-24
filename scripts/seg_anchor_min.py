"""MINIMAL change to seg_sequential: pick the run AT THE ANCHOR, not the first qualifying one.

seg_anchored.py changed nine things at once and lost on MMC, so it identifies nothing. This variant
is seg_sequential.segment_sequential with exactly two changes:

  (1) RUN SELECTION. Where the original takes `closing[0]` / `closing_cm[0]` / `opening_cm[0]` -- the
      FIRST run clearing the threshold -- this takes the run adjacent to the event anchor
      (argmin of the relevant distance): the closing run ending nearest before the anchor, and the
      opening run starting nearest after it.
  (2) GRASP AND RELEASE FROM ONE HOLD. The hand holds the cup from grasp to release, so both come
      from the same wrist->cup anchor: grasp = closing run before argmin(d_wc), release = opening run
      after it -- instead of two independent forward searches that can latch onto different
      excursions.

EVERYTHING ELSE IS THE ORIGINAL, deliberately: GRASP_FLAT_MMPS, the 0.3*span qualification, the
11-frame median smooth, one-sided np.diff, the absolute LEAVE_REST/ARRIVE_REST thresholds and the
_sustained/_tail rest logic are all untouched, so any difference is attributable to (1)+(2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from pipeline.segment import _butter_lp, _interp_nan_xyz, _median_smooth, _runs, FPS, GRASP_FLAT_MMPS  # noqa: E402
from seg_sequential import ARRIVE_REST, HOLD, LEAVE_REST, _sustained, _tail, _to_list  # noqa: E402


def _run_before(runs, anchor):
    """The run ENDING nearest at-or-before the anchor (else the first run starting after it)."""
    if not runs:
        return None
    before = [r for r in runs if r[1] - 1 <= anchor]
    return max(before, key=lambda r: r[1]) if before else min(runs, key=lambda r: r[0])


def _run_after(runs, anchor):
    """The run STARTING nearest at-or-after the anchor (else the last one before it)."""
    if not runs:
        return None
    after = [r for r in runs if r[0] >= anchor]
    return min(after, key=lambda r: r[0]) if after else max(runs, key=lambda r: r[1])


def segment_anchor_min(cup_xyz, hand_xyz, mouth_xyz, fps=FPS):
    cup = _butter_lp(_interp_nan_xyz(np.asarray(cup_xyz, float)), fps)
    hand = _butter_lp(_interp_nan_xyz(np.asarray(hand_xyz, float)), fps)
    mouth = _butter_lp(_interp_nan_xyz(np.asarray(mouth_xyz, float)), fps)
    T = min(len(cup), len(hand), len(mouth))
    if T < 20:
        return []
    cup, hand, mouth = cup[:T], hand[:T], mouth[:T]

    rest = np.median(hand[:max(int(0.5 * fps), 10)], axis=0)
    d_rest = np.linalg.norm(hand - rest, axis=1)
    d_wc = np.linalg.norm(hand - cup, axis=1)
    v_wc = np.r_[0.0, np.diff(_median_smooth(d_wc, 11))] * fps
    big_wc = 0.3 * float(d_wc.max() - d_wc.min())
    d_cm = np.linalg.norm(cup - mouth, axis=1)
    v_cm = np.r_[0.0, np.diff(_median_smooth(d_cm, 11))] * fps
    if not np.isfinite(d_wc).any() or not np.isfinite(d_cm).any():
        return []

    bounds = {}
    r0 = _sustained(d_rest > LEAVE_REST, 0, HOLD)                 # unchanged
    if r0 is None:
        return []
    bounds["rest_pre"] = (0, r0)

    # (1)+(2) hand-at-cup ANCHOR -> grasp and release from the SAME hold
    a_wc = int(np.nanargmin(np.where(np.arange(T) > r0, d_wc, np.inf)))
    closing = [(s, e) for s, e in _runs(v_wc < -GRASP_FLAT_MMPS)
               if e > r0 and (d_wc[s] - d_wc[e - 1]) >= big_wc]
    c = _run_before(closing, a_wc)
    grasp = c[1] if c else _sustained(np.abs(v_wc) < GRASP_FLAT_MMPS, r0 + HOLD, HOLD)
    if grasp is None or grasp <= r0:
        grasp = min(r0 + HOLD, T - 1)
    bounds["reaching"] = (r0, grasp)

    # (1) cup-at-mouth ANCHOR -> the closing run before it / opening run after it
    span_cm = float(d_cm.max() - d_cm.min()); big_cm = 0.3 * span_cm
    a_cm = int(np.nanargmin(np.where(np.arange(T) > grasp, d_cm, np.inf)))
    closing_cm = [(s, e) for s, e in _runs(v_cm < -GRASP_FLAT_MMPS)
                  if e > grasp and (d_cm[s] - d_cm[e - 1]) >= big_cm]
    cc = _run_before(closing_cm, a_cm)
    opening_cm = [(s, e) for s, e in _runs(v_cm > GRASP_FLAT_MMPS)
                  if s > grasp and (d_cm[e - 1] - d_cm[s]) >= big_cm]
    oc = _run_after(opening_cm, a_cm)
    if cc is None and oc is None:                                  # no drink -> original apex carve
        apex = a_cm
        bounds["forward_transport"] = (grasp, apex)
        rel = _release_anchored(v_wc, d_wc, big_wc, a_wc, apex, T)
        bounds["back_transport"] = (apex, rel)
        _tail(bounds, d_rest, rel, T, rule="pos")   # comparison variant: keep the old rule
        return _to_list(bounds)
    d_on = cc[1] if cc else max(grasp + 1, a_cm)
    d_off = oc[0] if oc else min(T - 1, max(d_on + 1, a_cm))
    d_on = max(d_on, grasp); d_off = max(d_off, d_on + 1)
    bounds["forward_transport"] = (grasp, d_on)
    bounds["drinking"] = (d_on, d_off)

    rel = _release_anchored(v_wc, d_wc, big_wc, a_wc, d_off, T)
    bounds["back_transport"] = (d_off, max(rel, d_off + 1))
    _tail(bounds, d_rest, max(rel, d_off + 1), T, rule="pos")   # comparison variant
    return _to_list(bounds)


def _release_anchored(v_wc, d_wc, big_wc, a_wc, floor, T):
    """(2) release ends the SAME hold the grasp opened: opening run after the wrist->cup anchor."""
    opening = [(s, e) for s, e in _runs(v_wc > GRASP_FLAT_MMPS)
               if (d_wc[e - 1] - d_wc[s]) >= big_wc]
    o = _run_after(opening, max(a_wc, floor))
    return o[0] if o else min(floor + HOLD, T - 1)
