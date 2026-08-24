"""ANCHOR-THEN-EXPAND drink-task segmenter.

Why, measured: paired against itself on OMC-vs-MMC input, the sequential segmenter's boundaries agree
to a median of ~0 frames but with a heavy tail -- grasp p90 61 frames, release p90 91, and 44% / 39%
of trials off by >15 frames. The two boundaries that behave (reach onset, settle: |median| 2-3f) are
the ones anchored on a STATE; the four that don't are the ones read off the edge of the FIRST run
clearing a threshold (`closing[0]`, `closing_cm[0]`, `opening_cm[0]` in seg_sequential.py). On noisier
markerless input a different run qualifies first and the boundary jumps by a second or more.

So: find the EVENT first, then expand to its edges.
  hand-at-cup anchor   = argmin(wrist->cup distance)     -> its plateau is grasp .. release
  cup-at-mouth anchor  = argmin(cup->mouth distance)     -> its plateau is drink_on .. drink_off
  reach onset          = walk BACK from the grasp plateau while the hand is still moving
  settle               = walk FORWARD from release to the post-release rest plateau

Consequences of the anchor: the whole-trial `0.3 * span` run-qualification disappears (the anchor
already identifies the event, so nothing has to be "big enough to be the one"); a multi-sip drink no
longer truncates at the first lowering, because plateau expansion MERGES across short re-openings; and
the silent fallbacks (`d_off = T-1`, `rel = start + HOLD`) stop firing, because an anchor exists even
when no run clears a threshold. Thresholds are span-relative, not absolute mm.

Returns the same (name, s, e) list as segment_sequential, plus a flags dict via segment_anchored_full
so a degenerate trial announces itself instead of returning confident nonsense.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from pipeline.segment import _butter_lp, _interp_nan_xyz, _median_smooth, FPS  # noqa: E402

PHASES = ["rest_pre", "reaching", "forward_transport", "drinking", "back_transport",
          "returning", "rest_post"]
PLATEAU_FRAC = 0.05      # a plateau is "within 5% of the distance span" of the anchor minimum
PLATEAU_MIN_MM = 8.0     # ...but never tighter than 8mm (tracking noise floor)
MERGE_S = 0.40           # bridge re-openings shorter than this (multi-sip, cup wobble)
MOVE_FRAC = 0.12         # "still moving" = speed above 12% of the local peak
MIN_PLATEAU_F = 3


def _speed(xyz, fps):
    v = np.gradient(xyz, axis=0) * fps            # CENTRED difference: no half-window lag
    return np.linalg.norm(v, axis=1)


def _plateau(d, anchor, tol, merge):
    """Widest interval around `anchor` where d <= d[anchor] + tol, bridging gaps < `merge` frames."""
    T = len(d)
    lo = float(d[anchor]) + tol
    inside = d <= lo
    s = anchor
    while s > 0:
        if inside[s - 1]:
            s -= 1; continue
        j = s - 1                                  # look back across a short excursion
        while j > 0 and not inside[j] and s - j <= merge:
            j -= 1
        if j >= 0 and inside[j] and s - j <= merge:
            s = j
        else:
            break
    e = anchor
    while e < T - 1:
        if inside[e + 1]:
            e += 1; continue
        j = e + 1
        while j < T - 1 and not inside[j] and j - e <= merge:
            j += 1
        if j <= T - 1 and inside[j] and j - e <= merge:
            e = j
        else:
            break
    return s, e + 1                                # half-open


def _walk_back_while_moving(sp, start, lo, frac):
    """From `start`, walk back to where motion began: last frame with speed < frac * local peak."""
    if start <= lo + 1:
        return lo
    seg = sp[lo:start]
    if not len(seg) or not np.isfinite(seg).any():
        return lo
    thr = frac * float(np.nanmax(seg))
    i = start
    while i > lo and (not np.isfinite(sp[i - 1]) or sp[i - 1] > thr):
        i -= 1
    return i


def _walk_fwd_while_moving(sp, start, hi, frac):
    if start >= hi - 1:
        return hi
    seg = sp[start:hi]
    if not len(seg) or not np.isfinite(seg).any():
        return hi
    thr = frac * float(np.nanmax(seg))
    i = start
    while i < hi - 1 and (not np.isfinite(sp[i + 1]) or sp[i + 1] > thr):
        i += 1
    return i


def segment_anchored_full(cup_xyz, hand_xyz, mouth_xyz, fps=FPS):
    """-> (list of (name, s, e), flags dict). Flags are for QA; they never change the boundaries."""
    cup = _butter_lp(_interp_nan_xyz(np.asarray(cup_xyz, float)), fps)
    hand = _butter_lp(_interp_nan_xyz(np.asarray(hand_xyz, float)), fps)
    mouth = _butter_lp(_interp_nan_xyz(np.asarray(mouth_xyz, float)), fps)
    T = min(len(cup), len(hand), len(mouth))
    flags = {}
    if T < 20:
        return [], {"too_short": True}
    cup, hand, mouth = cup[:T], hand[:T], mouth[:T]

    d_wc = _median_smooth(np.linalg.norm(hand - cup, axis=1), 7)
    d_cm = _median_smooth(np.linalg.norm(cup - mouth, axis=1), 7)
    # an all-NaN distance means a track is empty end-to-end (interp cannot fill that) -- say so
    # instead of anchoring on nothing, which is what the old code's silent fallbacks did.
    if not np.isfinite(d_wc).any() or not np.isfinite(d_cm).any():
        return [], {"no_track": True}
    sp_h = _speed(hand, fps)
    merge = max(int(MERGE_S * fps), 2)
    tol = lambda d: max(PLATEAU_MIN_MM, PLATEAU_FRAC * float(np.nanmax(d) - np.nanmin(d)))

    # --- hand holds the cup: ONE plateau of wrist->cup distance, grasp .. release -----------------
    a_wc = int(np.nanargmin(d_wc))
    grasp, release = _plateau(d_wc, a_wc, tol(d_wc), merge)
    # --- cup at the mouth: plateau of cup->mouth distance, drink_on .. drink_off ------------------
    inner = d_cm.copy()
    inner[:grasp] = np.inf                            # the cup can only reach the mouth once held
    a_cm = int(np.nanargmin(inner)) if np.isfinite(inner).any() else int(np.nanargmin(d_cm))
    d_on, d_off = _plateau(d_cm, a_cm, tol(d_cm), merge)
    flags["drink_outside_hold"] = not (grasp <= d_on and d_off <= release + merge)
    d_on = max(d_on, grasp); d_off = max(min(d_off, release), d_on + 1)
    # --- reach onset: walk back from the grasp plateau while the hand is still moving -------------
    r0 = _walk_back_while_moving(sp_h, grasp, 0, MOVE_FRAC)
    # --- settle: forward from release to the post-release rest ------------------------------------
    settle = _walk_fwd_while_moving(sp_h, release, T, MOVE_FRAC)

    flags.update(short_hold=(release - grasp) < MIN_PLATEAU_F,
                 short_drink=(d_off - d_on) < MIN_PLATEAU_F,
                 hold_touches_start=grasp <= 1, hold_touches_end=release >= T - 2,
                 no_rest_pre=r0 <= 1, no_rest_post=settle >= T - 2,
                 cup_nan_frac=float(np.mean(~np.isfinite(np.asarray(cup_xyz, float)).all(1))),
                 hand_nan_frac=float(np.mean(~np.isfinite(np.asarray(hand_xyz, float)).all(1))))
    b = [("rest_pre", 0, r0), ("reaching", r0, grasp), ("forward_transport", grasp, d_on),
         ("drinking", d_on, d_off), ("back_transport", d_off, release),
         ("returning", release, settle), ("rest_post", settle, T)]
    return [(nm, s, e) for nm, s, e in b if e > s], flags


def segment_anchored(cup_xyz, hand_xyz, mouth_xyz, fps=FPS):
    return segment_anchored_full(cup_xyz, hand_xyz, mouth_xyz, fps)[0]
