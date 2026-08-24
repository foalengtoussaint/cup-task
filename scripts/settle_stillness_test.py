"""EXPERIMENT: is the `settle` boundary hurt by defining rest as POSITION rather than STILLNESS?

WHY. `seg_sequential._tail` ends the trial when the hand comes back within ARRIVE_REST (40mm) of the
median hand position over the trial's first 0.5s -- a PROXIMITY test against where the hand started.
AutoMQ/Unger classify phases from END-EFFECTOR VELOCITY (Unger et al. II-D-2, "following the approach
in [8]"), i.e. a MOTION test. A participant who stops somewhere else -- a different spot on the table,
or nearer where the cup was set down -- is at rest but fails the proximity test, and `_tail` then
falls through to its `rel + HOLD` timeout instead of detecting an event.

Measured symptom: scoring the shipped OMC segmenter against AutoMQ, `settle` is the second-worst
boundary (median |err| 25f = 417ms) against 8-13f for grasp/drink/reach onset.

WHAT. Re-derives ONLY the settle boundary from the same cached OMC inputs, three ways, and scores each
against AutoMQ:
    pos       the shipped rule (proximity to start, `_tail`)
    still     hand speed below a threshold, sustained HOLD frames, searched from release
    pos_or_still   whichever fires first (a hand that returns home AND stops is still detected)
Speed thresholds are swept. Also reports how often the shipped rule hits its timeout fallback rather
than detecting a return, which is the mechanism this is testing.

Shipped segmenter is NOT modified -- this only recomputes the tail and compares. Data only.

    python scripts/settle_stillness_test.py [--out out/scoring/settle_stillness.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import cache_seg_inputs as CSI                                          # noqa: E402
from seg_sequential import (segment_sequential, _sustained, HOLD,       # noqa: E402
                            LEAVE_REST, ARRIVE_REST, FPS)
from pipeline.segment import _butter_lp, _interp_nan_xyz                # noqa: E402

BOUNDS = [("release", "back_transport", 1), ("settle", "returning", 1)]
SPEED_THR = [30.0, 50.0, 75.0, 100.0, 150.0]     # mm/s "hand is still"
PEAK_FRAC = [0.02, 0.05, 0.08, 0.12]             # same test, relative to the trial's own peak


def _hand_speed(hand, fps=FPS):
    """Low-passed hand speed (mm/s), same preprocessing the segmenter applies to positions."""
    h = _butter_lp(_interp_nan_xyz(np.asarray(hand, float)), fps)
    v = np.linalg.norm(np.diff(h, axis=0), axis=1) * fps
    return np.concatenate([v[:1], v])


def _rest_ref(hand, fps=FPS):
    """The shipped reference: median hand position over the first 0.5 s."""
    h = _butter_lp(_interp_nan_xyz(np.asarray(hand, float)), fps)
    return h, np.linalg.norm(h - np.median(h[:max(int(0.5 * fps), 10)], axis=0), axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "out/scoring/settle_stillness.csv"))
    a = ap.parse_args(argv)

    recs = CSI.load_all()
    print(f"{len(recs)} cached trials", flush=True)
    rows = []
    for r in recs:
        cup, hand, nose = r["cup_omc"], r["wrist_omc"], r["nose_omc"]
        seg = {nm: (s, e) for nm, s, e in segment_sequential(cup, hand, nose)}
        if "back_transport" not in seg or "returning" not in seg:
            continue
        rel = int(seg["back_transport"][1])
        settle_pos = int(seg["returning"][1])
        _, d_rest = _rest_ref(hand)
        T = len(d_rest)

        # did the shipped rule DETECT a return, or fall through to its rel+HOLD timeout?
        det = _sustained(d_rest < ARRIVE_REST, rel, HOLD)
        timed_out = bool(det is None or det <= rel)

        amq = {}
        for lab, ph, i in BOUNDS:
            w = r[f"amq_{ph}"]
            amq[lab] = float(w[i]) if w[0] >= 0 else np.nan

        row = dict(part=r["part"], trial=r["trial"], arm=r["arm"], rel=rel, T=T,
                   settle_pos=settle_pos, timed_out=timed_out, amq_settle=amq["settle"],
                   amq_release=amq["release"])
        sp = _hand_speed(hand)
        peak = float(np.nanmax(sp))
        row["peak_speed"] = peak
        # quietest speed the hand actually sustains after release -- the noise floor of this trial
        if len(sp[rel:]) >= HOLD:
            row["quietest"] = float(
                np.lib.stride_tricks.sliding_window_view(sp[rel:], HOLD).max(axis=1).min())
        for frac in PEAK_FRAC:
            s = _sustained(sp < frac * peak, rel, HOLD)
            row[f"settle_rel_{frac}"] = int(s) if s is not None else int(min(rel + HOLD, T))
            row[f"rel_detected_{frac}"] = s is not None
        for thr in SPEED_THR:
            s = _sustained(sp < thr, rel, HOLD)
            s_still = int(s) if s is not None else int(min(rel + HOLD, T))
            row[f"settle_still_{int(thr)}"] = s_still
            row[f"still_detected_{int(thr)}"] = s is not None
            row[f"settle_or_{int(thr)}"] = min(s_still, settle_pos)
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(df)} trials)\n", flush=True)

    g = df.dropna(subset=["amq_settle"])
    print(f"trials with an AutoMQ settle: {len(g)}")
    print(f"shipped rule hit its TIMEOUT fallback on "
          f"{100*g.timed_out.mean():.1f}% of trials\n")

    def score(col):
        e = (g[col] - g["amq_settle"]).abs()
        return e.median(), e.median() / FPS * 1000, e.quantile(.90)

    print(f"  {'settle rule':22s} {'med|err|f':>10s} {'ms':>7s} {'p90 f':>8s} {'detected':>10s}")
    m, ms, p90 = score("settle_pos")
    print(f"  {'pos (shipped)':22s} {m:10.1f} {ms:7.0f} {p90:8.1f} {100*(1-g.timed_out.mean()):9.0f}%")
    for thr in SPEED_THR:
        m, ms, p90 = score(f"settle_still_{int(thr)}")
        det = 100 * g[f"still_detected_{int(thr)}"].mean()
        print(f"  {'still <'+str(int(thr))+' mm/s':22s} {m:10.1f} {ms:7.0f} {p90:8.1f} {det:9.0f}%")
    for thr in SPEED_THR:
        m, ms, p90 = score(f"settle_or_{int(thr)}")
        print(f"  {'pos OR still<'+str(int(thr)):22s} {m:10.1f} {ms:7.0f} {p90:8.1f}")

    for frac in PEAK_FRAC:
        m, ms, p90 = score(f"settle_rel_{frac}")
        det = 100 * g[f"rel_detected_{frac}"].mean()
        print(f"  {'still <'+str(int(frac*100))+'% of peak':22s} {m:10.1f} {ms:7.0f} {p90:8.1f} {det:9.0f}%")

    # WHY detection is not 100%: three participants have an OMC wrist noise floor above any
    # usable threshold, so "still" never registers there. Not clip truncation, not impairment.
    print("\n  quietest sustained hand speed after release, by participant (mm/s):")
    q = g.groupby("part")["quietest"].median().sort_values()
    for part, val in q.items():
        print(f"    {part:6s} {val:8.1f}")

    print(f"\n  reference -- release (unchanged rule): "
          f"{(g['rel'] - g['amq_release']).abs().median():.1f} f")


if __name__ == "__main__":
    main()
