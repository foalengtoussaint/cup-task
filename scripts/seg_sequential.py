"""Prototype: SEQUENTIAL drink-task segmenter (one forward pass, ordering guaranteed).

Instead of independent detectors + staged refinement (which overwrite each other and collapse
zero-width phases), walk the trial ONCE from frame 0 and find each boundary AFTER the previous one,
each from the signal where the event is actually visible:

  rest_pre  --hand leaves rest-->        reaching
  reaching  --wrist->cup PLATEAU (grasp)--> forward_transport
  forward   --cup reaches MOUTH (van Andel)--> drinking
  drinking  --cup LEAVES mouth-->         back_transport
  back      --wrist->cup OPENS (release)--> returning
  returning --hand back at rest-->        rest_post

By construction no phase can be out of order or overwrite another, and the drink is detected by
MOUTH DISTANCE (works on small-excursion affected-arm drinks where the displacement proxy collapses).

Validates on OMC vs AutoMQ ground-truth phases, head-to-head against the staged pipeline.
Uses cup + wrist + mouth(nose) tracks. Data only.
"""
from __future__ import annotations
import sys, re, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from pipeline import segment as SEG
from pipeline.segment import (_butter_lp, _interp_nan_xyz, _median_smooth, _runs, FPS,
                              DRINK_SPEED, DRINK_FRAC, GRASP_FLAT_MMPS, MIN_PHASE)
from score_vs_automq import (load_automq, automq_phases_to_video, automq_part, automq_key, _win, COHORT_PARTS)
GRID = R._GRID_JOINTS
PHASES = ["reaching", "forward_transport", "drinking", "back_transport", "returning"]
import os
# Reach onset and settle rule. "pos" / "end" are the shipped distance-referenced rules; "speed" puts
# each boundary at a fraction of the trial's own peak hand speed. Measured 2026-08-24: the shipped
# onset fires at 47% of peak and the settle at 52%, i.e. mid-motion, which costs 200 ms and 450 ms of
# agreement with the reference protocol and truncates total movement time by 7.5%. "speed" at 0.10
# fires at 11.7% / 8.6% and recovers both. Defaults keep the published numbers reproducible; set
# OT_SEG_ONSET=speed OT_SEG_SETTLE=speed to switch.
ONSET_RULE = os.environ.get("OT_SEG_ONSET", "pos")
SETTLE_RULE = os.environ.get("OT_SEG_SETTLE", "end")
PEAK_FRAC = float(os.environ.get("OT_SEG_PEAK_FRAC", "0.10"))

LEAVE_REST = 30.0     # mm: hand has left its rest position
STILL_MMPS = 30.0     # mm/s: hand is "at rest" by the STILLNESS rule (see _tail)
ARRIVE_REST = 40.0    # mm: hand is back home
HOLD = max(int(0.15 * FPS), 3)


def _sustained(mask, start, hold, want=True):
    """First index >= start where `mask` equals `want` for >= hold consecutive frames."""
    run = 0
    for i in range(start, len(mask)):
        if bool(mask[i]) == want:
            run += 1
            if run >= hold:
                return i - hold + 1
        else:
            run = 0
    return None


def segment_sequential(cup_xyz, hand_xyz, mouth_xyz, fps=FPS, cup_mouth_xyz=None,
                       settle_rule=None, return_flags=False, cm_source="wrist"):
    """One forward pass -> list of (name, s, e) intervals, ordering guaranteed.

    cm_source: what drives the cup->MOUTH channel. "wrist" (DEFAULT since 2026-08-20) uses the hand
    itself; "cup" uses the cup track. The cup is needed for grasp and release -- they are read from
    the wrist->cup distance and nothing else can see them -- but the DRINK boundaries never need it:
    the hand carries the cup, so the hand's distance to the mouth closes, holds flat and opens on the
    same schedule. Using the wrist is also the better signal, because it is triangulated from nine
    confident body keypoints while the cup is a single occluded object that falls below the
    three-camera floor on 21% of drinking frames. Measured on Table IV's quantity, both sides on the
    wrist: drink onset 133->83 ms median (p90 403->267), drink offset 133->67 ms (p90 267->217);
    every other boundary bit-identical. It also removes the offset estimator, its hold-detection and
    its 20-frame minimum, which were undefined on 37 trials.

    cup_mouth_xyz: explicit SEPARATE track for the cup->mouth channel; overrides cm_source. The
    wrist-proxy fill (cup ~= wrist + const where the cup has no consensus) is valid for cup->mouth --
    the hand carries the cup, so its distance to the mouth stands in -- but DESTROYS wrist->cup, which
    becomes a constant, and grasp/release are read from exactly that opening/closing. So the drink
    boundaries may use the proxy-filled cup while grasp/release keep the observed-only one.

    settle_rule: "end" (DEFAULT since 2026-08-20) ends the trial when the hand comes within
    ARRIVE_REST of where it ENDS UP; "pos" (the previous default) measures against where it STARTED;
    "still" ends it when the hand stops moving. See _tail for why the reference matters.

    return_flags: also return {"settle_observed": bool}. False means the movement end was never
    observed inside the clip -- the returned settle is a timeout, so total_movement_time (and, to a
    lesser degree, number_of_movement_units, which sums over `returning`) is RIGHT-CENSORED on that
    trial and should be excluded rather than scored. Measured 19.2% of 636 under settle_rule="still".
    """
    cup = _butter_lp(_interp_nan_xyz(np.asarray(cup_xyz, float)), fps)
    hand = _butter_lp(_interp_nan_xyz(np.asarray(hand_xyz, float)), fps)
    mouth = _butter_lp(_interp_nan_xyz(np.asarray(mouth_xyz, float)), fps)
    T = min(len(cup), len(hand), len(mouth))
    cup, hand, mouth = cup[:T], hand[:T], mouth[:T]
    if T < 20:
        return ([], {"settle_observed": False}) if return_flags else []

    rest = np.median(hand[:max(int(0.5 * fps), 10)], axis=0)
    d_rest = np.linalg.norm(hand - rest, axis=1)
    d_wc = np.linalg.norm(hand - cup, axis=1)
    v_wc = np.r_[0.0, np.diff(_median_smooth(d_wc, 11))] * fps
    big_wc = 0.3 * float(d_wc.max() - d_wc.min())          # a "real" grasp/release travels >=30% of span
    if cup_mouth_xyz is not None:
        cup_cm = _butter_lp(_interp_nan_xyz(np.asarray(cup_mouth_xyz, float)), fps)[:T]
    elif cm_source == "wrist":
        cup_cm = hand                      # the hand carries the cup; see cm_source in the docstring
    else:
        cup_cm = cup
    d_cm = np.linalg.norm(cup_cm - mouth, axis=1)
    v_cm = np.r_[0.0, np.diff(_median_smooth(d_cm, 11))] * fps    # cup->mouth distance rate
    v_hand = np.r_[0.0, np.linalg.norm(np.diff(hand, axis=0), axis=1) * fps]   # hand SPEED, mm/s

    settle_rule = SETTLE_RULE if settle_rule is None else settle_rule
    bounds = {}
    # 1) reach onset = hand leaves rest (distance), or crosses a fraction of its own peak speed
    if ONSET_RULE == "speed":
        pk_h = float(np.nanmax(v_hand)) if np.isfinite(v_hand).any() else 0.0
        r0 = _sustained(v_hand > PEAK_FRAC * pk_h, 1, HOLD) if pk_h > 0 else None
        if r0 is None:                              # fall back rather than drop the trial
            r0 = _sustained(d_rest > LEAVE_REST, 0, HOLD)
    else:
        r0 = _sustained(d_rest > LEAVE_REST, 0, HOLD)
    if r0 is None:
        return []
    bounds["rest_pre"] = (0, r0)
    # 2) grasp = end of the BIG wrist->cup closing run after reach (hand arrives, distance goes flat)
    closing = [(s, e) for s, e in _runs(v_wc < -GRASP_FLAT_MMPS)
               if e > r0 and (d_wc[s] - d_wc[e - 1]) >= big_wc]
    grasp = closing[0][1] if closing else _sustained(np.abs(v_wc) < GRASP_FLAT_MMPS, r0 + HOLD, HOLD)
    if grasp is None or grasp <= r0:
        grasp = min(r0 + HOLD, T - 1)
    bounds["reaching"] = (r0, grasp)
    # 3+4) drink = the cup->mouth distance PLATEAU, detected EXACTLY like the grasp plateau but on
    # cup->mouth instead of wrist->cup: the distance CLOSES as the cup rises to the mouth, goes FLAT
    # while drinking (cup held at the mouth -- true even if the cup wobbles or the HEAD moves to the
    # cup, because a plateau measures the RELATIONSHIP, not absolute cup speed), then OPENS as the cup
    # is lowered. So drink_onset = end of the big closing run (arrival), drink_offset = start of the
    # big opening run (departure). No DRINK_SPEED / near-mouth absolute threshold -- scale-free.
    span_cm = float(d_cm.max() - d_cm.min())
    big_cm = 0.3 * span_cm
    closing_cm = [(s, e) for s, e in _runs(v_cm < -GRASP_FLAT_MMPS)
                  if e > grasp and (d_cm[s] - d_cm[e - 1]) >= big_cm]
    if not closing_cm:                     # cup never travelled to the mouth -> no drink; carve at apex
        apex = grasp + int(np.argmin(d_cm[grasp:])) if T > grasp else grasp
        bounds["forward_transport"] = (grasp, apex)
        rel = _release(v_wc, d_wc, big_wc, apex, T)
        bounds["back_transport"] = (apex, rel)
        obs = _tail(bounds, d_rest, rel, T, v_hand, settle_rule, hand)
        return (_to_list(bounds), {"settle_observed": obs}) if return_flags else _to_list(bounds)
    d_on = closing_cm[0][1]                 # cup arrived at the mouth, distance went flat
    bounds["forward_transport"] = (grasp, d_on)
    opening_cm = [(s, e) for s, e in _runs(v_cm > GRASP_FLAT_MMPS)
                  if s > d_on and (d_cm[e - 1] - d_cm[s]) >= big_cm]
    d_off = opening_cm[0][0] if opening_cm else T - 1    # cup left the mouth
    bounds["drinking"] = (d_on, d_off)
    # 5) release = start of the BIG wrist->cup opening run after drink (hand withdraws, distance ramps)
    rel = _release(v_wc, d_wc, big_wc, d_off, T)
    bounds["back_transport"] = (d_off, rel)
    obs = _tail(bounds, d_rest, rel, T, v_hand, settle_rule, hand)
    return (_to_list(bounds), {"settle_observed": obs}) if return_flags else _to_list(bounds)


def _release(v_wc, d_wc, big_wc, start, T):
    opening = [(s, e) for s, e in _runs(v_wc > GRASP_FLAT_MMPS)
               if s >= start and (d_wc[e - 1] - d_wc[s]) >= big_wc]
    return opening[0][0] if opening else min(start + HOLD, T - 1)


def _tail(bounds, d_rest, rel, T, v_hand=None, rule="end", hand=None):
    """Close the trial. Returns True if the end of movement was OBSERVED, False if timed out.

    rule="end" (DEFAULT): hand within ARRIVE_REST of where it ENDS UP -- the median position over the
    final 0.5 s. Adopted 2026-08-20: on Table IV's quantity (|MMC - OMC|, degenerate excluded) it puts
    settle at 17 ms median / 83 ms p90 / 1.5% beyond 0.25 s, against 33 / 153 / 7.0% for "pos", and
    leaves every other boundary untouched.

    rule="pos" (the previous default): hand back within ARRIVE_REST of where it STARTED. A PROXIMITY test,
    and it answers a different question than "is the hand at rest" -- a hand that stops somewhere else
    fails it. Measured against AutoMQ it puts `settle` at 417 ms median error, the second-worst
    boundary, and it is unstable: on trials where the hand is still travelling at clip end, whether it
    dips inside 40 mm is near-arbitrary, so OMC and MMC pick different RULES (one detects, one times
    out) on 10.3% of them vs 2.7% elsewhere -- which is what generates the 40-frame p90 tail.

    rule="still_end": stillness AND the late-proximity test.

    rule="still": hand SPEED below STILL_MMPS, sustained. This is the motion test AutoMQ/Unger use
    ("based on the end-effector velocity", Unger et al. II-D-2). 117 ms median error, 83 ms on the
    trials where it fires. It also never references the trial START, which matters because the OMC
    lag shift leaves up to 14 leading NaNs -- on ~9% of trials that is half the 0.5 s rest reference.

    Either way, when nothing fires the settle is a TIMEOUT at rel+HOLD, not a detection. That is not a
    tuning failure: on 45% of those trials AutoMQ's own settle lies past the last recorded frame, so
    the movement end is simply not in the video. Callers must treat those trials as censored.
    """
    if rule == "speed":
        # After the return peak, the first sustained fall below a fraction of the trial's own peak.
        # Scale-free, and it places the boundary near rest instead of mid-motion.
        if v_hand is None:
            raise ValueError('settle_rule="speed" needs the hand speed series')
        seg = v_hand[rel:T]
        settle = None
        if len(seg) > 2 and np.isfinite(seg).any():
            pk_h = float(np.nanmax(v_hand[:T]))
            k = int(np.nanargmax(seg))
            r2 = _sustained(seg[k:] < PEAK_FRAC * pk_h, 0, HOLD) if pk_h > 0 else None
            if r2 is not None:
                settle = rel + k + r2
    elif rule == "still":
        if v_hand is None:
            raise ValueError('settle_rule="still" needs the hand speed series')
        settle = _sustained(v_hand < STILL_MMPS, rel, HOLD)
    elif rule in ("end", "still_end"):
        # LATE reference: where the hand ends up, not where it began. `pos` measures against a
        # reference fixed in the trial's first 0.5 s and applies it ~6 s later, by which time the
        # hand has come to rest a median 15 mm away with a p90 sitting ON the 40 mm threshold -- so
        # the test is marginal exactly where the mass is, and OMC/MMC flip it independently.
        if hand is None:
            raise ValueError(f'settle_rule="{rule}" needs the hand position series')
        tail_n = max(int(0.5 * FPS), 10)
        ref_end = np.median(hand[max(T - tail_n, 0):T], axis=0)
        d_end = np.linalg.norm(hand[:T] - ref_end, axis=1)
        near = d_end < ARRIVE_REST
        if rule == "still_end":
            if v_hand is None:
                raise ValueError('settle_rule="still_end" needs the hand speed series')
            near = near & (v_hand[:T] < STILL_MMPS)
        settle = _sustained(near, rel, HOLD)
    else:
        settle = _sustained(d_rest < ARRIVE_REST, rel, HOLD)
    # WHERE the boundary is and WHETHER the movement ended are different questions, and the same
    # test cannot answer both. A proximity rule ("pos"/"end") places the boundary well but claims a
    # settle on 84-90% of the trials where the hand is still travelling when the video stops -- it
    # cannot distinguish "arrived" from "passing through". Stillness is the only rule that gets that
    # right (0% false positives), so `settle_observed` is ALWAYS the stillness test, whatever rule
    # placed the boundary. False means total_movement_time is right-censored on this trial.
    if settle is None or settle <= rel:
        settle = min(rel + HOLD, T)
    observed = False
    if v_hand is not None:
        s_still = _sustained(v_hand[:T] < STILL_MMPS, rel, HOLD)
        observed = s_still is not None and s_still > rel
    bounds["returning"] = (rel, settle)
    bounds["rest_post"] = (settle, T)
    return observed


def _to_list(b):
    return [(nm, s, e) for nm, (s, e) in b.items() if e > s]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["omc", "mmc"], default="omc",
                    help="omc = OMC cup+wrist+mouth (clean); mmc = markerless v3+SN cup + pose wrist/nose")
    args = ap.parse_args()
    SRC = args.source
    H.use_good_cams()
    amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([RL])_")
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    print(f"cohort {len(trials)}  source={SRC}", flush=True)
    rows = []; t0 = time.time(); n = 0; n_nocup = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get(automq_key(part, t["trial"]))   # block-aware truth row
        if rec is None or rec.get("phases") is None:
            continue
        nfr = t["mmc"].shape[0]
        omc = H._load_omc(part, trial, nfr); wr = f"{side}_wrist"
        if wr not in omc or not np.isfinite(omc[wr]).any() or "nose" not in omc:
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        ph_amq = automq_phases_to_video(rec["phases"], lag, nfr)
        if not ph_amq:
            continue
        truth = {p: _win(ph_amq, p) for p in PHASES}
        if SRC == "omc":
            ocup = R._omc_cup(part, trial, nfr)
            if not np.isfinite(ocup).any():
                n_nocup += 1; continue
            cup, hand, nose = R._shift(ocup, lag), R._shift(omc[wr], lag), R._shift(omc["nose"], lag)
        else:   # MMC: markerless v3+SmoothNet cup + markerless pose wrist/nose (video timeline, no shift)
            mcup = R._cup_v3(part, trial, R._calib(part), nfr)
            if not np.isfinite(mcup).any():
                n_nocup += 1; continue
            cup = R._smooth_joint(mcup)
            hand = R._smooth_joint(t["mmc"][:, GRID.index(wr)])
            nose = R._smooth_joint(t["mmc"][:, GRID.index("nose")])
        # SEQUENTIAL
        seq = {nm: (s, e) for nm, s, e in segment_sequential(cup, hand, nose)}
        # STAGED (current pipeline, displacement proxy) for comparison
        sg = SEG.segment_cup_only(R._fill(cup), fps=FPS)
        sg = SEG.refine_grasp_with_pose(sg, R._fill(cup), R._fill(hand), None, fps=FPS)
        stg = {nm: (s, e) for nm, s, e in SEG.to_murphy_phases(sg, R._fill(hand), R._fill(cup), fps=FPS)}
        for p in PHASES:
            if truth[p] is None:
                continue
            for name, d in [("sequential", seq), ("staged", stg)]:
                w = d.get(p)
                rows.append(dict(part=part, trial=trial, method=name, phase=p, present=int(w is not None),
                                 onset_f=(w[0] - truth[p][0]) if w else np.nan,
                                 offset_f=(w[1] - truth[p][1]) if w else np.nan))
        n += 1
        if n % 80 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / f"out/scoring/seg_sequential_{SRC}_vs_automq.csv"
    df.to_csv(out, index=False)
    print(f"\nPROCESSING CHECK: {SRC} trials {n}, rows {len(df)}, no-cup skipped {n_nocup}", flush=True)
    for meth in ("sequential", "staged"):
        g = df[df.method == meth]
        print(f"\n== {meth.upper()} vs AutoMQ ({SRC.upper()} cup+wrist+mouth) — median |err| frames / miss ==")
        print(f"  {'phase':<18}{'n':>5}{'miss':>8}{'|onset|f':>10}{'|offset|f':>10}")
        for p in PHASES:
            gp = g[g.phase == p]; det = gp[gp.present == 1]
            fo = det.onset_f.abs().median(); ff = det.offset_f.abs().median()
            print(f"  {p:<18}{len(gp):>5}{f'{int((gp.present==0).sum())}/{len(gp)}':>8}{fo:>10.1f}{ff:>10.1f}")
        det = g[g.present == 1]
        print(f"  {'ALL':<18}{'':5}{f'{int((g.present==0).sum())}/{len(g)}':>8}"
              f"{det.onset_f.abs().median():>10.1f}{det.offset_f.abs().median():>10.1f}")
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
