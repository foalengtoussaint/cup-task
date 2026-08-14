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
from cup_task import segment as SEG
from cup_task.segment import (_butter_lp, _interp_nan_xyz, _median_smooth, _runs, FPS,
                              DRINK_SPEED, DRINK_FRAC, GRASP_FLAT_MMPS, MIN_PHASE)
from score_vs_automq import (load_automq, automq_phases_to_video, automq_part, _win, COHORT_PARTS)
GRID = R._GRID_JOINTS
PHASES = ["reaching", "forward_transport", "drinking", "back_transport", "returning"]
LEAVE_REST = 30.0     # mm: hand has left its rest position
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


def segment_sequential(cup_xyz, hand_xyz, mouth_xyz, fps=FPS, cup_mouth_xyz=None):
    """One forward pass -> list of (name, s, e) intervals, ordering guaranteed.

    cup_mouth_xyz: optional SEPARATE cup track for the cup->mouth channel only. Needed because the
    wrist-proxy fill (cup ~= wrist + const where the cup has no consensus) is valid for cup->mouth --
    the hand carries the cup, so its distance to the mouth stands in -- but DESTROYS wrist->cup, which
    becomes a constant, and grasp/release are read from exactly that opening/closing. So the drink
    boundaries may use the proxy-filled cup while grasp/release keep the observed-only one.
    """
    cup = _butter_lp(_interp_nan_xyz(np.asarray(cup_xyz, float)), fps)
    hand = _butter_lp(_interp_nan_xyz(np.asarray(hand_xyz, float)), fps)
    mouth = _butter_lp(_interp_nan_xyz(np.asarray(mouth_xyz, float)), fps)
    T = min(len(cup), len(hand), len(mouth))
    cup, hand, mouth = cup[:T], hand[:T], mouth[:T]
    if T < 20:
        return []

    rest = np.median(hand[:max(int(0.5 * fps), 10)], axis=0)
    d_rest = np.linalg.norm(hand - rest, axis=1)
    d_wc = np.linalg.norm(hand - cup, axis=1)
    v_wc = np.r_[0.0, np.diff(_median_smooth(d_wc, 11))] * fps
    big_wc = 0.3 * float(d_wc.max() - d_wc.min())          # a "real" grasp/release travels >=30% of span
    cup_cm = cup if cup_mouth_xyz is None else _butter_lp(
        _interp_nan_xyz(np.asarray(cup_mouth_xyz, float)), fps)[:T]
    d_cm = np.linalg.norm(cup_cm - mouth, axis=1)
    v_cm = np.r_[0.0, np.diff(_median_smooth(d_cm, 11))] * fps    # cup->mouth distance rate

    bounds = {}
    # 1) reach onset = hand leaves rest
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
        _tail(bounds, d_rest, rel, T)
        return _to_list(bounds)
    d_on = closing_cm[0][1]                 # cup arrived at the mouth, distance went flat
    bounds["forward_transport"] = (grasp, d_on)
    opening_cm = [(s, e) for s, e in _runs(v_cm > GRASP_FLAT_MMPS)
                  if s > d_on and (d_cm[e - 1] - d_cm[s]) >= big_cm]
    d_off = opening_cm[0][0] if opening_cm else T - 1    # cup left the mouth
    bounds["drinking"] = (d_on, d_off)
    # 5) release = start of the BIG wrist->cup opening run after drink (hand withdraws, distance ramps)
    rel = _release(v_wc, d_wc, big_wc, d_off, T)
    bounds["back_transport"] = (d_off, rel)
    _tail(bounds, d_rest, rel, T)
    return _to_list(bounds)


def _release(v_wc, d_wc, big_wc, start, T):
    opening = [(s, e) for s, e in _runs(v_wc > GRASP_FLAT_MMPS)
               if s >= start and (d_wc[e - 1] - d_wc[s]) >= big_wc]
    return opening[0][0] if opening else min(start + HOLD, T - 1)


def _tail(bounds, d_rest, rel, T):
    settle = _sustained(d_rest < ARRIVE_REST, rel, HOLD)
    if settle is None or settle <= rel:
        settle = min(rel + HOLD, T)
    bounds["returning"] = (rel, settle)
    bounds["rest_post"] = (settle, T)


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
        rec = amq.get((automq_part(part), int(m.group(1)), m.group(2)))
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
    out = ROOT / f"out/automq/seg_sequential_{SRC}_vs_automq.csv"
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
