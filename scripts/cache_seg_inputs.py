"""Cache the SEGMENTER INPUT TRACKS once, so segmenter variants can be scored with no re-derivation.

Every segmenter experiment so far re-triangulated the cup and re-parsed the C3Ds (~4-5 min per source
per run, x2 sources). The inputs never change -- only the segmentation rule does. So cache, per trial:
    cup/wrist/nose from OMC  (lag-shifted onto the video timebase)
    cup/wrist/nose from MMC  (markerless v3+SmoothNet cup, BA+SmoothNet pose)
    the lag, side/arm, and the reference phase windows WHERE THEY EXIST
Then a new rule is scored in seconds. Same rule as the detection and omc_prep caches.

A reference phase row is NOT required (changed 2026-08-21). The paper scores our segmenter against
our segmenter, so those windows are only used by the definitional side-experiments; requiring one
dropped 161 otherwise-usable trials whose optical wrist, nose and cup are all present and whose
detections were already computed. `arm` comes from the trial name, which is the only thing the
reference was still needed for. Trials without one carry amq_* = (-1, -1), the same sentinel a
missing phase already used, so every reader that already handles it keeps working.

    python scripts/cache_seg_inputs.py            # ~10 min once
    -> cache/seg_inputs/<part>__<trial>.npz
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
# OT_SEG_INPUTS_DIR pairs with OT_TRACKS_DIR so an alternative cup track (e.g. the yolo26x-seg
# seeded tracks_uetrack_26x) gets its own inputs cache instead of overwriting the default one.
CACHE = ROOT / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs")
PHASES = ["reaching", "forward_transport", "drinking", "back_transport", "returning"]

# MISPAIRED video<->C3D. scripts/verify_blocks.py scores every trial's OMC candidates at name offsets
# -2..+2 with the stacked lag and the block repair off; the answer should be the shift _C3D_BLOCKS
# prescribes. 22 of 851 trials land elsewhere, but 14 of those are ties -- adjacent repetitions of the
# same stereotyped movement, where the runner-up is within 0.008 -- and a tie is not evidence of a
# mispairing. These three are: another C3D beats the assigned one by 0.02-0.05, which a tie between
# neighbouring reps does not produce. Two predate the 2026-08-21 expansion, so the rule is applied to
# old and new trials alike rather than only to what was just added.
#     P14/trial_56_L_affected     r 0.930 at offset 0 vs 0.950 at -2
#     P15/trial_88_L_affected     r 0.935 at offset 0 vs 0.970 at +1
#     P19/trial_56_R_affected     r 0.753 at offset 0 vs 0.805 at +2   (both low)
# UNUSABLE OPTICAL REFERENCE. `_destep` repairs a marker that jumps and stays displaced by keeping the
# longest run, but declines when that would cost more than 35% of the finite frames -- correctly, since
# deleting a third of a trial is worse than passing it through. The consequence is that those trials
# keep a corrupted marker in the GROUND TRUTH. P08/trial_1 has four markers stepping at once, `chest`
# among them, and reports a 360.6mm optical trunk displacement against our 23.8mm -- 0.07 off Table
# III's trunk correlation from that one trial. Where the reference is demonstrably wrong it is removed.
#
# Decided from the OPTICAL data alone (a marker exceeding ~6 m/s whose repair would cost >35% of the
# trial), never from agreement with our estimate. The threshold is not tuned: >0.35 selects these four
# trials and >0.15 selects six, with nothing in between.
#
# NOT the same kind of exclusion as the segmenter's declines, and must never be summed with them: a
# decline is detected from the markerless cup track and a deployed system has that signal, whereas this
# needs mocap and has no field equivalent. It cleans the reference; it says nothing about the pipeline.
_BAD_OMC = {("P08", "trial_1_R_unaffected"),      # shoulder_R, wrist_inner_R, hip_R, chest -- 52.2%
            ("P17", "trial_26_L_unaffected"),     # elbow_R  -- 45.2%
            ("P12", "trial_2_L_unaffected"),      # wrist_outer_L -- 44.4%
            ("P17", "trial_77_R_affected")}       # wrist_outer_R -- 42.2%

_MISPAIRED = {("P14", "trial_56_L_affected"),
              ("P15", "trial_88_L_affected"),
              ("P19", "trial_56_R_affected")}


def main():
    import compare_pose_omc_delta as H
    import gnn_train as GT
    import results_v3_delta as R
    from score_vs_automq import (load_automq, automq_part, automq_key, automq_phases_to_video, _win,
                                 COHORT_PARTS)
    GRID = R._GRID_JOINTS
    CACHE.mkdir(parents=True, exist_ok=True)
    H.use_good_cams()
    amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([RL])_")
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    print(f"caching segmenter inputs for {len(trials)} trials -> {CACHE}", flush=True)
    t0 = time.time(); n_ok = n_skip = n_nocup = n_bad = n_noref = n_mispaired = n_badomc = 0
    for i, t in enumerate(trials):
        part, trial, side = t["part"], t["trial"], t["side"]
        out = CACHE / f"{part}__{trial}.npz"
        if (part, trial) in _MISPAIRED:
            n_mispaired += 1
            out.unlink(missing_ok=True)          # drop a stale one built before the audit
        elif (part, trial) in _BAD_OMC:
            n_badomc += 1
            out.unlink(missing_ok=True)
        elif out.exists():
            n_skip += 1
        else:
            m = pat.search(trial)
            rec = amq.get(automq_key(part, trial)) if m else None   # block-aware truth row
            arm = ("affected" if "_affected" in trial else
                   "unaffected" if "_unaffected" in trial else None)
            if arm is None:
                n_bad += 1
            else:
                nfr = t["mmc"].shape[0]
                omc = H._load_omc(part, trial, nfr); wr = f"{side}_wrist"
                if wr not in omc or not np.isfinite(omc[wr]).any() or "nose" not in omc:
                    n_bad += 1
                else:
                    # STACKED multi-signal lag (H.find_lag_best), same as the measure scorer. The
                    # OMC cup/wrist/nose channels are shifted onto the MMC timebase by this lag, and
                    # Table IV measures OMC-vs-MMC boundary DISAGREEMENT -- so a lag error injects
                    # boundary error directly, on the OMC side only.
                    lag, _, _ = H.find_lag_best(
                        {j: t["mmc"][:, k] for k, j in enumerate(GRID)}, omc, side)
                    ph = (automq_phases_to_video(rec["phases"], lag, nfr)
                          if rec is not None and rec.get("phases") is not None else None)
                    ocup = R._omc_cup(part, trial, nfr)
                    mcup = R._cup_v3(part, trial, R._calib(part), nfr)
                    if not np.isfinite(ocup).any() or not np.isfinite(mcup).any():
                        n_nocup += 1
                    else:
                        # (-1, -1) where the reference has no window for this phase, and for every
                        # phase when it has no row for the trial at all
                        win = {p: (_win(ph, p) or (-1, -1)) if ph else (-1, -1) for p in PHASES}
                        if ph is None:
                            n_noref += 1
                        np.savez_compressed(
                            out, part=np.array(part), trial=np.array(trial), side=np.array(side),
                            arm=np.array(arm),
                            lag=np.array(lag), n=np.array(nfr),
                            cup_omc=R._shift(ocup, lag), wrist_omc=R._shift(omc[wr], lag),
                            nose_omc=R._shift(omc["nose"], lag),
                            # one batched SmoothNet forward for all three channels (R._smooth_pose);
                            # same numbers as three separate calls, ~3x less GPU launch overhead
                            **{f"{k}_mmc": v for k, v in R._smooth_pose({
                                "cup": mcup,
                                "wrist": t["mmc"][:, GRID.index(wr)],
                                "nose": t["mmc"][:, GRID.index("nose")]}).items()},
                            **{f"amq_{p}": np.array(win[p]) for p in PHASES})
                        n_ok += 1
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(trials)}] {time.time()-t0:4.0f}s  new {n_ok} skip {n_skip} "
                  f"nocup {n_nocup} unusable {n_bad} (of which no reference row: {n_noref})",
                  flush=True)
    print(f"\nPROCESSING CHECK: trials {len(trials)}, new {n_ok}, already cached {n_skip}, "
          f"no-cup {n_nocup}, unusable {n_bad}, mispaired {n_mispaired}, "
          f"bad-optical-reference {n_badomc}, files {len(list(CACHE.glob('*.npz')))}", flush=True)
    print(f"  cached WITHOUT a reference phase row: {n_noref} "
          f"(their amq_* windows are (-1, -1))", flush=True)
    print("DONE", flush=True)


def load_all():
    recs = []
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        for k in ("part", "trial", "side", "arm"):
            d[k] = str(d[k])
        recs.append(d)
    return recs


if __name__ == "__main__":
    main()
