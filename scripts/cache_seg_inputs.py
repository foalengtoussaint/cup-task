"""Cache the SEGMENTER INPUT TRACKS once, so segmenter variants can be scored with no re-derivation.

Every segmenter experiment so far re-triangulated the cup and re-parsed the C3Ds (~4-5 min per source
per run, x2 sources). The inputs never change -- only the segmentation rule does. So cache, per trial:
    cup/wrist/nose from OMC  (lag-shifted onto the video timebase)
    cup/wrist/nose from MMC  (markerless v3+SmoothNet cup, BA+SmoothNet pose)
    the AutoMQ phase windows, the lag, side/arm
Then a new rule is scored in seconds. Same rule as the detection and omc_prep caches.

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


def main():
    import compare_pose_omc_delta as H
    import gnn_train as GT
    import results_v3_delta as R
    from score_vs_automq import (load_automq, automq_part, automq_phases_to_video, _win,
                                 COHORT_PARTS)
    GRID = R._GRID_JOINTS
    CACHE.mkdir(parents=True, exist_ok=True)
    H.use_good_cams()
    amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([RL])_")
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    print(f"caching segmenter inputs for {len(trials)} trials -> {CACHE}", flush=True)
    t0 = time.time(); n_ok = n_skip = n_nocup = n_bad = 0
    for i, t in enumerate(trials):
        part, trial, side = t["part"], t["trial"], t["side"]
        out = CACHE / f"{part}__{trial}.npz"
        if out.exists():
            n_skip += 1
        else:
            m = pat.search(trial)
            rec = amq.get((automq_part(part), int(m.group(1)), m.group(2))) if m else None
            if rec is None or rec.get("phases") is None:
                n_bad += 1
            else:
                nfr = t["mmc"].shape[0]
                omc = H._load_omc(part, trial, nfr); wr = f"{side}_wrist"
                if wr not in omc or not np.isfinite(omc[wr]).any() or "nose" not in omc:
                    n_bad += 1
                else:
                    lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
                    ph = automq_phases_to_video(rec["phases"], lag, nfr)
                    ocup = R._omc_cup(part, trial, nfr)
                    mcup = R._cup_v3(part, trial, R._calib(part), nfr)
                    if not ph or not np.isfinite(ocup).any() or not np.isfinite(mcup).any():
                        n_nocup += 1
                    else:
                        win = {p: (_win(ph, p) or (-1, -1)) for p in PHASES}
                        np.savez_compressed(
                            out, part=np.array(part), trial=np.array(trial), side=np.array(side),
                            arm=np.array("affected" if rec.get("condition") == "affected"
                                         else "unaffected"),
                            lag=np.array(lag), n=np.array(nfr),
                            cup_omc=R._shift(ocup, lag), wrist_omc=R._shift(omc[wr], lag),
                            nose_omc=R._shift(omc["nose"], lag),
                            cup_mmc=R._smooth_joint(mcup),
                            wrist_mmc=R._smooth_joint(t["mmc"][:, GRID.index(wr)]),
                            nose_mmc=R._smooth_joint(t["mmc"][:, GRID.index("nose")]),
                            **{f"amq_{p}": np.array(win[p]) for p in PHASES})
                        n_ok += 1
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(trials)}] {time.time()-t0:4.0f}s  new {n_ok} skip {n_skip} "
                  f"nocup {n_nocup} nophase {n_bad}", flush=True)
    print(f"\nPROCESSING CHECK: trials {len(trials)}, new {n_ok}, already cached {n_skip}, "
          f"no-cup {n_nocup}, no-phase {n_bad}, files {len(list(CACHE.glob('*.npz')))}", flush=True)
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
