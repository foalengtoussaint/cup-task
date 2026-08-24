"""Same-segmenter OMC-cup vs MMC-cup, using the SEQUENTIAL segmenter (isolates the markerless
tracking cost, no AutoMQ definitional gap). Compare to the staged same-segmenter (3 frames / 50ms).
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
from seg_sequential import segment_sequential
from score_vs_automq import COHORT_PARTS, FPS
GRID = R._GRID_JOINTS
PHASES = ["rest_pre", "reaching", "forward_transport", "drinking", "back_transport",
          "returning", "rest_post"]


def _seq(cup, hand, nose):
    return {nm: (s, e) for nm, s, e in segment_sequential(cup, hand, nose)}


def _staged(cup, hand, nose):
    sg = SEG.segment_cup_only(R._fill(cup), fps=FPS)
    sg = SEG.refine_grasp_with_pose(sg, R._fill(cup), R._fill(hand), None, fps=FPS)
    return {nm: (s, e) for nm, s, e in SEG.to_murphy_phases(sg, R._fill(hand), R._fill(cup), fps=FPS)}


def main():
    H.use_good_cams()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    print(f"cohort {len(trials)}", flush=True)
    rows = []; n = 0; t0 = time.time()
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        nfr = t["mmc"].shape[0]; wr = f"{side}_wrist"
        omc = H._load_omc(part, trial, nfr)
        if wr not in omc or not np.isfinite(omc[wr]).any() or "nose" not in omc:
            continue
        ocup = R._omc_cup(part, trial, nfr); mcup = R._cup_v3(part, trial, R._calib(part), nfr)
        if not (np.isfinite(ocup).any() and np.isfinite(mcup).any()):
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        # OMC side (shift onto video timeline), MMC side (already video timeline)
        o_cup, o_hand, o_nose = R._shift(ocup, lag), R._shift(omc[wr], lag), R._shift(omc["nose"], lag)
        m_cup = R._smooth_joint(mcup)
        m_hand = R._smooth_joint(t["mmc"][:, GRID.index(wr)])
        m_nose = R._smooth_joint(t["mmc"][:, GRID.index("nose")])
        for meth, fn in [("sequential", _seq), ("staged", _staged)]:
            po = fn(o_cup, o_hand, o_nose); pm = fn(m_cup, m_hand, m_nose)
            for p in PHASES:
                if p not in po:
                    continue
                w = pm.get(p)
                rows.append(dict(part=part, trial=trial, method=meth, phase=p, present=int(w is not None),
                                 onset_f=(w[0] - po[p][0]) if w else np.nan,
                                 offset_f=(w[1] - po[p][1]) if w else np.nan))
        n += 1
        if n % 80 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "out/scoring/seg_sequential_same_cup.csv"
    df.to_csv(out, index=False)
    print(f"\nPROCESSING CHECK: trials {n}, rows {len(df)}", flush=True)
    for meth in ("sequential", "staged"):
        g = df[df.method == meth]
        print(f"\n== {meth.upper()}  OMC-cup vs MMC-cup  (median |err| frames / miss) ==")
        print(f"  {'phase':<18}{'n':>5}{'miss':>9}{'|onset|f':>10}{'|offset|f':>10}")
        for p in PHASES:
            gp = g[g.phase == p]; det = gp[gp.present == 1]
            if not len(gp):
                continue
            print(f"  {p:<18}{len(gp):>5}{f'{int((gp.present==0).sum())}/{len(gp)}':>9}"
                  f"{det.onset_f.abs().median():>10.1f}{det.offset_f.abs().median():>10.1f}")
        det = g[g.present == 1]
        print(f"  {'ALL':<18}{'':5}{f'{int((g.present==0).sum())}/{len(g)}':>9}"
              f"{det.onset_f.abs().median():>10.1f}{det.offset_f.abs().median():>10.1f}")
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
