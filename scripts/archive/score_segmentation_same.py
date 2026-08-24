"""Same-segmenter phase metric, COHORT-WIDE: MMC-cup vs OMC-cup, identical segmenter both sides
(segment_cup_only -> refine_grasp_with_pose -> to_murphy_phases). Isolates the CUP TRACK (any bias in
the segmentation RULE cancels, because both sides run the same code). This is the P07/P08 "17ms / 1
frame" metric extended to all 11 participants -- distinct from score_segmentation_v2.py, which scores
our rule against AutoMQ's INDEPENDENT phase definitions (a harder, definitional comparison).

OMC cup = QTM cluster_cup markers (_omc_cup), shifted onto the video timeline by the wrist-speed lag.
Saves per-(trial,phase) CSV, prints per-phase median |error| in frames. Data only.
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
from score_vs_automq import FPS, COHORT_PARTS
GRID = R._GRID_JOINTS
PHASES = ["rest_pre", "reaching", "forward_transport", "drinking", "back_transport",
          "returning", "rest_post"]


def phases_of(cup, hand, nose):
    try:
        seg = SEG.segment_cup_only(R._fill(cup), fps=FPS)
        seg = SEG.refine_grasp_with_pose(seg, R._fill(cup), R._fill(hand),
                                         None if nose is None else R._fill(nose), fps=FPS)
        ph = SEG.to_murphy_phases(seg, R._fill(hand), R._fill(cup), fps=FPS)
        return {nm: (s, e) for nm, s, e in ph}
    except Exception:
        return {}


def main():
    H.use_good_cams()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    print(f"cohort trials {len(trials)}", flush=True)
    rows = []; t0 = time.time(); n = 0; n_ok = 0; n_skip = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        if not pat.search(trial):
            continue
        nfr = t["mmc"].shape[0]
        calib = R._calib(part)
        mcup = R._cup_v3(part, trial, calib, nfr)
        ocup = R._omc_cup(part, trial, nfr)
        if not (np.isfinite(mcup).any() and np.isfinite(ocup).any()):
            n_skip += 1; continue
        omc = H._load_omc(part, trial, nfr)
        wr = f"{side}_wrist"
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        # MMC side (video timeline)
        mcup = R._smooth_joint(mcup)
        mhand = R._smooth_joint(t["mmc"][:, GRID.index(wr)])
        mnose = R._smooth_joint(t["mmc"][:, GRID.index("nose")]) if "nose" in GRID else None
        # OMC side, shifted onto the same timeline
        ocup = R._shift(ocup, lag)
        ohand = R._shift(omc[wr], lag)
        onose = R._shift(omc["nose"], lag) if "nose" in omc else None
        pm = phases_of(mcup, mhand, mnose)
        po = phases_of(ocup, ohand, onose)
        if not po or not pm:
            n_skip += 1; continue
        n_ok += 1
        for p in PHASES:
            if p not in po:
                continue
            op = pm.get(p)
            rows.append(dict(part=part, trial=trial, phase=p, present=int(op is not None),
                             onset_fr=(op[0] - po[p][0]) if op else np.nan,
                             offset_fr=(op[1] - po[p][1]) if op else np.nan))
        n += 1
        if n % 60 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s ok={n_ok}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "out/scoring/segmentation_same_cup.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nPROCESSING CHECK: trials scored {n_ok} ({n_skip} skipped no-cup/failed), rows {len(df)}",
          flush=True)

    fr2ms = 1000.0 / FPS
    print(f"\n== SAME-SEGMENTER  MMC-cup vs OMC-cup  (median |error|, frames @60fps / ms) ==")
    print(f"  {'phase':<18}{'n':>5}{'miss':>8}{'|onset|':>16}{'|offset|':>16}")
    for p in PHASES:
        gp = df[df.phase == p]
        if not len(gp):
            continue
        det = gp[gp.present == 1]; nmiss = int((gp.present == 0).sum())
        def fmt(c):
            v = det[c].abs().dropna()
            return f"{v.median():4.1f}f/{v.median()*fr2ms:5.0f}ms" if len(v) else f"{'--':>12}"
        print(f"  {p:<18}{len(gp):>5}{f'{nmiss}/{len(gp)}':>8}{fmt('onset_fr'):>16}{fmt('offset_fr'):>16}")
    det = df[df.present == 1]
    for c, lab in [("onset_fr", "ALL |onset|"), ("offset_fr", "ALL |offset|")]:
        v = det[c].abs().dropna()
        print(f"  {lab:<18}{'':5}{'':8}{v.median():4.1f} frames / {v.median()*fr2ms:.0f} ms   (n={len(v)})")
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
