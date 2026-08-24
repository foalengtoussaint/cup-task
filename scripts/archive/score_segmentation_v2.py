"""Full-cohort segmentation validation vs AutoMQ OMC phases, using the REAL pipeline segmenter
(segment_cup_only -> refine_grasp_with_pose [wrist->cup plateau] -> to_murphy_phases).

The shipped score_segmentation_vs_automq.py scores segment_cup_only ALONE (no plateau refine) and only
the drinking window -- a stripped-down segmenter. This scores the plateau version AND the cup-only
baseline, on EVERY Murphy phase AutoMQ provides, in ms AND frames (60fps -> 16.7ms/frame).

Truth = AutoMQ phases mapped to the video timeline (same wrist-speed lag as the kinematics scorer).
P07/P08 have no cached cup tracks -> excluded (~9 participants). Saves per-(trial,variant,phase) CSV,
prints a processing check + per-phase median |error|. Data only.
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
from score_vs_automq import load_automq, automq_phases_to_video, automq_part, _win, FPS, COHORT_PARTS
GRID = R._GRID_JOINTS
# AutoMQ provides these 5 (no rest_pre/rest_post)
PHASES = ["reaching", "forward_transport", "drinking", "back_transport", "returning"]


def _fill(x):
    return R._fill(x)


def phases_of(cup, hand, nose, refine):
    try:
        seg = SEG.segment_cup_only(_fill(cup), fps=FPS)
        if refine:
            seg = SEG.refine_grasp_with_pose(seg, _fill(cup), _fill(hand),
                                             None if nose is None else _fill(nose), fps=FPS)
        ph = SEG.to_murphy_phases(seg, _fill(hand), _fill(cup), fps=FPS)
        return {nm: (s, e) for nm, s, e in ph}     # frames
    except Exception:
        return {}


def main():
    H.use_good_cams()
    amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in COHORT_PARTS]
    print(f"AutoMQ rows {len(amq)}; cohort trials {len(trials)}", flush=True)
    rows = []; n_seg = n_nocup = 0; t0 = time.time(); n = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        nfr = t["mmc"].shape[0]
        calib = R._calib(part)
        cup = R._cup_v3(part, trial, calib, nfr)
        if not np.isfinite(cup).any():
            n_nocup += 1; continue
        cup = R._smooth_joint(cup)
        hand = R._smooth_joint(t["mmc"][:, GRID.index(f"{side}_wrist")])
        nose = R._smooth_joint(t["mmc"][:, GRID.index("nose")]) if "nose" in GRID else None
        omc = H._load_omc(part, trial, nfr)
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph_amq = automq_phases_to_video(rec["phases"], lag, nfr)
        if not ph_amq:
            continue
        truth = {p: _win(ph_amq, p) for p in PHASES}
        n_seg += 1
        for vname, refine in [("cup_only", False), ("plateau", True)]:
            ours = phases_of(cup, hand, nose, refine)
            for p in PHASES:
                if truth[p] is None:
                    continue
                op = ours.get(p)
                rows.append(dict(part=part, trial=trial, variant=vname, phase=p,
                                 present=int(op is not None),
                                 onset_ms=(op[0] - truth[p][0]) / FPS * 1000 if op else np.nan,
                                 offset_ms=(op[1] - truth[p][1]) / FPS * 1000 if op else np.nan,
                                 dur_ms=(((op[1]-op[0])-(truth[p][1]-truth[p][0]))/FPS*1000) if op else np.nan))
        n += 1
        if n % 60 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s segmented={n_seg}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "out/scoring/segmentation_v2_vs_automq.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nPROCESSING CHECK: trials segmented {n_seg} ({n_nocup} no-cup skipped), rows {len(df)}", flush=True)

    for vname in ("cup_only", "plateau"):
        g = df[df.variant == vname]
        print(f"\n== {vname.upper()} vs AutoMQ -- median |error| (ms / frames @60fps) ==")
        print(f"  {'phase':<18}{'n':>5}{'miss':>7}{'|onset|':>16}{'|offset|':>16}{'|dur|':>14}")
        for p in PHASES:
            gp = g[g.phase == p]
            npres = len(gp); nmiss = int((gp['present'] == 0).sum())
            det = gp[gp.present == 1]
            def fmt(col):
                v = det[col].abs().dropna()
                return f"{v.median():6.0f}ms/{v.median()/ (1000/FPS):4.1f}f" if len(v) else f"{'--':>12}"
            print(f"  {p:<18}{npres:>5}{f'{nmiss}/{npres}':>7}"
                  f"{fmt('onset_ms'):>16}{fmt('offset_ms'):>16}{fmt('dur_ms'):>14}")
        det = g[g.present == 1]
        for col, lab in [("onset_ms", "ALL |onset|"), ("offset_ms", "ALL |offset|")]:
            v = det[col].abs().dropna()
            print(f"  {lab:<18}{'':5}{'':7}{v.median():6.0f}ms / {v.median()/(1000/FPS):.1f} frames"
                  f"   (n={len(v)}, total miss {int((g['present']==0).sum())}/{len(g)})")
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
