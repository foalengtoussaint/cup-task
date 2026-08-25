"""Score segmenter variants by the ONLY comparison that matters here: same rule, OMC input vs MMC input.

A boundary bias shared by both sides cancels when MMC measures are compared against OMC measures, so
vs-AutoMQ error is the wrong target. What survives is the OMC-vs-MMC DISAGREEMENT per boundary, which
for the sequential segmenter is median ~0 with a heavy tail (grasp p90 61f, release p90 91f).

Runs every variant on both sources from cache/seg_inputs (no re-triangulation, no C3D parsing) and
reports per boundary: median / |median| / IQR / p90 / frac>5f / frac>15f, plus the vs-AutoMQ medians
for reference and the anchored segmenter's QA flag rates. Per-trial values are saved.

    python scripts/score_seg_boundaries.py   -> out/scoring/seg_boundaries.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cache_seg_inputs as CSI                                    # noqa: E402
from seg_anchored import segment_anchored_full                     # noqa: E402
from seg_sequential import segment_sequential                      # noqa: E402
from seg_anchor_min import segment_anchor_min                       # noqa: E402
from pipeline.triangulate import fill_cup_from_wrist, kf_fill_gaps  # noqa: E402

# SmoothNet is a DENOISER, not a gap filler -- pose_smooth.smooth_track restores X=None on every
# frame that was missing on input, so the cached MMC cup still has 21% NaN / 1.33s longest gap. The
# segmenter's _interp_nan_xyz then draws a straight line through the occlusion. kf_rts_smooth is the
# filler that already exists for exactly this (constant-velocity coast + RTS backward correction).
# mmc_c3 = the SAME pipeline cup with every frame below the >=3-agreeing-camera floor dropped.
# A 2-camera cup is DROPPED because its errors are undetectable, not because it carries nothing:
# measured in paper/scripts/two_camera_cup.py it is 60mm at the median, but 21% of those frames
# are >100mm out against 0.02% at >=3 cams, and two views give no redundancy to say which. (The
# older '>1m errors' note here did not reproduce: 0.00% of 2-cam frames exceed 1m.) On the bad
# trials this is most of the track. mmc_c3_kf then fills those holes with the KF.
# _wr  = wrist proxy on BOTH channels (wrong: kills wrist->cup, so grasp/release degrade)
# _wr2 = wrist proxy on the cup->MOUTH channel only; grasp/release keep the observed-only cup.
#        SUPERSEDED 2026-08-20: segment_sequential now drives cup->mouth from the WRIST ITSELF by
#        default (cm_source="wrist"), so `mmc_c3kf` IS the shipped configuration and _wr2 is kept
#        only as the comparison against the offset-fitted proxy.
SOURCES = ["omc", "mmc", "mmc_c3kf", "mmc_c3kf_wr", "mmc_c3kf_wr2"]
NCAMS = ROOT / "cache" / (__import__("os").environ.get("OT_NCAMS_DIR") or "cup_ncams")

# boundary name -> (phase, which edge)
BOUNDS = [("reach onset", "reaching", 0), ("grasp", "reaching", 1),
          ("drink onset", "drinking", 0), ("drink offset", "drinking", 1),
          ("release", "back_transport", 1), ("settle", "returning", 1)]


def edges(seg):
    d = {nm: (s, e) for nm, s, e in seg}
    return {lab: (d[ph][i] if ph in d else np.nan) for lab, ph, i in BOUNDS}


def main():
    recs = CSI.load_all()
    print(f"{len(recs)} cached trials", flush=True)
    rows = []; flagrows = []; t0 = time.time()
    for k, r in enumerate(recs):
        amq = {}
        for lab, ph, i in BOUNDS:
            w = r[f"amq_{ph}"]
            amq[lab] = float(w[i]) if w[0] >= 0 else np.nan
        row = dict(part=r["part"], trial=r["trial"], arm=r["arm"])
        for src in SOURCES:
            base = "mmc" if src.startswith("mmc") else "omc"
            cup_mouth = None                      # only set for the _wr2 variant
            cup, hand, nose = r[f"cup_{base}"], r[f"wrist_{base}"], r[f"nose_{base}"]
            if src.startswith("mmc_c3kf"):
                z = np.load(NCAMS / f"{r['part']}__{r['trial']}.npz")
                cup = np.asarray(cup, float).copy()
                cup[z["n_cams"][:len(cup)] < 3] = np.nan
                cup = kf_fill_gaps(cup)
                if src.endswith("_wr"):
                    cup, _ = fill_cup_from_wrist(cup, r[f"wrist_{base}"])
                elif src.endswith("_wr2"):
                    cup_mouth, _ = fill_cup_from_wrist(cup, r[f"wrist_{base}"])
            seq = edges(segment_sequential(cup, hand, nose, cup_mouth_xyz=cup_mouth))
            anc_seg, fl = segment_anchored_full(cup, hand, nose)
            anc = edges(anc_seg)
            mini = edges(segment_anchor_min(cup, hand, nose))
            for lab, _, _ in BOUNDS:
                row[f"seq_{src}_{lab}"] = seq[lab]
                row[f"anc_{src}_{lab}"] = anc[lab]
                row[f"min_{src}_{lab}"] = mini[lab]
                row[f"amq_{lab}"] = amq[lab]
            flagrows.append(dict(src=src, part=r["part"], trial=r["trial"], **fl))
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"  [{k+1}/{len(recs)}] {time.time()-t0:4.0f}s", flush=True)

    D = pd.DataFrame(rows); F = pd.DataFrame(flagrows)
    D.to_csv(ROOT / "out/scoring/seg_boundaries.csv", index=False)
    F.to_csv(ROOT / "out/scoring/seg_boundaries_flags.csv", index=False)
    n_seq = int(D[[f"seq_omc_{l}" for l, _, _ in BOUNDS]].notna().all(1).sum())
    n_anc = int(D[[f"anc_omc_{l}" for l, _, _ in BOUNDS]].notna().all(1).sum())
    print(f"\nPROCESSING CHECK: trials {len(D)}, all-boundaries-present sequential {n_seq}, "
          f"anchored {n_anc}", flush=True)

    for tag, name in (("seq", "SEQUENTIAL (current)"), ("min", "ANCHOR-MIN (run-selection + shared hold ONLY)"), ("anc", "ANCHORED (all 9 changes)")):
        for msrc in [s for s in SOURCES if s != "omc"]:
            print(f"\n== {name}: {msrc.upper()} input minus OMC input (frames @60Hz) ==")
            print(f"{'boundary':16}{'n':>5}{'median':>9}{'|median|':>10}{'IQR':>15}{'p90|.|':>9}"
                  f"{'>5f':>7}{'>15f':>7}{'  |med| vs AutoMQ (omc/x)':>26}")
            for lab, _, _ in BOUNDS:
                d = (D[f"{tag}_{msrc}_{lab}"] - D[f"{tag}_omc_{lab}"]).dropna()
                if not len(d):
                    continue
                q1, q3 = d.quantile([.25, .75])
                ao = (D[f"{tag}_omc_{lab}"] - D[f"amq_{lab}"]).abs().median()
                am = (D[f"{tag}_{msrc}_{lab}"] - D[f"amq_{lab}"]).abs().median()
                print(f"{lab:16}{len(d):>5}{d.median():>9.1f}{d.abs().median():>10.1f}"
                      f"{f'[{q1:.0f}, {q3:.0f}]':>15}{d.abs().quantile(.9):>9.1f}"
                      f"{(d.abs()>5).mean():>7.2f}{(d.abs()>15).mean():>7.2f}"
                      f"{f'{ao:.0f} / {am:.0f}':>26}", flush=True)

    print("\nANCHORED QA flag rate (fraction of trials):")
    cols = [c for c in F.columns if c not in ("src", "part", "trial")]
    print(F.groupby("src")[cols].mean().round(3).T.to_string())
    print("\nwrote out/scoring/seg_boundaries.csv + _flags.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
