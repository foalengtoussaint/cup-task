"""Segmentation validation: our v3 cup-only segmenter's drink-phase boundaries vs AutoMQ OMC phases.

For every trial with a cached UETrack cup track, run segment_cup_only on the markerless 3D cup, map
AutoMQ's OMC phases onto the video timeline (same wrist-speed lag as the kinematics scorer), and compare
the DRINKING (dwell) window + the transport boundaries. Reports onset/offset/duration bias + |err| in ms.

P07/P08 have no cup tracks cached (6 each) -> excluded; ~588 trials across 9 participants.
    python scripts/score_segmentation_vs_automq.py  ->  out/automq/segmentation_vs_automq.csv
"""
from __future__ import annotations
import sys, re, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from cup_task.segment import segment_cup_only, _intervals
from score_vs_automq import load_automq, automq_phases_to_video, automq_part, _win, FPS
GRID = R._GRID_JOINTS


def _phase_win(intervals, name):
    """(start,end) of the named phase from a list of (name,s,e)."""
    sel = [(s, e) for (n, s, e) in intervals if n == name]
    return (min(s for s, _ in sel), max(e for _, e in sel)) if sel else None


def main():
    H.use_good_cams()
    amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    rows = []
    n_seg = n_nocup = 0
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        rec = amq.get((automq_part(t["part"]), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        n = t["mmc"].shape[0]
        calib = R._calib(t["part"])
        cup = R._cup_v3(t["part"], t["trial"], calib, n)         # markerless 3D cup (UETrack)
        if not np.isfinite(cup).any():
            n_nocup += 1
            continue
        seg = segment_cup_only(cup, fps=FPS)
        our = seg["intervals"]
        # AutoMQ phases -> video timeline (same lag path as the kinematics scorer)
        omc = H._load_omc(t["part"], t["trial"], n)
        wr = f"{t['side']}_wrist"
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        od = _win(ph, "drinking")                                 # AutoMQ drinking window (video frames)
        ud = _phase_win(our, "drinking")                          # our drinking window
        if od is None:
            continue
        n_seg += 1
        def ms(fr):
            return fr / FPS * 1000.0
        if ud is not None:
            rows.append({"part": t["part"], "trial": t["trial"], "detected": 1,
                         "onset_bias_ms": ms(ud[0] - od[0]),
                         "offset_bias_ms": ms(ud[1] - od[1]),
                         "dur_bias_ms": ms((ud[1] - ud[0]) - (od[1] - od[0])),
                         "our_dur_s": (ud[1] - ud[0]) / FPS, "omc_dur_s": (od[1] - od[0]) / FPS})
        else:
            rows.append({"part": t["part"], "trial": t["trial"], "detected": 0,
                         "onset_bias_ms": np.nan, "offset_bias_ms": np.nan, "dur_bias_ms": np.nan,
                         "our_dur_s": np.nan, "omc_dur_s": (od[1] - od[0]) / FPS})

    out = ROOT / "out" / "automq" / "segmentation_vs_automq.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    det = [r for r in rows if r["detected"]]
    print(f"\nsegmented {n_seg} trials ({n_nocup} no-cup-track skipped); drink detected {len(det)}/{n_seg} "
          f"({len(det)/max(n_seg,1):.0%})", flush=True)
    def stat(key):
        v = np.array([r[key] for r in det]); v = v[np.isfinite(v)]
        return f"bias {np.median(v):+7.0f}ms  |err| {np.median(np.abs(v)):6.0f}ms"
    print(f"  drink ONSET  : {stat('onset_bias_ms')}")
    print(f"  drink OFFSET : {stat('offset_bias_ms')}")
    print(f"  drink DURATION:{stat('dur_bias_ms')}")
    from scipy.stats import spearmanr
    a = np.array([r["omc_dur_s"] for r in det]); o = np.array([r["our_dur_s"] for r in det])
    mask = np.isfinite(a) & np.isfinite(o)
    print(f"  drink-duration rs (our vs OMC): {spearmanr(a[mask], o[mask]).correlation:+.2f}  n={mask.sum()}")
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
