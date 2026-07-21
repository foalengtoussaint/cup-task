"""v2 pipeline results on the NEW dataset (DELTA), n=18 OMC-truth trials.

Two deliverables, v1 (raw) vs v2 (SmoothNet pose + detect-once cup), each checked against OMC:

  SEGMENTATION -- segment the drink phases and compare the DRINK DWELL to OMC's own dwell.
      cup source v1 = every-frame YOLO triangulation (fuse_3d path)
      cup source v2 = detect-once UETrack + greedy consensus (cup_track, from cached tracker points)
      OMC dwell     = from the OMC cup speed (same segmenter, mocap cup) = the truth
      -> reports dwell error (MMC - OMC) in ms, per trial + median |error|.

  MURPHY -- the position-derived measures on the wrist, v1 (raw pose) vs v2 (SmoothNet pose),
      each vs OMC, phases held fixed (OMC-cup phases so the pose is the only variable).
      -> reports each measure's |MMC-OMC| error, v1 vs v2, so we can see if de-jitter helps.

The 18 trials: P07/P08/P13 trial_10-15 (side L for P07/P13, R for P08).

    python scripts/results_v2_delta.py --what both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import compare_pose_omc_delta as H
from cup_task import segment, cup_track, pose_smooth
from cup_task.score import compute_position_measures, MurphyPositionMeasures

TRIALS = {
    "P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
    "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right"),
    "P13": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
}
FPS = H.VIDEO_FPS
TRACKS = ROOT / "cache" / "tracks_uetrack"


def _shift(v, lag):
    out = np.full_like(v, np.nan)
    if lag >= 0:
        out[lag:] = v[:len(v) - lag] if lag else v
    else:
        out[:lag] = v[-lag:]
    return out


def _smooth_joint(xyz):
    """Run one (T,3) track through the v2 pose_smooth stage."""
    tr = [{"frame": f, "X": (None if not np.isfinite(p).all() else [float(v) for v in p])}
          for f, p in enumerate(np.asarray(xyz, float))]
    out = pose_smooth.smooth_track(tr)
    return np.array([o["X"] if o["X"] is not None else [np.nan] * 3 for o in out])


def _dwell(xyz, fps=FPS):
    """The ACTUAL pipeline dwell: segment.segment_cup_only -> the 'drinking' interval. Same segmenter
    on MMC cup and OMC cup so the comparison isolates the cup source. Returns dwell seconds (0 if none).
    NOT a hand-rolled speed proxy -- this is what phases_from_3d produces."""
    seg = segment.segment_cup_only(_fill(np.asarray(xyz, float)), fps=fps)
    drink = [(s, e) for nm, s, e in seg.get("intervals", []) if nm == "drinking"]
    return (drink[0][1] - drink[0][0]) / fps if drink else 0.0


def seg_results():
    print(f"\n{'='*74}\n=== SEGMENTATION — drink dwell vs OMC (n=18) ===\n{'='*74}", flush=True)
    print(f"{'trial':14} {'OMC dwell':>9} {'v1 dwell':>9} {'v2 dwell':>9}   "
          f"{'v1 err ms':>9} {'v2 err ms':>9}", flush=True)
    rows = []
    for part, (trials, side) in TRIALS.items():
        calib = H._load_calib_mm(part)
        for trial in trials:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            # sync on wrist
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            # v1 cup = every-frame triangulation (already in mmc? no -- mmc is pose; load cup separately)
            # cup v2 from cached UETrack tracker points
            cf = TRACKS / f"{part}__{trial}__uetrack__fs1.json"
            cup_v2 = np.array([t["X"] if t["X"] else [np.nan] * 3
                               for t in cup_track.track_cup_3d_from_cache(cf, calib)])
            # v1 cup: the raw per-frame yolo triangulation is the "yolo" boxes in the same cache
            import json
            rec = json.loads(cf.read_text())
            from cup_task import consensus
            prev = None; cup_v1 = []
            for f in range(n):
                row = rec.get(str(f), {})
                obs = {c: (v["yolo"][0] + v["yolo"][2] / 2, v["yolo"][1] + v["yolo"][3] / 2)
                       for c, v in row.items() if v.get("yolo") and c in calib}
                X, kept, _ = consensus.consensus3(obs, calib, prev=prev)
                if X is not None: prev = X
                cup_v1.append(X if X is not None else [np.nan] * 3)
            cup_v1 = np.array(cup_v1, float)
            # OMC cup
            omc_cup = _omc_cup(part, trial, n)
            omc_cup = _shift(omc_cup, lag)

            d_omc = _dwell(omc_cup)
            d_v1 = _dwell(cup_v1)
            d_v2 = _dwell(cup_v2)
            e1 = (d_v1 - d_omc) * 1000; e2 = (d_v2 - d_omc) * 1000
            tag = f"{part}_{trial.split('_')[1]}"
            print(f"{tag:14} {d_omc:8.2f}s {d_v1:8.2f}s {d_v2:8.2f}s   "
                  f"{e1:+8.0f} {e2:+8.0f}", flush=True)
            rows.append((d_omc, d_v1, d_v2, e1, e2))
    R = np.array(rows)
    print(f"{'-'*66}", flush=True)
    print(f"{'MEDIAN |err|':14} {np.median(R[:,0]):8.2f}s {'':9} {'':9}   "
          f"{np.median(np.abs(R[:,3])):8.0f} {np.median(np.abs(R[:,4])):8.0f}", flush=True)


def _omc_cup(part, trial, n):
    """OMC cup cluster centroid (mm), resampled to video fps, from the c3d."""
    import ezc3d
    c = ezc3d.c3d(str(H.DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]
    mk = [nm for nm in L if "cluster_cup" in nm.lower() or nm.lower().startswith("cup")]
    if not mk:
        return np.full((n, 3), np.nan)
    raw = np.mean([P[:3, L.index(m), :].T for m in mk], axis=0)
    grid = H._resample(raw, H.C3D_RATE, FPS)
    if len(grid) < n:
        grid = np.vstack([grid, np.full((n - len(grid), 3), np.nan)])
    return grid[:n]


def murphy_results():
    print(f"\n{'='*74}\n=== MURPHY position measures — v1 raw vs v2 SmoothNet, error vs OMC ===\n{'='*74}",
          flush=True)
    measures = ["total_movement_time", "peak_velocity", "number_of_movement_units",
                "max_trunk_displacement"]
    agg = {m: {"v1": [], "v2": []} for m in measures}
    for part, (trials, side) in TRIALS.items():
        for trial in trials:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            omc = {j: _shift(v, lag) for j, v in omc.items()}
            omc_cup = _shift(_omc_cup(part, trial, n), lag)

            wr = f"{side}_wrist"
            other = "right" if side == "left" else "left"
            # ONE shared 7-phase set for all three arms: segment the OMC cup, derive reaching/
            # returning from the OMC hand. Pose source is then the only variable in the measures.
            cup_seg = segment.segment_cup_only(_fill(omc_cup), fps=FPS)
            try:
                phases = segment.to_murphy_phases(cup_seg, _fill(omc[wr]), _fill(omc_cup), fps=FPS)
            except Exception as e:
                print(f"  {part} {trial}: phase ERR {e}", flush=True)
                continue

            trunk = (mmc[f"{side}_shoulder"] + mmc[f"{other}_shoulder"]) / 2
            trunk_o = (omc[f"{side}_shoulder"] + omc[f"{other}_shoulder"]) / 2

            def measure(hand, trunk_xyz):
                try:
                    return compute_position_measures(hand, trunk_xyz, phases, side, fps=FPS)
                except Exception:
                    return None
            m_v1 = measure(mmc[wr], trunk)
            m_v2 = measure(_smooth_joint(mmc[wr]), trunk)
            m_omc = measure(omc[wr], trunk_o)
            if not (m_v1 and m_v2 and m_omc):
                continue
            for m in measures:
                o = getattr(m_omc, m)
                if o is None or (isinstance(o, float) and not np.isfinite(o)):
                    continue
                agg[m]["v1"].append(abs(getattr(m_v1, m) - o))
                agg[m]["v2"].append(abs(getattr(m_v2, m) - o))

    print(f"{'measure':28} {'v1 |err|':>10} {'v2 |err|':>10}  {'n':>3}", flush=True)
    for m in measures:
        v1 = np.median(agg[m]["v1"]) if agg[m]["v1"] else np.nan
        v2 = np.median(agg[m]["v2"]) if agg[m]["v2"] else np.nan
        print(f"{m:28} {v1:10.3f} {v2:10.3f}  {len(agg[m]['v1']):3d}", flush=True)


def _fill(xyz):
    x = np.asarray(xyz, float).copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    return x


def _to_murphy(cup_seg, n):
    """Map the cup-only phases into the 7-name Murphy set score.py expects. Minimal: use the
    reaching/drinking/returning the segmenter emits; pad rest."""
    iv = cup_seg.get("intervals", [])
    if not iv:
        return None
    # segment_cup_only emits rest/forward/drinking/back/rest -> rename to Murphy phase names
    rename = {"forward": "reaching", "forward_transport": "forward_transport",
              "drinking": "drinking", "back": "returning", "back_transport": "returning",
              "rest": "rest_pre", "reaching": "reaching"}
    out = []
    for nm, s, e in iv:
        out.append((rename.get(nm, nm), int(s), int(e)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["seg", "murphy", "both"], default="both")
    a = ap.parse_args(argv)
    H.use_good_cams()
    if a.what in ("seg", "both"):
        seg_results()
    if a.what in ("murphy", "both"):
        murphy_results()


if __name__ == "__main__":
    main()
