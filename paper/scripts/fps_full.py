"""The whole cohort at 30 Hz, with the cup tracker actually re-run -- one row per trial per measure.

`fps_ablation.py` decimated the cup track, which assumes UETrack is rate-invariant; `fps_cup.py`
showed it is not (6.8 mm median divergence, 40 trials). This runs the honest version over every
trial: the tracker sees every second frame and follows twice the inter-frame displacement, exactly
as a 30 Hz camera would deliver.

Only the 30 Hz arm is computed. The 60 Hz arm already exists in
`out/scoring/score_own_phases_anat12.csv` and is what the paper reports, so comparing against it
answers the question actually being asked --- what would change if we had recorded at 30 Hz --- rather
than comparing two fresh runs that would both differ from the published numbers.

Everything downstream of the tracker is re-run on the coarser grid: the BA trajectory is decimated,
SmoothNet re-solved on it, the segmenter run at fps=30, and every measure computed at 30 Hz with the
low-pass cutoffs taken from the actual rate rather than a module constant.

TWO SMOOTHING VARIANTS are emitted, because they are not the same experiment. `mmc30` re-solves
SmoothNet on the 30 Hz series, where its fixed 32-sample window covers 1.07 s instead of 0.53 s;
`mmc30r` keeps the real-time support constant by interpolating to 60 Hz, smoothing, and decimating.
The difference is large and is NOT a property of the capture rate: measured separately, the coarser
sampling costs 0.2% of peak velocity while the doubled smoothing support costs 7.4%.

    python paper/scripts/fps_full.py                      # all trials, one process
    python paper/scripts/fps_full.py --shard 0 --nshards 3   # one of three parallel workers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

import cv2                                          # noqa: E402
import compare_pose_omc_delta as H                   # noqa: E402
import results_v3_delta as R                         # noqa: E402
import score_own_phases as SOP                       # noqa: E402
from pipeline import consensus                       # noqa: E402
from pipeline.triangulate import kf_fill_gaps        # noqa: E402
from seg_sequential import segment_sequential        # noqa: E402
from fps_ablation import MEAS, _measures_at, declined   # noqa: E402

SEG = ROOT / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs_ship")
SEED26X = ROOT / "cache" / "cup_seed26x"
STEP = 2                       # 60 Hz -> 30 Hz
FPS30 = 30.0


def _vids(part, trial, calib):
    d = H.DELTA / part / "staged"
    out = {}
    for c in sorted(calib, key=lambda s: int(s.split("_")[1])):
        p = d / f"delta_{part}_{trial}.{int(c.split('_')[1])}.mp4"
        if p.exists():
            out[c] = p
    return out


def track30(vids, seed_frame, seed_boxes, uet_cls):
    """Track the cup on every SECOND frame from the seed onward. Returns {frame: {cam: [x,y]}}."""
    cams = [c for c in sorted(vids) if c in seed_boxes]
    if len(cams) < 2:
        return None, 0
    tr = uet_cls(len(cams))
    ci = {c: i for i, c in enumerate(cams)}
    caps = {c: cv2.VideoCapture(str(vids[c])) for c in cams}
    sf = seed_frame + (seed_frame % STEP)        # the 30 Hz stream holds only even frames
    got, f = {}, 0
    try:
        while True:
            keep = (f % STEP == 0) and f >= sf
            rgb = {}
            ok_any = False
            for c in cams:
                ok, im = caps[c].read()
                ok_any |= ok
                if ok and keep:
                    rgb[c] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if not ok_any:
                break
            if keep:
                if f == sf:
                    for c in cams:
                        if c in rgb:
                            tr.init(ci[c], rgb[c], seed_boxes[c])
                    got[f] = {c: [seed_boxes[c][0] + seed_boxes[c][2] / 2,
                                  seed_boxes[c][1] + seed_boxes[c][3] / 2]
                              for c in cams if c in rgb}
                else:
                    arr = [rgb.get(c) for c in cams]
                    out = tr.update(arr)
                    row = {}
                    for c in cams:
                        if arr[ci[c]] is not None and out[ci[c]] is not None:
                            xy = out[ci[c]]
                            row[c] = [xy[0] + xy[2] / 2, xy[1] + xy[3] / 2]
                    if row:
                        got[f] = row
            f += 1
    finally:
        for cp in caps.values():
            cp.release()
    return got, f


def cup3d(rows, calib, n_out):
    cup = np.full((n_out, 3), np.nan)
    ncam = np.zeros(n_out, int)
    for f, row in rows.items():
        i = f // STEP
        if i >= n_out:
            continue
        obs = {c: v for c, v in row.items() if v is not None}
        if len(obs) >= 2:
            X, kept, _ = consensus.consensus3(obs, calib)
            if X is not None:
                cup[i] = X
                ncam[i] = len(kept)
    cup[ncam < 3] = np.nan                       # the shipped three-camera floor
    return kf_fill_gaps(cup)


def smooth_rate_matched(P30, joints):
    """Smooth a 30 Hz trajectory with SmoothNet's window covering the SAME REAL TIME as at 60 Hz.

    The checkpoint has a fixed 32-sample window. At 60 Hz that is 0.53 s of support; feeding it 30 Hz
    samples doubles the support to 1.07 s and low-passes twice as hard in real terms, which flattens
    the velocity peak. Measured on 154 trials, that alone costs 7.4% of peak velocity, while the
    coarser sampling costs 0.2% -- so the naive port confounds an implementation choice with the
    capture rate. Here the 30 Hz series is interpolated onto a 60 Hz grid, smoothed, and decimated
    back, which is what a 30 Hz deployment would do with this checkpoint.
    """
    T30 = len(P30)
    T60 = (T30 - 1) * STEP + 1
    x30, x60 = np.arange(T30), np.arange(T60) / STEP
    near = np.clip(np.rint(x60).astype(int), 0, T30 - 1)
    up = {}
    for k, j in enumerate(joints):
        X = np.asarray(P30, float)[:, k]
        fin = np.isfinite(X).all(1)
        if fin.sum() < 2:
            up[j] = np.full((T60, 3), np.nan)
            continue
        o = np.empty((T60, 3))
        for c in range(3):
            o[:, c] = np.interp(x60, x30[fin], X[fin, c])
        o[~fin[near]] = np.nan          # a 60 Hz sample is real only where its 30 Hz neighbour was
        up[j] = o
    sm = R._smooth_pose(up)
    return {j: v[::STEP][:T30] for j, v in sm.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or str(ROOT / f"out/scoring/fps_full_{a.shard}of{a.nshards}.csv")

    from uetrack_wrap import UETrackBatch
    import gnn_train as GT
    H.use_good_cams()
    ba = R._ba_traj_cache()
    bad = declined()
    trials = {f"{t['part']}/{t['trial']}": t for t in GT.load_clean(need_reproj=False)}
    files = [f for f in sorted(SEG.glob("*.npz"))]
    files = [f for i, f in enumerate(files) if i % a.nshards == a.shard]
    print(f"shard {a.shard}/{a.nshards}: {len(files)} trials", flush=True)

    rows, t0, n_skip = [], time.time(), 0
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        part, trial, side = str(z["part"]), str(z["trial"]), str(z["side"])
        if (part, trial) in bad:
            continue
        t = trials.get(f"{part}/{trial}")
        P = ba.get(f"{part}/{trial}")
        sf = SEED26X / f"{part}__{trial}.json"
        if t is None or P is None or not sf.exists():
            n_skip += 1; continue
        sd = json.loads(sf.read_text())
        if not sd.get("boxes"):
            n_skip += 1; continue
        calib = H._load_calib_mm(part)
        vids = _vids(part, trial, calib)
        if len(vids) < 2:
            n_skip += 1; continue
        try:
            got, nfr = track30(vids, int(sd["frame"]), sd["boxes"], UETrackBatch)
            if got is None:
                n_skip += 1; continue
            n_out = int(np.ceil(nfr / STEP))
            cup = cup3d(got, calib, n_out)
            P30 = np.asarray(P, float)[::STEP]
            J = list(R._GRID_JOINTS)
            pose_n = R._smooth_pose({j: P30[:, m] for m, j in enumerate(J)})   # naive: window at 30
            pose_r = smooth_rate_matched(P30, J)                               # rate-matched window
            # The segmenter's channels are the shipped 60 Hz-smoothed cup/wrist/nose, decimated, so
            # they are rate-matched in BOTH arms and the only difference between them is the pose
            # the measures are computed from.
            wr, no = z["wrist_mmc"][::STEP], z["nose_mmc"][::STEP]
            m_ = min(len(cup), len(wr), len(no), len(pose_n[f"{side}_wrist"]))
            ph = segment_sequential(cup[:m_], wr[:m_], no[:m_], fps=FPS30)
            if not ph:
                n_skip += 1; continue
            vals = _measures_at(pose_n, ph, side, FPS30)
            vals_r = _measures_at(pose_r, ph, side, FPS30)
        except Exception as e:
            print(f"  {part}/{trial}: {type(e).__name__}: {e}", flush=True)
            n_skip += 1; continue
        for meas, _l in MEAS:
            rows.append(dict(part=part, trial=trial, arm=str(z["arm"]), measure=meas,
                             mmc30=vals.get(meas, np.nan),
                             mmc30r=vals_r.get(meas, np.nan)))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(files)}] {el:5.0f}s ({el/(i+1):.1f}s/trial)  "
                  f"kept {len(rows)//len(MEAS)} skip {n_skip}", flush=True)

    D = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(out, index=False)
    print(f"\nwrote {out}: {D.groupby(['part','trial']).ngroups} trials, {n_skip} skipped")
    print("DONE_FPS_FULL", flush=True)


if __name__ == "__main__":
    main()
