"""30 Hz vs 60 Hz with the CUP TRACKER ACTUALLY RE-RUN, not decimated.

`fps_ablation.py` decimates every markerless channel, which is exact for the pose (YOLO runs per
frame) and honest for SmoothNet (re-run on the coarser grid) but WRONG for the cup: UETrack is
recursive, so its state at frame f depends on every frame before it. A 30 Hz capture makes the
tracker follow twice the inter-frame displacement from a template updated half as often; the result
is a different track, not a subsample of the 60 Hz one. Decimating the cup silently assumes the
tracker is rate-invariant, which is the thing under test.

Here both rates are tracked in ONE decode pass, from the same seed, with the same batch size (so the
batch-dependence of cuDNN cannot masquerade as a rate effect):

    tracker60.update(frame)          every frame
    tracker30.update(frame)          every second frame

and then, per rate, the 3D cup goes through the same >=3-camera floor and Kalman fill as the shipped
`mmc_c3kf`, the same segmenter (at its own fps), and the same measures. The pose side is decimated
from the BA cache in both arms, so the ONLY difference between the two columns is the cup.

Cost is ~22 s per trial (1.5x a 60 Hz front end), so this runs on a spread subset rather than all 750.

    python paper/scripts/fps_cup.py --trials 40
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

import cv2                                          # noqa: E402
import compare_pose_omc_delta as H                   # noqa: E402
import results_v3_delta as R                         # noqa: E402
import score_own_phases as SOP                       # noqa: E402
import score_vs_automq as S                          # noqa: E402
from pipeline import consensus                       # noqa: E402
from pipeline.triangulate import kf_fill_gaps        # noqa: E402
from seg_sequential import segment_sequential        # noqa: E402
from fps_ablation import MEAS, _measures_at, declined   # noqa: E402

SEG = ROOT / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs_ship")
SEED26X = ROOT / "cache" / "cup_seed26x"
# reuse score_seg_boundaries' own mapping so Table IV and this table name the same six instants
from score_seg_boundaries import BOUNDS as _BMAP, edges as _edges   # noqa: E402
BOUNDS = [lab for lab, _p, _i in _BMAP]


def _vids(part, trial, calib):
    d = H.DELTA / part / "staged"
    out = {}
    for c in sorted(calib, key=lambda s: int(s.split("_")[1])):
        p = d / f"delta_{part}_{trial}.{int(c.split('_')[1])}.mp4"
        if p.exists():
            out[c] = p
    return out


def track_both(vids, calib, seed_frame, seed_boxes, uet_cls):
    """One decode pass, two trackers: every frame and every second frame. Returns {rate: {f: {cam: xy}}}."""
    cams = [c for c in sorted(vids) if c in seed_boxes]
    if len(cams) < 2:
        return None
    t60, t30 = uet_cls(len(cams)), uet_cls(len(cams))
    ci = {c: i for i, c in enumerate(cams)}
    caps = {c: cv2.VideoCapture(str(vids[c])) for c in cams}
    seed30 = seed_frame + (seed_frame % 2)          # the 30 Hz stream only contains even frames
    got = {60: {}, 30: {}}
    try:
        f = 0
        while True:
            rgb = {}
            ok_any = False
            for c in cams:
                ok, im = caps[c].read()
                if ok:
                    rgb[c] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    ok_any = True
            if not ok_any:
                break
            for rate, tr, sf in ((60, t60, seed_frame), (30, t30, seed30)):
                if rate == 30 and f % 2:
                    continue
                if f < sf:
                    continue
                if f == sf:
                    for c in cams:
                        if c in rgb:
                            tr.init(ci[c], rgb[c], seed_boxes[c])
                    b = {c: seed_boxes[c] for c in cams if c in rgb}
                    got[rate][f] = {c: [v[0] + v[2] / 2, v[1] + v[3] / 2] for c, v in b.items()}
                    continue
                arr = [rgb.get(c) for c in cams]
                out = tr.update(arr)
                row = {}
                for c in cams:
                    if arr[ci[c]] is not None and out[ci[c]] is not None:
                        xy = out[ci[c]]
                        row[c] = [xy[0] + xy[2] / 2, xy[1] + xy[3] / 2]
                if row:
                    got[rate][f] = row
            f += 1
    finally:
        for cp in caps.values():
            cp.release()
    return got, cams, f


def cup3d(rows, calib, n_out, step):
    """Per-frame >=3-camera gated consensus, then the shipped floor and Kalman fill."""
    cup = np.full((n_out, 3), np.nan)
    ncam = np.zeros(n_out, int)
    for f, row in rows.items():
        i = f // step
        if i >= n_out:
            continue
        obs = {c: v for c, v in row.items() if v is not None}
        if len(obs) >= 2:
            X, kept, _ = consensus.consensus3(obs, calib)
            if X is not None:
                cup[i] = X
                ncam[i] = len(kept)
    cup[ncam < 3] = np.nan                      # same three-camera floor as `mmc_c3kf`
    return kf_fill_gaps(cup), ncam


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "paper" / "table7_fps_cup"))
    a = ap.parse_args(argv)

    from uetrack_wrap import UETrackBatch
    H.use_good_cams()
    import gnn_train as GT
    ba = R._ba_traj_cache()
    bad = declined()
    trials = {f"{t['part']}/{t['trial']}": t for t in GT.load_clean(need_reproj=False)}
    files = [f for f in sorted(SEG.glob("*.npz"))]
    keep = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if (str(z["part"]), str(z["trial"])) not in bad:
            keep.append(f)
    sel = keep[::max(1, len(keep) // a.trials)][:a.trials]
    print(f"fps_cup: {len(sel)} trials of {len(keep)} eligible, re-tracking at both rates",
          flush=True)

    rows, brows, crows, t0 = [], [], [], time.time()
    for i, f in enumerate(sel):
        z = np.load(f, allow_pickle=True)
        part, trial, side = str(z["part"]), str(z["trial"]), str(z["side"])
        t = trials.get(f"{part}/{trial}")
        P = ba.get(f"{part}/{trial}")
        sf = SEED26X / f"{part}__{trial}.json"
        if t is None or P is None or not sf.exists():
            continue
        sd = json.loads(sf.read_text())
        if not sd.get("boxes"):
            continue
        calib = H._load_calib_mm(part)
        vids = _vids(part, trial, calib)
        if len(vids) < 2:
            continue
        try:
            got, cams, nfr = track_both(vids, calib, int(sd["frame"]), sd["boxes"], UETrackBatch)
        except Exception as e:
            print(f"  {part}/{trial}: track failed {type(e).__name__}: {e}", flush=True)
            continue
        P = np.asarray(P, float)
        res = {}
        for rate, step in ((60, 1), (30, 2)):
            n_out = int(np.ceil(nfr / step))
            cup, ncam = cup3d(got[rate], calib, n_out, step)
            pose = R._smooth_pose({j: P[::step, m] for m, j in enumerate(R._GRID_JOINTS)})
            wr, no = z["wrist_mmc"][::step], z["nose_mmc"][::step]
            m_ = min(len(cup), len(wr), len(no), len(pose[f"{side}_wrist"]))
            try:
                ph = segment_sequential(cup[:m_], wr[:m_], no[:m_], fps=float(rate))
            except Exception:
                ph = None
            if not ph:
                continue
            res[rate] = (ph, _measures_at(pose, ph, side, float(rate)),
                         float(np.isfinite(cup).all(1).mean()), int((ncam >= 3).sum()), cup)
        if 60 not in res or 30 not in res:
            continue
        # ---- does the CUP TRAJECTORY itself change, or only the segmenter's reading of it? ----
        # Compare the 30 Hz track against the 60 Hz track sampled at the SAME instants. If these
        # agree, the tracker is effectively rate-invariant and every boundary shift below belongs to
        # the segmenter running on a coarser grid rather than to the tracking.
        c60, c30 = res[60][4], res[30][4]
        k = min(len(c60[::2]), len(c30))
        A, Bc = c60[::2][:k], c30[:k]
        m = np.isfinite(A).all(1) & np.isfinite(Bc).all(1)
        if m.sum() >= 20:
            dd = np.linalg.norm(A[m] - Bc[m], axis=1)
            crows.append(dict(part=part, trial=trial, n=int(m.sum()),
                              med_mm=float(np.median(dd)),
                              p90_mm=float(np.percentile(dd, 90)),
                              max_mm=float(dd.max()),
                              frac_over_10mm=float((dd > 10).mean())))

        # boundary comparison, both expressed in SECONDS on the shared clock
        e60, e30 = _edges(res[60][0]), _edges(res[30][0])
        for b in BOUNDS:
            if np.isfinite(e60.get(b, np.nan)) and np.isfinite(e30.get(b, np.nan)):
                brows.append(dict(part=part, trial=trial, boundary=b,
                                  s60=e60[b] / 60.0, s30=e30[b] / 30.0))
        for meas, _l in MEAS:
            rows.append(dict(part=part, trial=trial, arm=str(z["arm"]), measure=meas,
                             v60=res[60][1].get(meas, np.nan),
                             v30=res[30][1].get(meas, np.nan)))
        rows.append(dict(part=part, trial=trial, arm=str(z["arm"]), measure="_cup_coverage",
                         v60=res[60][2], v30=res[30][2]))
        print(f"  [{i+1}/{len(sel)}] {part}/{trial} {len(cams)}cam {nfr}fr "
              f"{time.time()-t0:5.0f}s  cup coverage {res[60][2]:.2f} / {res[30][2]:.2f}",
              flush=True)

    if not rows:
        print("nothing scored"); return
    D, B, C = pd.DataFrame(rows), pd.DataFrame(brows), pd.DataFrame(crows)
    D.to_csv(f"{a.out}_measures.csv", index=False)
    B.to_csv(f"{a.out}_bounds.csv", index=False)
    C.to_csv(f"{a.out}_cupdist.csv", index=False)
    if not C.empty:
        print("\nCUP TRAJECTORY, 30 Hz track vs 60 Hz track at the same instants (mm)")
        print(f"  per-trial median : median {C.med_mm.median():7.2f}  "
              f"p90 {C.med_mm.quantile(.9):7.2f}  worst {C.med_mm.max():7.2f}")
        print(f"  per-trial p90    : median {C.p90_mm.median():7.2f}  worst {C.p90_mm.max():7.2f}")
        print(f"  per-trial max    : median {C.max_mm.median():7.2f}  worst {C.max_mm.max():7.2f}")
        print(f"  trials with >5% of frames beyond 10 mm: "
              f"{int((C.frac_over_10mm > 0.05).sum())}/{len(C)}")
    ntr = D.groupby(["part", "trial"]).ngroups
    print(f"\nPROCESSING CHECK: {ntr} trials tracked at both rates, {len(B)} boundary pairs")

    print(f"\nCUP TRACK re-run at 30 Hz vs 60 Hz -- boundary shift (ms)")
    print(f"{'boundary':14}{'median':>9}{'p90':>8}{'>0.25s':>9}{'n':>6}")
    if B.empty:
        print("  (no boundary pairs)")
    for b in (BOUNDS if not B.empty else []):
        g = B[B.boundary == b]
        if not len(g):
            continue
        d = (g.s30 - g.s60).abs() * 1000
        print(f"{b:14}{d.median():9.0f}{d.quantile(.9):8.0f}"
              f"{100*(d > 250).mean():8.1f}%{len(g):6d}")

    cov = D[D.measure == "_cup_coverage"]
    print(f"\ncup coverage (finite 3D frames): 60 Hz {cov.v60.median():.3f}, "
          f"30 Hz {cov.v30.median():.3f}")
    print(f"\nMEASURE shift, 30 Hz cup vs 60 Hz cup (same pose treatment)")
    print(f"{'measure':34}{'med 60':>10}{'med 30':>10}{'signed':>10}{'% of 60':>9}{'n':>6}")
    for meas, label in MEAS:
        g = D[D.measure == meas].dropna(subset=["v60", "v30"])
        if len(g) < 5:
            continue
        sc = np.abs(np.median(g.v60))
        sd = np.median(g.v30 - g.v60)
        print(f"{label:34}{np.median(g.v60):>10.2f}{np.median(g.v30):>10.2f}{sd:>10.3f}"
              f"{100*sd/sc if sc > 1e-9 else np.nan:>8.1f}%{len(g):6d}")
    print(f"\nwrote {a.out}_measures.csv and _bounds.csv")
    print("DONE_FPS_CUP", flush=True)


if __name__ == "__main__":
    main()
