"""Re-audit DELTA cameras by REPROJECTION (geometry), not wrist-speed correlation.

WHY (learned the hard way, prompted by the user's three questions):
  * The old cam_quality_delta.py classified cameras by 2D wrist-speed correlation vs a reference.
    That is VIEWPOINT-CONFOUNDED: a perfectly good camera whose wrist moves mostly toward/away
    from it has low apparent 2D speed -> low correlation -> mislabeled "DESYNCED". Measured:
    P10 cam2 & P19 cam2 had corr 0.64 but reproject at 16-18px (FINE); they were wrongly dropped.
  * Low correlation also cannot tell desync from miscalibration.

THE RELIABLE TEST is reprojection of the camera against the consensus 3D wrist, split by motion:
  reproj when STILL (<3mm/frame)   reproj when MOVING     best-lag   verdict
  low                              low                    ~0         FINE
  low                              high                   small lag  DESYNC-small (a shift fixes)
  high (at lag0) -> low (at lag)   -                      large lag  DESYNC-large (re-cut fixes)
  high                             high, no lag helps     none       MISCALIB (needs recalibration)
A still-frame is the discriminator: timing errors vanish when nothing moves, geometry errors don't.

Consensus is built per frame by robust_triangulate over ALL cams (iteratively ejects the worst
reprojector), so it survives a minority of bad cams. Flag participants where <50% of frames reach
a >=3-cam consensus (reference itself untrustworthy).

    python scripts/reaudit_cam_quality.py --parts P10 P13 P14 P15 P17 P19
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from cup_task.triangulate import robust_triangulate  # noqa: E402  (kept for import compat)
from cup_task.kalman_3d import project, triangulate_dlt  # noqa: E402
from sync_fix_delta import _w2d, _spd, _best_lag  # noqa: E402
import itertools  # noqa: E402

STILL_MM = 3.0        # wrist speed below this (mm/frame) = "still"
REPROJ_OK = 20.0      # px; still-reproj below this = geometry fine
MOVE_HIGH = 30.0      # px; move-reproj above this with LOW still-reproj = desync (timing) signature
DESYNC_R = 0.80       # best-lag correlation above this at a consistent nonzero lag = true desync
RANSAC_THR = 15.0     # px; a camera is an inlier to a candidate 3D if it reprojects within this
MIN_INLIERS = 3       # need >=3 mutually-agreeing cams for a trustworthy reference


def ransac_point(cams_d, pts_d, thr=RANSAC_THR):
    """RANSAC 3D point: seed from every camera PAIR, keep the candidate with the most inliers,
    refit on them. Finds the largest mutually-agreeing camera subset even when >50% are bad --
    which all-camera iterative ejection (robust_triangulate) cannot. Returns (X, inlier_set)."""
    ks = [k for k in pts_d if k in cams_d]
    if len(ks) < 2:
        return None, set()
    best_X, best_in = None, set()
    for i, j in itertools.combinations(ks, 2):
        X = triangulate_dlt([cams_d[i], cams_d[j]], [np.array(pts_d[i]), np.array(pts_d[j])])
        inl = {k for k in ks
               if np.hypot(*(project(cams_d[k], X)[0] - np.array(pts_d[k]))) <= thr}
        if len(inl) > len(best_in):
            best_in, best_X = inl, X
    if len(best_in) < MIN_INLIERS:
        return None, best_in
    X = triangulate_dlt([cams_d[k] for k in best_in], [np.array(pts_d[k]) for k in best_in])
    return X, best_in


def _side(part):
    return "right" if glob.glob(str(C.DELTA / part / "dets" / "*_R_*.pose.json")) else "left"


def _trials(part, n):
    s = "R" if _side(part) == "right" else "L"
    return sorted({Path(f).name.split(".")[0].replace(f"delta_{part}_", "")
                   for f in glob.glob(str(C.DELTA / part / "dets" / f"*_{s}_*.pose.json"))})[:n]


def audit(part, n_trials=6, max_cam=5):
    side = _side(part)
    cams = C._load_calib_mm(part)
    cams = {c: cams[c] for c in cams if int(c.split("_")[1]) <= max_cam}  # only the trusted 1-5 set
    fn = C._kp_point(f"{side}_wrist")
    trials = _trials(part, n_trials)
    rep = {}          # cam -> {'still':[], 'move':[]}
    inrate = {}       # cam -> [n_inlier, n_seen]  (how often it agrees with the RANSAC subset)
    laginfo = {}      # cam -> [(lag, r) per trial]
    cons_ok = cons_tot = 0
    for t in trials:
        per = {}
        for pj in sorted(glob.glob(str(C.DELTA / part / "dets" / f"delta_{part}_{t}.*.pose.json"))):
            per["cam_" + Path(pj).name.split(".")[1]] = json.loads(Path(pj).read_text())["frames"]
        n = min(len(v) for v in per.values())
        # per-frame RANSAC consensus (largest mutually-agreeing subset) + speed
        X3 = [None] * n
        for f in range(n):
            pts_d = {c: fn(per[c][f]) for c in per if c in cams and fn(per[c][f]) is not None}
            cons_tot += 1
            if len(pts_d) >= 2:
                X, inl = ransac_point({c: cams[c] for c in pts_d}, pts_d)
                if X is not None:
                    X3[f] = X; cons_ok += 1
                    for c in pts_d:
                        r = inrate.setdefault(c, [0, 0])
                        r[1] += 1
                        if c in inl:
                            r[0] += 1
        spd = [None] * n
        for f in range(1, n):
            if X3[f] is not None and X3[f - 1] is not None:
                spd[f] = float(np.linalg.norm(X3[f] - X3[f - 1]))
        # per-camera reproj stratified by motion
        for c in per:
            if c not in cams:
                continue
            rep.setdefault(c, {"still": [], "move": []})
            for f in range(1, n):
                if X3[f] is None or spd[f] is None:
                    continue
                p = fn(per[c][f])
                if p is None:
                    continue
                e = float(np.hypot(*(project(cams[c], X3[f])[0] - p)))
                rep[c]["still" if spd[f] < STILL_MM else "move"].append(e)
        # desync discriminator: best-lag correlation of wrist speed
        sp = {c: C._lp(_spd(_w2d(per[c], f"{side}_wrist"))) for c in per}
        L = min(len(v) for v in sp.values())
        M = np.nanmedian(np.vstack([v[:L] for v in sp.values()]), axis=0)
        for c in per:
            lag, r = _best_lag(M, sp[c][:L], maxlag=300)
            laginfo.setdefault(c, []).append((lag, r))
    return side, rep, laginfo, inrate, (cons_ok, cons_tot)


def classify(rep_c, lag_c):
    s = np.median(rep_c["still"]) if rep_c["still"] else float("nan")
    mv = np.median(rep_c["move"]) if rep_c["move"] else float("nan")
    lags = [x[0] for x in lag_c]
    rs = [x[1] for x in lag_c]
    best_r = float(np.median(rs)) if rs else 0.0
    lag_med = int(np.median(lags)) if lags else 0
    lag_sd = float(np.std(lags)) if lags else 0.0
    consistent_lag = abs(lag_med) >= 2 and lag_sd <= 5 and best_r >= DESYNC_R

    if np.isfinite(s) and s <= REPROJ_OK:
        # geometry sound when still. If it degrades in motion, that's a TIMING (desync) tell,
        # not FINE -- the bug last pass was stopping at "still ok -> FINE" without this check.
        if np.isfinite(mv) and mv >= MOVE_HIGH and (consistent_lag or mv > 2 * s):
            return "DESYNC (re-cut)", s, mv, best_r, lag_med, lag_sd
        return "FINE", s, mv, best_r, lag_med, lag_sd
    # still-reproj high: geometry wrong. Desync only if a CONSISTENT lag actually explains it.
    if consistent_lag:
        return "DESYNC (re-cut)", s, mv, best_r, lag_med, lag_sd
    return "MISCALIB (recalib)", s, mv, best_r, lag_med, lag_sd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P10", "P13", "P14", "P15", "P17", "P19"])
    ap.add_argument("--n-trials", type=int, default=6)
    ap.add_argument("--max-cam", type=int, default=5, help="restrict consensus+audit to cams 1..MAX")
    a = ap.parse_args(argv)
    summary = {}
    for part in a.parts:
        try:
            side, rep, laginfo, inrate, (cok, ctot) = audit(part, a.n_trials, a.max_cam)
        except Exception as e:
            print(f"\n### {part}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        print(f"\n### {part} (side={side}, RANSAC consensus reached {100*cok/max(ctot,1):.0f}% of "
              f"cam-frames)", flush=True)
        print(f"{'cam':7}{'still':>8}{'move':>8}{'inlier%':>8}{'bestR':>7}{'lag':>6}{'lagSD':>7}"
              f"   verdict")
        usable = []
        cls = {}
        for c in sorted(rep, key=lambda z: int(z.split("_")[1])):
            v, s, mv, br, lg, sd = classify(rep[c], laginfo[c])
            cls[c] = v
            ir = inrate.get(c, [0, 1])
            print(f"{c:7}{s:7.0f}px{mv:6.0f}px{100*ir[0]/max(ir[1],1):7.0f}%{br:7.2f}{lg:>6}"
                  f"{sd:7.1f}   {v}")
            if v == "FINE" or v.startswith("DESYNC"):
                usable.append(c)
        print(f"  USABLE now (FINE): {sorted([c for c in cls if cls[c]=='FINE'], key=lambda z:int(z.split('_')[1]))}")
        recut = [c for c in cls if cls[c].startswith('DESYNC')]
        recal = [c for c in cls if cls[c].startswith('MISCALIB')]
        if recut:
            print(f"  fixable by RE-CUT: {sorted(recut, key=lambda z:int(z.split('_')[1]))}")
        if recal:
            print(f"  need RECALIBRATION: {sorted(recal, key=lambda z:int(z.split('_')[1]))}")
        summary[part] = {"fine": [c for c in cls if cls[c] == "FINE"],
                         "recut": recut, "recalib": recal}
    out = C.DELTA / "reaudit_cam_quality.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
