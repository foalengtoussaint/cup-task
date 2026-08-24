"""Per-camera quality for DELTA: separate DESYNC from MISCALIBRATION, then re-triangulate.

WHY: P15/P17/P19 looked like "our pose pipeline fails". It doesn't. HALF their cameras
disagree, which sits exactly on robust_triangulate's 50% breakdown point -- the all-cam DLT
seed is then corrupted and the iterative ejection throws out the GOOD cameras. Measured on
P15 t10: coverage 25% (all 10 cams) -> 100% (cams 1-5 only), reproj 10.8 -> 7.5px. The
cameras were the problem; the pipeline was fine. Confirmed visually by the user.

THE TWO FAILURE MODES ARE DIFFERENT AND NEED DIFFERENT TESTS:
  DESYNC        -- camera shows a different INSTANT. Its 2D wrist tracks the arm perfectly,
                   so 2D confidence/quality look fine, but it disagrees in 3D whenever the
                   arm MOVES (and agrees while it is static -> the bimodal 0-or-10 signature).
                   TEST: cross-correlate this camera's 2D wrist SPEED against a reference
                   camera. Low r = desynced. (User's idea; it is what finally cracked this.)
  MISCALIBRATION-- camera's 2D is right and its GEOMETRY is wrong. Sync test CANNOT see it
                   (r stays high). TEST: build the consensus from trusted cameras only, then
                   reproject it into this camera and measure the residual.
Running only the sync test would silently pass a miscalibrated camera.

    python scripts/cam_quality_delta.py --parts P14 P15 P17 P19
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
from pipeline.kalman_3d import project  # noqa: E402
from pipeline.triangulate import robust_triangulate  # noqa: E402

SYNC_R = 0.65     # below this vs the reference camera => desynced
REPROJ_OK = 30.0  # px, same gate the triangulator uses


def _w2d(frames, side):
    P = []
    for fr in frames:
        k = fr.get("kps", {})
        P.append(k[side][:2] if side in k else [np.nan, np.nan])
    return np.array(P, float)


def _speed(P):
    return np.concatenate([[np.nan], np.linalg.norm(np.diff(P, axis=0), axis=1)])


def _best_r(a, b, maxlag=12):
    # Cameras WITHIN one trial can have different frame counts (seen: cam_3=518 vs 263) --
    # another symptom of the cut/slicing defect. Truncate to the common length instead of
    # broadcasting (which just raises), and let the caller report the mismatch.
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best = -2.0
    for L in range(-maxlag, maxlag + 1):
        bs = np.full_like(b, np.nan)
        if L > 0:
            bs[L:] = b[:-L]
        elif L < 0:
            bs[:L] = b[-L:]
        else:
            bs = b.copy()
        m = np.isfinite(a) & np.isfinite(bs)
        if m.sum() < 60 or np.std(a[m]) < 1e-9 or np.std(bs[m]) < 1e-9:
            continue
        best = max(best, float(np.corrcoef(a[m], bs[m])[0, 1]))
    return best


def _dets(part, trial):
    per = {}
    for pj in sorted(glob.glob(str(C.DELTA / part / "dets" / f"delta_{part}_{trial}.*.pose.json"))):
        per["cam_" + Path(pj).name.split(".")[1]] = json.loads(Path(pj).read_text())["frames"]
    return per


def analyse(part, trials, side="right", ref="cam_3"):
    cams = C._load_calib_mm(part)
    sync, ragged = {}, []
    for t in trials:
        per = _dets(part, t)
        if ref not in per:
            continue
        lens = {c: len(v) for c, v in per.items()}
        if len(set(lens.values())) > 1:
            ragged.append((t, lens))
        # Reference-free: correlate each camera against the MEDIAN wrist-speed across all
        # cameras, not a single anchor. Anchoring to cam_3 assumed cam_3 was good -- but on
        # P19 the reference itself is suspect, which would poison every "vs cam_3" number.
        # The per-frame median is robust as long as <50% of cameras are bad in the same way.
        sp = {c: C._lp(_speed(_w2d(fr, f"{side}_wrist"))) for c, fr in per.items()}
        L = min(len(v) for v in sp.values())
        M = np.nanmedian(np.vstack([v[:L] for v in sp.values()]), axis=0)
        for c, v in sp.items():
            sync.setdefault(c, []).append(_best_r(M, v[:L]))
    sync = {c: float(np.nanmedian(v)) for c, v in sync.items()}
    good = sorted([c for c, r in sync.items() if r >= SYNC_R], key=lambda z: int(z.split("_")[1]))

    # calibration probe: consensus from GOOD cams only, reprojected into EVERY cam
    rep = {}
    cov_all = cov_good = ntot = 0
    for t in trials:
        per = _dets(part, t)
        if not per:
            continue
        fn = C._kp_point(f"{side}_wrist")
        n = min(len(v) for v in per.values())
        for f in range(n):
            obs = {c: fn(per[c][f]) for c in per if fn(per[c][f]) is not None and c in cams}
            if not obs:
                continue
            ntot += 1
            Xa, ka, _ = robust_triangulate([cams[c] for c in obs], list(obs.values()))
            if Xa is not None:
                cov_all += 1
            g = {c: p for c, p in obs.items() if c in good}
            if len(g) < 3:
                continue
            Xg, kg, _ = robust_triangulate([cams[c] for c in g], list(g.values()))
            if Xg is None:
                continue
            cov_good += 1
            for c, p in obs.items():
                e = float(np.hypot(*(project(cams[c], Xg)[0] - p)))
                rep.setdefault(c, []).append(e)
    rep = {c: float(np.median(v)) for c, v in rep.items() if v}
    return sync, rep, good, (cov_all, cov_good, ntot), ragged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P14", "P15", "P17", "P19"])
    ap.add_argument("--n-trials", type=int, default=6)
    a = ap.parse_args(argv)

    summary = {}
    for part in a.parts:
        # Side-flexible: some participants have only L trials (e.g. P13 = *_L_unaffected). The
        # camera sync/calib audit doesn't care which arm -- pick whichever side has trials, and
        # match the wrist we correlate to it.
        rt = sorted({Path(f).name.split(".")[0].replace(f"delta_{part}_", "")
                     for f in glob.glob(str(C.DELTA / part / "dets" / "*_R_*.pose.json"))})
        lt = sorted({Path(f).name.split(".")[0].replace(f"delta_{part}_", "")
                     for f in glob.glob(str(C.DELTA / part / "dets" / "*_L_*.pose.json"))})
        side, trials = ("right", rt) if len(rt) >= len(lt) else ("left", lt)
        trials = trials[:a.n_trials]
        if not trials:
            print(f"{part}: no R trials", flush=True)
            continue
        sync, rep, good, (ca, cg, nt), ragged = analyse(part, trials, side)
        print(f"\n### {part}  ({len(trials)} trials, sync=vs-median-of-cameras, side={side})",
              flush=True)
        for t, lens in ragged:
            print(f"  RAGGED frame counts in {t}: {dict(sorted(lens.items()))}", flush=True)
        print(f"{'cam':8}{'sync r':>9}{'reproj px':>11}   verdict")
        for c in sorted(sync, key=lambda z: int(z.split("_")[1])):
            r = sync[c]
            e = rep.get(c, float("nan"))
            if r < SYNC_R:
                v = "DESYNCED"
            elif np.isfinite(e) and e > REPROJ_OK:
                v = "MISCALIBRATED (sync ok, geometry off)"
            else:
                v = "ok"
            print(f"{c:8}{r:9.2f}{e:11.1f}   {v}")
        print(f"  GOOD cams: {len(good)}/{len(sync)}  {good}")
        print(f"  wrist coverage: ALL cams {100*ca/max(nt,1):5.1f}%   GOOD cams only "
              f"{100*cg/max(nt,1):5.1f}%")
        summary[part] = {"sync": sync, "reproj": rep, "good": good,
                         "cov_all": ca / max(nt, 1), "cov_good": cg / max(nt, 1)}
    out = C.DELTA / "cam_quality.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
