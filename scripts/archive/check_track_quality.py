"""Per-trial INTRINSIC quality of the cached UETrack cup 3D tracks (no OMC needed).

Answers "is every trial providing a good track?" from the track itself -- the properties a usable
cup 3D must have, using the SAME consensus geometry the pipeline uses (cup_track.track_cup_3d_from_cache
-> consensus3, reprojection-gated). NOT an accuracy-vs-OMC check (that needs lag-sync + trial-length
matching; done elsewhere). Flags:

  valid%   : fraction of frames with a consensus 3D point (>=2 agreeing cams)
  med_cams : median #cameras kept by consensus per valid frame (>=3 = robust, ==2 = fragile)
  p99_step : 99th-pct per-frame 3D step (mm/frame). A TELEPORT (huge step) = bad triangulation
             hijacked by a drifted camera; a smooth track has a modest step.
  span     : XYZ bounding-box of the track (mm) -- a plausible drink is ~300-600mm; tiny=stuck,
             huge=blown up.

A trial is FLAGGED if valid% < 60, or med_cams < 2, or p99_step > 150 mm/fr (a physical cup moves
<~80mm/fr even at peak). Prints a per-participant summary + the flagged trials.

    python scripts/check_track_quality.py --parts P10 P12 P13 P14 P251 P252
"""
from __future__ import annotations
import sys, argparse, glob, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
from pipeline import cup_track

TRACKS = ROOT / "cache" / "tracks_uetrack"


def trial_quality(part, trial, calib):
    cf = TRACKS / f"{part}__{trial}__uetrack__fs1.json"
    if not cf.exists():
        return None
    tr = cup_track.track_cup_3d_from_cache(cf, calib)
    X = np.array([t["X"] if t.get("X") else [np.nan] * 3 for t in tr], float)
    ncam = np.array([t.get("n_cams", 0) for t in tr])
    n = len(X)
    v = np.isfinite(X).all(1)
    if v.sum() < 5:
        return dict(n=n, valid=v.mean(), med_cams=0, p99_step=np.nan, span=[0, 0, 0], ok=False)
    # per-frame step on valid frames only (consecutive valids)
    steps = []
    idx = np.flatnonzero(v)
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            steps.append(np.linalg.norm(X[b] - X[a]))
    steps = np.array(steps) if steps else np.array([np.nan])
    span = (np.nanmax(X[v], 0) - np.nanmin(X[v], 0))
    med_cams = int(np.median(ncam[v])) if v.any() else 0
    p99 = float(np.nanpercentile(steps, 99)) if np.isfinite(steps).any() else np.nan
    valid = float(v.mean())
    ok = (valid >= 0.60) and (med_cams >= 2) and (np.isfinite(p99) and p99 <= 150) \
        and (30 <= max(span) <= 1500)
    return dict(n=n, valid=valid, med_cams=med_cams, p99_step=p99,
                span=[round(float(s)) for s in span], ok=bool(ok))


def _trials(part):
    ids = set()
    for f in glob.glob(str(TRACKS / f"{part}__*__uetrack__fs1.json")):
        m = re.match(rf"{part}__(.+)__uetrack__fs1\.json$", Path(f).name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P10", "P12", "P13", "P14", "P251", "P252"])
    a = ap.parse_args(argv)
    H.use_good_cams()

    print(f"{'part':6s} {'trials':>6s} {'good':>6s} {'flagged':>8s}  {'med valid%':>10s} "
          f"{'med cams':>8s} {'med p99step':>11s}", flush=True)
    allflag = []
    for p in a.parts:
        calib = H._load_calib_mm(p)
        if p in H.GOOD_CAMS:
            calib = {c: v for c, v in calib.items() if c in H.GOOD_CAMS[p]}
        rows = []
        for t in _trials(p):
            q = trial_quality(p, t, calib)
            if q is None:
                continue
            rows.append((t, q))
            if not q["ok"]:
                allflag.append((p, t, q))
        if not rows:
            print(f"{p:6s}  (no tracks)"); continue
        good = sum(1 for _, q in rows if q["ok"])
        mv = np.median([q["valid"] for _, q in rows]) * 100
        mc = int(np.median([q["med_cams"] for _, q in rows]))
        mp = np.median([q["p99_step"] for _, q in rows if np.isfinite(q["p99_step"])])
        print(f"{p:6s} {len(rows):>6d} {good:>6d} {len(rows)-good:>8d}  {mv:>9.0f}% "
              f"{mc:>8d} {mp:>10.0f}", flush=True)

    print(f"\n=== {len(allflag)} FLAGGED trials (valid<60% OR <2 cams OR teleport>150mm/fr OR bad span) ===",
          flush=True)
    for p, t, q in allflag[:60]:
        why = []
        if q["valid"] < 0.60: why.append(f"valid {q['valid']*100:.0f}%")
        if q["med_cams"] < 2: why.append(f"{q['med_cams']}cam")
        if not (np.isfinite(q["p99_step"]) and q["p99_step"] <= 150): why.append(f"step {q['p99_step']:.0f}")
        if not (30 <= max(q["span"]) <= 1500): why.append(f"span {q['span']}")
        print(f"  {p} {t:30s} {', '.join(why)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
