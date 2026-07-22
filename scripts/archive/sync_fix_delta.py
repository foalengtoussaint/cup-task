"""Fix camera desync by estimating a constant per-camera frame offset and re-indexing.

Measured (P13): cam2 is a CLEAN constant +140-frame offset (SD 2 across 6 trials) that
correlates at r>=0.83 once shifted -- a fixed 2.3s clock offset, fully recoverable without
re-cutting video. Contrast P10 cam2: low correlation (0.49-0.71) at ANY lag -> NOT a time
shift (bad detection/viewpoint), and P10 cam4: synced but miscalibrated -> not a sync issue.
So the fix applies only where the offset is (a) consistent across trials and (b) yields a high
correlation when applied. Everything else is flagged, not silently "fixed".

The offset is applied by RE-INDEXING detections at triangulation time (cam c frame f -> f+lag),
NOT by re-cutting video. Cheaper, no uncut footage needed. Cost: ~lag frames of coverage lost
at one clip end (the shifted camera runs out) -- acceptable when lag << clip length; for large
lags a real re-cut from the uncut video would recover those too.

Writes cache/delta/sync_offsets.json = {part: {cam: {lag, conf, apply}}}.

    python scripts/sync_fix_delta.py --parts P10 P13 --verify
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
from cup_task.triangulate import robust_triangulate  # noqa: E402

CONF_MIN = 0.65    # shifted correlation must clear this to trust the offset
LAG_SD_MAX = 4     # frames; offset must be consistent across trials to be a real clock shift


def _w2d(frames, side):
    P = []
    for fr in frames:
        k = fr.get("kps", {})
        P.append(k[side][:2] if side in k else [np.nan, np.nan])
    return np.array(P, float)


def _spd(P):
    return np.concatenate([[np.nan], np.linalg.norm(np.diff(P, axis=0), axis=1)])


def _best_lag(a, b, maxlag=180):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best, bl = -2.0, 0
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
        c = float(np.corrcoef(a[m], bs[m])[0, 1])
        if c > best:
            best, bl = c, L
    return bl, best


def _trials(part):
    return sorted({Path(f).name.split(".")[0].replace(f"delta_{part}_", "")
                   for f in glob.glob(str(C.DELTA / part / "dets" / "*.pose.json"))})


def _side(part):
    r = glob.glob(str(C.DELTA / part / "dets" / "*_R_*.pose.json"))
    return "right" if r else "left"


def _load_trial(part, t):
    per = {}
    for pj in sorted(glob.glob(str(C.DELTA / part / "dets" / f"delta_{part}_{t}.*.pose.json"))):
        per[int(Path(pj).name.split(".")[1])] = json.loads(Path(pj).read_text())["frames"]
    return per


def estimate(part):
    side = _side(part)
    trials = _trials(part)
    lags = {}
    confs = {}
    for t in trials:
        per = _load_trial(part, t)
        sp = {c: C._lp(_spd(_w2d(per[c], f"{side}_wrist"))) for c in per}
        L = min(len(v) for v in sp.values())
        M = np.nanmedian(np.vstack([v[:L] for v in sp.values()]), axis=0)
        for c in per:
            lg, r = _best_lag(M, sp[c][:L])
            lags.setdefault(c, []).append(lg)
            confs.setdefault(c, []).append(r)
    out = {}
    for c in sorted(lags):
        lag_med = int(np.median(lags[c]))
        lag_sd = float(np.std(lags[c]))
        # confidence = correlation achievable at the MEDIAN lag (re-scored), not the per-trial best
        conf = float(np.median(confs[c]))
        apply = lag_med != 0 and lag_sd <= LAG_SD_MAX and conf >= CONF_MIN
        out[c] = {"lag": lag_med, "lag_sd": round(lag_sd, 1), "conf": round(conf, 2),
                  "apply": bool(apply)}
    return out, side


def _shift_frames(frames, lag):
    """Return a frame list re-indexed so that output[f] = input[f+lag] (NaN/empty padded)."""
    n = len(frames)
    empty = {"kps": {}}
    if lag == 0:
        return frames
    if lag > 0:
        return frames[lag:] + [empty] * lag
    return [empty] * (-lag) + frames[:lag]


def verify(part, offsets, side):
    """Coverage + reproj on the wrist, before vs after applying the sync offsets."""
    cams = C._load_calib_mm(part)
    fn = C._kp_point(f"{side}_wrist")
    trials = _trials(part)[:6]
    res = {}
    for label, apply in [("before", False), ("after", True)]:
        cov = 0
        ntot = 0
        reps = []
        for t in trials:
            per = _load_trial(part, t)
            if apply:
                per = {c: _shift_frames(v, offsets.get(c, {}).get("lag", 0)
                                        if offsets.get(c, {}).get("apply") else 0)
                       for c, v in per.items()}
            n = min(len(v) for v in per.values())
            for f in range(n):
                cs, pts = [], []
                for c in per:
                    key = f"cam_{c}"
                    if key not in cams:
                        continue
                    p = fn(per[c][f])
                    if p is not None:
                        cs.append(cams[key])
                        pts.append(p)
                if not cs:
                    continue
                ntot += 1
                if len(cs) < 2:
                    continue
                X, keep, med = robust_triangulate(cs, pts)
                if X is not None:
                    cov += 1
                    reps.append(med)
        res[label] = (100 * cov / max(ntot, 1), float(np.median(reps)) if reps else float("nan"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P10", "P13"])
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)

    allofs = {}
    for part in a.parts:
        offsets, side = estimate(part)
        allofs[part] = offsets
        print(f"\n### {part} (side={side})")
        print(f"{'cam':6}{'lag':>6}{'lag_sd':>8}{'conf':>7}   action")
        for c, d in offsets.items():
            act = f"SHIFT {d['lag']:+d}fr" if d["apply"] else (
                "ok (aligned)" if d["lag"] == 0 else
                f"NOT sync-fixable (lag {d['lag']:+d}, sd {d['lag_sd']}, conf {d['conf']})")
            print(f"cam_{c:<3}{d['lag']:>6}{d['lag_sd']:>8}{d['conf']:>7}   {act}")
        if a.verify:
            v = verify(part, offsets, side)
            print(f"  wrist coverage: before {v['before'][0]:5.1f}% ({v['before'][1]:.1f}px)"
                  f"  ->  after {v['after'][0]:5.1f}% ({v['after'][1]:.1f}px)")
    outp = C.DELTA / "sync_offsets.json"
    outp.write_text(json.dumps(allofs, indent=1))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
