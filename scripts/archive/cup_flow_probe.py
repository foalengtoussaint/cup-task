"""Does the WRIST flow-speed method transfer to the CUP?

The v3 cup track has near-perfect POSITION (displacement corr 0.9995) but noisy SPEED: ~1mm of
frame-to-frame positional wobble is invisible on a 700mm trajectory yet becomes ~60mm/s once
differentiated. OMC sits at ~11mm/s at rest, v3 at ~60. That noise floor is what breaks the
segmenter's BACK_OFF=10mm/s gate, so returning/rest_post never fire (11/12 trials).

This is the SAME failure the wrist had, and the wrist fix was optical flow: PyrLK measures pixel
DISPLACEMENT between two frames directly, so it never differentiates a noisy position. Here we point
the identical machinery (pipeline.flow_speed, unchanged -- it is target-agnostic) at the CUP pixel
from the UETrack cache instead of the YOLO wrist keypoint.

Compares, per trial, cup speed vs OMC:
    pos-diff    : differentiate the v3 cup track                  -- the current pipeline
    smoothnet   : differentiate the SmoothNet-refined cup track   -- the cheap fix
    flow        : PyrLK at the cup pixel -> triangulated velocity -- the wrist method
    blend       : speed-gated flow/SmoothNet                      -- the wrist winner

Also reports the two things that actually matter downstream: speed at REST (must fall under the
10mm/s gate) and error on MOVING frames.

    python scripts/cup_flow_probe.py                 # all 12 trials, cached flow reused
    python scripts/cup_flow_probe.py --part P07 --trial trial_10_L_unaffected
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import compare_pose_omc_delta as H
import results_v3_delta as R
from pipeline import flow_speed, speed_blend

FPS = H.VIDEO_FPS
TRACKS = ROOT / "cache" / "tracks_uetrack"
CUPFLOW = ROOT / "cache" / "cup_flow_vel"          # separate from the wrist's cache/flow_vel
MOVING = 50.0                                       # mm/s: "the cup is actually moving"


def cup_px(part: str, trial: str, n: int) -> dict[str, np.ndarray]:
    """Per-camera CUP pixel (T,2) from the UETrack cache — the tracker's own 2D points."""
    cf = TRACKS / f"{part}__{trial}__uetrack__fs1.json"
    rec = json.loads(cf.read_text())
    cams = sorted({c for f in rec.values() for c in f})
    out = {}
    for cam in cams:
        p = np.full((n, 2), np.nan)
        for f in range(n):
            v = (rec.get(str(f)) or {}).get(cam)
            if v and v.get("trk") is not None:
                p[f] = v["trk"]
        out[cam] = p
    return out


def cup_flow_speed(part: str, trial: str, calib: dict, n: int) -> np.ndarray:
    """PyrLK at the cup pixel in every camera -> triangulated 3D speed. Cached per clip."""
    px = cup_px(part, trial, n)
    flow = {}
    for cam, p in px.items():
        clip = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{cam.split('_')[1]}.mp4"
        if clip.exists() and cam in calib:
            flow[cam] = flow_speed.flow_track_from_clip(clip, p, cache_dir=CUPFLOW)
    if not flow:
        return np.full(n, np.nan)
    return flow_speed.speed_from_cached_flow(px, flow, calib, n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part"); ap.add_argument("--trial")
    a = ap.parse_args(argv)
    H.use_good_cams()

    todo = []
    for part, (trials, side) in R.TRIALS.items():
        for trial in trials:
            if a.part and part != a.part:
                continue
            if a.trial and trial != a.trial:
                continue
            todo.append((part, trial, side))

    M = ["pos-diff", "smoothnet", "flow", "blend"]
    agg = {m: {"all": [], "moving": [], "rest": [], "corr": []} for m in M}
    print(f"{'trial':16} " + " ".join(f"{m:>10}" for m in M) + "   (moving-frame |err| mm/s)",
          flush=True)
    print("-" * 74, flush=True)

    for part, trial, side in todo:
        calib = R._calib(part)
        mmc, n = H._load_mmc(part, trial)
        omc = H._load_omc(part, trial, n)
        lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
        oc = R._shift(R._omc_cup(part, trial, n), lag)
        c3 = R._cup_v3(part, trial, calib, n)

        so = H._lp(H._speed(oc))
        raw = H._lp(H._speed(c3))
        sn = H._lp(H._speed(R._smooth_joint(c3)))
        fl = H._lp(cup_flow_speed(part, trial, calib, n))
        bl = speed_blend.blend(fl, sn)

        mv = np.isfinite(so) & (so > MOVING)
        rest = np.isfinite(so) & (so < 20)
        line = []
        for m, sig in zip(M, (raw, sn, fl, bl)):
            ok = np.isfinite(sig) & np.isfinite(so)
            if ok.sum() < 30:
                line.append(float("nan")); continue
            agg[m]["all"].append(float(np.median(np.abs(sig[ok] - so[ok]))))
            agg[m]["corr"].append(float(np.corrcoef(sig[ok], so[ok])[0, 1]))
            q = ok & mv
            e = float(np.median(np.abs(sig[q] - so[q]))) if q.sum() > 20 else float("nan")
            agg[m]["moving"].append(e)
            r = ok & rest
            if r.sum() > 20:
                agg[m]["rest"].append(float(np.median(sig[r])))
            line.append(e)
        print(f"{part}_{trial.split('_')[1]:>10} " + " ".join(f"{v:10.1f}" for v in line),
              flush=True)

    print("\n" + "=" * 74)
    print(f"{'method':12} {'|err| all':>10} {'MOVING':>9} {'corr':>7} {'REST speed':>12} "
          f"{'under 10mm/s gate?':>19}")
    print("-" * 74)
    f = lambda v: np.median([x for x in v if np.isfinite(x)]) if v else float("nan")
    for m in M:
        rest = f(agg[m]["rest"])
        print(f"{m:12} {f(agg[m]['all']):10.1f} {f(agg[m]['moving']):9.1f} "
              f"{f(agg[m]['corr']):7.3f} {rest:11.1f}  {'YES' if rest < 10 else 'no':>18}")
    print("\n  REST speed = median reported speed where the OMC cup is under 20mm/s. The segmenter's")
    print("  BACK_OFF gate is 10mm/s: a method above that can never end back_transport, which is why")
    print("  returning/rest_post go missing. OMC's own rest speed is ~11mm/s for scale.")


if __name__ == "__main__":
    main()
