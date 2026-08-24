"""COHORT validation of the mocap-free Murphy measures on DELTA. Multi-participant, side-aware.

Consolidates everything the P14 investigation established, so the findings can be checked on
participants they were not derived from:

  1. CONCURRENT VALIDITY  r(OMC, MMC) across trials, WITHIN side. The test that decides
     usefulness: if MMC tracks the participant's real trial-to-trial variation, a constant bias
     is just an offset to calibrate. (Pooling sides would fake a correlation out of the
     affected-vs-unaffected group difference -- always compute within side.)
  2. ESTIMATOR FIX. The P14 finding: the signals are fine, the ESTIMATORS are broken.
       * max()    on a noisy signal -> selects a jitter spike BY CONSTRUCTION (peak_elbow_ang_vel
         r=0.07 -> 0.85 using p90; and p90 is a valid proxy for max: r(OMC max, OMC p90)=0.97)
       * argmax() on a flat plateau -> selects arbitrarily (time_to_peak r=0.64 -> 0.86 using the
         centroid of the >80%-of-max region)
     Both estimators are reported side by side so the fix is re-tested, not assumed.
  3. L-R BIAS ASYMMETRY. P14: MMC effect = true effect + (left bias - right bias), exact to 2dp,
     which SIGN-FLIPPED peak_velocity. P17/P19 are MIRRORED (affected=R, unaffected=L), so if the
     asymmetry is a real landmark property it must FLIP DIRECTION on them. A falsifiable test.

Phases are computed ONCE from the OMC wrist+cup and applied to BOTH sides, isolating POSE error
from SEGMENTATION error (with each side using its own phases, MMC's jittery wrist moved the reach
boundary by 1.3s and destroyed the timing measures).

    python scripts/validate_cohort_delta.py --parts P14 P15 P17 P19
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import ezc3d
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import segment
from pipeline.score import compute_position_measures, _smoothed_xyz
import scripts.compare_pose_omc_delta as C
from scripts.compare_pose_omc_delta import (_load_mmc, _despike, _resample, _lp, _find_lag,
                                            VIDEO_FPS, C3D_RATE, DELTA)
from scripts.score_omc_delta import _shift

OMC_MAP = {"shoulder": "shoulder_{S}", "elbow": "elbow_{S}", "hip": "hip_{S}",
           "shoulder_o": "shoulder_{O}", "hip_o": "hip_{O}", "head": "head"}


def _omc(part, trial, side):
    c = ezc3d.c3d(str(DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]
    S = "R" if side == "right" else "L"
    O = "L" if side == "right" else "R"

    def mk(n):
        return P[:3, L.index(n), :].T
    out = {k: mk(v.format(S=S, O=O)) for k, v in OMC_MAP.items()}
    out["wrist"] = np.mean([mk(f"wrist_inner_{S}"), mk(f"wrist_outer_{S}")], axis=0)
    out["cup"] = np.mean([mk(f"cluster_cup_{i}") for i in (1, 2, 3, 4)], axis=0)
    return {k: _despike(_resample(v, C3D_RATE, VIDEO_FPS)) for k, v in out.items()}


def _mmc(part, trial, side):
    M, n = _load_mmc(part, trial)
    S, O = side, ("left" if side == "right" else "right")
    return {"shoulder": M[f"{S}_shoulder"], "elbow": M[f"{S}_elbow"], "wrist": M[f"{S}_wrist"],
            "shoulder_o": M[f"{O}_shoulder"], "hip": M[f"{S}_hip"], "hip_o": M[f"{O}_hip"],
            "head": M["nose"]}, n


def _fill(x):
    x = np.asarray(x, float).copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    return x


def _centroid_time(s, frac=0.80):
    """WHEN did the peak happen -- centroid of the >frac*max region, not argmax.

    The reach velocity profile is a broad bell (~8 frames within 5% of the max on a ~62-frame
    window), so argmax picks one arbitrary sample off a plateau and inherits the plateau width as
    error (4.8 fr) -- while the real between-trial signal is only 4.6 fr. The centroid averages
    over the flat top: r 0.64 -> 0.86 on P14, with the signal SD preserved.
    """
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return np.nan
    i = np.arange(len(s))
    m = ok & (s >= frac * np.nanmax(s))
    return float((i[m] * s[m]).sum() / s[m].sum())


def _robust_peak(s, q=90):
    """HOW BIG was the peak -- p90, not max. max() selects the single most extreme sample, which
    on a jittery signal is always noise (P14: MMC/OMC ratio 1.30, r=0.07). p90 is unbiased
    (ratio 1.03), tracks at r=0.85, AND is a valid proxy for the max (r(OMC max, OMC p90)=0.97)."""
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    return float(np.percentile(s, q)) if len(s) else np.nan


def _signals(J):
    def U(v):
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    trunk = (J["shoulder"] + J["shoulder_o"]) / 2
    hip_mid = (J["hip"] + J["hip_o"]) / 2
    arm = U(J["elbow"] - J["shoulder"])
    flex = _lp(np.degrees(np.arccos(np.clip((arm * U(hip_mid - trunk)).sum(1), -1, 1))))
    u, v = J["shoulder"] - J["elbow"], J["wrist"] - J["elbow"]
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    elb = _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))
    eav = np.abs(np.gradient(elb)) * VIDEO_FPS
    spd = np.r_[0.0, np.linalg.norm(np.diff(_smoothed_xyz(_fill(J["wrist"]), VIDEO_FPS, 4, 2),
                                            axis=0), axis=1) * VIDEO_FPS]
    return dict(flex=flex, elb=elb, eav=eav, spd=spd, trunk=trunk)


def _measures(J, phases, side):
    m = compute_position_measures(_fill(J["wrist"]),
                                  _fill((J["shoulder"] + J["shoulder_o"]) / 2),
                                  phases, side, fps=VIDEO_FPS).to_dict()
    m.pop("side", None)
    S = _signals(J)

    def ph(n):
        for nm, s, e in phases:
            if nm == n:
                return s, e
        return None
    r, fw = ph("reaching"), ph("forward_transport")
    if not r:
        return None
    rf = (r[0], fw[1]) if fw else r
    m["elbow_extension_reaching"] = float(np.nanmax(S["elb"][rf[0]:rf[1]]))
    m["shoulder_flexion_reaching"] = float(np.nanmax(S["flex"][r[0]:r[1]]))
    # the two estimators, side by side -- OLD (as shipped) vs FIXED
    m["peak_elbow_ang_vel__max"] = float(np.nanmax(S["eav"][rf[0]:rf[1]]))
    m["peak_elbow_ang_vel__p90"] = _robust_peak(S["eav"][rf[0]:rf[1]], 90)
    seg_spd = S["spd"][r[0]:r[1]]
    m["time_to_peak__argmax"] = float(np.nanargmax(seg_spd)) / VIDEO_FPS
    m["time_to_peak__centroid"] = _centroid_time(seg_spd, 0.80) / VIDEO_FPS
    m["peak_velocity__max"] = float(np.nanmax(seg_spd))
    m["peak_velocity__p90"] = _robust_peak(seg_spd, 90)
    return m


def _trial(part, trial, side):
    """Phases from OMC (shared by both sides) -> isolates pose error from segmentation error."""
    O = _omc(part, trial, side)
    M, n = _mmc(part, trial, side)
    lag, _ = _find_lag(M["wrist"], O["wrist"][:n] if len(O["wrist"]) >= n else O["wrist"])
    Osh = {k: _shift(v[:n] if len(v) >= n else np.vstack([v, np.full((n - len(v), 3), np.nan)]),
                     lag) for k, v in O.items()}
    cup = Osh["cup"]
    T = min(len(M["wrist"]), len(cup))
    Osh = {k: v[:T] for k, v in Osh.items()}
    M = {k: v[:T] for k, v in M.items()}
    cup = cup[:T]
    cr = _lp(np.linalg.norm(cup - np.nanmedian(cup[:30], 0), axis=1))
    wd = _lp(np.linalg.norm(Osh["wrist"] - np.nanmedian(Osh["wrist"][:30], 0), axis=1))
    lift = np.flatnonzero(cr > 150)
    if not len(lift):
        return None
    b = np.flatnonzero((np.arange(T) > lift[0]) & (cr < 60) & (wd < 60))
    if not len(b):
        return None
    cut = min(b[0] + 20, T)
    Osh = {k: v[:cut] for k, v in Osh.items()}
    M = {k: v[:cut] for k, v in M.items()}
    cup = cup[:cut]
    seg = segment.segment_cup_only(cup, fps=VIDEO_FPS)
    seg = segment.refine_grasp_with_pose(seg, cup, Osh["wrist"], Osh["head"], fps=VIDEO_FPS)
    phases = segment.to_murphy_phases(seg, Osh["wrist"], cup, fps=VIDEO_FPS)
    if not any(n_ == "reaching" for n_, _, _ in phases):
        return None
    o, m = _measures(Osh, phases, side), _measures(M, phases, side)
    if o is None or m is None:
        return None
    return o, m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["P14", "P15", "P17", "P19"])
    ap.add_argument("--good-cams", action="store_true",
                    help="triangulate only verified-good cams per participant (cam_quality.json). "
                         "The all-cams 3D on P15/P17/P19 is poisoned by desynced cameras.")
    a = ap.parse_args(argv)
    if a.good_cams:
        C.use_good_cams()
        print(f"[good-cams] {{{', '.join(f'{k}:{len(v)}' for k,v in C.GOOD_CAMS.items())}}}",
              flush=True)

    KEYS = ["total_movement_time", "max_trunk_displacement", "elbow_extension_reaching",
            "shoulder_flexion_reaching", "number_of_movement_units",
            "peak_velocity__max", "peak_velocity__p90",
            "time_to_peak__argmax", "time_to_peak__centroid",
            "peak_elbow_ang_vel__max", "peak_elbow_ang_vel__p90"]
    store = {}
    for part in a.parts:
        skips = {}
        dd = DELTA / part / "dets"
        if not dd.exists():
            print(f"{part}: no dets, skip", flush=True)
            continue
        trials = sorted({re.sub(r"\.\d+\.pose\.json$", "", f.name).replace(f"delta_{part}_", "")
                         for f in dd.glob("*.pose.json")})
        for t in trials:
            m = re.search(r"_(R|L)_(unaffected|affected)$", t)
            if not m:
                continue
            side = "right" if m.group(1) == "R" else "left"
            cond = m.group(2)
            try:
                r = _trial(part, t, side)
            except Exception as e:
                # NEVER swallow silently: a bare `except: continue` here hid the fact that
                # P19 has NO left wrist markers (0/94 c3d) -> P19 reported {} as if it were
                # a null result rather than absent data.
                skips.setdefault(f"{type(e).__name__}: {e}", []).append(t)
                continue
            if r:
                store.setdefault((part, side, cond), []).append(r)
        got = {k: len(v) for k, v in store.items() if k[0] == part}
        print(f"{part}: {got}", flush=True)
        for reason, ts in sorted(skips.items(), key=lambda kv: -len(kv[1])):
            print(f"    SKIPPED {len(ts):3d} trials -- {reason}  (e.g. {ts[0]})", flush=True)

    print("\n" + "=" * 96)
    print("1. CONCURRENT VALIDITY  r(OMC,MMC) across trials, WITHIN side")
    print("=" * 96)
    hdr = f"{'measure':32}" + "".join(f"{p:>15}" for p in a.parts)
    print(hdr); print("-" * len(hdr))
    for k in KEYS:
        row = f"{k:32}"
        for part in a.parts:
            rs = []
            for (p, side, cond), v in store.items():
                if p != part:
                    continue
                arr = np.array([[o[k], m[k]] for o, m in v
                                if np.isfinite(o.get(k, np.nan)) and np.isfinite(m.get(k, np.nan))])
                if len(arr) >= 4 and np.std(arr[:, 0]) > 1e-9 and np.std(arr[:, 1]) > 1e-9:
                    rs.append(pearsonr(arr[:, 0], arr[:, 1])[0])
            row += f"{np.mean(rs):>+15.2f}" if rs else f"{'--':>15}"
        print(row)
    print("\n  __max/__argmax = as shipped.  __p90/__centroid = the P14 estimator fix.")

    print("\n" + "=" * 96)
    print("2. L-R BIAS ASYMMETRY  (bias = MMC-OMC per side).  P14/P15 affected=L; P17/P19 affected=R")
    print("   PREDICTION: if this is a landmark property, the L-R gap FLIPS SIGN on P17/P19.")
    print("=" * 96)
    print(f"{'measure':32}" + "".join(f"{p:>15}" for p in a.parts))
    print("-" * len(hdr))
    for k in ["peak_velocity__max", "peak_velocity__p90", "elbow_extension_reaching",
              "max_trunk_displacement"]:
        row = f"{k:32}"
        for part in a.parts:
            b = {}
            for side in ("left", "right"):
                e = [m[k] - o[k] for (p, s, c), v in store.items() if p == part and s == side
                     for o, m in v if np.isfinite(o.get(k, np.nan)) and np.isfinite(m.get(k, np.nan))]
                if e:
                    b[side] = float(np.mean(e))
            row += f"{b['left'] - b['right']:>+15.1f}" if len(b) == 2 else f"{'--':>15}"
        print(row)
    print("\n  value = (left bias - right bias). P14 peak_velocity was +100.8 and flipped the")
    print("  clinical effect's SIGN (true -83.2 -> measured +17.7).")


if __name__ == "__main__":
    main()
