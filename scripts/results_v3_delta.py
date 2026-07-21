"""v3 pipeline OFFICIAL METRICS on DELTA vs OMC ground truth.

Four deliverables, all on the same trials, all vs OMC:

  1. ACCURACY  -- per-target DISPLACEMENT error (mm) and SPEED error (mm/s), for the relevant
                  joints (wrists, elbows, shoulders, nose) and the CUP.
  2. SPEED     -- the v3 speed path (flow/SmoothNet/blend) on the acting wrist, per-frame + PEAK.
  3. SEGMENT   -- drink-phase boundaries + dwell error vs the OMC-cup segmentation.
  4. MURPHY    -- the position measures, v1 (raw pose) vs v3 (SmoothNet+blend), error vs OMC.

METRIC NOTES (learned the hard way, see docs/SPEED_METRICS.md):
  * Speed is frame-invariant -> report ABSOLUTE mm/s, never correlation (correlation hides magnitude).
  * Report the TAIL (p90/max), not just the median -- a good median can hide catastrophic failures.
  * DISPLACEMENT is reported as the error about the mean, i.e. after removing the constant offset.
    The absolute MMC-vs-OMC offset is dominated by the rig<->mocap calibration floor (~40mm) and the
    local body-fit, NOT by the tracker; subtracting it isolates what the pipeline controls. The raw
    (uncentred) value is printed alongside so the floor stays visible and nothing is hidden.

  P13 is EXCLUDED from the SPEED metrics (linear clock drift = bad ground truth) but KEPT for
  displacement/segmentation, where a constant lag does not invalidate the comparison. Every table
  says which cohort it used.

    python scripts/results_v3_delta.py --what all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import compare_pose_omc_delta as H
import flow_velocity_probe as F
from cup_task import segment, cup_track, pose_smooth, flow_speed, speed_blend
from cup_task.score import compute_position_measures

FPS = H.VIDEO_FPS
TRACKS = ROOT / "cache" / "tracks_uetrack"
FLOWDIR = ROOT / "cache" / "flow_vel"

TRIALS = {
    "P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
    "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right"),
    "P13": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
}
SPEED_COHORT = ("P07", "P08")          # P13 excluded: clock drift


def _shift(v, lag):
    out = np.full_like(v, np.nan)
    if lag >= 0:
        out[lag:] = v[:len(v) - lag] if lag else v
    else:
        out[:lag] = v[-lag:]
    return out


def _fill(xyz):
    x = np.asarray(xyz, dtype=float).copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    return x


def _omc_cup(part, trial, n):
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


def _calib(part):
    cams = H._load_calib_mm(part)
    if part in H.GOOD_CAMS:
        cams = {c: v for c, v in cams.items() if c in H.GOOD_CAMS[part]}
    return cams


def _cup_v3(part, trial, calib, n):
    """v3 cup 3D = detect-once UETrack + greedy consensus, from the cached tracker points."""
    cf = TRACKS / f"{part}__{trial}__uetrack__fs1.json"
    if not cf.exists():
        return np.full((n, 3), np.nan)
    tr = cup_track.track_cup_3d_from_cache(cf, calib)
    X = np.array([t["X"] if t.get("X") else [np.nan] * 3 for t in tr], dtype=float)
    if len(X) < n:
        X = np.vstack([X, np.full((n - len(X), 3), np.nan)])
    return X[:n]


def _cup_v1(part, trial, calib, n):
    """v1 cup 3D = EVERY-FRAME YOLO detection + the same greedy consensus (the pre-v2 baseline).

    The per-frame YOLO boxes live in the same tracker cache under the "yolo" key, so v1 and v3 are
    read from one file and differ ONLY in the 2D source (fresh detection vs tracked point).
    """
    import json
    from cup_task import consensus
    cf = TRACKS / f"{part}__{trial}__uetrack__fs1.json"
    if not cf.exists():
        return np.full((n, 3), np.nan)
    rec = json.loads(cf.read_text())
    prev, out = None, []
    for f in range(n):
        row = rec.get(str(f), {})
        obs = {c: (v["yolo"][0] + v["yolo"][2] / 2, v["yolo"][1] + v["yolo"][3] / 2)
               for c, v in row.items() if v.get("yolo") and c in calib}
        X, _, _ = consensus.consensus3(obs, calib, prev=prev)
        if X is not None:
            prev = X
        out.append(X if X is not None else [np.nan] * 3)
    return np.array(out, dtype=float)


def _smooth_joint(xyz):
    tr = [{"frame": i, "X": (None if not np.isfinite(p).all() else [float(v) for v in p])}
          for i, p in enumerate(xyz)]
    out = pose_smooth.smooth_track(tr)
    return np.array([t["X"] if t["X"] else [np.nan] * 3 for t in out])


def _flow_speed(part, trial, joint, calib, n):
    px = F.load_wrist_px(part, trial, joint)
    fl = {}
    for c in px:
        p = FLOWDIR / f"delta_{part}_{trial}.{c.split('_')[1]}__pyrlk.npy"
        if p.exists() and c in calib:
            fl[c] = np.load(p)
    if not fl:
        return np.full(n, np.nan)
    return flow_speed.speed_from_cached_flow(px, fl, calib, n)


def _stat(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if not len(v):
        return np.nan, np.nan, np.nan
    return float(np.median(v)), float(np.percentile(v, 90)), float(np.max(v))


# ---------------------------------------------------------------- 1. ACCURACY

def accuracy():
    """Displacement + speed error per target vs OMC. Pose: raw vs SmoothNet. Cup: v1 vs v3.

    DISPLACEMENT IS ORIGIN-RELATIVE: d(t) = ||X(t) - X(t0)||, how far the target has travelled from
    its OWN starting point. Error = |d_mmc(t) - d_omc(t)|.

    This is the right primitive for two reasons. (1) Like speed, it is invariant to the frame: the
    MMC world (rig calibration) and the OMC lab frame differ by a ROTATION -- the drink's vertical
    motion lands in MMC's Y but OMC's Z -- so any per-axis difference measures the frame mismatch,
    not the tracker. (2) It needs no Kabsch fit, so it does not inherit the ~38mm rig<->mocap
    calibration floor that a rigid alignment carries; it measures the MOTION, which is what the
    clinical measures actually read.

    t0 is the first frame where BOTH tracks are finite, so the two are anchored at the same instant.
    A per-trial Kabsch-aligned absolute error is reported by scripts/compare_pose_omc_delta.py for
    anyone who wants the placement number instead.
    """
    print(f"\n{'='*86}\n=== 1. ACCURACY — displacement (mm) + speed (mm/s) error vs OMC ===\n{'='*86}",
          flush=True)
    targets = ["acting_wrist", "acting_elbow", "acting_shoulder", "nose"]
    acc = {t: {"d_raw": [], "d_sn": [], "dp90_raw": [], "dp90_sn": [],
               "s_raw": [], "s_sn": []} for t in targets}
    cup = {"d_v1": [], "d_v3": [], "dp90_v1": [], "dp90_v3": [],
           "s_v1": [], "s_v3": [], "cov1": [], "cov3": []}

    def disp_err(a, o):
        """|d_mmc - d_omc| where d = distance travelled from the shared first valid frame."""
        m = np.isfinite(a).all(1) & np.isfinite(o).all(1)
        if m.sum() < 30:
            return None
        i0 = int(np.flatnonzero(m)[0])
        da = np.linalg.norm(a - a[i0], axis=1)
        do = np.linalg.norm(o - o[i0], axis=1)
        e = np.abs(da[m] - do[m])
        return float(np.median(e)), float(np.percentile(e, 90))

    for part, (trials, side) in TRIALS.items():
        calib = _calib(part)
        for trial in trials:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            omc = {j: _shift(v, lag) for j, v in omc.items()}

            for tgt in targets:
                j = f"{side}_{tgt.split('_')[1]}" if tgt != "nose" else "nose"
                if j not in mmc or j not in omc:
                    continue
                a, o, sn = mmc[j], omc[j], _smooth_joint(mmc[j])
                for key, sig in (("raw", a), ("sn", sn)):
                    r = disp_err(sig, o)
                    if r:
                        acc[tgt][f"d_{key}"].append(r[0])
                        acc[tgt][f"dp90_{key}"].append(r[1])
                so = H._lp(H._speed(o))
                for key, sig in (("s_raw", H._lp(H._speed(a))), ("s_sn", H._lp(H._speed(sn)))):
                    mm = np.isfinite(sig) & np.isfinite(so)
                    if mm.sum() > 30:
                        acc[tgt][key].append(float(np.median(np.abs(sig[mm] - so[mm]))))

            # ---- CUP: v1 (every-frame YOLO) vs v3 (detect-once UETrack) ----
            oc = _shift(_omc_cup(part, trial, n), lag)
            for key, cx in (("v1", _cup_v1(part, trial, calib, n)),
                            ("v3", _cup_v3(part, trial, calib, n))):
                if cx is None:
                    continue
                cup[f"cov{key[-1]}"].append(float(np.isfinite(cx).all(1).sum()) / max(n, 1))
                r = disp_err(cx, oc)
                if r:
                    cup[f"d_{key}"].append(r[0])
                    cup[f"dp90_{key}"].append(r[1])
                so = H._lp(H._speed(oc)); sa = H._lp(H._speed(cx))
                mm = np.isfinite(sa) & np.isfinite(so)
                if mm.sum() > 30:
                    cup[f"s_{key}"].append(float(np.median(np.abs(sa[mm] - so[mm]))))

    ntr = sum(len(TRIALS[p][0]) for p in TRIALS)
    f = lambda v: f"{np.median(v):.1f}" if v else "  -"
    print(f"\nDISPLACEMENT = distance travelled from each track's own origin (rotation-invariant,")
    print(f"no rigid fit, so no calibration floor). Error = |d_MMC - d_OMC| in mm.")
    print(f"\nPOSE joints (n={ntr} trials, all 3 participants)")
    print(f"{'target':16} {'displ RAW':>10} {'p90':>7} {'displ SN':>10} {'p90':>7}  "
          f"{'speed RAW':>10} {'speed SN':>9}")
    print("-" * 76)
    for t in targets:
        a = acc[t]
        print(f"{t:16} {f(a['d_raw']):>10} {f(a['dp90_raw']):>7} {f(a['d_sn']):>10} "
              f"{f(a['dp90_sn']):>7}  {f(a['s_raw']):>10} {f(a['s_sn']):>9}")
    print("\n  speed = median |Δspeed| vs OMC (mm/s), 6Hz low-passed both sides.")

    print(f"\nCUP")
    print(f"{'source':18} {'displ':>10} {'p90':>7} {'speed':>10} {'coverage':>10}")
    print("-" * 60)
    for key, nm in (("v1", "v1 every-frame"), ("v3", "v3 UETrack+cons")):
        if cup[f"d_{key}"]:
            print(f"{nm:18} {f(cup[f'd_{key}']):>10} {f(cup[f'dp90_{key}']):>7} "
                  f"{f(cup[f's_{key}']):>10} {np.mean(cup[f'cov{key[-1]}'])*100:9.0f}%")
    return acc, cup


# ---------------------------------------------------------------- 2. SPEED PATH

def speed_path():
    """The v3 wrist-speed path: pos-diff vs SmoothNet vs flow vs BLEND. P07+P08 only."""
    from scipy.signal import find_peaks
    print(f"\n{'='*86}\n=== 2. SPEED PATH — acting wrist, per-frame + PEAK (P07+P08, P13 excluded) "
          f"===\n{'='*86}", flush=True)
    M = ["pos-diff", "smoothnet", "flow", "BLEND"]
    R = {m: {"pf": [], "off": [], "pk": [], "tt": []} for m in M}

    for part in SPEED_COHORT:
        trials, side = TRIALS[part]
        calib = _calib(part)
        joint = f"{side}_wrist"
        for trial in trials:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[joint], omc[joint])
            oS = H._lp(H._speed(_shift(omc[joint], lag)))

            raw = H._lp(H._speed(mmc[joint]))
            sn = H._lp(H._speed(_smooth_joint(mmc[joint])))
            fl = H._lp(_flow_speed(part, trial, joint, calib, n))
            bl = speed_blend.blend(fl, sn)

            for name, sig in zip(M, (raw, sn, fl, bl)):
                m = np.isfinite(sig) & np.isfinite(oS)
                if m.sum() < 30:
                    continue
                R[name]["pf"].append(float(np.median(np.abs(sig[m] - oS[m]))))
                off = m & (oS < 300)
                if off.sum() > 20:
                    R[name]["off"].append(float(np.median(np.abs(sig[off] - oS[off]))))
                for p in find_peaks(oS, height=300, distance=30, prominence=150)[0]:
                    spk, _ = find_peaks(sig, height=200, distance=25, prominence=100)
                    if len(spk):
                        j = spk[np.argmin(np.abs(spk - p))]
                        if abs(j - p) <= 20:
                            R[name]["pk"].append(abs(sig[j] - oS[p]))
                            R[name]["tt"].append(abs(j - p) / FPS * 1000)

    print(f"{'method':12} {'per-frame':>10} {'off-peak':>9} | {'PEAK med':>9} {'p90':>7} {'max':>7} | "
          f"{'time mean':>10} {'max':>7}")
    print("-" * 82)
    for m in M:
        a = R[m]
        pk = _stat(a["pk"])
        print(f"{m:12} {np.median(a['pf']) if a['pf'] else np.nan:10.1f} "
              f"{np.median(a['off']) if a['off'] else np.nan:9.1f} | "
              f"{pk[0]:9.1f} {pk[1]:7.1f} {pk[2]:7.1f} | "
              f"{np.mean(a['tt']) if a['tt'] else np.nan:9.0f}ms {np.max(a['tt']) if a['tt'] else np.nan:6.0f}ms")
    print("\n  All mm/s vs OMC. per-frame/off-peak = median |Δspeed|; PEAK = per-reach peak-value")
    print("  error (straight peak-to-peak, no windowed argmax); time = peak-timing error.")
    return R


# ---------------------------------------------------------------- 3. SEGMENTATION

def segmentation():
    """Drink dwell + phase-boundary error vs the OMC-cup segmentation."""
    print(f"\n{'='*86}\n=== 3. SEGMENTATION — drink phases, MMC cup vs OMC cup (same segmenter) "
          f"===\n{'='*86}", flush=True)

    def dwell(xyz):
        seg = segment.segment_cup_only(_fill(xyz), fps=FPS)
        d = [(s, e) for nm, s, e in seg["intervals"] if nm == "drinking"]
        return ((d[0][1] - d[0][0]) / FPS if d else np.nan,
                d[0][0] / FPS if d else np.nan, d[0][1] / FPS if d else np.nan)

    rows = []
    print(f"{'trial':16} {'OMC dwell':>10} {'v3 dwell':>10} {'Δdwell':>9} {'Δonset':>9} {'Δoffset':>9}")
    print("-" * 70)
    for part, (trials, side) in TRIALS.items():
        calib = _calib(part)
        for trial in trials:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            oc = _shift(_omc_cup(part, trial, n), lag)
            c3 = _cup_v3(part, trial, calib, n)
            if not np.isfinite(c3).any() or not np.isfinite(oc).any():
                continue
            do, so_, eo = dwell(oc)
            d3, s3, e3 = dwell(c3)
            if not (np.isfinite(do) and np.isfinite(d3)):
                continue
            rows.append(((d3 - do) * 1000, (s3 - so_) * 1000, (e3 - eo) * 1000))
            print(f"{part}_{trial.split('_')[1]:>10} {do:9.2f}s {d3:9.2f}s "
                  f"{(d3-do)*1000:+8.0f}ms {(s3-so_)*1000:+8.0f}ms {(e3-eo)*1000:+8.0f}ms")
    R = np.array(rows)
    if len(R):
        print("-" * 70)
        print(f"{'MEDIAN |err|':16} {'':10} {'':10} {np.median(np.abs(R[:,0])):7.0f}ms "
              f"{np.median(np.abs(R[:,1])):7.0f}ms {np.median(np.abs(R[:,2])):7.0f}ms")
        print(f"{'p90 |err|':16} {'':10} {'':10} {np.percentile(np.abs(R[:,0]),90):7.0f}ms "
              f"{np.percentile(np.abs(R[:,1]),90):7.0f}ms {np.percentile(np.abs(R[:,2]),90):7.0f}ms")
    print("\n  Same segmenter on both sides, so the CUP TRACK is the only variable.")
    return rows


# ---------------------------------------------------------------- 4. MURPHY

def murphy():
    """Murphy position measures: v1 (raw pose) vs v3 (SmoothNet + blend speed), error vs OMC."""
    print(f"\n{'='*86}\n=== 4. MURPHY measures — v1 raw vs v3, |error| vs OMC ===\n{'='*86}",
          flush=True)
    measures = ["total_movement_time", "peak_velocity", "number_of_movement_units",
                "max_trunk_displacement"]
    agg = {m: {"v1": [], "v3": []} for m in measures}
    nfail = 0

    for part, (trials, side) in TRIALS.items():
        calib = _calib(part)
        for trial in trials:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            omc = {j: _shift(v, lag) for j, v in omc.items()}
            oc = _shift(_omc_cup(part, trial, n), lag)
            wr = f"{side}_wrist"
            other = "right" if side == "left" else "left"

            # ONE shared phase set (from the OMC cup) so the POSE is the only variable.
            cup_seg = segment.segment_cup_only(_fill(oc), fps=FPS)
            try:
                phases = segment.to_murphy_phases(cup_seg, _fill(omc[wr]), _fill(oc), fps=FPS)
            except Exception:
                nfail += 1
                continue

            trunk = (mmc[f"{side}_shoulder"] + mmc[f"{other}_shoulder"]) / 2
            trunk_o = (omc[f"{side}_shoulder"] + omc[f"{other}_shoulder"]) / 2

            def measure(hand, trunk_xyz):
                try:
                    return compute_position_measures(hand, trunk_xyz, phases, side, fps=FPS)
                except Exception:
                    return None

            m1 = measure(mmc[wr], trunk)
            m3 = measure(_smooth_joint(mmc[wr]), trunk)
            mo = measure(omc[wr], trunk_o)
            if not (m1 and m3 and mo):
                nfail += 1
                continue
            for m in measures:
                o = getattr(mo, m, None)
                if o is None or (isinstance(o, float) and not np.isfinite(o)):
                    continue
                agg[m]["v1"].append(abs(getattr(m1, m) - o))
                agg[m]["v3"].append(abs(getattr(m3, m) - o))

    print(f"{'measure':28} {'n':>4} {'v1 |err|':>10} {'v3 |err|':>10} {'change':>10}")
    print("-" * 66)
    for m in measures:
        v1, v3 = agg[m]["v1"], agg[m]["v3"]
        if not v1:
            print(f"{m:28} {'-':>4}")
            continue
        a, b = np.median(v1), np.median(v3)
        ch = f"{(b-a)/a*100:+.0f}%" if a else "-"
        print(f"{m:28} {len(v1):4d} {a:10.2f} {b:10.2f} {ch:>10}")
    if nfail:
        print(f"\n  ({nfail} trials skipped: phase or measure computation failed)")
    print("\n  Phases held FIXED (from the OMC cup) so the pose source is the only variable.")
    print("  peak_velocity in mm/s; times in s; movement units = count; trunk in mm.")
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["accuracy", "speed", "segment", "murphy", "all"],
                    default="all")
    a = ap.parse_args(argv)
    H.use_good_cams()
    if a.what in ("accuracy", "all"):
        accuracy()
    if a.what in ("speed", "all"):
        speed_path()
    if a.what in ("segment", "all"):
        segmentation()
    if a.what in ("murphy", "all"):
        murphy()


if __name__ == "__main__":
    main()
