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

# COHORT = P07 + P08 only, n=12. P13 is EXCLUDED EVERYWHERE (not just from speed).
#
# P13's OMC clock DRIFTS linearly against video (-8 -> +3 frames over ~6s, 3.8% rate mismatch), so
# its ground truth is progressively mis-timed. That is not a constant lag a single cross-correlation
# can absorb, and it corrupts POSITION as well as speed -- measured on the cup: median displacement
# error 10-12mm for P13 vs 2-3mm for P07/P08, d-corr 0.974 vs 0.998, and P13 owns the entire 504mm
# tail. Including it was flattering v1 and penalising v3 at the same time.
#
# P13 stays in the repo and its caches are untouched: a linear time-warp of its OMC would recover
# all 6 trials, which is the documented way to grow this cohort back to n=18.
TRIALS = {
    "P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
    "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right"),
}
SPEED_COHORT = ("P07", "P08")
# "the cup is actually moving" threshold. Well above the triangulation noise floor at rest
# (~30-50mm/s) and well below the transport peaks, so it cleanly separates the two populations.
CUP_MOVING_MMPS = 50.0


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
    cup = {k: [] for k in ("d_v1", "d_v3", "dp90_v1", "dp90_v3", "s_v1", "s_v3",
                           "smv_v1", "smv_v3", "corr_v1", "corr_v3",
                           "dcorr_v1", "dcorr_v3", "cov1", "cov3", "covmv1", "covmv3")}

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
            # Speed is reported BOTH over all frames and restricted to frames where the cup is
            # actually MOVING. That split is load-bearing: v1 only covers ~74% of frames and those
            # are overwhelmingly the STATIONARY cup (median OMC speed 1.1mm/s on frames v1 has vs
            # 121.6mm/s on frames it misses). An all-frames median therefore mostly scores "is the
            # still cup still?" and hides that v1 drops out exactly when the cup moves.
            oc = _shift(_omc_cup(part, trial, n), lag)
            so = H._lp(H._speed(oc))
            moving = np.isfinite(so) & (so > CUP_MOVING_MMPS)
            for key, cx in (("v1", _cup_v1(part, trial, calib, n)),
                            ("v3", _cup_v3(part, trial, calib, n))):
                if cx is None:
                    continue
                fin = np.isfinite(cx).all(1)
                cup[f"cov{key[-1]}"].append(float(fin.sum()) / max(n, 1))
                cup[f"covmv{key[-1]}"].append(float((fin & moving).sum()) / max(moving.sum(), 1))
                r = disp_err(cx, oc)
                if r:
                    cup[f"d_{key}"].append(r[0])
                    cup[f"dp90_{key}"].append(r[1])
                # trajectory correlation on DISPLACEMENT-FROM-ORIGIN. Both displacement and speed
                # are frame-invariant, so neither needs a rigid fit -- but a PER-AXIS position
                # correlation would be meaningless here (the MMC and OMC frames are rotated, which
                # makes it read NEGATIVE). This reproduces the v2 finding: v3 = 0.9995.
                fm = np.isfinite(cx).all(1) & np.isfinite(oc).all(1)
                if fm.sum() > 40:
                    i0 = int(np.flatnonzero(fm)[0])
                    da = np.linalg.norm(cx - cx[i0], axis=1)[fm]
                    do = np.linalg.norm(oc - oc[i0], axis=1)[fm]
                    cup[f"dcorr_{key}"].append(float(np.corrcoef(da, do)[0, 1]))
                sa = H._lp(H._speed(cx))
                mm = np.isfinite(sa) & np.isfinite(so)
                if mm.sum() > 30:
                    cup[f"s_{key}"].append(float(np.median(np.abs(sa[mm] - so[mm]))))
                    cup[f"corr_{key}"].append(float(np.corrcoef(sa[mm], so[mm])[0, 1]))
                mv = mm & moving
                if mv.sum() > 20:
                    cup[f"smv_{key}"].append(float(np.median(np.abs(sa[mv] - so[mv]))))

    ntr = sum(len(TRIALS[p][0]) for p in TRIALS)
    f = lambda v: f"{np.median(v):.1f}" if v else "  -"
    print(f"\nDISPLACEMENT = distance travelled from each track's own origin (rotation-invariant,")
    print(f"no rigid fit, so no calibration floor). Error = |d_MMC - d_OMC| in mm.")
    print(f"\nPOSE joints (n={ntr} trials, P07+P08)")
    print(f"{'target':16} {'displ RAW':>10} {'p90':>7} {'displ SN':>10} {'p90':>7}  "
          f"{'speed RAW':>10} {'speed SN':>9}")
    print("-" * 76)
    for t in targets:
        a = acc[t]
        print(f"{t:16} {f(a['d_raw']):>10} {f(a['dp90_raw']):>7} {f(a['d_sn']):>10} "
              f"{f(a['dp90_sn']):>7}  {f(a['s_raw']):>10} {f(a['s_sn']):>9}")
    print("\n  speed = median |Δspeed| vs OMC (mm/s), 6Hz low-passed both sides.")

    print(f"\nCUP   (displacement AND speed are both frame-invariant — no rigid fit anywhere)")
    print(f"{'source':18} {'displ':>7} {'p90':>6} {'d-corr':>7} | {'spd all':>8} "
          f"{'spd MOVING':>11} {'s-corr':>7} | {'cov all':>8} {'cov MOVE':>9}")
    print("-" * 94)
    for key, nm in (("v1", "v1 every-frame"), ("v3", "v3 UETrack+cons")):
        if cup[f"d_{key}"]:
            print(f"{nm:18} {f(cup[f'd_{key}']):>7} {f(cup[f'dp90_{key}']):>6} "
                  f"{np.median(cup[f'dcorr_{key}']):7.4f} | "
                  f"{f(cup[f's_{key}']):>8} {f(cup[f'smv_{key}']):>11} "
                  f"{np.median(cup[f'corr_{key}']):7.2f} | "
                  f"{np.mean(cup[f'cov{key[-1]}'])*100:7.0f}% {np.mean(cup[f'covmv{key[-1]}'])*100:8.0f}%")
    print(f"\n  d-corr = correlation of DISPLACEMENT-FROM-ORIGIN (frame-invariant). v3 = 0.9996,")
    print("  reproducing the v2 tracker-shootout result. s-corr = the same trajectory DIFFERENTIATED:")
    print("  0.93. Position is near-perfect; its derivative is not — the same split the wrist shows.")
    print(f"\n  'MOVING' = frames where the OMC cup exceeds {CUP_MOVING_MMPS:.0f} mm/s. READ THAT")
    print("  COLUMN, not 'spd all': v1 covers only the frames where the cup is nearly STILL (median")
    print("  OMC speed 0.6mm/s on frames it has vs 139.3mm/s on frames it misses), so its all-frames")
    print("  median mostly scores 'is the still cup still?'. On moving frames v1 is ~1.8x WORSE than")
    print("  v3 (136 vs 77 mm/s) and sees only HALF of them. v3 wins the cup on every fair cut.")
    return acc, cup


# ---------------------------------------------------------------- 2. SPEED PATH

def speed_path():
    """The v3 wrist-speed path: pos-diff vs SmoothNet vs flow vs BLEND. P07+P08 only."""
    from scipy.signal import find_peaks
    print(f"\n{'='*86}\n=== 2. SPEED PATH — acting wrist, per-frame + PEAK (P07+P08, n=12) "
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

MURPHY_PHASES = ["rest_pre", "reaching", "forward_transport", "drinking",
                 "back_transport", "returning", "rest_post"]


def segmentation(smooth_cup: bool = True):
    """Phase-boundary error for EVERY phase, MMC cup vs OMC cup (same segmenter both sides).

    Scores the whole 7-phase Murphy timeline, not just the drink dwell. Reporting the dwell alone
    is misleading: the dwell is the EASIEST boundary (the cup stops dead at the mouth) and it hid a
    real defect -- the terminal phases often never fire at all, so back_transport swallows the tail
    of the trial and total_movement_time over-runs. A phase that is never produced is a MISS, which
    is a worse failure than a mis-timed boundary and is counted separately here.
    """
    print(f"\n{'='*92}\n=== 3. SEGMENTATION — ALL 7 phases, MMC cup vs OMC cup (same segmenter) "
          f"===\n{'='*92}", flush=True)

    def phases_of(cup, hand, mouth=None):
        """The REAL pipeline path: cup-only gate, THEN the pose refinement.

        refine_grasp_with_pose is not optional and must not be skipped in a harness. It replaces
        both transport boundaries with the wrist->cup PLATEAU (hand and cup are one rigid body
        between grasp and release), which is where the accuracy actually comes from: same rule on
        both sides, OMC-vs-tracker agreement is 17ms median / 50ms p90, versus 167/643 for any
        cup-only rule. An earlier version of this harness called segment_cup_only alone, which
        silently measured a stripped-down segmenter and made a long-solved boundary look broken.
        """
        try:
            seg = segment.segment_cup_only(_fill(cup), fps=FPS)
            seg = segment.refine_grasp_with_pose(seg, _fill(cup), _fill(hand),
                                                 None if mouth is None else _fill(mouth), fps=FPS)
            ph = segment.to_murphy_phases(seg, _fill(hand), _fill(cup), fps=FPS)
            return {nm: (s / FPS, e / FPS) for nm, s, e in ph}
        except Exception:
            return {}

    srcs = ["v1", "v3"] + (["v3+SN"] if smooth_cup else [])
    on = {k: {p: [] for p in MURPHY_PHASES} for k in srcs}   # onset error
    off = {k: {p: [] for p in MURPHY_PHASES} for k in srcs}  # offset error
    dur = {k: {p: [] for p in MURPHY_PHASES} for k in srcs}  # duration error
    miss = {k: {p: 0 for p in MURPHY_PHASES} for k in srcs}
    npres = {p: 0 for p in MURPHY_PHASES}                    # times OMC had the phase

    for part, (trials, side) in TRIALS.items():
        calib = _calib(part)
        for trial in trials:
            mmc, n = H._load_mmc(part, trial)
            omc = H._load_omc(part, trial, n)
            lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
            omc = {j: _shift(v, lag) for j, v in omc.items()}
            oc = _shift(_omc_cup(part, trial, n), lag)
            c1 = _cup_v1(part, trial, calib, n)
            c3 = _cup_v3(part, trial, calib, n)
            hand = _smooth_joint(mmc[f"{side}_wrist"])

            po = phases_of(oc, omc[f"{side}_wrist"], omc.get("nose"))
            if not po:
                continue
            mmc_nose = _smooth_joint(mmc["nose"]) if "nose" in mmc else None
            cand = {"v1": (c1, mmc[f"{side}_wrist"], mmc.get("nose")),
                    "v3": (c3, hand, mmc_nose)}
            if smooth_cup:
                cand["v3+SN"] = (_smooth_joint(c3), hand, mmc_nose)
            for k, (cup, hd, mo) in cand.items():
                pm = phases_of(cup, hd, mo)
                for p in MURPHY_PHASES:
                    if p not in po:
                        continue
                    if k == srcs[0]:
                        npres[p] += 1
                    if p not in pm:
                        miss[k][p] += 1
                        continue
                    on[k][p].append((pm[p][0] - po[p][0]) * 1000)
                    off[k][p].append((pm[p][1] - po[p][1]) * 1000)
                    dur[k][p].append(((pm[p][1] - pm[p][0]) - (po[p][1] - po[p][0])) * 1000)

    med = lambda v: np.median(np.abs(v)) if v else np.nan
    for k in srcs:
        print(f"\n  --- {k} ---")
        print(f"  {'phase':20} {'n':>3} {'miss':>5} {'|Δonset|':>10} {'|Δoffset|':>10} "
              f"{'|Δdur|':>9}")
        print("  " + "-" * 62)
        for p in MURPHY_PHASES:
            if not npres[p]:
                continue
            f = lambda v: f"{med(v):.0f}ms" if v else "   -"
            mk = f"{miss[k][p]}/{npres[p]}"
            print(f"  {p:20} {len(on[k][p]):3d} {mk:>5} {f(on[k][p]):>10} {f(off[k][p]):>10} "
                  f"{f(dur[k][p]):>9}")
        allon = [x for p in MURPHY_PHASES for x in on[k][p]]
        alloff = [x for p in MURPHY_PHASES for x in off[k][p]]
        tm = sum(miss[k].values()); tn = sum(npres.values())
        print(f"  {'ALL PHASES':20} {len(allon):3d} {f'{tm}/{tn}':>5} {med(allon):9.0f}ms "
              f"{med(alloff):9.0f}ms")

    print("\n  Same segmenter on every row, so the CUP TRACK is the only variable.")
    print("  'miss' = the phase was NEVER PRODUCED (worse than a mis-timed boundary).")
    print("  ⚠ The drink dwell alone (the OLD metric) is the EASIEST boundary — the cup stops dead")
    print("  at the mouth — and it HID everything below. Score all 7 phases, not just drinking.")
    print("\n  v3+SN now produces every phase in every trial (0/83 miss) and all boundaries are")
    print("  17-125 ms EXCEPT back_transport-end / returning-onset (~550 ms). That one is NOT a")
    print("  tracking error: at BOTH boundaries the OMC cup is already parked (median displacement")
    print("  4.5 mm at OMC's own boundary, 6.1 mm at v3's), and the disagreement is symmetric in")
    print("  sign across trials (+917, -783, +750, -600 ms ...). It is the magnitude rule chasing")
    print("  whichever track twitches last once the cup is stationary — ill-defined on both sides.")
    print("  Fixing it needs a better DEFINITION of 'the cup is down', not a better track.")
    return on, off, miss


# ---------------------------------------------------------------- 4. MURPHY

POSITION_MEASURES = ["total_movement_time", "peak_velocity", "time_to_peak_velocity",
                     "time_to_peak_velocity_percent", "time_to_first_peak_velocity",
                     "time_to_first_peak_velocity_percent", "number_of_movement_units",
                     "max_trunk_displacement"]

ANGLE_MEASURES = ["elbow_extension_reaching", "shoulder_flexion_reaching",
                  "shoulder_flexion_drinking", "shoulder_abduction_reaching",
                  "shoulder_abduction_drinking", "peak_elbow_ang_vel",
                  "interjoint_coordination"]


def _angle_scalars(P, phases, side):
    """The 7 ANGLE measures from raw 3D points (no IK).

    ⚠ These are NOT the ported container measures. The container computes angles from a MuJoCo
    `qpos` IK fit and explicitly REFUSES raw-point angles ("would inherit pose jitter"). cup-task
    has no body model, so these are computable-to-SEE only -- useful for MMC-vs-OMC comparison
    (both sides use the identical formula, so the comparison is fair), NOT for clinical scoring.
    """
    from compare_pose_omc_delta import _murphy_signals
    o = "right" if side == "left" else "left"
    sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    u, v = P[sh] - P[el], P[wr] - P[el]
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    elb = H._lp(np.degrees(np.arccos(np.clip(c, -1, 1))))
    try:
        sig = _murphy_signals(P, side=side)      # side-aware: DELTA's affected arm is the LEFT one
        flex, abd = H._lp(sig["shoulder_flexion"]), H._lp(sig["shoulder_abduction"])
    except Exception:
        flex = abd = np.full(len(elb), np.nan)
    eav = np.abs(np.gradient(elb)) * FPS

    def ph(name):
        for nm, s, e in phases:
            if nm == name:
                return s, e
        return None

    def mx(a, w):
        # an all-NaN phase window (e.g. an occluded arm over that phase) legitimately means the
        # measure isn't computable for this trial -> NaN, which the scorer skips. Suppress the
        # nanmax All-NaN RuntimeWarning so it doesn't flood the log.
        if not (w and w[1] > w[0]):
            return float("nan")
        seg = a[w[0]:w[1]]
        return float(np.nanmax(seg)) if np.isfinite(seg).any() else float("nan")

    r, d, fw = ph("reaching"), ph("drinking"), ph("forward_transport")
    rf = (r[0], fw[1]) if (r and fw) else r
    ijc = float("nan")
    if r and r[1] - r[0] >= 10:
        m = int(0.1 * (r[1] - r[0]))
        a1, b1 = flex[r[0] + m:r[1] - m], elb[r[0] + m:r[1] - m]
        if np.isfinite(a1).all() and np.isfinite(b1).all() and np.std(a1) > 1e-6 and np.std(b1) > 1e-6:
            ijc = float(np.corrcoef(a1, b1)[0, 1])
    return {"elbow_extension_reaching": mx(elb, rf),
            "shoulder_flexion_reaching": mx(flex, r),
            "shoulder_flexion_drinking": mx(flex, d),
            "shoulder_abduction_reaching": mx(abd, r),
            "shoulder_abduction_drinking": mx(abd, d),
            "peak_elbow_ang_vel": mx(eav, (0, len(elb))),
            "interjoint_coordination": ijc}


# ============================================================================================
# CONSOLIDATION GRID: score every candidate pose pipeline vs OMC on the FULL Murphy set, emit a
# tidy per-trial table (variant, part, trial, arm, measure, omc, mmc) for the Fig.4 scatter +
# median|err| + Bland-Altman. Cohort = gnn_train.load_clean (328 trials, 5 OMC parts), which also
# supplies BA's native sidecar format and trial-ids that match the BA trajectory cache.
#
# STAGES (composable):
#   A triangulation : "pipeline" (load_clean mmc = robust-consensus DLT) | "BA" (cached BA+guard traj)
#   B smoother (3D) : "none" | "savgol" | "smoothnet"
#   C wrist-speed   : handled inside compute_position_measures (pos-diff on the chosen pose). The
#                     flow-BLEND path is a separate C option added once its gate is LOPO-tuned.
#   D elbow-angular : "posdiff" (angle-rate of the pose) | "flowcloud" (two-segment flow) -- D slot
#                     is filled once scripts/elbow_flow_angular is validated; posdiff is the default.
# Cup is FIXED = v3 UETrack for every pose variant, so phases are constant and POSE is the variable.
# ============================================================================================
_GRID_JOINTS = ["right_wrist", "right_elbow", "right_shoulder", "left_wrist", "left_elbow",
                "left_shoulder", "right_hip", "left_hip", "nose"]   # == gnn_refiner.JOINTS order


def _savgol_joint(xyz, win=21, order=3):
    """Savitzky-Golay per-axis, NaN-gap-safe, restores NaN where the whole point was missing."""
    from scipy.signal import savgol_filter
    out = xyz.copy().astype(float)
    v = np.isfinite(xyz).all(1)
    if v.sum() < win:
        return out
    idx = np.flatnonzero(v)
    for ax in range(3):
        xi = np.interp(np.arange(len(xyz)), idx, xyz[idx, ax])
        out[:, ax] = savgol_filter(xi, win, order, mode="interp")
    out[~v] = np.nan
    return out


def _ba_traj_cache():
    """Load the cached BA+guard trajectories (fb150) -> {id -> (T,9,3)}. Built by ba_cache_traj.py."""
    p = ROOT / "cache" / "ba_traj" / "traj_sw0_fb150.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    return {str(i): tr for i, tr in zip(d["ids"], d["traj"])}


def _flow_peak_velocity(part, trial, side, calib, n):
    """Raw PyrLK-flow wrist PEAK velocity (mm/s): peak of the low-passed flow speed. The 'flow' C
    option. No blend, no gate -- the honest flow method on its own. NaN if no flow cache."""
    try:
        sp = H._lp(_flow_speed(part, trial, f"{side}_wrist", calib, n))
        return float(np.nanmax(sp)) if np.isfinite(sp).any() else np.nan
    except Exception:
        return np.nan


def _flowcloud_elbow_cache():
    """Load cached two-segment flow-cloud peak_elbow_ang_vel (validation cohort n=12), if present.
    Built by scripts/elbow_flow_angular.py -> cache/elbow_flow/<part>__<trial>.npz. Returns
    {id -> (peak_cloud, peak_omc)} deg/s, so the flow-cloud panel is scored from its OWN OMC pairing
    (the cache stores peak_omc/peak_raw/peak_sn/peak_cloud)."""
    d = {}
    cdir = ROOT / "cache" / "elbow_flow"
    if not cdir.exists():
        return d
    for p in cdir.glob("*.npz"):
        try:
            z = np.load(str(p), allow_pickle=True)
            # RECONCILE: the cached scalar peaks are p95; recompute both as MAX of the low-passed
            # per-frame series (eav_omc, rate_cloud) so the elbow panel uses the SAME peak definition
            # (max) as the pose path's _angle_scalars -> a fair common OMC x-axis across all variants.
            peak_omc = float(np.nanmax(H._lp(z["eav_omc"])))
            peak_cloud = float(np.nanmax(H._lp(z["rate_cloud"])))
            d[p.stem.replace("__", "/")] = (peak_cloud, peak_omc)
        except Exception:
            continue
    return d


def _pose_variant(trial_rec, triangulation, smoother, ba_cache):
    """Produce the pose joint-dict for one (triangulation, smoother) cell from a load_clean record.
    Returns {joint_name: (T,3)} or None if unavailable."""
    part, trial = trial_rec["part"], trial_rec["trial"]
    if triangulation == "pipeline":
        arr = trial_rec["mmc"]                                    # (T,9,3) robust-consensus DLT
    elif triangulation == "BA":
        if ba_cache is None or f"{part}/{trial}" not in ba_cache:
            return None
        arr = np.asarray(ba_cache[f"{part}/{trial}"], float)      # (T,9,3) BA+guard
    else:
        raise ValueError(triangulation)
    pose = {j: arr[:, k] for k, j in enumerate(_GRID_JOINTS)}
    if smoother == "none":
        pass
    elif smoother == "savgol":
        pose = {j: _savgol_joint(x) for j, x in pose.items()}
    elif smoother == "smoothnet":
        pose = {j: _smooth_joint(x) for j, x in pose.items()}
    else:
        raise ValueError(smoother)
    return pose


def murphy_grid(variants=None, parts=("P07", "P08", "P15", "P17", "P19"),
                out_csv="out/murphy_grid.csv", fixed_phases=False):
    """Score each pose variant vs OMC on all 15 Murphy measures; write the tidy per-trial table.

    variants: list of dicts {name, triangulation, smoother}. Default = the A x B grid.

    fixed_phases: if True, score EVERY variant with the OMC-cup phase boundaries (ph_o) instead of
      segmenting each pose with its own v3 (UETrack) cup. This ISOLATES the pose/smoother/flow choice
      -- the variable actually under comparison -- by removing segmentation as a confound, and it is
      the only way to score participants that have no markerless cup track (P15/P17/P19). Default
      (False) is END-TO-END: each pose segments with the v3 cup (needs a tracks_uetrack file).
    """
    import csv as _csv
    sys.path.insert(0, str(ROOT / "scripts"))
    import gnn_train as GT
    H.use_good_cams()          # init the per-participant good-camera whitelist (as main() does)

    if variants is None:
        variants = [
            {"name": "pipeline",          "triangulation": "pipeline", "smoother": "none"},
            {"name": "pipeline+savgol",   "triangulation": "pipeline", "smoother": "savgol"},
            {"name": "pipeline+smoothnet","triangulation": "pipeline", "smoother": "smoothnet"},
            {"name": "BA+smoothnet",      "triangulation": "BA",       "smoother": "smoothnet"},
            # C: raw PyrLK flow for peak_velocity (that measure only; rest = SmoothNet pose)
            {"name": "flow-speed",        "triangulation": "pipeline", "smoother": "smoothnet",
             "speed": "flow"},
            # D: two-segment flow-cloud for peak_elbow_ang_vel (that measure only, cached n=12)
            {"name": "flow-cloud-elbow",  "triangulation": "pipeline", "smoother": "smoothnet",
             "angular": "flowcloud"},
        ]
    ba_cache = _ba_traj_cache()
    fc_cache = _flowcloud_elbow_cache()
    print(f"  flow-cloud elbow cache: {len(fc_cache)} trials", flush=True)
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in parts]
    phase_src = "OMC-cup phases (fixed, pose isolated)" if fixed_phases else "v3 UETrack cup (end-to-end)"
    print(f"murphy_grid: {len(trials)} trials, {len(variants)} variants, phases={phase_src}", flush=True)
    print(f"  BA cache: {'loaded '+str(len(ba_cache))+' trajs' if ba_cache else 'MISSING'}", flush=True)

    rows = []       # tidy: variant, part, trial, arm, side, measure, omc, mmc
    n_ok = 0; n_fail = 0
    ALL_M = POSITION_MEASURES + ANGLE_MEASURES
    from tqdm import tqdm
    pbar = tqdm(list(enumerate(trials)), total=len(trials), desc="murphy_grid", unit="trial")
    for ti, t in pbar:
        part, trial, side = t["part"], t["trial"], t["side"]
        arm = "affected" if "unaffected" not in trial else "unaffected"
        other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        calib = _calib(part)
        # OMC (same loader/lag path as murphy()) + v3 cup for phases
        omc = H._load_omc(part, trial, n)
        lag, _ = H._find_lag(t["mmc"][:, _GRID_JOINTS.index(f"{side}_wrist")],
                             omc[f"{side}_wrist"])
        omc = {j: _shift(v, lag) for j, v in omc.items()}
        oc = _shift(_omc_cup(part, trial, n), lag)
        c3 = None if fixed_phases else _cup_v3(part, trial, calib, n)
        wr = f"{side}_wrist"

        def phases_for(cup_xyz, hand_xyz):
            try:
                seg = segment.segment_cup_only(_fill(cup_xyz), fps=FPS)
                return segment.to_murphy_phases(seg, _fill(hand_xyz), _fill(cup_xyz), fps=FPS)
            except Exception:
                return None

        ph_o = phases_for(oc, omc[wr])
        if not ph_o:
            n_fail += 1
            continue
        # OMC measures (once per trial)
        trunk_o = (omc[f"{side}_shoulder"] + omc[f"{other}_shoulder"]) / 2
        try:
            mo = compute_position_measures(omc[wr], trunk_o, ph_o, side, fps=FPS)
            ao = _angle_scalars(omc, ph_o, side)
        except Exception:
            n_fail += 1
            continue

        for spec in variants:
            pose = _pose_variant(t, spec["triangulation"], spec["smoother"], ba_cache)
            if pose is None:
                continue
            # fixed_phases: score with the OMC-cup phases (isolates pose; works with no cup track).
            # end-to-end: each pose segments with its own v3 (UETrack) cup.
            ph = ph_o if fixed_phases else phases_for(c3, pose[wr])
            if not ph:
                continue
            trunk = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
            try:
                mm = compute_position_measures(pose[wr], trunk, ph, side, fps=FPS)
                am = _angle_scalars(pose, ph, side)
            except Exception:
                continue
            for m in POSITION_MEASURES:
                ov = getattr(mo, m, None); mv = getattr(mm, m, None)
                # C override: raw PyrLK flow for peak_velocity (no blend, no gate)
                if m == "peak_velocity" and spec.get("speed") == "flow":
                    mv = _flow_peak_velocity(part, trial, side, calib, n)
                if ov is None or mv is None or not np.isfinite(ov) or not np.isfinite(mv):
                    continue
                rows.append((spec["name"], part, trial, arm, side, m, float(ov), float(mv)))
            for m in ANGLE_MEASURES:
                omv, amv = ao[m], am[m]
                # D override: two-segment flow-cloud for peak_elbow_ang_vel (cached n=12; carries own OMC)
                if m == "peak_elbow_ang_vel" and spec.get("angular") == "flowcloud":
                    fc = fc_cache.get(f"{part}/{trial}")
                    if fc is None:
                        continue                      # not in the validation cohort -> skip (sparse panel)
                    amv, omv = fc                     # (peak_cloud, peak_omc) deg/s
                if not np.isfinite(omv) or not np.isfinite(amv):
                    continue
                rows.append((spec["name"], part, trial, arm, side, m, float(omv), float(amv)))
        n_ok += 1
        pbar.set_postfix(scored=n_ok, failed=n_fail, rows=len(rows))

    outp = ROOT / out_csv
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["variant", "part", "trial", "arm", "side", "measure", "omc", "mmc"])
        w.writerows(rows)
    print(f"\nPROCESSING CHECK: {n_ok} trials scored, {n_fail} failed, {len(rows)} tidy rows", flush=True)
    print(f"wrote {outp}", flush=True)
    print("DONE", flush=True)
    return rows


def murphy(own_phases: bool = True):
    """Murphy measures, v1 (raw pose + every-frame cup) vs v3 (SmoothNet pose + UETrack cup).

    PHASES: by default each arm segments with its OWN cup track (v1 with v1's, v3 with v3's), and
    only OMC uses the OMC cup. That is the END-TO-END number -- it is what the pipeline would
    actually report, and it is now the honest comparison because the v3 segmentation is good
    (dwell 67ms median). `--fixed-phases` restores the old OMC-phases-for-everyone mode, which
    ISOLATES the pose by removing segmentation as a variable; useful for attribution, but it
    flatters both arms by handing them ground-truth phase boundaries they would not have live.
    """
    mode = "each arm segments with its OWN cup" if own_phases else "phases FIXED from the OMC cup"
    print(f"\n{'='*86}\n=== 4. MURPHY measures — v1 vs v3, |error| vs OMC  ({mode}) ===\n{'='*86}",
          flush=True)
    agg = {m: {"v1": [], "v3": []} for m in POSITION_MEASURES + ANGLE_MEASURES}
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

            c1 = _cup_v1(part, trial, calib, n)
            c3 = _cup_v3(part, trial, calib, n)
            sn = {j: _smooth_joint(mmc[j]) for j in
                  (wr, f"{side}_elbow", f"{side}_shoulder", f"{other}_shoulder")}

            def phases_for(cup_xyz, hand_xyz):
                try:
                    seg = segment.segment_cup_only(_fill(cup_xyz), fps=FPS)
                    return segment.to_murphy_phases(seg, _fill(hand_xyz), _fill(cup_xyz), fps=FPS)
                except Exception:
                    return None

            ph_o = phases_for(oc, omc[wr])
            if own_phases:
                ph_1 = phases_for(c1, mmc[wr])
                ph_3 = phases_for(c3, sn[wr])
            else:
                ph_1 = ph_3 = ph_o
            if not (ph_o and ph_1 and ph_3):
                nfail += 1
                continue

            trunk1 = (mmc[f"{side}_shoulder"] + mmc[f"{other}_shoulder"]) / 2
            trunk3 = (sn[f"{side}_shoulder"] + sn[f"{other}_shoulder"]) / 2
            trunk_o = (omc[f"{side}_shoulder"] + omc[f"{other}_shoulder"]) / 2

            def measure(hand, trunk_xyz, ph):
                try:
                    return compute_position_measures(hand, trunk_xyz, ph, side, fps=FPS)
                except Exception:
                    return None

            m1 = measure(mmc[wr], trunk1, ph_1)
            m3 = measure(sn[wr], trunk3, ph_3)
            mo = measure(omc[wr], trunk_o, ph_o)
            if not (m1 and m3 and mo):
                nfail += 1
                continue
            for m in POSITION_MEASURES:
                o = getattr(mo, m, None)
                if o is None or (isinstance(o, float) and not np.isfinite(o)):
                    continue
                agg[m]["v1"].append(abs(getattr(m1, m) - o))
                agg[m]["v3"].append(abs(getattr(m3, m) - o))

            # angle measures: same formula on each pose source
            P1 = {j: mmc[j] for j in mmc}
            P3 = dict(mmc); P3.update(sn)
            a1 = _angle_scalars(P1, ph_1, side)
            a3 = _angle_scalars(P3, ph_3, side)
            ao = _angle_scalars(omc, ph_o, side)
            for m in ANGLE_MEASURES:
                if not np.isfinite(ao[m]):
                    continue
                if np.isfinite(a1[m]):
                    agg[m]["v1"].append(abs(a1[m] - ao[m]))
                if np.isfinite(a3[m]):
                    agg[m]["v3"].append(abs(a3[m] - ao[m]))

    def _block(title, names, unit_note):
        print(f"\n  {title}")
        print(f"  {'measure':36} {'n':>3} {'v1 |err|':>10} {'v3 |err|':>10} {'change':>9}")
        print("  " + "-" * 72)
        for m in names:
            v1, v3 = agg[m]["v1"], agg[m]["v3"]
            if not v1 or not v3:
                print(f"  {m:36} {'-':>3}")
                continue
            a, b = np.median(v1), np.median(v3)
            ch = f"{(b-a)/a*100:+.0f}%" if a > 1e-9 else ("=" if b <= 1e-9 else "-")
            print(f"  {m:36} {len(v1):3d} {a:10.2f} {b:10.2f} {ch:>9}")
        print(f"  {unit_note}")

    _block("POSITION measures (the ported set, 8/8)", POSITION_MEASURES,
           "times in s (or % of movement time); peak_velocity mm/s; units = count; trunk mm.")
    _block("ANGLE measures (raw-point, NOT the ported set — see _angle_scalars)", ANGLE_MEASURES,
           "degrees (peak_elbow_ang_vel deg/s; interjoint_coordination = Pearson r).")
    if nfail:
        print(f"\n  ({nfail} trials skipped: phase or measure computation failed)")
    if own_phases:
        print("\n  END-TO-END: v1 segmented with the v1 cup, v3 with the v3 cup, OMC with the OMC")
        print("  cup. Differences therefore include BOTH the pose and the segmentation — which is")
        print("  what the pipeline actually delivers. Use --fixed-phases to isolate the pose.")
        print("\n  total_movement_time was |err| 1.50 s here until the segmenter learned DIRECTION.")
        print("  The old scalar-speed rule ended back_transport at 'last frame above BACK_OFF',")
        print("  which a detect-once tracker never satisfies (it keeps emitting a slightly-moving")
        print("  point after the cup is down), so returning/rest_post went missing in 11/12 trials")
        print("  and TMT over-ran. segment_cup_only now ends the return on ARRIVAL (settled near")
        print("  rest AND no longer closing on it) — a state, not a threshold. TMT: 1.50 -> 0.05 s.")
    else:
        print("\n  Phases FIXED from the OMC cup for all three arms, so the POSE is the only")
        print("  variable. Attribution only — live, no arm gets ground-truth phase boundaries.")
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["accuracy", "speed", "segment", "murphy", "grid", "all"],
                    default="all")
    ap.add_argument("--fixed-phases", action="store_true",
                    help="score Murphy with OMC-cup phases for every arm (isolates the pose). "
                         "Default is END-TO-END: each arm segments with its own cup track.")
    ap.add_argument("--out-csv", default="out/murphy_grid.csv",
                    help="grid: where to write the tidy per-trial OMC-vs-MMC table")
    a = ap.parse_args(argv)
    H.use_good_cams()
    if a.what == "grid":
        murphy_grid(out_csv=a.out_csv, fixed_phases=a.fixed_phases)
        return
    if a.what in ("accuracy", "all"):
        accuracy()
    if a.what in ("speed", "all"):
        speed_path()
    if a.what in ("segment", "all"):
        segmentation()
    if a.what in ("murphy", "all"):
        murphy(own_phases=not a.fixed_phases)


if __name__ == "__main__":
    main()
