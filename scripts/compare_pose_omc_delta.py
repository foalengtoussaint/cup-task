"""MMC pose (yolo26 triangulated) vs OMC pose (mocap markers), DELTA cohort.

Two comparisons, because the two live in DIFFERENT coordinate frames (MMC = calib-camera world,
metres; OMC = mocap lab, mm -> m). See cache/delta/README.md.

  FRAME-INVARIANT (no alignment needed -- what the Murphy measures actually use):
    * inter-joint DISTANCES   |wrist-shoulder|, |elbow-wrist|, |wrist-head|
    * joint SPEEDS            (masked, no interp across gaps -- the qtm despike lesson)
    * ELBOW ANGLE            shoulder-elbow-wrist
  ABSOLUTE (needs one rigid Kabsch lab->world fit, solved here on the shared joints):
    * per-joint 3D error in mm after alignment

Joint map (OMC anatomical marker -> COCO keypoint):
    wrist_inner_R + wrist_outer_R midpoint -> right_wrist
    elbow_R                                -> right_elbow
    shoulder_R                             -> right_shoulder
    head                                   -> nose (head-proxy; fixed offset, cancels in dist)

Sync: OMC is 100Hz, video 60fps; resample OMC to 60fps then find integer lag by max
cross-correlation of the RIGHT-WRIST SPEED (same principle as the BRIO cup-speed sync, and the
same keypoint DELTA's own Pose2Sim config syncs on: keypoints_to_consider=['RWrist']).

    python scripts/compare_pose_omc_delta.py --part P14 --trial trial_1_R_unaffected
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import ezc3d
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import triangulate
from cup_task.kalman_3d import load_calibration

ROOT = Path(__file__).resolve().parents[1]
DELTA = ROOT / "cache" / "delta"
VIDEO_FPS = 60.0
C3D_RATE = 100.0
LP_HZ = 6.0   # DELTA's own filter_cutoff

# COCO joints to compare, and the OMC marker(s) that map to each
JOINTS = {
    "right_wrist":    ("wrist_inner_R", "wrist_outer_R"),   # midpoint
    "right_elbow":    ("elbow_R",),
    "right_shoulder": ("shoulder_R",),
    "left_wrist":     ("wrist_inner_L", "wrist_outer_L"),   # the AFFECTED side is L in DELTA
    "left_elbow":     ("elbow_L",),
    "left_shoulder":  ("shoulder_L",),   # also the trunk axis / abduction plane
    "right_hip":      ("hip_R",),
    "left_hip":       ("hip_L",),
    "nose":           ("head",),
}

# The Murphy scoring (imove_extensions/murphy_measures.py) uses MORE than wrist+elbow:
# hand velocity, elbow angle, shoulder FLEXION, shoulder ABDUCTION, trunk Y displacement,
# plus scalars (peak velocity, movement units). NOTE: the production pipeline takes joint
# ANGLES from a MuJoCo IK fit (qpos), NOT from raw points ("would inherit pose jitter"). Here
# BOTH sides use raw-point geometry, so this is a fair MMC-markers-vs-OMC-markers comparison,
# not a reproduction of the qpos-based scoring. Flagged in the output.


def _kp_point(name):
    def fn(fr):
        k = fr.get("kps", {})
        return np.array(k[name][:2], float) if name in k else None
    return fn


def _resample(xyz, from_hz, to_hz):
    n = xyz.shape[0]
    t_old = np.arange(n) / from_hz
    t_new = np.arange(0, t_old[-1], 1.0 / to_hz)
    out = np.empty((len(t_new), 3))
    for k in range(3):
        out[:, k] = np.interp(t_new, t_old, xyz[:, k])
    return out


def _despike(xyz, max_step_mm=40.0):
    out = xyz.copy()
    step = np.r_[0.0, np.linalg.norm(np.diff(out, axis=0), axis=1)]
    out[step > max_step_mm] = np.nan
    return out


def _lp(x):
    v = np.isfinite(x)
    if v.sum() < 8:
        return x
    idx = np.flatnonzero(v)
    xi = np.interp(np.arange(len(x)), idx, x[idx])
    b, a = butter(2, LP_HZ / (VIDEO_FPS / 2))
    return filtfilt(b, a, xi)


def _speed(xyz):
    """Masked per-frame speed (mm or m per s), NaN across gaps -- no interp (qtm despike lesson)."""
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * VIDEO_FPS
    both = np.isfinite(xyz[:-1]).all(1) & np.isfinite(xyz[1:]).all(1)
    out = np.full(len(xyz), np.nan)
    out[1:][both] = d[both]
    return out


def _kabsch(A, B):
    """Rigid R,t mapping A->B (both (N,3), paired). Returns R,t and per-point residual mm."""
    both = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[both], B[both]
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cB - R @ cA
    resid = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return R, t, resid


def _load_calib_mm(part):
    """DELTA calib translations are in METRES; the shared triangulate rounds its output X to
    1 decimal, which is 0.1mm in BRIO's mm world but a catastrophic 100mm in metres. Scale
    each camera's translation to mm so the whole pipeline works in mm (round -> 0.1mm again)
    and both sides of the comparison are mm."""
    cams = load_calibration(str(DELTA / part / "calib" / f"{part}_calibration.toml"),
                            target_size=(1920, 1080))
    for c in cams.values():
        c.t = c.t * 1000.0
    return cams


# Per-participant good-camera whitelist. When set (by cam_quality.json via use_good_cams),
# _load_mmc triangulates ONLY these cameras. The DELTA rig has systematically desynced
# high-numbered cameras + a few miscalibrated ones; with ~5/10 bad, robust_triangulate hits
# its 50% breakdown point and ejects the GOOD cameras instead. Restricting to the verified-good
# set took P15 elbow-angle error 24.2 -> 4.2deg. See project_delta_cohort_transfer.
GOOD_CAMS: dict[str, set] | None = None


def use_good_cams(path=None):
    """Load the good-camera whitelist from cam_quality.json into the module global."""
    global GOOD_CAMS
    p = Path(path) if path else (DELTA / "cam_quality.json")
    if not p.exists():
        raise SystemExit(f"no camera-quality audit at {p}; run scripts/cam_quality_delta.py")
    GOOD_CAMS = {k: set(v["good"]) for k, v in json.loads(p.read_text()).items()}
    return GOOD_CAMS


# which per-camera dets subdir to triangulate from. Default "dets" = YOLO pose; the alt-model
# accuracy comparison sets this to "dets_rtmpose" / "dets_blazepose" (same JSON schema + filenames).
DETS_SUBDIR = "dets"


def _load_mmc(part, trial):
    d = DELTA / part
    cams = _load_calib_mm(part)
    per_cam = {}
    for pj in sorted(glob.glob(str(d / DETS_SUBDIR / f"*{trial}*.pose.json"))):
        cam = Path(pj).name.split(".")[1]
        per_cam[f"cam_{cam}"] = json.loads(Path(pj).read_text())["frames"]
    if GOOD_CAMS is not None and part in GOOD_CAMS:
        keep = GOOD_CAMS[part]
        per_cam = {c: v for c, v in per_cam.items() if c in keep}
        cams = {c: v for c, v in cams.items() if c in keep}
        if not per_cam:
            raise ValueError(f"{part} {trial}: no good cams present")
    n = max(len(v) for v in per_cam.values())
    out = {}
    for joint in JOINTS:
        tr = triangulate.triangulate_target(per_cam, cams, _kp_point(joint), n)
        X = np.array([t["X"] if t.get("X") else [np.nan] * 3 for t in tr])
        # X is now in mm (calib translations pre-scaled to mm). Despike triangulation
        # teleports (isolated >40mm/frame jumps) that otherwise dominate the speed sync --
        # the BRIO qtm despike lesson.
        out[joint] = _despike(X)
    return out, n


def _load_omc(part, trial, n_video):
    c = ezc3d.c3d(str(DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]   # (4, m, T) mm

    T = P.shape[2]

    def marker(nm):
        # NaN for a label this C3D doesn't have (e.g. some P15 R-arm trials lack wrist_outer_L, the
        # NON-scored arm). Averaging below then falls back to the present marker(s), or leaves the
        # whole joint NaN -- which the validity masks downstream already handle. A missing marker on
        # the arm actually being scored surfaces later as NaN measures, not a hard crash on load.
        if nm not in L:
            return np.full((T, 3), np.nan)
        return P[:3, L.index(nm), :].T

    out = {}
    for joint, mks in JOINTS.items():
        with np.errstate(invalid="ignore"):          # all-NaN joint (missing non-scored arm) -> NaN
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = np.nanmean([marker(m) for m in mks], axis=0)
        grid = _despike(_resample(raw, C3D_RATE, VIDEO_FPS))
        # pad/trim to video length
        if len(grid) < n_video:
            grid = np.vstack([grid, np.full((n_video - len(grid), 3), np.nan)])
        out[joint] = grid[:n_video]
    return out


def _find_lag(mmc_wrist, omc_wrist, max_lag=180):
    a = _lp(_speed(mmc_wrist)); b = _lp(_speed(omc_wrist))
    best, blag = -2, 0
    for lag in range(-max_lag, max_lag + 1):
        bs = np.roll(b, lag)
        m = np.isfinite(a) & np.isfinite(bs)
        if m.sum() < 40:
            continue
        c = np.corrcoef(a[m], bs[m])[0, 1]
        if c > best:
            best, blag = c, lag
    return blag, best


def _disp_from_start(X):
    """Distance travelled from the first valid frame (rotation-invariant, less jitter than speed)."""
    X = np.asarray(X, float)
    v = np.isfinite(X).all(1)
    out = np.full(len(X), np.nan)
    if v.sum() >= 2:
        i0 = np.flatnonzero(v)[0]
        out[v] = np.linalg.norm(X[v] - X[i0], axis=1)
    return out


def _corr_at_lag(a, b, max_lag=180):
    """Best correlation of low-passed a vs lag-shifted b, returns (lag, corr)."""
    a = _lp(a); best, blag = -2, 0
    for lag in range(-max_lag, max_lag + 1):
        bs = np.roll(_lp(b), lag)
        m = np.isfinite(a) & np.isfinite(bs)
        if m.sum() < 40:
            continue
        c = np.corrcoef(a[m], bs[m])[0, 1]
        if np.isfinite(c) and c > best:
            best, blag = c, lag
    return blag, best


def _find_lag_multi(mmc, omc, side, mmc_cup=None, omc_cup=None, max_lag=180):
    """Robust MMC<->OMC sync: try SEVERAL signals and keep the best-correlating one.

    The single-wrist-SPEED gate (_find_lag) fails on trials that are actually fine -- a poorly-detected
    wrist keypoint (P17: elbow syncs 0.94 where wrist fails 0.22), or the speed derivative being fragile
    even when position is perfect (P251/P252: displacement corr 0.90-0.98 vs speed 0.54). So correlate
    {wrist,elbow,shoulder} x {speed,displacement} plus the CUP (arm-independent), and take the max.

    Returns (lag, best_corr, signal_name). Falls back to the wrist-speed result if nothing beats it, so
    it is never worse than _find_lag.
    """
    cands = []
    for j in ("wrist", "elbow", "shoulder"):
        jn = f"{side}_{j}"
        if jn in mmc and jn in omc:
            cands.append((f"{j}_speed", _speed(mmc[jn]), _speed(omc[jn])))
            cands.append((f"{j}_disp", _disp_from_start(mmc[jn]), _disp_from_start(omc[jn])))
    if mmc_cup is not None and omc_cup is not None:
        cands.append(("cup_speed", _speed(mmc_cup), _speed(omc_cup)))
        cands.append(("cup_disp", _disp_from_start(mmc_cup), _disp_from_start(omc_cup)))
    best_c, best_lag, best_sig = -2.0, 0, "none"
    for name, a, b in cands:
        lag, c = _corr_at_lag(a, b, max_lag)
        if c > best_c:
            best_c, best_lag, best_sig = c, lag, name
    return best_lag, float(best_c), best_sig


def _murphy_signals(P, side="right"):
    """side-aware wrapper: DELTA's affected arm is the LEFT one, so the signals must be able to
    follow either arm (the trunk axis uses both shoulders/hips regardless)."""
    if side == "left":
        P = dict(P)
        for a, b in [("right_shoulder", "left_shoulder"), ("right_elbow", "left_elbow"),
                     ("right_wrist", "left_wrist"), ("right_hip", "left_hip")]:
            P[a], P[b] = P[b], P[a]
    return _murphy_signals_right(P)


def _murphy_signals_right(P):
    """Raw-point proxies for the angular/trunk Murphy signals.
      shoulder_flexion  : angle of the (shoulder->elbow) vector below the trunk (vertical) axis
                          -- arm raised forward/up. Uses shoulder->hip as the trunk-down ref.
      shoulder_abduction: angle of (shoulder->elbow) away from the trunk midline in the frontal
                          plane (shoulder-to-shoulder direction).
      trunk_y_disp      : forward lean = shoulder-midpoint minus its start, projected on the
                          shoulder-line normal (the horizontal axis perpendicular to both the
                          shoulder line and vertical). Sign arbitrary; correlation is what counts.
    """
    sh, el = P["right_shoulder"], P["right_elbow"]
    hipR, hipL = P["right_hip"], P["left_hip"]
    shL = P["left_shoulder"]
    trunk_mid = (P["right_shoulder"] + shL) / 2.0
    hip_mid = (hipR + hipL) / 2.0
    down = hip_mid - trunk_mid                       # trunk axis (shoulder->hip)
    down = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    arm = el - sh
    arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    # flexion: angle between arm and the DOWN axis -> 0 arm hanging, ~180 raised
    flex = np.degrees(np.arccos(np.clip((arm * down).sum(1), -1, 1)))
    # abduction: component of arm along the shoulder-line (right<-left) direction
    side = (shL - P["right_shoulder"])
    side = side / (np.linalg.norm(side, axis=1, keepdims=True) + 1e-9)
    abd = np.degrees(np.arcsin(np.clip((arm * side).sum(1), -1, 1)))
    # trunk forward lean: shoulder-mid displacement along (down x side) = forward normal
    fwd = np.cross(down, side)
    disp = trunk_mid - trunk_mid[np.isfinite(trunk_mid).all(1)][0]
    trunk = (disp * fwd).sum(1)
    return {"shoulder_flexion": flex, "shoulder_abduction": abd, "trunk_y_disp": trunk}


def _movement_units(vel, amp_thr=20.0, gap=3):
    """Count min->max velocity oscillations (Murphy smoothness proxy)."""
    from scipy.signal import find_peaks
    v = vel[np.isfinite(vel)]
    if len(v) < 3:
        return 0
    mins, _ = find_peaks(-v); maxs, _ = find_peaks(v)
    n = 0
    for mn in mins:
        later = maxs[maxs > mn]
        if len(later) and (v[later[0]] - v[mn]) > amp_thr and (later[0] - mn) >= gap:
            n += 1
    return n


def _render_signals(mmc, omc, sig, part, trial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    smc, som = sig
    rows = [
        ("|wrist-nose| (reach) mm", np.linalg.norm(mmc["right_wrist"] - mmc["nose"], axis=1),
         np.linalg.norm(omc["right_wrist"] - omc["nose"], axis=1)),
        ("wrist speed mm/s", _speed(mmc["right_wrist"]), _speed(omc["right_wrist"])),
        ("elbow angle deg", None, None),
        ("shoulder flexion deg", smc["shoulder_flexion"], som["shoulder_flexion"]),
        ("shoulder abduction deg", smc["shoulder_abduction"], som["shoulder_abduction"]),
        ("trunk fwd disp mm", smc["trunk_y_disp"], som["trunk_y_disp"]),
    ]
    def elbow(P):
        u, v = P["right_shoulder"] - P["right_elbow"], P["right_wrist"] - P["right_elbow"]
        c = (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
        return np.degrees(np.arccos(np.clip(c,-1,1)))
    rows[2] = ("elbow angle deg", elbow(mmc), elbow(omc))
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.7*len(rows)), sharex=True)
    t = np.arange(len(mmc["right_wrist"])) / VIDEO_FPS
    for ax, (lab, m, o) in zip(axes, rows):
        ax.plot(t, _lp(m), label="MMC (cameras)", lw=1.4)
        ax.plot(t, _lp(o), label="OMC (mocap)", lw=1.4, alpha=0.8)
        mm = np.isfinite(_lp(m)) & np.isfinite(_lp(o))
        c = np.corrcoef(_lp(m)[mm], _lp(o)[mm])[0,1] if mm.sum()>20 else float("nan")
        ax.set_ylabel(lab, fontsize=8); ax.set_title(f"corr {c:.3f}", fontsize=8, loc="right")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"DELTA {part} {trial}: MMC pose vs OMC markers (low-passed)", fontsize=10)
    fig.tight_layout()
    out = ROOT / "out" / f"pose_signals_{part}_{trial}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110); print(f"\nrendered {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="P14")
    ap.add_argument("--trial", default="trial_1_R_unaffected")
    a = ap.parse_args(argv)

    mmc, n = _load_mmc(a.part, a.trial)
    omc = _load_omc(a.part, a.trial, n)
    for j in JOINTS:
        fin = np.isfinite(mmc[j]).all(1).mean()
        print(f"  MMC {j:15} 3D {fin*100:3.0f}% of {n} frames", flush=True)

    lag, synccorr = _find_lag(mmc["right_wrist"], omc["right_wrist"])
    print(f"\nsync: lag {lag:+d} frames ({lag/VIDEO_FPS*1000:+.0f} ms), "
          f"wrist-speed corr {synccorr:.3f}", flush=True)

    def _shift(v, lag):
        """Integer shift by lag, NaN-filling the vacated edge -- NOT np.roll, which is
        circular and wraps the far end onto the near end, fabricating a one-frame teleport
        (the phantom OMC peak-velocity spike). Segmentation masks NaN, and so do we."""
        out = np.full_like(v, np.nan)
        if lag >= 0:
            out[lag:] = v[:len(v) - lag] if lag else v
        else:
            out[:lag] = v[-lag:]
        return out
    omc = {j: _shift(v, lag) for j, v in omc.items()}

    # ---- FRAME-INVARIANT ----
    print("\n=== FRAME-INVARIANT (no alignment) ===", flush=True)
    pairs = [("right_wrist", "right_shoulder"), ("right_elbow", "right_wrist"),
             ("right_wrist", "nose")]
    for j1, j2 in pairs:
        dm = _lp(np.linalg.norm(mmc[j1] - mmc[j2], axis=1))
        do = _lp(np.linalg.norm(omc[j1] - omc[j2], axis=1))
        m = np.isfinite(dm) & np.isfinite(do)
        if m.sum() < 20:
            print(f"  dist {j1}-{j2}: too few frames"); continue
        corr = np.corrcoef(dm[m], do[m])[0, 1]
        off = np.median(dm[m] - do[m])
        adiff = np.median(np.abs((dm[m] - off) - do[m]))
        print(f"  |{j1}-{j2}|  corr {corr:.3f}  offset {off:+.0f}mm  |Δ|(offset-rm) {adiff:.1f}mm",
              flush=True)

    # elbow angle (shoulder-elbow-wrist)
    def angle(P):
        sh, el, wr = P["right_shoulder"], P["right_elbow"], P["right_wrist"]
        u, v = sh - el, wr - el
        cosang = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    am, ao = _lp(angle(mmc)), _lp(angle(omc))
    m = np.isfinite(am) & np.isfinite(ao)
    print(f"  elbow angle  corr {np.corrcoef(am[m], ao[m])[0,1]:.3f}  "
          f"|Δ| {np.median(np.abs(am[m]-ao[m])):.1f}°", flush=True)

    # speed agreement on the wrist -- raw AND low-passed. The raw masked speed carries MMC's
    # frame-to-frame keypoint jitter (see realtime.md ~3.5mm), so the smoothed profile (what
    # score.py actually differentiates, after its 4-6Hz low-pass) is the fair number.
    sm, so = _speed(mmc["right_wrist"]), _speed(omc["right_wrist"])
    m = np.isfinite(sm) & np.isfinite(so)
    smL, soL = _lp(sm), _lp(so)
    mL = np.isfinite(smL) & np.isfinite(soL)
    print(f"  wrist speed  corr raw {np.corrcoef(sm[m], so[m])[0,1]:.3f}  "
          f"low-passed {np.corrcoef(smL[mL], soL[mL])[0,1]:.3f}", flush=True)

    # ---- the OTHER Murphy signals (shoulder flexion/abduction proxies, trunk) ----
    sig = _murphy_signals(mmc), _murphy_signals(omc)
    print("\n=== MORE MURPHY SIGNALS (raw-point proxies; scoring uses qpos IK, see header) ===",
          flush=True)
    for name in ["shoulder_flexion", "shoulder_abduction", "trunk_y_disp"]:
        am, ao = _lp(sig[0][name]), _lp(sig[1][name])
        mm = np.isfinite(am) & np.isfinite(ao)
        if mm.sum() < 20:
            print(f"  {name:20} too few frames"); continue
        unit = "mm" if "trunk" in name else "°"
        off = np.median(am[mm] - ao[mm])
        print(f"  {name:20} corr {np.corrcoef(am[mm],ao[mm])[0,1]:.3f}  "
              f"|Δ|(offset-rm) {np.median(np.abs((am[mm]-off)-ao[mm])):.1f}{unit}", flush=True)

    # ---- scalar Murphy measures the signals feed ----
    print("\n=== SCALAR MEASURES (MMC vs OMC) ===", flush=True)
    for lab, arr_m, arr_o in [("peak wrist vel mm/s", smL, soL)]:
        print(f"  {lab:22} MMC {np.nanmax(arr_m):7.0f}   OMC {np.nanmax(arr_o):7.0f}", flush=True)
    mu_m = _movement_units(smL); mu_o = _movement_units(soL)
    print(f"  {'movement units (#)':22} MMC {mu_m:7d}   OMC {mu_o:7d}", flush=True)

    # render every signal trace, MMC vs OMC
    _render_signals(mmc, omc, sig, a.part, a.trial)

    # ---- ABSOLUTE (Kabsch lab->world) ----
    # ONE rigid transform relates the two frames for the whole trial. Fit it on ALL shared
    # joint-frames at once: the constellation of 4 joints moving over time over-determines a
    # single R,t. (A per-FRAME fit would instead measure shape agreement -- reported separately
    # below as Procrustes -- because it re-aligns every frame and so cannot see mis-placement.)
    print("\n=== ABSOLUTE (single rigid lab->world, fit on arm joints) ===", flush=True)
    # Hips are scaffolding for the trunk axis, NOT an accuracy target: their COCO keypoint (a
    # learned pelvis-region centroid) and the OMC hip_R/L skin marker are 10-15cm-apart
    # landmarks, so hip absolute error is convention offset, not tracking -- and it drags the
    # rigid fit. Fit + report on the arm + head only.
    ABS_JOINTS = [j for j in JOINTS if "hip" not in j]
    A = np.vstack([omc[j] for j in ABS_JOINTS])
    B = np.vstack([mmc[j] for j in ABS_JOINTS])
    R, t, resid = _kabsch(A, B)
    print(f"  fit residual: median {np.median(resid):.1f}mm  p90 {np.percentile(resid,90):.1f}mm  "
          f"(n={len(resid)} joint-frames, hips excluded)", flush=True)
    for j in ABS_JOINTS:
        aligned = omc[j] @ R.T + t
        e = np.linalg.norm(aligned - mmc[j], axis=1)
        e = e[np.isfinite(e)]
        print(f"    {j:15} median {np.median(e):5.1f}mm  p90 {np.percentile(e,90):5.1f}mm", flush=True)

    # ---- SHAPE (per-frame Procrustes: aligns each frame's 4 joints, so it isolates the
    # relative arm CONFIGURATION agreement from any lab->world placement error) ----
    perr = []
    for f in range(n):
        Af = np.array([omc[j][f] for j in ABS_JOINTS])
        Bf = np.array([mmc[j][f] for j in ABS_JOINTS])
        if not (np.isfinite(Af).all() and np.isfinite(Bf).all()):
            continue
        _, _, r = _kabsch(Af, Bf)
        perr.append(np.median(r))
    perr = np.array(perr)
    print(f"\n=== SHAPE (per-frame Procrustes, n={len(perr)} frames) ===", flush=True)
    print(f"  arm-configuration residual: median {np.median(perr):.1f}mm  "
          f"p90 {np.percentile(perr,90):.1f}mm", flush=True)


if __name__ == "__main__":
    main()
