"""How accurate is the MMC pose, straight up? (DELTA cohort, no derived measures.)

Deliberately NOT the Murphy measures. Those went down a hole where the headline quantity was
an algebraic identity. This script only measures things that can come out wrong.

THE CORE PROBLEM: a COCO keypoint and a mocap marker are DIFFERENT PHYSICAL POINTS (COCO hip =
learned joint centre; mocap hip = lateral pelvis marker -- 57% apart in width). So a raw
MMC-vs-OMC distance is  landmark-definition-offset + tracking-error  mixed together, and the
offset part is HARMLESS (a constant offset cancels in any within-subject comparison).
So we split them:

    mean residual per joint  = LANDMARK DEFINITION  (systematic; cancels; not our error)
    SD   residual per joint  = TRACKING PRECISION   (<-- the honest accuracy number)

Two families of number, both falsifiable:

  A. FRAME-INVARIANT (needs NO alignment -- immune to the lab-vs-calib frame mess entirely)
       * elbow angle (shoulder-elbow-wrist)     -> deg error
       * segment lengths |sh-el|, |el-wr|       -> pure landmark definition, ~0 tracking content
       * |wrist-shoulder|, |wrist-nose|         -> the geometry the measures actually use
  B. ABSOLUTE, via ONE SESSION rigid transform per participant (lab->calib-world). Justified
     because BOTH rigs are static: the transform is a physical constant, not a per-trial fudge.
     Fitting it per-TRIAL would silently absorb real error (and can flip on symmetric configs).

Sync: OMC 100Hz -> 60fps, integer lag from RIGHT-WRIST SPEED cross-correlation.
NOTE: the lag search uses a NaN-FILLING shift, never np.roll -- a circular shift wraps the far
end onto the near end and fabricates correlation (it previously invented a 3669 mm/s peak).

    python scripts/pose_accuracy_delta.py --parts P14 P15 P17 P19
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ezc3d  # noqa: E402

from compare_pose_omc_delta import (  # noqa: E402
    C3D_RATE, DELTA, JOINTS, VIDEO_FPS, _despike, _load_mmc, _lp, _resample, _speed,
)


def _resolve_marker(name, labels):
    """DELTA marker labels contain a TYPO: P19 spells every wrist marker `wirst_*` on ALL 94
    c3d (only `wrist_outer_R` is correct). Taking the labels literally made P19 look like it
    had NO wrist markers -- I reported it as unusable absent data when the markers are all
    there. Resolve known spelling variants before declaring anything missing.
    """
    if name in labels:
        return name
    for alt in (name.replace("wrist", "wirst"), name.replace("wirst", "wrist")):
        if alt in labels:
            return alt
    return None


def _load_omc_tolerant(part, trial, n_video):
    """Per-joint tolerant OMC loader.

    compare_pose_omc_delta._load_omc builds ALL joints and raises if ANY marker is absent, so
    one missing `wrist_outer_L` killed RIGHT-side trials that never referenced it -- and P19
    (no left wrist markers on ANY of its 94 c3d) would vanish entirely. That is the same
    swallow-the-participant failure as the bare `except` in validate_cohort_delta.py: absent
    data must be VISIBLE per joint, never silently delete the trial.
    """
    c = ezc3d.c3d(str(DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]
    out, missing = {}, {}
    for joint, mks in JOINTS.items():
        res = {m: _resolve_marker(m, L) for m in mks}
        absent = [m for m, r in res.items() if r is None]
        if absent:
            missing[joint] = absent
            continue
        raw = np.mean([P[:3, L.index(res[m]), :].T for m in mks], axis=0)
        grid = _despike(_resample(raw, C3D_RATE, VIDEO_FPS))
        if len(grid) < n_video:
            grid = np.vstack([grid, np.full((n_video - len(grid), 3), np.nan)])
        out[joint] = grid[:n_video]
    return out, missing

ARMS = ["wrist", "elbow", "shoulder"]
# P17's calibration reprojection error is 13.32px vs P14's 3.58 (DELTA's own
# 2024_10_23_1138_Calibration_errors.csv). Reported but flagged -- it is NOT a peer of P14.
CALIB_ERR = {"P14": 3.58, "P15": 4.31, "P19": 6.34, "P17": 13.32}


def _shift(x, lag):
    """Shift along axis 0 filling with NaN. NEVER np.roll (circular -> fabricates signal)."""
    out = np.full_like(x, np.nan, dtype=float)
    if lag == 0:
        out[:] = x
    elif lag > 0:
        out[lag:] = x[:-lag]
    else:
        out[:lag] = x[-lag:]
    return out


def _find_lag_nc(mmc_wrist, omc_wrist, max_lag=180):
    a = _lp(_speed(mmc_wrist))
    b = _lp(_speed(omc_wrist))
    best, blag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        bs = _shift(b.reshape(-1, 1), lag).ravel()
        m = np.isfinite(a) & np.isfinite(bs)
        if m.sum() < 60:
            continue
        if np.std(a[m]) < 1e-9 or np.std(bs[m]) < 1e-9:
            continue
        c = float(np.corrcoef(a[m], bs[m])[0, 1])
        if c > best:
            best, blag = c, lag
    return blag, best


def _angle(a, b, c):
    """Angle at b, degrees. NaN-safe."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


def _kabsch_w(A, B):
    """Rigid transform mapping A->B (both (n,3), finite). Returns R, t."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _trial_pair(part, trial):
    """Return synced (mmc, omc) joint dicts in mm, or None."""
    M, n = _load_mmc(part, trial)
    O, missing = _load_omc_tolerant(part, trial, n)
    side = "right" if "_R_" in trial else "left"
    w = f"{side}_wrist"
    if w not in M or w not in O:
        raise ValueError(f"no {w}: OMC missing {missing.get(w, 'n/a')}")
    lag, corr = _find_lag_nc(M[w], O[w])
    if not np.isfinite(corr) or corr < 0.3:
        raise ValueError(f"sync corr {corr:.2f} < 0.3 (lag {lag})")
    O = {k: _shift(v, lag) for k, v in O.items()}
    return (M, O, side, lag, corr, missing)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P14", "P15", "P17", "P19"])
    ap.add_argument("--out", default=str(DELTA / "pose_accuracy.json"))
    a = ap.parse_args(argv)

    rows = []
    for part in a.parts:
        dd = DELTA / part / "dets"
        trials = sorted({f.name.split(".")[0].replace(f"delta_{part}_", "")
                         for f in dd.glob("*.pose.json")})
        print(f"\n### {part}  ({len(trials)} trials)  calib_err={CALIB_ERR.get(part,'?')}px",
              flush=True)
        ok, skips = 0, {}
        for i, t in enumerate(trials):
            try:
                r = _trial_pair(part, t)
            except Exception as e:
                skips.setdefault(f"{type(e).__name__}: {e}", []).append(t)
                continue
            M, O, side, lag, corr, missing = r
            rec = {"part": part, "trial": t, "side": side, "lag": lag, "sync_corr": corr}

            sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
            absent = [k for k in (sh, el, wr) if k not in M or k not in O]
            if absent:
                skips.setdefault(f"arm chain incomplete: {absent}", []).append(t)
                continue

            # --- A. frame-invariant (no alignment) -------------------------------------
            rec["elbow_ang_err"] = float(np.nanmedian(np.abs(
                _angle(M[sh], M[el], M[wr]) - _angle(O[sh], O[el], O[wr]))))
            for nm, (p, q) in {"upperarm": (sh, el), "forearm": (el, wr),
                               "wrist_shoulder": (wr, sh)}.items():
                dm, do = _dist(M[p], M[q]), _dist(O[p], O[q])
                rec[f"{nm}_mmc"] = float(np.nanmedian(dm))
                rec[f"{nm}_omc"] = float(np.nanmedian(do))
                rec[f"{nm}_err"] = float(np.nanmedian(np.abs(dm - do)))
            if "nose" in M and "nose" in O:
                rec["wrist_nose_err"] = float(np.nanmedian(
                    np.abs(_dist(M[wr], M["nose"]) - _dist(O[wr], O["nose"]))))

            # --- coverage --------------------------------------------------------------
            rec["tri_cov"] = float(np.mean(np.isfinite(M[wr][:, 0])))

            # --- stash points for the SESSION Kabsch -----------------------------------
            rec["_pts"] = {j: (M[j].tolist(), O[j].tolist()) for j in JOINTS
                           if j in M and j in O}
            rec["omc_missing"] = {k: v for k, v in missing.items()}
            rows.append(rec)
            ok += 1
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(trials)}  ok={ok}", flush=True)
        print(f"  {part}: ok={ok}/{len(trials)}", flush=True)
        for reason, ts in sorted(skips.items(), key=lambda kv: -len(kv[1])):
            print(f"    SKIPPED {len(ts):3d} -- {reason}  (e.g. {ts[0]})", flush=True)

    # ---------------- A. FRAME-INVARIANT (no alignment at all) ----------------------
    print("\n" + "=" * 92)
    print("A. FRAME-INVARIANT  (no alignment -- immune to the lab-vs-calib frame question)")
    print("   segment lengths: MMC vs OMC medians. A gap here is LANDMARK DEFINITION, since a")
    print("   bone length is fixed -- it contains almost no tracking content.")
    print("=" * 92)
    print(f"{'part':6}{'side':7}{'n':>4}{'elbAngErr':>11}{'upArm mmc/omc':>18}{'fore mmc/omc':>17}"
          f"{'wr-nose err':>13}{'sync r':>8}{'cov':>7}")
    print("-" * 92)
    for part in a.parts:
        for side in ("right", "left"):
            R = [r for r in rows if r["part"] == part and r["side"] == side]
            if not R:
                continue
            g = lambda k: np.nanmedian([r[k] for r in R if k in r])  # noqa: E731
            print(f"{part:6}{side:7}{len(R):4d}{g('elbow_ang_err'):10.1f}°"
                  f"{g('upperarm_mmc'):9.0f}/{g('upperarm_omc'):<8.0f}"
                  f"{g('forearm_mmc'):8.0f}/{g('forearm_omc'):<8.0f}"
                  f"{g('wrist_nose_err'):11.0f}mm{g('sync_corr'):8.2f}{g('tri_cov'):7.2f}")

    # ---------------- B. SESSION rigid fit per participant --------------------------
    print("\n" + "=" * 92)
    print("B. ABSOLUTE ERROR after ONE session lab->world transform per participant")
    print("   mean = LANDMARK DEFINITION (cancels within-subject) | SD = TRACKING PRECISION")
    print("=" * 92)
    print("   UPPER BOUND: 'precision' also absorbs sync error + POSE-DEPENDENT offset drift")
    print("   (the COCO-vs-marker offset rotates with forearm pronation), so true jitter <= this.")
    print(f"{'part':6}{'joint':16}{'n_fr':>7}{'mean|off|':>11}{'RMS(prec)':>10}{'p90':>9}")
    print("-" * 92)
    sess = {}
    for part in a.parts:
        R = [r for r in rows if r["part"] == part]
        if not R:
            continue
        A, B = [], []
        for r in R:
            for j, (m, o) in r["_pts"].items():
                m, o = np.array(m), np.array(o)
                k = np.isfinite(m[:, 0]) & np.isfinite(o[:, 0])
                if k.sum():
                    A.append(o[k]); B.append(m[k])
        if not A:
            continue
        A, B = np.vstack(A), np.vstack(B)
        Rm, t = _kabsch_w(A, B)
        sess[part] = (Rm.tolist(), t.tolist())
        for j in JOINTS:
            res = []
            for r in R:
                if j not in r["_pts"]:
                    continue
                m, o = np.array(r["_pts"][j][0]), np.array(r["_pts"][j][1])
                k = np.isfinite(m[:, 0]) & np.isfinite(o[:, 0])
                if k.sum():
                    res.append((Rm @ o[k].T).T + t - m[k])
            if not res:
                continue
            res = np.vstack(res)
            mu = res.mean(0)                       # systematic part = landmark definition
            dev = np.linalg.norm(res - mu, axis=1)  # deviation FROM that offset
            # NOT std of the magnitude ||res||: a residual swinging on a cone at constant
            # length has ~zero magnitude-variance while the point is in fact moving a lot.
            # RMS about the mean VECTOR is the honest dispersion.
            rms = float(np.sqrt((dev ** 2).mean()))
            print(f"{part:6}{j:16}{len(res):7d}{np.linalg.norm(mu):11.1f}"
                  f"{rms:10.1f}{np.percentile(dev,90):9.1f}")

    for r in rows:
        r.pop("_pts", None)
    Path(a.out).write_text(json.dumps({"rows": rows, "session_fit": sess}, indent=1))
    print(f"\nwrote {a.out}  ({len(rows)} trials)", flush=True)


if __name__ == "__main__":
    main()
