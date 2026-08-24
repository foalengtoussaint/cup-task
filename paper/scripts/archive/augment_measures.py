"""Augmentation expressed in MURPHY MEASURE units (not just 3D pose error).

Re-triangulates the pose under each augmentation condition, then runs it through the SCORER'S OWN
measure functions (imported from score_vs_automq -- no reimplementation) against the AutoMQ truth.
Measures are frame-invariant (angles / speed / excursion) so the rig-frame triangulation is fed
directly, no alignment.

CONDITIONS (per trial, measures averaged over the subsets/perturbations in each):
  full     all cameras in the sidecar (my no-BA/no-SmoothNet baseline)
  drop1    leave-one-camera-out  (mean over the n LOO variants)
  min2     every 2-camera subset (mean over C(n,2))
  miscal2  one camera rotated 2 deg in place (mean over cam x 2 axes)  + records induced reproj px
  miscal4  one camera rotated 4 deg in place (mean over cam x 2 axes)

PIPELINE NOTE: this uses triangulate-only (the measure functions' own built-in low-pass), NOT BA +
SmoothNet, for tractability -- so absolute r_s differs from the shipped scorer. Every condition is
baselined against the SAME triangulate-only 'full' condition, so the DELTAS are apples-to-apples.

Saves per-(trial,condition,measure) long CSV + npz; prints LIVE progress + r_s / median|err| by
condition. Data only.
"""
from __future__ import annotations
import sys, glob, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import score_vs_automq as S           # measure funcs + phase/lag path (module-level, no main() at import)
import compare_pose_omc_delta as H
import gnn_train as GT
from pipeline.score import compute_position_measures

PAIRS = REPO / "cache/delta/gnn_pairs"
OUT = REPO / "paper"
FPS = S.FPS
JIDX = {"right_wrist": 0, "right_elbow": 1, "right_shoulder": 2,
        "left_wrist": 3, "left_elbow": 4, "left_shoulder": 5, "right_hip": 6, "left_hip": 7}
NEED = list(JIDX)                     # 8 joints the measures use (nose dropped)
ROT_AXES = [np.array([1., 0, 0]), np.array([0., 1, 0])]


def _Pmats(K, R, t):
    return np.stack([K[c] @ np.hstack([R[c], t[c].reshape(3, 1)]) for c in range(len(K))])


def tri_batch(P_sub, uv, val):
    F, m, _ = uv.shape
    P0, P1, P2 = P_sub[:, 0], P_sub[:, 1], P_sub[:, 2]
    A = np.zeros((F, 2 * m, 4))
    u, v = uv[..., 0], uv[..., 1]
    A[:, 0::2, :] = u[..., None] * P2[None] - P0[None]
    A[:, 1::2, :] = v[..., None] * P2[None] - P1[None]
    w = val.astype(float)[..., None]
    A[:, 0::2, :] *= w; A[:, 1::2, :] *= w
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[:, -1, :]
    X = h[:, :3] / h[:, 3:4]
    X[val.sum(1) < 2] = np.nan
    return X


def _rodrigues(axis, deg):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    th = np.radians(deg)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def _pose_from_P(P, uv, uvv):
    """Build the joint-name -> (T,3) dict the measure functions expect."""
    return {name: tri_batch(P, uv[:, :, JIDX[name]], uvv[:, :, JIDX[name]]) for name in NEED}


def _measures(pose, ph, side, rec):
    """Return {measure: (mmc_value, automq_truth)} using the SCORER'S functions."""
    out = {}
    wr = f"{side}_wrist"
    other = "right" if side == "left" else "left"
    ang = S.angle_measures_automq(pose, ph, side, "max")
    for k in ("max_shoulder_flexion", "max_shoulder_abduction", "max_elbow_angle",
              "peak_elbow_angular_velocity", "interjoint_coordination", "max_trunk_displacement"):
        gt = rec.get(k)
        if gt is not None and np.isfinite(gt) and np.isfinite(ang.get(k, np.nan)):
            out[k] = (ang[k], float(gt))
    pv = S.peak_velocity_reduce(pose, ph, side, "max")
    gtpv = rec.get("peak_velocity_wrist")
    if gtpv is not None and np.isfinite(gtpv) and np.isfinite(pv):
        out["peak_velocity"] = (pv, float(gtpv))
    try:
        trunk_xyz = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
        mm = compute_position_measures(pose[wr], trunk_xyz, ph, side, fps=FPS)
    except Exception:
        mm = None
    if mm is not None:
        for k in ("total_movement_time", "time_to_peak_velocity", "time_to_first_peak_velocity",
                  "number_of_movement_units"):
            gt = rec.get(k); mv = getattr(mm, k, None)
            if gt is not None and mv is not None and np.isfinite(gt) and np.isfinite(mv):
                out[k] = (float(mv), float(gt))
    return out


def _avg_measures(list_of_dicts):
    """Average mmc values across variant re-triangulations (truth is constant)."""
    keys = set().union(*[d.keys() for d in list_of_dicts]) if list_of_dicts else set()
    out = {}
    for k in keys:
        mv = [d[k][0] for d in list_of_dicts if k in d and np.isfinite(d[k][0])]
        gt = next((d[k][1] for d in list_of_dicts if k in d), np.nan)
        if mv:
            out[k] = (float(np.mean(mv)), gt)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    S.H.use_good_cams()
    amq = S.load_automq()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in S.COHORT_PARTS]
    print(f"AutoMQ rows {len(amq)}; scorable cohort trials {len(trials)}", flush=True)
    import re
    pat = re.compile(r"trial_(\d+)_([LR])_")
    rows = []
    t0 = time.time(); n = 0; n_ok = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((S.automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        rjf = PAIRS / part / f"{trial}.reproj.npz"
        if not rjf.exists():
            continue
        nfr = t["mmc"].shape[0]
        omc = H._load_omc(part, trial, nfr)
        lag, _ = H._find_lag(t["mmc"][:, S.GRID_JOINTS.index(f"{side}_wrist")], omc[f"{side}_wrist"])
        ph = S.automq_phases_to_video(rec["phases"], lag, nfr)
        if ph is None:
            continue
        rj = np.load(rjf, allow_pickle=True)
        uv, uvv, K, R, tt = rj["uv"], rj["uv_valid"], rj["K"], rj["R"], rj["t"]
        T = min(len(uv), nfr)
        uv, uvv = uv[:T], uvv[:T]
        ncam = len(K)
        Pfull = _Pmats(K, R, tt)

        cond = {}
        cond["full"] = _measures(_pose_from_P(Pfull, uv, uvv), ph, side, rec)
        # drop1
        loo = []
        for c in range(ncam):
            sub = [i for i in range(ncam) if i != c]
            loo.append(_measures(_pose_from_P(Pfull[sub], uv[:, sub], uvv[:, sub]), ph, side, rec))
        cond["drop1"] = _avg_measures(loo)
        # min2
        two = []
        for sub in itertools.combinations(range(ncam), 2):
            sub = list(sub)
            two.append(_measures(_pose_from_P(Pfull[sub], uv[:, sub], uvv[:, sub]), ph, side, rec))
        cond["min2"] = _avg_measures(two)
        # miscal 2 / 4 deg
        for deg, tag in [(2.0, "miscal2"), (4.0, "miscal4")]:
            mset = []
            for c in range(ncam):
                for ax in ROT_AXES:
                    dR = _rodrigues(ax, deg)
                    Rp, tp = R.copy(), tt.copy()
                    Rp[c] = dR @ R[c]; tp[c] = dR @ tt[c]
                    mset.append(_measures(_pose_from_P(_Pmats(K, Rp, tp), uv, uvv), ph, side, rec))
            cond[tag] = _avg_measures(mset)

        for cname, d in cond.items():
            for meas, (mv, gt) in d.items():
                rows.append(dict(part=part, trial=trial, cond=cname, measure=meas, mmc=mv, automq=gt))
        n += 1; n_ok += 1
        if n % 40 == 0:
            print(f"[{n}/{len(trials)}] {time.time()-t0:5.0f}s rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "augment_measures.csv", index=False)
    print(f"\nPROCESSING CHECK: trials {n_ok}, rows {len(df)}, "
          f"nonfinite {int((~np.isfinite(df['mmc'])).sum())}", flush=True)

    CONDS = ["full", "drop1", "min2", "miscal2", "miscal4"]
    MEAS = ["max_shoulder_flexion", "max_shoulder_abduction", "max_elbow_angle",
            "peak_elbow_angular_velocity", "interjoint_coordination", "max_trunk_displacement",
            "peak_velocity", "total_movement_time", "time_to_peak_velocity",
            "time_to_first_peak_velocity", "number_of_movement_units"]

    def tbl(fn, title):
        print(f"\n== {title} ==")
        print(f"{'measure':<32}" + "".join(f"{c:>10}" for c in CONDS))
        for meas in MEAS:
            cells = []
            for c in CONDS:
                g = df[(df.measure == meas) & (df.cond == c)].dropna(subset=["mmc", "automq"])
                cells.append(f"{fn(g):>10.2f}" if len(g) > 3 else f"{'--':>10}")
            print(f"{meas:<32}" + "".join(cells))

    tbl(lambda g: spearmanr(g.automq, g.mmc).correlation, "Spearman r_s vs AutoMQ, by condition")
    tbl(lambda g: (g.mmc - g.automq).abs().median(), "median |error| (measure units), by condition")
    print(f"\nwrote {OUT/'augment_measures.csv'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
