"""Re-score the full Murphy set with MMC peaks summarized by p95 (OMC stays at max), for EVERY
measure, from the cached smoothed poses -- then the existing plotter regenerates all figures.

WHY. The 8 PEAK-TYPE measures (a max of a per-frame signal) were scored MMC-max in murphy_grid, which
OVER-READS a spiky estimator of the smooth OMC peak (elbow +25-29%). The fair pairing is OMC=max (the
clinical peak) vs MMC=p95 (a robust summary of the same spiky signal). The other 7 measures
(timing/count/correlation/duration) have no max in them, so p95 is a no-op -- computed identically.

Cache-fed (cache/pose_smoothed/), so all 328 trials in seconds, no GPU. Emits the SAME tidy schema as
murphy_grid (variant,part,trial,arm,side,measure,omc,mmc) for two variants (SmoothNet, BA+SmoothNet)
so murphy_grid_plot.py works unchanged.

    python scripts/rescore_p95.py --summ p95 --out out/murphy_grid_p95.csv     # the fair version
    python scripts/rescore_p95.py --summ max --out out/murphy_grid_maxcheck.csv # reproduce the grid
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import results_v3_delta as R
from cup_task.score import compute_position_measures

FPS = H.VIDEO_FPS
# the 8 peak-type measures where MMC-summary (max vs p95) is a real question
PEAK_TYPE = {"peak_velocity", "peak_elbow_ang_vel", "elbow_extension_reaching",
             "shoulder_flexion_reaching", "shoulder_flexion_drinking",
             "shoulder_abduction_reaching", "shoulder_abduction_drinking",
             "max_trunk_displacement"}
POS = R.POSITION_MEASURES; ANG = R.ANGLE_MEASURES


def _lp_hz(x, hz, order=2):
    """Zero-phase Butterworth at `hz` (NaN-safe: interp gaps, filter, keep). MMC cutoff knob."""
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, float)
    if x.ndim == 2:                                   # per-column for (T,3)
        return np.stack([_lp_hz(x[:, k], hz, order) for k in range(x.shape[1])], 1)
    v = np.isfinite(x)
    if v.sum() < 8:
        return x
    idx = np.flatnonzero(v)
    xi = np.interp(np.arange(len(x)), idx, x[idx])
    b, a = butter(order, hz / (FPS / 2))
    return filtfilt(b, a, xi)


def _summ(a, w, how):
    """max|p95 of signal a over window w=(s,e). how in {'max','p95'}. NaN-safe."""
    if not (w and w[1] > w[0]):
        return float("nan")
    seg = np.asarray(a)[w[0]:w[1]]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 1:
        return float("nan")
    return float(np.max(seg)) if how == "max" else float(np.nanpercentile(seg, 95))


def _phase(phases, *names):
    for nm, s, e in phases:
        if nm in names:
            return (s, e)
    return None


def _angle_measures(P, phases, side, how, hz):
    """The angle measures + peak_elbow_ang_vel, peak summary `how`, MMC low-pass at `hz`."""
    from compare_pose_omc_delta import _murphy_signals
    sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    u, v = P[sh] - P[el], P[wr] - P[el]
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    elb = _lp_hz(np.degrees(np.arccos(np.clip(c, -1, 1))), hz)
    try:
        sig = _murphy_signals(P, side=side)
        flex, abd = _lp_hz(sig["shoulder_flexion"], hz), _lp_hz(sig["shoulder_abduction"], hz)
    except Exception:
        flex = abd = np.full(len(elb), np.nan)
    eav = np.abs(np.gradient(elb)) * FPS
    r = _phase(phases, "reaching"); d = _phase(phases, "drinking"); fw = _phase(phases, "forward_transport")
    rf = (r[0], fw[1]) if (r and fw) else r
    ijc = float("nan")
    if r and r[1] - r[0] >= 10:
        m = int(0.1 * (r[1] - r[0]))
        a1, b1 = flex[r[0] + m:r[1] - m], elb[r[0] + m:r[1] - m]
        if np.isfinite(a1).all() and np.isfinite(b1).all() and np.std(a1) > 1e-6 and np.std(b1) > 1e-6:
            ijc = float(np.corrcoef(a1, b1)[0, 1])
    return {"elbow_extension_reaching": _summ(elb, rf, how),
            "shoulder_flexion_reaching": _summ(flex, r, how),
            "shoulder_flexion_drinking": _summ(flex, d, how),
            "shoulder_abduction_reaching": _summ(abd, r, how),
            "shoulder_abduction_drinking": _summ(abd, d, how),
            "peak_elbow_ang_vel": _summ(eav, (0, len(elb)), how),
            "interjoint_coordination": ijc}


def _wrist_pv(hand_xyz, phases, how, hz):
    """peak_velocity with summary `how`, over the reaching window, MMC position low-passed at `hz`."""
    from cup_task.score import _smoothed_xyz, _hand_speed_mmps
    sp = _hand_speed_mmps(_smoothed_xyz(np.asarray(hand_xyz, float), FPS, hz, 2), FPS)
    r = _phase(phases, "reaching")
    return _summ(sp, r, how)


def _load_sm(part, trial):
    p = ROOT / "cache" / "pose_smoothed" / f"{part}__{trial}.npz"
    if not p.exists():
        return None
    z = np.load(str(p), allow_pickle=True)
    joints = list(z["joints"])
    out = {}
    for var, key in [("smoothnet", "pipeline_sn"), ("BA+smoothnet", "ba_sn")]:
        arr = z[key]
        if np.isfinite(arr).any():
            out[var] = {j: arr[:, i, :] for i, j in enumerate(joints)}
    return out


def run(summ, out_csv, mmc_hz=4.0, trunk_summ="max"):
    import gnn_train as GT
    from tqdm import tqdm
    H.use_good_cams()
    trials = GT.load_clean(need_reproj=False)
    rows = []; nok = nfail = 0
    print(f"rescore: {len(trials)} trials, MMC peak summary={summ}, MMC cutoff={mmc_hz}Hz, "
          f"trunk={trunk_summ} (OMC=max @default), all 15 measures", flush=True)
    for t in tqdm(trials, desc="rescore", unit="trial"):
        part, trial, side = t["part"], t["trial"], t["side"]
        arm = "affected" if "unaffected" not in trial else "unaffected"
        other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]; wr = f"{side}_wrist"
        omc = H._load_omc(part, trial, n)
        lag, _ = H._find_lag(t["mmc"][:, R._GRID_JOINTS.index(wr)], omc[wr])
        omc = {j: R._shift(v, lag) for j, v in omc.items()}
        oc = R._shift(R._omc_cup(part, trial, n), lag)
        try:
            seg = R.segment.segment_cup_only(R._fill(oc), fps=FPS)
            ph = R.segment.to_murphy_phases(seg, R._fill(omc[wr]), R._fill(oc), fps=FPS)
        except Exception:
            ph = None
        if not ph:
            nfail += 1; continue
        # OMC: peaks at MAX (ground truth); position measures via the scorer, angles via _angle_measures(max)
        trunk_o = (omc[f"{side}_shoulder"] + omc[f"{other}_shoulder"]) / 2
        try:
            mo = compute_position_measures(omc[wr], trunk_o, ph, side, fps=FPS)
            ao = _angle_measures(omc, ph, side, "max", H.LP_HZ)          # OMC truth: default 6Hz, max
            o_pv = _wrist_pv(omc[wr], ph, "max", 4.0)
            oty = _lp_hz(trunk_o[:, 1], H.LP_HZ)
            o_trunk = _summ(np.abs(oty - oty[0]), (0, n), "max")
        except Exception:
            nfail += 1; continue

        cached = _load_sm(part, trial)
        if cached is None:
            nfail += 1; continue
        for vname, pose in cached.items():
            trunk = (pose[f"{side}_shoulder"] + pose[f"{other}_shoulder"]) / 2
            try:
                mm = compute_position_measures(pose[wr], trunk, ph, side, fps=FPS,
                                               lowpass_hz=mmc_hz)
                am = _angle_measures(pose, ph, side, summ, mmc_hz)
                m_pv = _wrist_pv(pose[wr], ph, summ, mmc_hz)
                mty = _lp_hz(trunk[:, 1], mmc_hz)
                m_trunk = _summ(np.abs(mty - mty[0]), (0, n), trunk_summ)   # trunk keeps MAX
            except Exception:
                continue
            for m in POS:
                ov = getattr(mo, m, None); mv = getattr(mm, m, None)
                if m == "peak_velocity":
                    ov, mv = o_pv, m_pv                       # summary-controlled
                if m == "max_trunk_displacement":
                    ov, mv = o_trunk, m_trunk
                if ov is None or mv is None or not np.isfinite(ov) or not np.isfinite(mv):
                    continue
                rows.append((vname, part, trial, arm, side, m, float(ov), float(mv)))
            for m in ANG:
                ov, mv = ao[m], am[m]
                if not np.isfinite(ov) or not np.isfinite(mv):
                    continue
                rows.append((vname, part, trial, arm, side, m, float(ov), float(mv)))
        nok += 1

    outp = ROOT / out_csv; outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "part", "trial", "arm", "side", "measure", "omc", "mmc"])
        w.writerows(rows)
    print(f"\nPROCESSING CHECK: {nok} scored, {nfail} failed, {len(rows)} rows -> {outp}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summ", choices=["max", "p95"], default="p95")
    ap.add_argument("--out", default="out/murphy_grid_p95.csv")
    a = ap.parse_args()
    run(a.summ, a.out)
