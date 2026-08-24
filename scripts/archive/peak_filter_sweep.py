"""How do MMC low-pass cutoff and the peak SUMMARY (max vs p95) change the peak Murphy measures?

Two questions, tested together because they interact (heavier filtering ALSO removes the jitter spikes
that make MMC-max over-read, so it's a 2-D grid not two 1-D sweeps):

  1. peak_velocity (wrist) and peak_elbow_ang_vel: OMC truth = MAX of the OMC signal (that's what the
     clinical measure means). But MMC is a SPIKY ESTIMATOR of that smooth peak -- so the best MMC
     SUMMARY of the OMC-max need not be MMC-max. Sweep MMC summary in {max, p95}.
  2. Sweep the MMC low-pass CUTOFF in {4,3,2,1.5 Hz} (the score.py default is 4). OMC keeps its own
     (default) filtering -- we're calibrating the ESTIMATOR, not moving the truth.

Reuses the cached poses (pipeline MMC + BA traj) + OMC + fixed OMC-cup phases from results_v3_delta,
so it's a few minutes, no re-triangulation. Cohort = the 12 elbow-flow trials for elbow (so flow-cloud
is comparable) AND the full 5-part cohort for wrist peak_velocity.

    python scripts/peak_filter_sweep.py            # prints both tables + saves npz
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import results_v3_delta as R
from scipy.signal import butter, filtfilt, medfilt

FPS = H.VIDEO_FPS
CLEAN = {"P07", "P08", "P15"}; MISCAL = {"P17", "P19"}


def _lp_at(x, hz, order=2):
    """Zero-phase Butterworth at `hz`, NaN-safe (interp gaps, filter, keep). Mirrors score.py."""
    x = np.asarray(x, float)
    v = np.isfinite(x)
    if v.sum() < 8:
        return x
    idx = np.flatnonzero(v)
    xi = np.interp(np.arange(len(x)), idx, x[idx])
    b, a = butter(order, hz / (FPS / 2))
    return filtfilt(b, a, xi)


def _wrist_speed_peak(xyz, phase_win, hz, summ):
    """Peak wrist speed over the reaching window, MMC filtered at `hz`, summarized by max|p95."""
    p = xyz.copy()
    for k in range(3):                                   # median-3 then lowpass the POSITION
        good = np.isfinite(p[:, k])
        if good.sum() >= 3:
            pk = p[:, k].copy(); pk[good] = medfilt(pk[good], 3)
            p[:, k] = _lp_at(pk, hz)
    sp = np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS
    s, e = phase_win
    seg = sp[max(s, 0):max(e - 1, 0)]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 3:
        return np.nan
    return float(np.max(seg)) if summ == "max" else float(np.nanpercentile(seg, 95))


def _elbow_av_peak(P, side, hz, summ):
    """Peak elbow angular velocity, elbow angle filtered at `hz`, summarized by max|p95 (whole trial)."""
    sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    u, v = P[sh] - P[el], P[wr] - P[el]
    c = (u * v).sum(1) / (np.linalg.norm(u, 1) * np.linalg.norm(v, axis=1) + 1e-9) \
        if False else (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    elb = _lp_at(np.degrees(np.arccos(np.clip(c, -1, 1))), hz)
    eav = np.abs(np.gradient(elb)) * FPS
    eav = eav[np.isfinite(eav)]
    if len(eav) < 3:
        return np.nan
    return float(np.max(eav)) if summ == "max" else float(np.nanpercentile(eav, 95))


def _ph_reaching(phases):
    for nm, s, e in phases:
        if nm == "reaching":
            return (s, e)
    return None


def _load_smoothed(part, trial):
    """Read the cached SmoothNet poses (built by cache_smoothed_pose.py) -> {variant: {joint:(T,3)}}.
    None if not cached (so we can fall back / report coverage)."""
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


def run():
    import gnn_train as GT
    H.use_good_cams()
    ba = R._ba_traj_cache()
    trials = GT.load_clean(need_reproj=False)
    HZ = [4.0, 3.0, 2.0, 1.5]
    SUMM = ["max", "p95"]

    # accumulate per (variant, hz, summ): lists of (omc_truth, mmc_est) for wrist + elbow
    wrist = {}   # key -> list
    elbow = {}
    fc12 = set()  # the 12 elbow-flow trials
    for f in (ROOT / "cache" / "elbow_flow").glob("*.npz"):
        z = np.load(str(f), allow_pickle=True)
        fc12.add(f"{z['part']}/{z['trial']}")

    print(f"sweep: {len(trials)} trials, HZ={HZ}, SUMM={SUMM}", flush=True)
    from tqdm import tqdm
    for t in tqdm(trials, desc="peak_filter_sweep", unit="trial"):
        part, trial, side = t["part"], t["trial"], t["side"]
        n = t["mmc"].shape[0]
        wr = f"{side}_wrist"
        omc = H._load_omc(part, trial, n)
        lag, _ = H._find_lag(t["mmc"][:, R._GRID_JOINTS.index(wr)], omc[wr])
        omc = {j: R._shift(v, lag) for j, v in omc.items()}
        oc = R._shift(R._omc_cup(part, trial, n), lag)
        try:
            seg = R.segment.segment_cup_only(R._fill(oc), fps=FPS)
            ph = R.segment.to_murphy_phases(seg, R._fill(omc[wr]), R._fill(oc), fps=FPS)
        except Exception:
            continue
        if not ph:
            continue
        reach = _ph_reaching(ph)
        if reach is None:
            continue

        # OMC truth: its OWN max at the default 6Hz H._lp (that IS the ground-truth peak)
        o_wrist_speed = H._lp(H._speed(omc[wr]))
        rs, re = reach
        segw = o_wrist_speed[max(rs, 0):max(re - 1, 0)]; segw = segw[np.isfinite(segw)]
        omc_wrist_max = float(np.max(segw)) if len(segw) >= 3 else np.nan
        omc_elbow_max = _elbow_av_peak(omc, side, 6.0, "max")   # OMC elbow truth = its max

        cached = _load_smoothed(part, trial)      # pure-CPU read of the smoothed pose (no GPU)
        if cached is None:
            continue
        for smoother in ("smoothnet", "BA+smoothnet"):
            pose = cached.get(smoother)
            if pose is None:
                continue
            for hz in HZ:
                for summ in SUMM:
                    kw = (smoother, hz, summ)
                    if np.isfinite(omc_wrist_max):
                        mv = _wrist_speed_peak(pose[wr], (rs, re), hz, summ)
                        if np.isfinite(mv):
                            wrist.setdefault(kw, []).append((omc_wrist_max, mv))
                    if np.isfinite(omc_elbow_max) and f"{part}/{trial}" in fc12:
                        me = _elbow_av_peak(pose, side, hz, summ)
                        if np.isfinite(me):
                            elbow.setdefault(kw, []).append((omc_elbow_max, me))

    def med_err(pairs):
        if len(pairs) < 3:
            return None, None, len(pairs)
        o = np.array([a for a, b in pairs]); m = np.array([b for a, b in pairs])
        ae = np.median(np.abs(m - o) / o * 100)
        bias = np.median((m - o) / o * 100)
        return ae, bias, len(o)

    for name, D in [("WRIST peak_velocity (all 5 parts, OMC truth=max)", wrist),
                    ("ELBOW peak_ang_vel (12 flow trials, OMC truth=max)", elbow)]:
        print(f"\n=== {name} : median|err|% (bias%) ===", flush=True)
        print(f"  {'variant':13s} {'summ':4s} " + "".join(f"{h}Hz".rjust(16) for h in HZ), flush=True)
        for smoother in ["smoothnet", "BA+smoothnet"]:
            for summ in SUMM:
                cells = []
                for hz in HZ:
                    ae, bias, npr = med_err(D.get((smoother, hz, summ), []))
                    cells.append("-".rjust(16) if ae is None else f"{ae:.1f}({bias:+.0f})".rjust(16))
                print(f"  {smoother:13s} {summ:4s} " + "".join(cells), flush=True)

    np.savez(str(ROOT / "out" / "peak_filter_sweep.npz"),
             wrist=np.array(list(wrist.items()), dtype=object),
             elbow=np.array(list(elbow.items()), dtype=object))
    print(f"\nsaved out/peak_filter_sweep.npz", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
