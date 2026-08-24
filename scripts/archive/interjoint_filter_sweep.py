"""Does extra smoothing of the flexion/elbow SERIES (only for interjoint) recover its MMC-vs-OMC r_s?

interjoint = corr(flexion, elbow) over the inner 80% of reaching. Pose jitter in those two series
injects a low tail (seq_mmc p5=0.59). Sweep an extra Savitzky-Golay filter on the two series before
correlating, on BOTH sides (OMC pose+phase, MMC pose+phase), and report r_s(MMC, OMC) per window.
Baseline (window 0 = no extra filter) should reproduce the 0.24 product number. Data only.
"""
from __future__ import annotations
import sys, re, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.signal import savgol_filter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import score_vs_automq as S
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from seg_sequential import segment_sequential
GRID = S.GRID_JOINTS; FPS = S.FPS
WINDOWS = [0, 7, 15, 31, 61, 91]           # extra savgol window (frames); 0 = none


def _series(pose, side):
    other = "right" if side == "left" else "left"
    elb = S._elbow_series(pose, side)
    try:
        flex, _ = S._planar_body_angles(pose, side, other)
    except Exception:
        flex = np.full(len(elb), np.nan)
    return flex, elb


def _interjoint(flex, elb, ph, w):
    reach = S._win(ph, "reaching")
    if not reach or reach[1] - reach[0] < 12:
        return np.nan
    if w >= 5:
        ww = min(w if w % 2 else w + 1, (len(flex) // 2) * 2 - 1)
        if ww >= 5:
            flex = savgol_filter(np.nan_to_num(flex, nan=np.nanmedian(flex)), ww, 3)
            elb = savgol_filter(np.nan_to_num(elb, nan=np.nanmedian(elb)), ww, 3)
    rr0, rr1 = reach; mg = int(0.1 * (rr1 - rr0))
    a, b = flex[rr0 + mg:rr1 - mg], elb[rr0 + mg:rr1 - mg]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() >= 10 and np.std(a[m]) > 1e-6 and np.std(b[m]) > 1e-6:
        return float(np.corrcoef(a[m], b[m])[0, 1])
    return np.nan


def main():
    S.H.use_good_cams(); amq = S.load_automq(); ba = R._ba_traj_cache()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in S.COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    print(f"cohort {len(trials)}", flush=True)
    rows = []; n = 0; t0 = time.time()
    JN = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
          "right_wrist", "left_wrist", "right_hip", "left_hip", "nose"]
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((S.automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        nfr = t["mmc"].shape[0]
        pose = S._pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        omc = H._load_omc(part, trial, nfr); wr = f"{side}_wrist"
        if wr not in omc or not np.isfinite(omc[wr]).any() or not all(j in omc for j in JN):
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        ocup = R._omc_cup(part, trial, nfr); mcup = R._cup_v3(part, trial, R._calib(part), nfr)
        if not (np.isfinite(ocup).any() and np.isfinite(mcup).any()):
            continue
        seq_omc = {nm: (s, e) for nm, s, e in segment_sequential(
            R._shift(ocup, lag), R._shift(omc[wr], lag), R._shift(omc["nose"], lag))}
        seq_mmc = {nm: (s, e) for nm, s, e in segment_sequential(
            R._smooth_joint(mcup), R._smooth_joint(t["mmc"][:, GRID.index(wr)]),
            R._smooth_joint(t["mmc"][:, GRID.index("nose")]))}
        ph_o = [(k, s, e) for k, (s, e) in seq_omc.items()]
        ph_m = [(k, s, e) for k, (s, e) in seq_mmc.items()]
        pose_omc = {j: R._shift(omc[j], lag) for j in JN}
        fo, eo = _series(pose_omc, side)          # OMC pose
        fm, em = _series(pose, side)              # MMC pose
        for w in WINDOWS:
            io = _interjoint(fo, eo, ph_o, w)
            im = _interjoint(fm, em, ph_m, w)
            rows.append(dict(part=part, trial=trial, w=w, omc=io, mmc=im))
        n += 1
        if n % 80 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s", flush=True)

    df = pd.DataFrame(rows); df.to_csv(ROOT / "out/scoring/interjoint_filter_sweep.csv", index=False)
    print(f"\nPROCESSING CHECK: trials {n}, rows {len(df)}\n", flush=True)
    print(f"{'extra savgol window':>20}{'r_s(MMC,OMC)':>14}{'|err|':>8}{'MMC p5':>9}{'n':>6}")
    for w in WINDOWS:
        g = df[df.w == w].dropna(subset=["omc", "mmc"])
        rs = spearmanr(g.omc, g.mmc).correlation if len(g) > 3 else np.nan
        err = (g.mmc - g.omc).abs().median()
        p5 = g.mmc.quantile(0.05)
        lbl = "none" if w == 0 else f"{w} fr ({w/FPS*1000:.0f}ms)"
        print(f"{lbl:>20}{rs:>14.2f}{err:>8.3f}{p5:>9.3f}{len(g):>6}")
    print("\nwrote out/scoring/interjoint_filter_sweep.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
