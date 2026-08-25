"""Bootstrap CIs for the movement-quality measures -> paper/table3_cis.csv.

Why this exists: interjoint coordination's r_s spans 0.27--0.98 across random subsamples of the same
trials (see VERIFY.md), so a bare point estimate cannot be read. Every r in Table II needs an
interval, and the optical-vs-markerless comparison needs a PAIRED one -- resampling the two
conditions independently would ignore that they are computed on the same trials and would widen the
interval on their difference for no reason.

  r_s     resamples TRIALS with replacement
  r_av    resamples the 21 participant x arm GROUPS, which is the unit that statistic averages over
  d_mmc   the paired change, markerless windows minus optical, resampled on the same trial draw

    python paper/scripts/measure_cis.py [--boot 3000]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from fps_ablation import declined                                    # noqa: E402

OUT = ROOT / "paper" / "table3_cis.csv"
MEAS = [("peak_velocity", "PV [mm/s]"),
        ("peak_elbow_angular_velocity", "Elbow angular PV [deg/s]"),
        ("time_to_peak_velocity", "Time to PV [s]"),
        ("time_to_first_peak_velocity", "Time to first PV [s]"),
        ("number_of_movement_units", "Number of movement units [n]"),
        ("total_movement_time", "Total movement time [s]"),
        ("interjoint_coordination", "Interjoint coordination"),
        ("max_trunk_displacement", "Trunk displacement [mm]"),
        ("max_shoulder_flexion", "Shoulder flexion [deg]"),
        ("max_shoulder_flexion_drink", "Shoulder flexion, drinking [deg]"),
        ("max_elbow_angle", "Elbow extension [deg]"),
        ("max_shoulder_abduction", "Shoulder abduction [deg]")]


def _r(a, b):
    if np.std(a) <= 0 or np.std(b) <= 0:
        return np.nan
    return pearsonr(a, b)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260825)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    a_boot = a.boot
    bad = declined()
    import glob
    f30 = sorted(glob.glob(str(ROOT / "out/scoring/fps_full_*of3.csv")))
    F30 = pd.concat([pd.read_csv(x) for x in f30]) if f30 else None
    if F30 is None:
        print("  no fps_full_*of3.csv -- 30 Hz columns will be blank", flush=True)
    d = pd.read_csv(ROOT / "out/scoring/score_own_phases_anat12.csv")
    d = d[[(p, t) not in bad for p, t in zip(d.part, d.trial)]]

    rows = []
    for key, label in MEAS:
        g = d[d.measure == key].dropna(subset=["omc_omc", "mmc_omc", "mmc_mmc"])
        x = g.omc_omc.values
        opt, mmc = g.mmc_omc.values, g.mmc_mmc.values
        rec = dict(measure=label, n=len(g),
                   r_s_opt=_r(x, opt), r_s_mmc=_r(x, mmc))
        bo, bm, bd = [], [], []
        for _ in range(a.boot):
            i = rng.integers(0, len(x), len(x))
            ro, rm = _r(x[i], opt[i]), _r(x[i], mmc[i])
            if np.isfinite(ro) and np.isfinite(rm):
                bo.append(ro); bm.append(rm); bd.append(rm - ro)
        for tag, arr in (("opt", bo), ("mmc", bm)):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rec[f"r_s_{tag}_lo"], rec[f"r_s_{tag}_hi"] = lo, hi
        rec["d_mmc"] = rec["r_s_mmc"] - rec["r_s_opt"]
        rec["d_lo"], rec["d_hi"] = np.percentile(bd, [2.5, 97.5])
        # strictly excludes zero. NOT `(lo>0)==(hi>0)`, which is True for a zero-width interval
        # at zero -- trunk displacement and elbow extension are identical between the two
        # conditions by construction, both being reduced over the whole movement.
        rec["d_detectable"] = bool(rec["d_lo"] > 0 or rec["d_hi"] < 0)
        rec["d_identical"] = bool(abs(rec["d_mmc"]) < 1e-12)

        # r_av resamples GROUPS, not trials
        for tag, col in (("opt", "mmc_omc"), ("mmc", "mmc_mmc")):
            av = g.groupby(["part", "arm"])[["omc_omc", col]].mean().values
            rec[f"r_av_{tag}"] = _r(av[:, 0], av[:, 1])
            ba = []
            for _ in range(a.boot):
                i = rng.integers(0, len(av), len(av))
                r = _r(av[i, 0], av[i, 1])
                if np.isfinite(r):
                    ba.append(r)
            lo, hi = np.percentile(ba, [2.5, 97.5])
            rec[f"r_av_{tag}_lo"], rec[f"r_av_{tag}_hi"] = lo, hi
            rec[f"n_groups_{tag}"] = len(av)
        # --- 30 Hz, paired against 60 Hz on the intersection of trials.
        # `mmc30r` is the rate-matched variant the paper reports (SmoothNet's window scaled to the
        # rate); `mmc30` is the naive port, whose peak loss is a filter artefact, not sampling.
        if F30 is not None:
            f = F30[F30.measure == key][["part", "trial", "mmc30r"]]
            M = g[["part", "trial", "omc_omc", "mmc_mmc"]].merge(f, on=["part", "trial"],
                                                                 how="inner").dropna()
            if len(M) > 20:
                xx = M.omc_omc.values
                y60, y30 = M.mmc_mmc.values, M.mmc30r.values
                rec["n_30"] = len(M)
                rec["r_s_30"] = _r(xx, y30)
                b30, bd30 = [], []
                for _ in range(a_boot):
                    i = rng.integers(0, len(xx), len(xx))
                    r3, r6 = _r(xx[i], y30[i]), _r(xx[i], y60[i])
                    if np.isfinite(r3) and np.isfinite(r6):
                        b30.append(r3); bd30.append(r3 - r6)
                rec["r_s_30_lo"], rec["r_s_30_hi"] = np.percentile(b30, [2.5, 97.5])
                rec["d30"] = rec["r_s_30"] - _r(xx, y60)
                rec["d30_lo"], rec["d30_hi"] = np.percentile(bd30, [2.5, 97.5])
                rec["d30_detectable"] = bool(rec["d30_lo"] > 0 or rec["d30_hi"] < 0)
        rows.append(rec)

    R = pd.DataFrame(rows)
    num = R.select_dtypes(include=[float]).columns
    R[num] = R[num].round(4)
    R.to_csv(OUT, index=False)
    print(f"wrote {OUT}\n")
    print(f"{'measure':>32s} {'r_s mmc [95% CI]':>24s} {'width':>6s} {'paired change [CI]':>26s}")
    for _, r in R.iterrows():
        det = "*" if r.d_detectable else " "
        print(f"{r.measure:>32s}  {r.r_s_mmc:5.2f} [{r.r_s_mmc_lo:5.2f},{r.r_s_mmc_hi:5.2f}] "
              f"{r.r_s_mmc_hi - r.r_s_mmc_lo:6.2f}  {r.d_mmc:+7.3f} "
              f"[{r.d_lo:+6.3f},{r.d_hi:+6.3f}]{det}")
    if "d30" in R.columns:
        print(f"\n{'measure':>32s} {'r_s 30Hz [95% CI]':>24s} {'60->30 change [CI]':>26s}")
        for _, r in R.iterrows():
            if not np.isfinite(r.get("d30", np.nan)):
                continue
            det = "*" if r.d30_detectable else " "
            print(f"{r.measure:>32s}  {r.r_s_30:5.2f} [{r.r_s_30_lo:5.2f},{r.r_s_30_hi:5.2f}] "
                  f"{r.d30:+7.3f} [{r.d30_lo:+6.3f},{r.d30_hi:+6.3f}]{det}")
        print(f"  detectable 60->30 changes: {int(R.d30_detectable.sum())} of "
              f"{int(R.d30.notna().sum())}")
    print("\n  * = the change excludes zero")
    print(f"  detectable changes: {int(R.d_detectable.sum())} of {len(R)}")
    print(f"  widest CI on r_s: {R.loc[(R.r_s_mmc_hi-R.r_s_mmc_lo).idxmax(),'measure']}")


if __name__ == "__main__":
    main()
