"""Aggregate score_vs_automq.csv into the per-measure agreement report (MMC vs AutoMQ ground truth).

Reports, per variant, per measure: n, median|err|, bias (median MMC-AutoMQ), Spearman rs, and the
OLS slope (rs high + slope!=1 = we capture the RANK but not the RANGE -- the [[project_murphy_estimators]]
trap). PEAK measures (peak_velocity, peak_elbow_angular_velocity) LEAD WITH p95 (the robust choice the
consolidation grid + AutoMQ smoke both support); `max` shown only as the "matches-AutoMQ-definition"
reference. The 2 shoulder-angle measures are FLAGGED as the known W0(up=-Y)-vs-mocap(up=+Z) frame /
calibration floor (consolidation grid: "tight rs but systematic Bland-Altman offset, unfixable by any
smoother") -- reported, not re-investigated.

    python scripts/analyze_automq_scores.py [--csv out/scoring/score_vs_automq.csv]
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

# report order; peak measures carry their preferred metric
POS = ["total_movement_time", "peak_velocity", "time_to_peak_velocity_percent",
       "time_to_first_peak_velocity_percent", "time_to_peak_velocity",
       "time_to_first_peak_velocity", "number_of_movement_units", "max_trunk_displacement"]
ANG = ["max_elbow_angle", "peak_elbow_angular_velocity", "interjoint_coordination",
       "max_shoulder_flexion", "max_shoulder_abduction"]
# LEAD WITH max: AutoMQ's ground-truth peak_velocity/peak_elbow_angular_velocity are np.MAX over the
# reaching window. p95 discards the top 5% by construction, so our-p95-vs-their-max builds in a ~13mm/s
# spurious negative bias. max is the DEFINITION-MATCHED reduction (bias -27 vs -40 on BA). p95 stays as
# a robustness reference (marginally higher rs). NB the consolidation grid preferred p95 for INTERNAL
# pose-robustness where both sides were ours; against an external max-based GT, max is correct.
PEAK_PREF = {"peak_velocity": "max", "peak_elbow_angular_velocity": "max"}
FRAME_FLOOR = {"max_shoulder_flexion", "max_shoulder_abduction"}
UNIT = {"total_movement_time": "s", "peak_velocity": "mm/s", "time_to_peak_velocity": "s",
        "time_to_peak_velocity_percent": "%", "time_to_first_peak_velocity": "s",
        "time_to_first_peak_velocity_percent": "%", "number_of_movement_units": "count",
        "max_trunk_displacement": "mm", "max_elbow_angle": "deg",
        "peak_elbow_angular_velocity": "deg/s", "interjoint_coordination": "r",
        "max_shoulder_flexion": "deg", "max_shoulder_abduction": "deg"}


def agg(g):
    a, m = g["automq"].values.astype(float), g["mmc"].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(m)
    a, m = a[ok], m[ok]
    # drop non-physical MMC blow-ups (BA divergence: |coord|->1e21 => derived measure is astronomical).
    # These are guarded out at the scorer now (_pose_variant), but strip here too so re-analysis of an
    # older CSV is also clean and the OLS slope can't be wrecked by a single 1e19 point.
    phys = np.abs(m) < 1e6
    a, m = a[phys], m[phys]
    if len(a) < 3:
        return None
    err = np.abs(m - a)
    rs = spearmanr(a, m).correlation if (np.std(a) > 0 and np.std(m) > 0) else np.nan
    slope = np.polyfit(a, m, 1)[0] if np.std(a) > 0 else np.nan
    return dict(n=len(a), med_err=np.median(err), bias=np.median(m - a),
                rs=rs, slope=slope, p90_err=np.percentile(err, 90))


def measure_rows(df, measure):
    """Peaks -> definition-matched primary (PEAK_PREF, = max) + the other metric as reference (tagged *)."""
    if measure in PEAK_PREF:
        pref = PEAK_PREF[measure]                      # primary reduction (max, = AutoMQ's definition)
        ref = "p95" if pref == "max" else "max"        # the other one, shown as a reference
        return [(f"{measure} ({pref})", df[(df.measure == measure) & (df.peak_metric == pref)]),
                (f"{measure} ({ref}*)", df[(df.measure == measure) & (df.peak_metric == ref)])]
    return [(measure, df[df.measure == measure])]


def report_variant(df, variant):
    sub = df[df.variant == variant]
    print(f"\n{'='*94}\n=== {variant}  vs AutoMQ ground truth ===\n{'='*94}", flush=True)
    print(f"{'measure':38s}{'unit':7s}{'n':>4s}{'med|err|':>10s}{'p90|err|':>10s}"
          f"{'bias':>10s}{'rs':>7s}{'slope':>7s}", flush=True)
    for group, names in (("POSITION", POS), ("ANGLE", ANG)):
        print(f"  -- {group} --", flush=True)
        for meas in names:
            for label, g in measure_rows(sub, meas):
                r = agg(g)
                base = meas
                flag = "  <frame floor>" if base in FRAME_FLOOR else ""
                if r is None:
                    print(f"{label:38s}{UNIT.get(base,''):7s}{'--':>4s}  (insufficient)", flush=True)
                    continue
                print(f"{label:38s}{UNIT.get(base,''):7s}{r['n']:4d}{r['med_err']:10.2f}"
                      f"{r['p90_err']:10.2f}{r['bias']:+10.2f}{r['rs']:+7.2f}{r['slope']:+7.2f}{flag}",
                      flush=True)


def variant_shootout(df):
    """Which variant wins each measure on median|err| (the grid's question, vs the real GT now)."""
    print(f"\n{'='*94}\n=== VARIANT SHOOTOUT: median|err| per measure (best in <>) ===\n{'='*94}", flush=True)
    variants = ["pipeline", "pipeline+savgol", "pipeline+smoothnet", "BA+smoothnet"]
    hdr = f"{'measure':34s}" + "".join(f"{v.replace('pipeline','pipe'):>16s}" for v in variants)
    print(hdr, flush=True)
    for group, names in (("POSITION", POS), ("ANGLE", ANG)):
        print(f"  -- {group} --", flush=True)
        for meas in names:
            metric = PEAK_PREF.get(meas)
            label = f"{meas} ({metric})" if metric else meas
            vals = {}
            for v in variants:
                sel = df[(df.variant == v) & (df.measure == meas)]
                if metric:
                    sel = sel[sel.peak_metric == metric]
                r = agg(sel)
                vals[v] = r["med_err"] if r else np.nan
            best = min((x for x in vals.values() if np.isfinite(x)), default=np.nan)
            cells = ""
            for v in variants:
                x = vals[v]
                s = f"{x:.2f}" if np.isfinite(x) else "--"
                s = f"<{s}>" if (np.isfinite(x) and np.isfinite(best) and abs(x-best) < 1e-9) else s
                cells += f"{s:>16s}"
            flag = " *floor*" if meas in FRAME_FLOOR else ""
            print(f"{label:34s}{cells}{flag}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/scoring/score_vs_automq.csv")
    args = ap.parse_args(argv)
    df = pd.read_csv(ROOT / args.csv)
    print(f"loaded {len(df)} rows | variants={sorted(df.variant.unique())} | "
          f"parts={sorted(df.part.unique())} | trials={df.trial.nunique()}", flush=True)
    # coverage: matched trials per part
    cov = df[df.variant == "BA+smoothnet"].groupby("part")["trial"].nunique()
    print(f"matched trials per part (BA+smoothnet): {cov.to_dict()}", flush=True)

    for v in ["BA+smoothnet", "pipeline+smoothnet", "pipeline+savgol", "pipeline"]:
        if v in df.variant.unique():
            report_variant(df, v)
    variant_shootout(df)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
