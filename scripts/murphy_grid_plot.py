"""Fig.4-style OMC-vs-MMC validation figures from the murphy_grid tidy CSV.

Per VARIANT, one multi-panel scatter (OMC x, MMC y), one panel per Murphy measure:
  - points colored PER PARTICIPANT, shaped PER ARM (affected=triangle, unaffected=circle)
  - identity line (solid grey) + OLS regression (dashed)
  - Spearman rs annotated (the paper's metric)
Also emits, across variants:
  - a median|error| table per measure (which stage wins each measure)
  - a Bland-Altman panel-set per variant (bias +/- 1.96 SD limits of agreement)

Reads out/gnn/murphy_grid.csv (cols: variant,part,trial,arm,side,measure,omc,mmc).
Writes out/gnn/fig_murphy_<variant>.png, fig_bland_<variant>.png, murphy_rs_table.csv.
Cohort stratum (clean P07/P08/P15 vs miscalib P17/P19) is encoded in color; --stratum filters.
"""
import sys, csv, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CLEAN = {"P07", "P08", "P15"}
MISCAL = {"P17", "P19"}
# measure -> (nice label, unit) in a sensible panel order (position then angle, ~ Fig.4)
MEASURES = [
    ("peak_velocity", "PV [mm/s]"),
    ("peak_elbow_ang_vel", "Elbow angular PV [deg/s]"),
    ("time_to_peak_velocity", "Time to PV [s]"),
    ("time_to_first_peak_velocity", "Time to first PV [s]"),
    ("number_of_movement_units", "Number of movement units"),
    ("total_movement_time", "Total movement time [s]"),
    ("interjoint_coordination", "Interjoint coordination"),
    ("max_trunk_displacement", "Trunk displacement [mm]"),
    ("shoulder_flexion_reaching", "Shoulder flexion [deg]"),
    ("elbow_extension_reaching", "Elbow extension [deg]"),
    ("shoulder_abduction_reaching", "Shoulder abduction [deg]"),
    ("shoulder_flexion_drinking", "Shoulder flexion D [deg]"),
]
PART_COLORS = {"P07": "#1f77b4", "P08": "#2ca02c", "P15": "#9467bd",
               "P17": "#d62728", "P19": "#ff7f0e"}


def load(csv_path):
    rows = defaultdict(lambda: defaultdict(list))   # variant -> measure -> [(part,arm,omc,mmc)]
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows[r["variant"]][r["measure"]].append(
                (r["part"], r["arm"], float(r["omc"]), float(r["mmc"])))
    return rows


def _panel_scatter(ax, data, label):
    """data: list of (part, arm, omc, mmc). Draw the OMC-vs-MMC panel + rs + identity + fit."""
    if len(data) < 3:
        ax.set_title(f"{label}\n(n<3)", fontsize=8); ax.set_xticks([]); ax.set_yticks([]); return
    o = np.array([d[2] for d in data]); m = np.array([d[3] for d in data])
    fin = np.isfinite(o) & np.isfinite(m)
    o, m = o[fin], m[fin]; data = [d for d, k in zip(data, fin) if k]
    for part in sorted(set(d[0] for d in data)):
        for arm, mk in [("affected", "^"), ("unaffected", "o")]:
            xs = [d[2] for d in data if d[0] == part and d[1] == arm]
            ys = [d[3] for d in data if d[0] == part and d[1] == arm]
            if xs:
                ax.scatter(xs, ys, s=12, marker=mk, alpha=0.5,
                           color=PART_COLORS.get(part, "#888"), linewidths=0)
    lo = float(min(o.min(), m.min())); hi = float(max(o.max(), m.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", lw=1)      # identity
    if len(o) >= 3 and np.std(o) > 1e-9:
        b, a = np.polyfit(o, m, 1)
        ax.plot([lo, hi], [a + b * lo, a + b * hi], "k--", lw=1)                  # OLS fit
        rs = stats.spearmanr(o, m).statistic
        ax.text(0.97, 0.05, f"$r_s$={rs:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("OMC", fontsize=7); ax.set_ylabel("MMC", fontsize=7)
    ax.tick_params(labelsize=6); ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)


def fig_variant(variant, mdata, outdir, stratum_parts):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, (mk, lab) in zip(axes.ravel(), MEASURES):
        data = [d for d in mdata.get(mk, []) if d[0] in stratum_parts]
        _panel_scatter(ax, data, lab)
    # legend
    handles = [plt.Line2D([], [], marker="o", ls="", color=PART_COLORS[p], label=p)
               for p in sorted(stratum_parts) if p in PART_COLORS]
    handles += [plt.Line2D([], [], marker="^", ls="", color="grey", label="affected"),
                plt.Line2D([], [], marker="o", ls="", color="grey", label="unaffected")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle(f"OMC vs MMC — {variant}   (each point = one trial)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    p = outdir / f"fig_murphy_{variant.replace('+','_')}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig_bland(variant, mdata, outdir, stratum_parts):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, (mk, lab) in zip(axes.ravel(), MEASURES):
        data = [d for d in mdata.get(mk, []) if d[0] in stratum_parts]
        if len(data) < 3:
            ax.set_title(f"{lab}\n(n<3)", fontsize=8); ax.set_xticks([]); ax.set_yticks([]); continue
        o = np.array([d[2] for d in data]); m = np.array([d[3] for d in data])
        fin = np.isfinite(o) & np.isfinite(m); o, m = o[fin], m[fin]
        diff = m - o; mean = (m + o) / 2; bias = diff.mean(); sd = diff.std()
        for part in sorted(set(d[0] for d in data)):
            for arm, mrk in [("affected", "^"), ("unaffected", "o")]:
                idx = [i for i, d in enumerate([d for d, k in zip(data, fin) if k])
                       if d[0] == part and d[1] == arm]
                if idx:
                    ax.scatter(mean[idx], diff[idx], s=12, marker=mrk, alpha=0.5,
                               color=PART_COLORS.get(part, "#888"), linewidths=0)
        ax.axhline(bias, color="k", lw=1)
        ax.axhline(bias + 1.96 * sd, color="r", ls="--", lw=0.8)
        ax.axhline(bias - 1.96 * sd, color="r", ls="--", lw=0.8)
        ax.set_title(f"{lab}\nbias {bias:+.1f}, LoA ±{1.96*sd:.1f}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Bland-Altman (MMC-OMC) — {variant}", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    p = outdir / f"fig_bland_{variant.replace('+','_')}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def rs_and_err_table(rows, stratum_parts, outdir):
    """rs + median|err| per (variant, measure) across all variants -> CSV + printed summary.

    Covers EVERY measure in the data (the union across variants), not just the 12 plotted panels --
    this table is what the pipeline decision is read off, so it must not silently drop the 3 measures
    the figure omits for space (the % timing variants + shoulder_abduction_drinking)."""
    out = []
    variants = list(rows.keys())
    labeled = [mk for mk, _ in MEASURES]
    all_measures = labeled + sorted({mk for v in variants for mk in rows[v]} - set(labeled))
    for mk in all_measures:
        row = {"measure": mk}
        for v in variants:
            data = [d for d in rows[v].get(mk, []) if d[0] in stratum_parts]
            if len(data) >= 3:
                o = np.array([d[2] for d in data]); m = np.array([d[3] for d in data])
                fin = np.isfinite(o) & np.isfinite(m); o, m = o[fin], m[fin]
                rs = stats.spearmanr(o, m).statistic if np.std(o) > 1e-9 else np.nan
                mederr = float(np.median(np.abs(m - o)))
                row[f"{v}__rs"] = round(rs, 3); row[f"{v}__mederr"] = round(mederr, 2)
        out.append(row)
    p = outdir / "murphy_rs_table.csv"
    keys = ["measure"] + [f"{v}__{s}" for v in variants for s in ("rs", "mederr")]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(f"\n=== rs by measure x variant (stratum={sorted(stratum_parts)}) ===", flush=True)
    print("  " + "measure".ljust(30) + "".join(v[:14].rjust(15) for v in variants), flush=True)
    for r in out:
        print("  " + r["measure"].ljust(30) +
              "".join((f"{r.get(v+'__rs','-')}").rjust(15) for v in variants), flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/gnn/murphy_grid.csv")
    ap.add_argument("--stratum", choices=["all", "clean", "miscal"], default="all")
    a = ap.parse_args()
    rows = load(ROOT / a.csv)
    parts = {"all": CLEAN | MISCAL, "clean": CLEAN, "miscal": MISCAL}[a.stratum]
    # per-stratum subfolder so all figures + the table for one stratum live together and the three
    # strata don't overwrite each other's identically-named files.
    outdir = ROOT / "out" / "gnn" / f"murphy_{a.stratum}"; outdir.mkdir(parents=True, exist_ok=True)
    print(f"variants: {list(rows.keys())}   stratum={a.stratum} ({sorted(parts)}) -> {outdir}", flush=True)
    for v in rows:
        p1 = fig_variant(v, rows[v], outdir, parts)
        p2 = fig_bland(v, rows[v], outdir, parts)
        print(f"  {v}: wrote {p1.name}, {p2.name}", flush=True)
    rs_and_err_table(rows, parts, outdir)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
