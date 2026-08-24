"""Figure: does smoothing trade absolute error for between-trial correlation?

Per-phase peak wrist speed, MMC vs OMC, one point per (trial, phase).
Panels A-D: the scatter for four variants. Panel E: the smooth_w sweep.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import gnn_train as T                    # noqa: E402
import gnn_refiner as G                  # noqa: E402
import compare_pose_omc_delta as H       # noqa: E402

# reference categorical palette, fixed order (slots 1,2)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
PH = ["amq_reaching", "amq_forward_transport", "amq_back_transport", "amq_returning"]
SWEEP = [("fix", 0.0), ("smooth_0.3", 0.3), ("smooth_1.0", 1.0),
         ("smooth_3.0", 3.0), ("smooth_10.0", 10.0)]
PANELS = ["fix", "smooth_0.3", "smooth_10.0", "freebone_0.2_s10.0"]
TITLE = {"fix": "no smoothing", "smooth_0.3": "smooth_w = 0.3  (OMC-jitter matched)",
         "smooth_10.0": "smooth_w = 10", "freebone_0.2_s10.0": "free bone + smooth_w = 10"}


def collect():
    tags = sorted({t for t, _ in SWEEP} | set(PANELS))
    V = {}
    for tg in tags:
        z = np.load(ROOT / f"cache/ba_variants/{tg}.npz", allow_pickle=True)
        V[tg] = dict(zip(z["ids"], z["traj"]))
    ids = set(V["freebone_0.2_s10.0"])
    out = {t: [] for t in tags}
    for t in T.load_clean(need_reproj=True):
        tid = f"{t['part']}/{t['trial']}"
        if tid not in ids:
            continue
        f = ROOT / f"cache/seg_inputs_26x/{t['part']}__{t['trial']}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        s = t["side"]; wi = G.JOINTS.index(f"{s}_wrist"); v = t["valid"][:, wi]
        wo = t["omc"][:, wi].copy(); wo[~v] = np.nan
        so = H._lp(H._speed(wo))
        for ph in PH:
            try:
                a, b = int(d[ph][0]), int(d[ph][1])
            except Exception:
                continue
            if b - a < 8 or b > len(so):
                continue
            po = np.nanmax(so[a:b])
            if not np.isfinite(po) or po <= 0:
                continue
            for tg in tags:
                P = V[tg].get(tid)
                if P is None or P.shape[0] != len(so):
                    continue
                wm = P[:, wi].copy(); wm[~v] = np.nan
                pm = np.nanmax(H._lp(H._speed(wm))[a:b])
                if np.isfinite(pm):
                    out[tg].append((po, pm))
    return {k: np.array(v) for k, v in out.items() if v}


def r2_of(A):
    po, pm = A[:, 0], A[:, 1]
    sl, ic = np.polyfit(po, pm, 1)
    return 1 - ((pm - (sl * po + ic)) ** 2).sum() / ((pm - pm.mean()) ** 2).sum(), sl


def main() -> None:
    D = collect()
    fig = plt.figure(figsize=(15, 4.2))
    gs = fig.add_gridspec(1, 5, wspace=0.32)
    lim = (0, 1600)
    for i, tg in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        A = D[tg]
        ax.plot(lim, lim, "-", color=MUTED, lw=1.4, zorder=1)          # identity
        ax.scatter(A[:, 0], A[:, 1], s=14, color=BLUE, alpha=.55,
                   edgecolors="none", zorder=3)
        sl, ic = np.polyfit(A[:, 0], A[:, 1], 1)
        xs = np.array(lim)
        ax.plot(xs, sl * xs + ic, "-", color=ORANGE, lw=2, zorder=4)   # fit
        rs = spearmanr(A[:, 0], A[:, 1]).correlation
        r2, _ = r2_of(A)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_title(TITLE[tg], fontsize=9.5, color=INK, pad=7)
        ax.text(.04, .95, f"$r_s$ = {rs:.3f}\n$R^2$ = {r2:.3f}\nslope = {sl:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK2)
        ax.set_xlabel("OMC peak speed (mm/s)", fontsize=8.5, color=INK2)
        if i == 0:
            ax.set_ylabel("MMC peak speed (mm/s)", fontsize=8.5, color=INK2)
        ax.grid(True, color=GRID, lw=.7); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)

    ax = fig.add_subplot(gs[0, 4])
    xs = [w for _, w in SWEEP]
    rs = [spearmanr(D[t][:, 0], D[t][:, 1]).correlation for t, _ in SWEEP]
    r2 = [r2_of(D[t])[0] for t, _ in SWEEP]
    xp = np.arange(len(xs))
    ax.plot(xp, rs, "-o", color=BLUE, lw=2, ms=8, label="$r_s$ (rank agreement)")
    ax.plot(xp, r2, "-o", color=ORANGE, lw=2, ms=8, label="$R^2$ (variance explained)")
    ax.set_xticks(xp); ax.set_xticklabels([f"{w:g}" for w in xs], fontsize=8)
    ax.set_xlabel("smooth_w", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 1); ax.set_title("cost of smoothing", fontsize=9.5, color=INK, pad=7)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(True, color=GRID, lw=.7); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)

    fig.suptitle("Per-phase peak wrist speed, MMC vs OMC  —  "
                 f"{len(D['fix'])} phase-peaks, 91 trials", fontsize=11.5, color=INK, y=1.02)
    out = ROOT / "out/figures/peak_agreement.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}", flush=True)
    for t, w in SWEEP:
        A = D[t]; r, s = r2_of(A)
        print(f"  smooth_w={w:<5g} r_s {spearmanr(A[:,0],A[:,1]).correlation:.3f}  "
              f"R2 {r:.3f}  slope {s:.3f}", flush=True)


if __name__ == "__main__":
    main()
