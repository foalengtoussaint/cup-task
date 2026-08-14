"""Figure: vector-coding PROXIMAL fraction -- MMC vs OMC, and affected vs unaffected.

Panels
  A  per participant x arm mean: OMC (x) vs MMC (y), identity line, r_av
  B  per-trial ECDF of proximal by arm, OMC solid / MMC dashed (zero-inflated -> ECDF not violin)
  C  participant-level slopegraph unaffected -> affected, OMC and MMC side by side
Input : out/automq/vector_coding.csv   (from scripts/vector_coding.py)
Output: paper/figures/proximal_vc.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
C_UN, C_AF = "#2C6FBB", "#E07B39"          # CVD-safe blue / orange
INK, MUTED = "#1a1a1a", "#8a8a8a"


def ecdf(v):
    v = np.sort(np.asarray(v, float))
    return v, np.arange(1, len(v) + 1) / len(v)


def main():
    df = pd.read_csv(ROOT / "out/automq/vector_coding.csv")
    df = df.dropna(subset=["proximal_mmc", "proximal_omc"])
    av = df.groupby(["part", "arm"])[["proximal_mmc", "proximal_omc"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values():
            s.set_color(MUTED)

    # ---- A: MMC vs OMC ----
    ax = axes[0]
    lim = max(av[["proximal_mmc", "proximal_omc"]].max()) * 1.15
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=":", zorder=1)
    for arm, c in (("unaffected", C_UN), ("affected", C_AF)):
        g = av[av.arm == arm]
        ax.scatter(g.proximal_omc, g.proximal_mmc, s=55, color=c, edgecolor="white",
                   linewidth=1.2, label=arm, zorder=3)
    r_av = pearsonr(av.proximal_omc, av.proximal_mmc)[0]
    r_s = pearsonr(df.proximal_omc, df.proximal_mmc)[0]
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("OMC  proximal fraction", color=INK)
    ax.set_ylabel("MMC  proximal fraction", color=INK)
    ax.set_title(f"A  MMC vs OMC   $r_{{av}}$ = {r_av:.2f}  (per-trial $r$ = {r_s:.2f})",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.text(0.03, 0.95, f"n = {len(av)} participant x arm", transform=ax.transAxes,
            fontsize=8, color=MUTED, va="top")

    # ---- B: ECDF by arm ----
    ax = axes[1]
    for arm, c in (("unaffected", C_UN), ("affected", C_AF)):
        for src, ls in (("omc", "-"), ("mmc", "--")):
            x, y = ecdf(df[df.arm == arm][f"proximal_{src}"])
            ax.step(x, y, where="post", color=c, lw=2 if src == "omc" else 1.4,
                    ls=ls, label=f"{arm} ({src.upper()})")
    u = df[df.arm == "unaffected"].proximal_omc; a = df[df.arm == "affected"].proximal_omc
    U, p = mannwhitneyu(u, a); rb = 1 - 2 * U / (len(u) * len(a))
    ax.set_xlim(0, 0.35); ax.set_ylim(0, 1.02)
    ax.set_xlabel("proximal fraction (per trial)", color=INK)
    ax.set_ylabel("cumulative fraction of trials", color=INK)
    ax.set_title("B  affected vs unaffected (per trial)", fontsize=10.5, color=INK, loc="left")
    ax.text(0.03, 0.99, f"OMC: p = {p:.1e},  rank-biserial = {rb:+.2f}", transform=ax.transAxes,
            fontsize=8.5, color=INK, va="top")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right",
              bbox_to_anchor=(1.0, 0.42))
    ax.text(0.97, 0.02, f"zero-valued trials: OMC {np.mean(df.proximal_omc==0):.0%} / "
            f"MMC {np.mean(df.proximal_mmc==0):.0%}", transform=ax.transAxes,
            fontsize=8, color=MUTED, va="bottom", ha="right")

    # ---- C: paired slopegraph ----
    ax = axes[2]
    piv = {s: av.pivot(index="part", columns="arm", values=f"proximal_{s}").dropna()
           for s in ("omc", "mmc")}
    xs = {"omc": (0, 1), "mmc": (2.2, 3.2)}
    for s, (x0, x1) in xs.items():
        P = piv[s]
        for _, r in P.iterrows():
            ax.plot([x0, x1], [r.unaffected, r.affected], color=MUTED, lw=1, alpha=.6, zorder=1)
        ax.scatter([x0] * len(P), P.unaffected, s=45, color=C_UN, edgecolor="white",
                   linewidth=1, zorder=3)
        ax.scatter([x1] * len(P), P.affected, s=45, color=C_AF, edgecolor="white",
                   linewidth=1, zorder=3)
        ax.plot([x0, x1], [P.unaffected.median(), P.affected.median()], color=INK,
                lw=2.5, zorder=4)
    ax.set_xticks([0, 1, 2.2, 3.2])
    ax.set_xticklabels(["unaff.", "aff.", "unaff.", "aff."], color=INK, fontsize=9)
    ax.text(0.5, -0.14, "OMC", ha="center", transform=ax.get_xaxis_transform(),
            color=INK, fontsize=10)
    ax.text(2.7, -0.14, "MMC", ha="center", transform=ax.get_xaxis_transform(),
            color=INK, fontsize=10)
    ax.set_ylabel("proximal fraction (participant mean)", color=INK)
    ax.set_title("C  paired by participant (thick = median)", fontsize=10.5,
                 color=INK, loc="left")

    fig.tight_layout()
    out = ROOT / "paper/figures/proximal_vc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", out, flush=True)
    print(f"participants {av.part.nunique()}, trials {len(df)}", flush=True)


if __name__ == "__main__":
    main()
