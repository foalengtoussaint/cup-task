"""Visualise what the POSITION-loss GNN did to the wrist trajectory: raw MMC vs GNN vs OMC, per axis.

For a good fold (P07) and a broken fold (P17): pick the trial with the most valid wrist frames, put
all three (raw, GNN, OMC) into ONE common frame via a single rigid Kabsch fit of RAW->OMC on the arm
(so the plot shows the SAME transform applied to raw and GNN -- you see the GNN's ADDITIONAL move on
top of raw, not a per-series realignment). Then plot x/y/z over time.

    python scripts/gnn_viz_correction.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gnn_train as T
import gnn_refiner as G

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 31


def load(held):
    m = G.GNNRefiner(hidden=64, blocks=3, t_kernel=5, dropout=0.1).to(DEV)
    m.load_state_dict(torch.load(f"out/gnn/gnn_{held}.pt", map_location=DEV)); m.eval()
    return m


def main():
    trials = T.load_clean()
    folds = [("P07", "WORKED: wrist 22→8mm"), ("P17", "BROKE: wrist 33→54mm")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")
    for col, (held, note) in enumerate(folds):
        mdl = load(held)
        test = [t for t in trials if t["part"] == held]
        wi = G.JOINTS.index(f'{test[0]["side"]}_wrist')
        # trial with most valid wrist frames
        t = max(test, key=lambda z: z["valid"][:, wi].sum())
        gnn = T.refine_trial(mdl, t["mmc"], t["valid"], WIN)
        arm_i = T.ARM_I
        # ONE Kabsch raw->omc on arm joints, applied to raw AND gnn (common frame)
        A = np.vstack([t["mmc"][:, j] for j in arm_i])
        B = np.vstack([t["omc"][:, j] for j in arm_i])
        R, tt, _ = T.H._kabsch(A, B)
        def xform(P):  # (T,3)
            return P @ R.T + tt
        raw_w = xform(t["mmc"][:, wi])
        gnn_w = xform(gnn[:, wi])
        omc_w = t["omc"][:, wi]
        v = t["valid"][:, wi]
        raw_w[~v] = np.nan; gnn_w[~v] = np.nan; omc_w[~v] = np.nan
        tt_ax = np.arange(len(raw_w)) / T.H.VIDEO_FPS
        for ax_i, axname in enumerate("XYZ"):
            ax = axes[ax_i][col]
            ax.plot(tt_ax, omc_w[:, ax_i], color="k", lw=2.0, label="OMC (truth)")
            ax.plot(tt_ax, raw_w[:, ax_i], color="tab:red", lw=1.3, alpha=0.8, label="raw MMC")
            ax.plot(tt_ax, gnn_w[:, ax_i], color="tab:blue", lw=1.3, label="GNN")
            ax.set_ylabel(f"wrist {axname} (mm)")
            if ax_i == 0:
                ax.set_title(f"{held}  ({note})\n{t['trial']}", fontsize=10)
                ax.legend(fontsize=8, loc="best")
        axes[2][col].set_xlabel("time (s)")
    fig.suptitle("What the POSITION-loss GNN did to the wrist (common Kabsch frame: raw→OMC)",
                 fontsize=12)
    fig.tight_layout()
    out = Path("out/gnn/position_correction_wrist.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
