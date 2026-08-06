"""Is AutoMQ's world-vertical shoulder flexion a DIFFERENT quantity from a trunk-referenced one?

AutoMQ ships shoulder_flexion = angle(shoulder-elbow, GLOBAL vertical Z) -- it has hip_L/hip_R markers
but does NOT use them. For a stroke cohort that LEANS (trunk compensation), a world-vertical angle folds
trunk lean into the "shoulder" number. This test recomputes flexion from AutoMQ's OWN sub-mm markers,
TWO ways, and checks whether they diverge on high-lean trials:

  vertical : angle(shoulder-elbow, [0,0,1])                       [AutoMQ's shipped definition]
  trunk    : angle(shoulder-elbow, shoulder_mid - hip_mid)        [trunk-referenced, uses their hips]

Both from OMC markers only -- ZERO MMC. Per trial: max of each over the reach->drink window, the trunk
LEAN (hip->shoulder tilt from vertical, deg), and (vertical - trunk) flexion difference. If the diff
grows with lean, the two definitions measure different things and AutoMQ's is trunk-contaminated.
    python paper/scripts/automq_flexion_ref_test.py -> paper/automq_flexion_ref_test.png + prints
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from score_vs_automq import load_automq, automq_part, AUTOMQ
COHORT = ["P07", "P08", "P10", "P12", "P13", "P14", "P15", "P17", "P19", "P25"]


def _ang(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _marker_series(markers_df, name):
    """(T,3) array for one marker from AutoMQ's long (frame, marker)->xyz table."""
    try:
        sub = markers_df.xs(name, level=1)
    except (KeyError, TypeError):
        return None
    return sub[["x", "y", "z"]].to_numpy()


def main():
    amq = load_automq()          # for phase windows (per trial)
    Z = np.array([0.0, 0.0, 1.0])
    rows = []
    pat_side = {"L": ("shoulder_L", "elbow_L"), "R": ("shoulder_R", "elbow_R")}
    for p in COHORT:
        try:
            cdf = pd.read_pickle(AUTOMQ / p / "combined_data_with_kinematics.pkl")
        except Exception:
            continue
        for fk in cdf.index:
            tn, sd = int(fk[1]), fk[2]
            rec = amq.get((automq_part(p), tn, sd))
            if rec is None or rec.get("phases") is None:
                continue
            mk = cdf.loc[fk, "markers"]
            sh_name, el_name = pat_side[sd]
            sh = _marker_series(mk, sh_name); el = _marker_series(mk, el_name)
            hL = _marker_series(mk, "hip_L"); hR = _marker_series(mk, "hip_R")
            shL = _marker_series(mk, "shoulder_L"); shR = _marker_series(mk, "shoulder_R")
            if any(x is None for x in (sh, el, hL, hR, shL, shR)):
                continue
            T = min(len(sh), len(el), len(hL), len(hR), len(shL), len(shR))
            sh, el, hL, hR, shL, shR = (a[:T] for a in (sh, el, hL, hR, shL, shR))
            arm = sh - el                                   # AutoMQ upper-arm vector (el->sh)
            sh_mid = (shL + shR) / 2.0
            hip_mid = (hL + hR) / 2.0
            up_trunk = sh_mid - hip_mid                     # trunk UP axis (hip->shoulder)
            flex_vert = _ang(arm, Z[None, :])               # AutoMQ shipped
            flex_trunk = _ang(arm, up_trunk)                # trunk-referenced
            lean = _ang(up_trunk, Z[None, :])               # trunk tilt from vertical (deg)
            # phases are 100Hz frames; take the reach->drink span on the 100Hz series directly
            ph = rec["phases"]
            def span(names):
                s = [ph[k] for k in names if k in ph and ph[k] is not None]
                idx = [i for seg in s for i in (seg if isinstance(seg, (list, tuple)) else [seg])]
                return None
            # AutoMQ phases dict: keys are phase names -> (start,end) in 100Hz frames. Take reaching[0]..drinking[1]
            def win(a, b):
                ra = ph.get(a); rb = ph.get(b)
                if ra is None or rb is None:
                    return None
                s = int(ra[0]); e = int(rb[1])
                return (s, e) if e > s else None
            w = win("Reaching", "Drinking") or win("reaching", "drinking")
            if w is None:
                # fallback: whole trial
                w = (0, T)
            s, e = max(0, w[0]), min(T, w[1])
            if e - s < 10:
                continue
            def mx(a):
                seg = a[s:e]; seg = seg[np.isfinite(seg)]
                return float(np.max(seg)) if len(seg) else np.nan
            rows.append({"part": p, "trial": tn, "side": sd,
                         "flex_vert": mx(flex_vert), "flex_trunk": mx(flex_trunk),
                         "lean_max": mx(lean), "lean_med": float(np.nanmedian(lean[s:e]))})

    df = pd.DataFrame(rows).dropna(subset=["flex_vert", "flex_trunk", "lean_med"])
    df["diff"] = df["flex_vert"] - df["flex_trunk"]
    df.to_csv(PAPER / "automq_flexion_ref_test.csv", index=False)

    print(f"\nn = {len(df)} trials (OMC markers only, zero MMC)\n")
    print(f"flexion VERTICAL (AutoMQ shipped): median {df.flex_vert.median():.1f}  IQR [{df.flex_vert.quantile(.25):.1f},{df.flex_vert.quantile(.75):.1f}]")
    print(f"flexion TRUNK-referenced         : median {df.flex_trunk.median():.1f}  IQR [{df.flex_trunk.quantile(.25):.1f},{df.flex_trunk.quantile(.75):.1f}]")
    print(f"the two definitions differ by     : median {df['diff'].median():+.1f} deg  |diff| med {df['diff'].abs().median():.1f}  max {df['diff'].abs().max():.1f}")
    print(f"trunk lean (hip->shoulder from vertical): median {df.lean_med.median():.1f} deg  max {df.lean_med.max():.1f}")
    r = pearsonr(df.lean_med, df['diff'])[0]
    rs = spearmanr(df.lean_med, df['diff']).correlation
    print(f"\n*** does the definition DIFFERENCE grow with trunk LEAN?  pearson r={r:+.2f}  spearman={rs:+.2f} ***")
    print("   (strong positive => AutoMQ's world-vertical flexion is trunk-CONTAMINATED; the two are different quantities)")
    # per participant lean
    print(f"\n{'part':6}{'n':>4}{'lean_med':>10}{'|def diff|':>12}")
    for p in sorted(df.part.unique()):
        g = df[df.part == p]
        print(f"{p:6}{len(g):>4}{g.lean_med.median():>10.1f}{g['diff'].abs().median():>12.1f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.get_cmap("tab10"); parts = sorted(df.part.unique())
    pcol = {p: cmap(i % 10) for i, p in enumerate(parts)}
    for p in parts:
        g = df[df.part == p]
        ax[0].scatter(g.lean_med, g['diff'], s=16, alpha=0.6, color=pcol[p], label=p)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("trunk lean (deg from vertical)"); ax[0].set_ylabel("flexion: vertical - trunk-ref (deg)")
    ax[0].set_title(f"Definition gap grows with lean (r={r:+.2f})"); ax[0].legend(fontsize=6, ncol=2, frameon=False)
    ax[1].scatter(df.flex_trunk, df.flex_vert, s=16, alpha=0.5, color="#c1440e")
    lo = min(df.flex_trunk.min(), df.flex_vert.min()); hi = max(df.flex_trunk.max(), df.flex_vert.max())
    ax[1].plot([lo, hi], [lo, hi], "0.55"); ax[1].set_aspect("equal", "box")
    ax[1].set_xlabel("flexion, trunk-referenced (deg)"); ax[1].set_ylabel("flexion, world-vertical (deg)")
    ax[1].set_title("Two flexion definitions, OMC markers")
    fig.suptitle("AutoMQ flexion: world-vertical vs trunk-referenced (both from OMC markers)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PAPER / "automq_flexion_ref_test.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {PAPER/'automq_flexion_ref_test.png'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
