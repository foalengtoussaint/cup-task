"""Same question as flexion, for ABDUCTION: is AutoMQ's world-frame abduction a different quantity from a
body-referenced one, and does the gap grow with TORSO YAW (rotation)?

AutoMQ abduction = angle( [arm_x, 0, arm_z], world-Z ) -- projects the arm onto the world X-Z plane and
measures from world vertical. This hard-assumes world-X = the participant's SIDEWAYS axis and world-Y =
FORWARD. If the torso is YAWED (rotated toward the cup) or LEANING, world-X/Y no longer match anatomy and
flexion<->abduction bleed. A body-referenced abduction uses the actual SHOULDER LINE as sideways.

From AutoMQ's OWN markers (zero MMC), per trial (reach->drink max):
  abd_world : AutoMQ shipped ( [arm_x,0,arm_z] vs Z )
  abd_body  : angle of arm's component along the shoulder-line (lateral), i.e. arcsin(arm . side_unit)
  yaw       : torso rotation = angle between the shoulder-line (projected to the world X-Y/horizontal
              plane) and the world-X axis (deg). lean = trunk tilt from vertical (deg).
Reports whether (abd_world - abd_body) grows with yaw and/or lean.
    python paper/scripts/automq_abduction_ref_test.py -> paper/automq_abduction_ref_test.png + prints
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
Z = np.array([0.0, 0.0, 1.0]); Xw = np.array([1.0, 0.0, 0.0])


def _ang(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def _maxseg(a, w):
    if not (w and w[1] > w[0]):
        return np.nan
    s = a[w[0]:w[1]]; s = s[np.isfinite(s)]
    return float(np.max(s)) if len(s) else np.nan


def main():
    amq = load_automq()
    rows = []
    for p in COHORT:
        try:
            cdf = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        except Exception:
            continue
        for fk in cdf.index:
            tn, sd = int(fk[1]), fk[2]
            rec = amq.get((automq_part(p), tn, sd))
            if rec is None or rec.get("phases") is None:
                continue
            mk = cdf.loc[fk, "markers"]
            sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}")
            shL = _marker(mk, "shoulder_L"); shR = _marker(mk, "shoulder_R")
            hL = _marker(mk, "hip_L"); hR = _marker(mk, "hip_R")
            if any(x is None for x in (sh, el, shL, shR, hL, hR)):
                continue
            T = min(map(len, (sh, el, shL, shR, hL, hR)))
            sh, el, shL, shR, hL, hR = (a[:T] for a in (sh, el, shL, shR, hL, hR))
            arm = sh - el
            # AutoMQ world-frame abduction: project arm onto X-Z plane, angle vs Z
            arm_xz = arm.copy(); arm_xz[:, 1] = 0.0
            abd_world = _ang(arm_xz, Z[None, :])
            # BODY-referenced abduction: arm component along the shoulder-line (lateral). Sign via arcsin.
            side = (shR - shL) if sd == "R" else (shL - shR)     # acting-side OUTWARD
            side_u = side / (np.linalg.norm(side, axis=1, keepdims=True) + 1e-9)
            arm_u = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
            abd_body = np.degrees(np.abs(np.arcsin(np.clip((arm_u * side_u).sum(1), -1, 1))))
            # TORSO YAW: shoulder-line projected to horizontal (X-Y) plane, angle from world-X
            sl = shR - shL; sl_h = sl.copy(); sl_h[:, 2] = 0.0
            yaw = _ang(sl_h, Xw[None, :])
            up_trunk = (shL + shR) / 2.0 - (hL + hR) / 2.0
            lean = _ang(up_trunk, Z[None, :])
            rr = rec["phases"].get("Reaching"); dd = rec["phases"].get("Drinking")
            w = (int(rr[0]), int(dd[1])) if (rr is not None and dd is not None) else (0, T)
            s, e = max(0, w[0]), min(T, w[1])
            if e - s < 10:
                continue
            rows.append({"part": p, "trial": tn, "side": sd,
                         "abd_world": _maxseg(abd_world, (s, e)), "abd_body": _maxseg(abd_body, (s, e)),
                         "yaw": float(np.nanmedian(yaw[s:e])), "lean": float(np.nanmedian(lean[s:e]))})

    df = pd.DataFrame(rows).dropna(subset=["abd_world", "abd_body"])
    df["diff"] = df["abd_world"] - df["abd_body"]
    df.to_csv(PAPER / "automq_abduction_ref_test.csv", index=False)
    print(f"\nn = {len(df)} trials (OMC markers only, zero MMC)\n")
    print(f"abduction WORLD-frame (AutoMQ): median {df.abd_world.median():.1f}  IQR [{df.abd_world.quantile(.25):.1f},{df.abd_world.quantile(.75):.1f}]")
    print(f"abduction BODY-referenced     : median {df.abd_body.median():.1f}  IQR [{df.abd_body.quantile(.25):.1f},{df.abd_body.quantile(.75):.1f}]")
    print(f"definition difference          : median {df['diff'].median():+.1f}  |diff| med {df['diff'].abs().median():.1f}  max {df['diff'].abs().max():.1f}")
    print(f"torso YAW (shoulder-line from world-X): median {df.yaw.median():.1f} deg  IQR [{df.yaw.quantile(.25):.1f},{df.yaw.quantile(.75):.1f}]")
    print(f"trunk LEAN: median {df.lean.median():.1f} deg")
    for var in ["yaw", "lean"]:
        m = np.isfinite(df[var]) & np.isfinite(df['diff'])
        r = pearsonr(df[var][m], df['diff'][m])[0]
        print(f"  *** definition diff vs {var.upper():5}: pearson r = {r:+.2f} ***")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.get_cmap("tab10"); parts = sorted(df.part.unique()); pcol = {p: cmap(i % 10) for i, p in enumerate(parts)}
    for p in parts:
        g = df[df.part == p]; ax[0].scatter(g.yaw, g['diff'], s=16, alpha=0.6, color=pcol[p], label=p)
    ax[0].axhline(0, color="0.6", lw=0.8)
    r = pearsonr(df.yaw[np.isfinite(df.yaw)], df['diff'][np.isfinite(df.yaw)])[0]
    ax[0].set_xlabel("torso yaw (deg from world-X)"); ax[0].set_ylabel("abduction: world - body (deg)")
    ax[0].set_title(f"Abduction gap vs torso yaw (r={r:+.2f})"); ax[0].legend(fontsize=6, ncol=2, frameon=False)
    ax[1].scatter(df.abd_body, df.abd_world, s=16, alpha=0.5, color="#4c72b0")
    lo = min(df.abd_body.min(), df.abd_world.min()); hi = max(df.abd_body.max(), df.abd_world.max())
    ax[1].plot([lo, hi], [lo, hi], "0.55"); ax[1].set_aspect("equal", "box")
    ax[1].set_xlabel("abduction, body-referenced (deg)"); ax[1].set_ylabel("abduction, world-frame (deg)")
    ax[1].set_title("Two abduction definitions, OMC markers")
    fig.suptitle("AutoMQ abduction: world-frame vs body-referenced (both from OMC markers)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PAPER / "automq_abduction_ref_test.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {PAPER/'automq_abduction_ref_test.png'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
