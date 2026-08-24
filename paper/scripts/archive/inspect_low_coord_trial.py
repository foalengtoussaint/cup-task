"""Look at ONE low-coordination trial: plot the actual flexion + elbow series frame-by-frame (OMC only).

Find the OMC trials with the LOWEST trunk-referenced interjoint, print them with IDs, and plot the chosen
one: shoulder flexion (world-vertical AND trunk-referenced) + elbow extension over the reach, so we can
SEE what 'low coordination' is doing. Sign convention matches shipped _murphy_signals:
  arm  = elbow - shoulder      (points down, shoulder->elbow)
  down = hip_mid - sh_mid      (points down, shoulder->hip)  -> flexion = angle(arm, down), 0=aligned
  world flexion = angle(arm, -Z) so it also grows as the arm raises (Z is up; -Z is down, matches 'down')
    python paper/scripts/inspect_low_coord_trial.py [--rank N]
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from compare_pose_omc_delta import _lp
from score_vs_automq import load_automq, automq_part, AUTOMQ
Zdown = np.array([0.0, 0.0, -1.0])       # DOWN in the lab (so world-flexion grows as arm raises, like trunk)


def _ang(u, v):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or np.std(a[m]) < 1e-6 or np.std(b[m]) < 1e-6:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def series(mk, sd, r0, r1):
    sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}"); wr = _marker(mk, f"hand_{sd}")
    shO = _marker(mk, "shoulder_L" if sd == "R" else "shoulder_R")
    hR = _marker(mk, "hip_R"); hL = _marker(mk, "hip_L")
    if any(x is None for x in (sh, el, wr, shO, hR, hL)):
        return None
    T = min(map(len, (sh, el, wr, shO, hR, hL))); r1 = min(r1, T)
    sh, el, wr, shO, hR, hL = (a[:T] for a in (sh, el, wr, shO, hR, hL))
    arm = el - sh                                   # shoulder->elbow (down-ish)
    flex_world = _lp(_ang(arm, Zdown[None, :]))     # vs lab DOWN
    flex_trunk = _lp(_ang(arm, (hR + hL) / 2 - (sh + shO) / 2))   # vs shoulder->hip (down)
    trunk_lean = _lp(_ang((sh + shO) / 2 - (hR + hL) / 2, -Zdown[None, :]))  # trunk tilt from vertical
    u, v = sh - el, wr - el
    elb = _lp(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9), -1, 1))))
    return dict(flex_world=flex_world, flex_trunk=flex_trunk, trunk_lean=trunk_lean, elbow=elb, r0=r0, r1=r1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=0); a = ap.parse_args()
    amq = load_automq()
    cand = []
    mkc = {}
    for p in ["P07", "P08", "P10", "P12", "P13", "P14", "P15", "P17", "P19", "P25"]:
        try:
            cdf = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        except Exception:
            continue
        mkc[p] = cdf
        for fk in cdf.index:
            tn, sd, cond = int(fk[1]), fk[2], fk[3]
            rec = amq.get((automq_part(p), tn, sd))
            if rec is None or rec.get("phases") is None or rec["phases"].get("Reaching") is None:
                continue
            rr = rec["phases"]["Reaching"]; r0, r1 = int(rr[0]), int(rr[1])
            if r1 - r0 < 15:
                continue
            s = series(cdf.loc[fk, "markers"], sd, r0, r1)
            if s is None:
                continue
            ij_t = _corr(s["flex_trunk"][r0:r1], s["elbow"][r0:r1])
            ij_w = _corr(s["flex_world"][r0:r1], s["elbow"][r0:r1])
            if np.isfinite(ij_t):
                cand.append((ij_t, ij_w, p, tn, sd, cond))
    cand.sort()
    print("LOWEST trunk-referenced interjoint trials (OMC):")
    print(f"{'ij_trunk':>9}{'ij_world':>9}  part trial side cond")
    for c in cand[:8]:
        print(f"{c[0]:>9.2f}{c[1]:>9.2f}  {c[2]} {c[3]} {c[4]} {c[5]}")
    ijt, ijw, p, tn, sd, cond = cand[a.rank]
    print(f"\nCHOSEN (rank {a.rank}): {p} trial_{tn}_{sd} ({cond})  ij_trunk={ijt:.2f}  ij_world={ijw:.2f}")

    cdf = mkc[p]
    fk = next(fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd)
    rec = amq.get((automq_part(p), tn, sd)); rr = rec["phases"]["Reaching"]
    s = series(cdf.loc[fk, "markers"], sd, int(rr[0]), int(rr[1]))
    r0, r1 = s["r0"], s["r1"]; t = np.arange(r0, r1) / 100.0

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(t, s["flex_trunk"][r0:r1], color="#c1440e", lw=1.8, label="shoulder flexion (trunk-ref)")
    ax[0].plot(t, s["flex_world"][r0:r1], color="#e0a200", lw=1.8, ls="--", label="shoulder flexion (world-vert)")
    ax[0].plot(t, s["trunk_lean"][r0:r1], color="#2a7", lw=1.3, ls=":", label="trunk lean (from vertical)")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("angle (deg)")
    ax[0].set_title("Flexion definitions + trunk lean over the reach"); ax[0].legend(fontsize=8, frameon=False)
    # coordination view: flexion vs elbow (the interjoint scatter within the trial)
    ax[1].plot(s["elbow"][r0:r1], s["flex_trunk"][r0:r1], color="#c1440e", lw=1.5, marker="o", ms=3, label=f"trunk-ref (r={ijt:.2f})")
    ax[1].plot(s["elbow"][r0:r1], s["flex_world"][r0:r1], color="#e0a200", lw=1.5, ls="--", marker="s", ms=3, label=f"world-vert (r={ijw:.2f})")
    ax[1].set_xlabel("elbow extension (deg)"); ax[1].set_ylabel("shoulder flexion (deg)")
    ax[1].set_title("Coordination: flexion vs elbow (interjoint = corr)"); ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Low-coordination trial: {p} trial_{tn}_{sd} ({cond})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PAPER / f"low_coord_trial_{p}_{tn}_{sd}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
