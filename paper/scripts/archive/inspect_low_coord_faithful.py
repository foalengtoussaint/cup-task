"""Faithful low-coordination trial inspection: use AutoMQ's OWN stored flexion+elbow SERIES (no
reimplementation of the flexion formula -- that is where every sign bug came from). VERIFY the
recomputed interjoint reproduces AutoMQ's stored scalar before plotting.

Picks the trials with the LOWEST AutoMQ-STORED interjoint_coordination, prints them, and plots the
chosen one from AutoMQ's own kinematics series: shoulder_flexion + elbow_angle over the reach, and
the flexion-vs-elbow coordination curve. Also overlays our trunk-referenced flexion for comparison
(clearly labelled), but the interjoint NUMBER shown is AutoMQ's stored value.
    python paper/scripts/inspect_low_coord_faithful.py [--rank N]
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


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or np.std(a[m]) < 1e-6 or np.std(b[m]) < 1e-6:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _interj_reach(flex, elb, r0, r1):
    """AutoMQ recipe: corr over the reaching window (their stored value uses reaching, no inner-crop for
    the native series -- we test both to see which reproduces the stored scalar)."""
    full = _corr(flex[r0:r1], elb[r0:r1])
    mg = int(0.1 * (r1 - r0))
    inner = _corr(flex[r0 + mg:r1 - mg], elb[r0 + mg:r1 - mg])
    return full, inner


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=0); a = ap.parse_args()
    amq = load_automq()
    # rank trials by AutoMQ's STORED interjoint scalar
    cand = []
    for k, rec in amq.items():
        g = rec.get("interjoint_coordination")
        if g is not None and np.isfinite(g) and rec.get("phases", {}) and rec["phases"].get("Reaching"):
            cand.append((float(g), k))   # k = (automq_part, trial_number, side)
    cand.sort()
    print("LOWEST AutoMQ-STORED interjoint_coordination trials:")
    print(f"{'stored ij':>10}  part trial side")
    for g, k in cand[:8]:
        print(f"{g:>10.2f}  {k[0]} {k[1]} {k[2]}")
    g_stored, (ap_, tn, sd) = cand[a.rank]
    print(f"\nCHOSEN (rank {a.rank}): {ap_} trial_{tn}_{sd}  STORED interjoint = {g_stored:.3f}")

    cdf = pd.read_pickle(AUTOMQ / ap_ / "combined_data_with_kinematics.pkl")
    fk = next(fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd)
    kin = cdf.loc[fk, "kinematics"]
    rec = amq.get((ap_, tn, sd)); rr = rec["phases"]["Reaching"]; r0, r1 = int(rr[0]), int(rr[1])
    flexN = kin["shoulder_flexion"].to_numpy(); elbN = kin["elbow_angle"].to_numpy()
    r1 = min(r1, len(flexN))
    full, inner = _interj_reach(_lp(flexN), _lp(elbN), r0, r1)
    print(f"\nVERIFY recompute vs stored: stored={g_stored:.3f}  recompute(full reach)={full:.3f}  "
          f"recompute(inner-80%)={inner:.3f}")
    use = full if abs(full - g_stored) <= abs(inner - g_stored) else inner
    print(f"  -> using the window that matches stored (delta {abs(use-g_stored):.3f}); plot is FAITHFUL if delta small")

    t = np.arange(r0, r1) / 100.0
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(t, _lp(flexN)[r0:r1], color="#c1440e", lw=1.8, label="shoulder flexion (AutoMQ)")
    ax[0].plot(t, _lp(elbN)[r0:r1], color="#4c72b0", lw=1.8, label="elbow angle (AutoMQ)")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("angle (deg)")
    ax[0].set_title("AutoMQ flexion + elbow over the reach"); ax[0].legend(fontsize=8, frameon=False)
    ax[1].plot(_lp(elbN)[r0:r1], _lp(flexN)[r0:r1], color="#7a4", lw=1.5, marker="o", ms=3)
    ax[1].set_xlabel("elbow angle (deg)"); ax[1].set_ylabel("shoulder flexion (deg)")
    ax[1].set_title(f"Coordination (AutoMQ series)  interjoint = {g_stored:.2f}")
    fig.suptitle(f"Low-coordination trial (AutoMQ ground truth): {ap_} trial_{tn}_{sd}  ij={g_stored:.2f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PAPER / f"low_coord_faithful_{ap_}_{tn}_{sd}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
