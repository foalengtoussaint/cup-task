"""Which abduction definition does our MMC match: AutoMQ's WORLD-frame, or a BODY-referenced one?

Parallel to mmc_vs_flexion_defs.py. AutoMQ abduction = angle([arm_x,0,arm_z], world-Z) -- assumes world
axes = anatomical. A body-referenced abduction = arm's component along the SHOULDER LINE (lateral),
which is what OUR MMC _murphy_signals already computes. Per matched trial (BA+SmoothNet, reach->drink max):
  mmc_abd    : OUR shipped abduction (arcsin(arm . shoulder-line), from OUR pose)
  omc_world  : arm projected to X-Z plane vs Z, from OMC markers          [AutoMQ shipped]
  omc_body   : arcsin(arm . shoulder-line), from OMC markers              [body-referenced]
Reports rs/bias/med|err| of MMC vs each. If MMC ~ OMC_body, our pipeline computes the body-referenced
angle and the gap to AutoMQ's shipped number is definitional. OMC read-only; scoring uses our pose only.
    python paper/scripts/mmc_vs_abduction_defs.py -> paper/mmc_vs_abduction_defs.png + prints
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
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from compare_pose_omc_delta import _murphy_signals
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, AUTOMQ)
GRID = R._GRID_JOINTS
Z = np.array([0.0, 0.0, 1.0])


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def _ang(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _abd_body(arm, side):
    """arcsin(arm . side_unit) magnitude -- component of the arm along the lateral shoulder line."""
    su = side / (np.linalg.norm(side, axis=1, keepdims=True) + 1e-9)
    au = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    return np.degrees(np.abs(np.arcsin(np.clip((au * su).sum(1), -1, 1))))


def _maxseg(a, w):
    if not (w and w[1] > w[0]):
        return np.nan
    s = a[w[0]:w[1]]; s = s[np.isfinite(s)]
    return float(np.max(s)) if len(s) else np.nan


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    mk_cache = {}
    rows = []
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        p = t["part"]; tn = int(m.group(1)); sd = m.group(2)
        rec = amq.get((automq_part(p), tn, sd))
        if rec is None or rec.get("phases") is None:
            continue
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; n = t["mmc"].shape[0]
        try:
            sig = _murphy_signals(pose, side=side)
            mmc_abd_series = H._lp(sig["shoulder_abduction"])
        except (IndexError, ValueError):
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], H._load_omc(p, t["trial"], n)[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd_v = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        mmc_abd = _maxseg(np.abs(mmc_abd_series), rd_v)          # our shipped abduction (abs, body-ref)

        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        if key is None:
            continue
        mk = cdf.loc[key, "markers"]
        sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}")
        shL = _marker(mk, "shoulder_L"); shR = _marker(mk, "shoulder_R")
        if any(x is None for x in (sh, el, shL, shR)):
            continue
        T = min(map(len, (sh, el, shL, shR)))
        sh, el, shL, shR = (a[:T] for a in (sh, el, shL, shR))
        arm = sh - el
        arm_xz = arm.copy(); arm_xz[:, 1] = 0.0
        omc_world_s = _ang(arm_xz, Z[None, :])
        side_v = (shR - shL) if sd == "R" else (shL - shR)      # acting-side OUTWARD
        omc_body_s = _abd_body(arm, side_v)
        rr = rec["phases"].get("Reaching"); dd = rec["phases"].get("Drinking")
        w100 = (int(rr[0]), int(dd[1])) if (rr is not None and dd is not None) else (0, T)
        rows.append({"part": p, "trial": tn, "side": sd, "mmc_abd": mmc_abd,
                     "omc_world": _maxseg(omc_world_s, w100), "omc_body": _maxseg(omc_body_s, w100)})

    df = pd.DataFrame(rows).dropna(subset=["mmc_abd", "omc_world", "omc_body"])
    df.to_csv(PAPER / "mmc_vs_abduction_defs.csv", index=False)
    print(f"\nn = {len(df)} matched trials\n")

    def agree(a, b):
        mm = np.isfinite(a) & np.isfinite(b)
        return (spearmanr(a[mm], b[mm]).correlation, float(np.median(b[mm] - a[mm])), float(np.median(np.abs(b[mm] - a[mm]))))
    print(f"{'MMC abduction vs ...':30}{'rs':>7}{'bias':>8}{'med|err|':>10}")
    for lab, col in [("OMC WORLD-frame (AutoMQ)", "omc_world"), ("OMC BODY-referenced", "omc_body")]:
        rs, bias, err = agree(df[col].values, df["mmc_abd"].values)
        print(f"{lab:30}{rs:>+7.2f}{bias:>+8.1f}{err:>10.1f}")
    print("\n(if MMC agrees BETTER with BODY-referenced, our shoulder-line abduction computes the body-relative")
    print(" angle and the gap to AutoMQ's world-frame number is definitional, like flexion)")
    # per-participant bias vs OMC-body (is the residual a consistent per-participant offset?)
    print(f"\n{'part':6}{'n':>4}{'bias(MMC-OMCbody)':>19}{'IQR':>9}")
    for pp in sorted(df.part.unique()):
        g = df[df.part == pp]; b = g["mmc_abd"] - g["omc_body"]
        print(f"{pp:6}{len(g):>4}{b.median():>+19.1f}{b.quantile(.75)-b.quantile(.25):>9.1f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, col, ttl in [(ax[0], "omc_world", "MMC vs OMC world-frame"), (ax[1], "omc_body", "MMC vs OMC body-referenced")]:
        a.scatter(df[col], df["mmc_abd"], s=14, alpha=0.5, color="#4c72b0")
        lo, hi = 0, 90; a.plot([lo, hi], [lo, hi], "0.55"); a.set_aspect("equal", "box")
        rs = spearmanr(df[col], df["mmc_abd"]).correlation
        a.set_xlabel(f"OMC abduction [{col.split('_')[1]}] (deg)"); a.set_ylabel("MMC abduction (deg)")
        a.set_title(f"{ttl}\n$r_s$={rs:.2f}"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    fig.suptitle("Which abduction definition does MMC match?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PAPER / "mmc_vs_abduction_defs.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {PAPER/'mmc_vs_abduction_defs.png'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
