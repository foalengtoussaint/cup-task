"""anat12 landmark offsets: a CONSTANT definition difference, or session-to-session variance?

THE QUESTION. A skin marker and a detected keypoint label different points on the body, so the
`anat12` fit (shoulder TRUNK+SEG, elbow SEG, wrist SEG; 12 params) removes a constant offset before
the angles are compared. Two very different things could be producing that offset:

  (a) a COHORT-WIDE definition difference -- the acromion marker vs the COCO shoulder keypoint sit in
      the same relative place on everyone. Then one offset, fitted once, transfers to a new
      participant, and a deployed system can bake it in.
  (b) SESSION-TO-SESSION variance -- YOLO placing the keypoint differently per participant, per
      calibration, per camera geometry, plus marker-placement variation by the operator. Then the fit
      is a per-session calibration and does NOT transfer, and the paper cannot claim it does.

Four conditions separate them. All four score the SAME measures the same way; only theta changes:

  raw          theta = 0.                                    no correction
  within_in    theta fitted on the participant x arm it is applied to.   CIRCULAR upper bound
  within_oos   split-half inside a participant x arm: theta fitted on one half, applied to the other.
               Honest WITHIN a session -- this is the 0.007 held-out check the paper cites.
  lopo         LEAVE ONE PARTICIPANT OUT: theta fitted on every OTHER participant's trials, applied
               to the held-out participant, who contributed nothing to it. Honest ACROSS sessions.

Read the three gaps:
    raw -> lopo          what a single cohort-wide offset buys.  This is (a).
    lopo -> within_oos   what per-session refitting buys ON TOP.  This is (b).
    within_oos -> within_in   overfit; the distance between the honest and circular numbers.

If lopo lands near within_oos the offset is a transferable constant. If it lands near raw it is a
per-session calibration and Sec. II-F's transferability caveat has to stay.

Also written out: the fitted theta per participant x arm (out/scoring/anat12_lopo_theta.csv), whose
across-group MEAN is the constant part and whose across-group SD is the session-to-session part, in
millimetres, per landmark block.

THE LOSS. Three residual blocks, each RMS-normalised at theta=0 so a block's scale does not decide
its influence, then weighted:
    angles   flexion, abduction and elbow-extension series          (always, weight 1)
    speed    wrist speed series, the signal PEAK VELOCITY reduces   (--wv)
    angvel   elbow angular-velocity series, which PEAK ELBOW        (--wa)
             ANGULAR VELOCITY reduces
An angles-only fit (--wv 0 --wa 0) is free to move the wrist landmark anywhere that helps an angle,
and wrecks the two derivative measures -- peak wrist velocity fell 0.758 -> 0.002 held-out. Putting
those two signals in the loss is what stops that, at whatever cost to the angles the trade implies.

Reads cache/omc_prep2 (all 11 participants, full series + window indices, lag from find_lag_best).
Build it first: python paper/scripts/anat12/prep_cache2.py build

    python paper/scripts/anat12/anat12_lopo.py [--cap 8] [--out out/scoring/anat12_lopo.csv]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prep_cache2                                                      # noqa: E402
import anat_frame as AF                                                 # noqa: E402
from joint_fit import speed_series                                      # noqa: E402

MODEL = "anat12"
NPAR = AF.NPAR[MODEL]
FPS = AF.FPS
JN = prep_cache2.JN
KEYS = AF.KEYS
NAMES = AF.NAMES
# the 12 params in `apply` order, grouped by landmark block
BLOCKS = [("shoulder_trunk", 0, 3), ("shoulder_seg", 3, 6),
          ("elbow_seg", 6, 9), ("wrist_seg", 9, 12)]


def as_v1(d):
    """v2 cache stores FULL series + window indices; anat_frame expects the reach->drink slice."""
    s, e = (int(x) for x in d["rd"])
    rs, re_ = (int(x) for x in d["reach"])
    out = {k: d[k] for k in ("B", "side", "other", "arm", "part", "trial")}
    out["n_reach"] = max(re_ - rs, 2)
    for k in ("flex_mmc", "abd_mmc", "elb_mmc", "mmc_wrist"):
        out[k] = d[k][s:e]
    for j in JN:
        out[f"omc_{j}"] = d[f"omc_{j}"][s:e]
    return out


def blocks3(r, th):
    """(angle, wrist-speed, elbow-angular-velocity) residuals for one trial.

    Same first two blocks as anat_frame.blocks; the third is new. Elbow angular velocity is the frame
    difference of the elbow-angle series -- exactly what anat_frame._meas reduces for `pav` -- so the
    loss and the measure read the same signal. Written out rather than wrapping anat_frame.blocks so
    that `apply` (the expensive part, and the inner loop of every Jacobian column) runs once."""
    P = AF.apply(r, th, MODEL)
    side = r["side"]
    fl, ab = AF._planar_body_angles(P, side, r["other"])
    el = AF._elbow_series(P, side)
    k = min(len(fl), len(r["flex_mmc"]))
    ang = []
    for x, y in ((fl[:k], r["flex_mmc"][:k]), (ab[:k], r["abd_mmc"][:k]),
                 (el[:k], r["elb_mmc"][:k])):
        v = x - y
        ang.append(v[np.isfinite(v)])
    so, sm = speed_series(P[f"{side}_wrist"]), r.get("_sm_cached")
    vel = np.zeros(0)
    if so is not None and sm is not None:
        kk = min(len(so), len(sm))
        dv = so[:kk] - sm[:kk]
        vel = dv[np.isfinite(dv)]
    da = (np.diff(el[:k]) - np.diff(r["elb_mmc"][:k])) * FPS
    return (np.concatenate(ang) if ang else np.zeros(0)), vel, da[np.isfinite(da)]


def _scales(recs):
    """RMS of each block at theta = 0, so the three enter the loss on comparable footing."""
    acc = [[], [], []]
    for r in recs:
        for i, b in enumerate(blocks3(r, np.zeros(NPAR))):
            acc[i].append(b)
    out = []
    for c in acc:
        v = np.concatenate(c) if c else np.zeros(1)
        out.append(float(np.sqrt(np.mean(v ** 2))) or 1.0)
    return out


def fit(recs, wv, wa):
    """theta for `recs`. wv = wa = 0 reproduces the angles-only anat_frame fit."""
    sA, sV, sW = _scales(recs)

    def resid(th):
        A, V, W = [], [], []
        for r in recs:
            a, v, w = blocks3(r, th)
            A.append(a); V.append(v); W.append(w)
        parts = [np.concatenate(A) / sA if A else np.zeros(1)]
        if wv > 0:
            parts.append(wv * np.concatenate(V) / sV)
        if wa > 0:
            parts.append(wa * np.concatenate(W) / sW)
        return np.concatenate(parts)

    return least_squares(resid, x0=np.zeros(NPAR), method="lm", max_nfev=900).x


def rows_for(recs, th, cond):
    out = []
    for r in recs:
        o = AF.measures_omc(r, th, MODEL)
        m = AF.measures_mmc(r)
        out.append({"cond": cond, "part": r["part"], "arm": r["arm"], "trial": r["trial"],
                    **{f"o_{k}": o[k] for k in KEYS}, **{f"m_{k}": m[k] for k in KEYS}})
    return out


def stride_cap(recs, cap):
    """Evenly strided subsample -- keeps the spread of trials, bounds the fit cost."""
    if cap <= 0 or len(recs) <= cap:
        return recs
    idx = np.linspace(0, len(recs) - 1, cap).round().astype(int)
    return [recs[i] for i in sorted(set(idx.tolist()))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=8,
                    help="max training trials per participant x arm inside a LOPO fold (0 = all)")
    ap.add_argument("--no-lopo", action="store_true",
                    help="skip the leave-one-participant-out folds; report raw + within only")
    ap.add_argument("--wv", type=float, default=1.0,
                    help="weight on the wrist-speed block (0 = the old angles-only fit)")
    ap.add_argument("--wa", type=float, default=1.0,
                    help="weight on the elbow-angular-velocity block")
    ap.add_argument("--out", default=str(ROOT / "out/scoring/anat12_lopo.csv"))
    a = ap.parse_args(argv)

    recs = [as_v1(d) for d in prep_cache2.load_all()]
    for r in recs:
        r["_sm_cached"] = speed_series(r["mmc_wrist"])
    groups: dict[tuple[str, str], list] = {}
    for r in recs:
        groups.setdefault((r["part"], r["arm"]), []).append(r)
    parts = sorted({p for p, _ in groups})
    print(f"{len(recs)} trials, {len(groups)} participant x arm groups, {len(parts)} participants; "
          f"loss = angles + {a.wv:g}*wrist-speed + {a.wa:g}*elbow-angular-velocity", flush=True)

    t0 = time.time()
    out, theta_rows = [], []

    # ---- raw ---------------------------------------------------------------
    out += rows_for(recs, np.zeros(NPAR), "raw")

    # ---- within-participant: in-sample and split-half -----------------------
    for (p, arm), rs in sorted(groups.items()):
        th_full = fit(rs, a.wv, a.wa)
        out += rows_for(rs, th_full, "within_in")
        theta_rows.append(dict(part=p, arm=arm, n=len(rs),
                               **{f"th{i}": float(v) for i, v in enumerate(th_full)}))
        for parity in (0, 1):
            tr = [r for i, r in enumerate(rs) if i % 2 != parity]
            te = [r for i, r in enumerate(rs) if i % 2 == parity]
            if len(tr) < 4 or not te:
                continue
            out += rows_for(te, fit(tr, a.wv, a.wa), "within_oos")
        print(f"  [{time.time()-t0:5.0f}s] within  {p} {arm}  ({len(rs)} trials)", flush=True)

    # ---- leave one participant out -----------------------------------------
    for p in ([] if a.no_lopo else parts):
        tr = [r for (q, arm), rs in sorted(groups.items()) if q != p
              for r in stride_cap(rs, a.cap)]
        te = [r for (q, _), rs in groups.items() if q == p for r in rs]
        th = fit(tr, a.wv, a.wa)
        out += rows_for(te, th, "lopo")
        theta_rows.append(dict(part=f"LOPO_not_{p}", arm="pooled", n=len(tr),
                               **{f"th{i}": float(v) for i, v in enumerate(th)}))
        print(f"  [{time.time()-t0:5.0f}s] lopo    hold out {p}: fit on {len(tr)} trials "
              f"from {len(parts)-1} participants, applied to {len(te)}", flush=True)

    d = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    T = pd.DataFrame(theta_rows)
    tout = Path(a.out).with_name(Path(a.out).stem + "_theta.csv")
    T.to_csv(tout, index=False)
    print(f"\nwrote {a.out} ({len(d)} rows) and {tout}", flush=True)

    n_raw = d[d.cond == "raw"].trial.nunique()
    print(f"PROCESSING CHECK: {n_raw} trials scored raw; per condition "
          + ", ".join(f"{c} {len(g)}" for c, g in d.groupby('cond')), flush=True)

    # ---- the table ---------------------------------------------------------
    conds = [c for c in ("raw", "lopo", "within_oos", "within_in") if (d.cond == c).any()]
    print(f"\nAgreement (Pearson r over single trials), MMC vs landmark-matched OMC:\n")
    hdr = (f"{'raw->lopo':>11}{'lopo->oos':>11}" if "lopo" in conds else f"{'raw->oos':>11}")
    print(f"{'measure':22}" + "".join(f"{c:>13}" for c in conds) + hdr)
    for k in KEYS:
        cells = {}
        for c in conds:
            g = d[d.cond == c].dropna(subset=[f"o_{k}", f"m_{k}"])
            cells[c] = pearsonr(g[f"o_{k}"], g[f"m_{k}"])[0] if len(g) > 3 else np.nan
        line = f"{NAMES[k]:22}" + "".join(f"{cells[c]:>13.3f}" for c in conds)
        if "lopo" in cells:
            line += f"{cells['lopo']-cells['raw']:>11.3f}{cells['within_oos']-cells['lopo']:>11.3f}"
        else:
            line += f"{cells['within_oos']-cells['raw']:>11.3f}"
        print(line)

    print(f"\nSame, Spearman (comparable with the earlier anat_frame logs):\n")
    print(f"{'measure':22}" + "".join(f"{c:>13}" for c in conds))
    for k in KEYS:
        cells = []
        for c in conds:
            g = d[d.cond == c].dropna(subset=[f"o_{k}", f"m_{k}"])
            cells.append(spearmanr(g[f"o_{k}"], g[f"m_{k}"]).correlation if len(g) > 3 else np.nan)
        print(f"{NAMES[k]:22}" + "".join(f"{c:>13.3f}" for c in cells))

    # ---- constant vs session-to-session, in millimetres ---------------------
    W_ = T[~T.part.str.startswith("LOPO_")]
    print("\nFitted offsets per landmark block (mm), across the "
          f"{len(W_)} participant x arm fits:")
    print(f"  {'block':16}{'|mean|':>9}{'mean SD':>9}{'|mean|/SD':>11}   "
          f"the constant part vs the session-to-session part")
    for nm, i, j in BLOCKS:
        V = W_[[f"th{k}" for k in range(i, j)]].values
        mu = np.linalg.norm(V.mean(axis=0))
        sd = float(np.mean(V.std(axis=0, ddof=1)))
        print(f"  {nm:16}{mu:>9.1f}{sd:>9.1f}{mu/(sd+1e-9):>11.2f}")
    print("\n  |mean| large and SD small -> a cohort-wide landmark definition difference.")
    print("  SD comparable to |mean| -> the offset is per-session and does not transfer.")
    print("DONE_ANAT12_LOPO", flush=True)


if __name__ == "__main__":
    main()
