"""anat12 for TRUNK DISPLACEMENT: displace the OPTICAL sternum in the body frame.

Exactly parallel to anat12, which displaces the optical shoulder/elbow/wrist and fits those offsets
to the optical-vs-markerless MEASURE difference. Here the landmark is the sternum, the frame is the
trunk (down / fwd / lat, rebuilt per frame from the shoulders and hips so the offset rotates with the
torso), and the loss is the trunk-displacement difference:

    OMC trunk point = sternum + d . B_omc(t)        3 parameters, fitted
    MMC trunk point = shoulder midpoint             unchanged -- COCO has no sternum keypoint
    loss            = sum_t (excursion_omc(d) - excursion_mmc)^2

A world-fixed offset would do nothing: trunk displacement is a distance from the trial's own rest
position, so a constant offset cancels. The body-frame offset does not cancel, because it rotates
with the trunk and so changes the arc the point sweeps during a forward lean -- which is the actual
discrepancy between a sternum and a shoulder midpoint.

    raw          d = 0
    within_oos   d fitted on half a participant x arm's trials, applied to the other half
    lopo         d fitted on every OTHER participant, applied to the held-out one

    python paper/scripts/anat12/trunk_offset_fit.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import compare_pose_omc_delta as H                                   # noqa: E402
import results_v3_delta as R                                         # noqa: E402
import gnn_train as GT                                               # noqa: E402
from score_vs_automq import _pose_variant_cached                     # noqa: E402

SEG = ROOT / "cache" / "seg_inputs_ship"


def basis(pose):
    """Per-frame torso triad (down, fwd, lat); rows of B(t) are the axes."""
    nn = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    sh, shO = pose["right_shoulder"], pose["left_shoulder"]
    down = nn((pose["right_hip"] + pose["left_hip"]) / 2.0 - (sh + shO) / 2.0)
    fwd = nn(np.cross(nn(sh - shO), down))
    lat = nn(np.cross(down, fwd))
    return np.stack([down, fwd, lat], axis=1)


def excursion(pt):
    """Trunk displacement: 3D travel from the trial's own rest position (as paper_trajectories does)."""
    fin = np.isfinite(pt).all(1)
    if fin.sum() < 15:
        return None
    return np.linalg.norm(pt - np.median(pt[fin][:15], 0), axis=1)


def agree(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 20:
        return None
    x, y = a[k], b[k]
    bias = float(np.mean(x - y))
    return (float(np.sqrt(np.mean(((x - bias) - y) ** 2))),
            float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else np.nan,
            bias, float(np.nanmax(x) - np.nanmax(y)))


def build():
    """Per trial: the optical body-frame offset, plus everything needed to score a candidate d."""
    H.use_good_cams()
    ba = R._ba_traj_cache()
    lookup = {f"{t['part']}/{t['trial']}": t for t in GT.load_clean(need_reproj=False)}
    out, t0 = [], time.time()
    files = sorted(SEG.glob("*.npz"))
    for i, f in enumerate(files):
        z = np.load(f)
        part, trial = str(z["part"]), str(z["trial"])
        t = lookup.get(f"{part}/{trial}")
        if t is None:
            continue
        n, lag = int(z["n"]), int(z["lag"])
        try:
            p = H._load_omc(part, trial, n)
        except Exception:
            continue
        p = {j: R._shift(v, lag) for j, v in p.items()}
        st = p.get("trunk")
        if st is None:
            continue
        mm = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if mm is None:
            continue
        sh_o = (p["left_shoulder"] + p["right_shoulder"]) / 2.0
        B_o = basis(p)
        ok = np.isfinite(st).all(1) & np.isfinite(sh_o).all(1) & np.isfinite(B_o).all((1, 2))
        if ok.sum() < 50:
            continue
        d_body = np.einsum('tij,tj->ti', B_o[ok], (st - sh_o)[ok])
        try:
            sh_m = (mm["left_shoulder"] + mm["right_shoulder"]) / 2.0
            B_m = basis(mm)
        except KeyError:
            continue
        out.append(dict(part=part, trial=trial, arm=str(z["arm"]),
                        d=np.median(d_body, axis=0),
                        st=st, sh_o=sh_o, B_o=B_o, sh_m=sh_m, B_m=B_m))
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:4.0f}s  kept {len(out)}", flush=True)
    return out


def apply_d(sh, B, d):
    """shoulder midpoint + d expressed in the per-frame torso triad."""
    return sh + np.einsum('tij,i->tj', B, np.asarray(d, float))


def fit_d(recs):
    """d displacing the OPTICAL sternum, fitted to the trunk-displacement difference (anat12's rule)."""
    pre = []
    for r in recs:
        e_m = excursion(r["sh_m"])          # markerless side, never modified
        if e_m is not None:
            pre.append((r["st"], r["B_o"], e_m))
    if not pre:
        return np.zeros(3)

    def resid(d):
        out = []
        for st, B, e_m in pre:
            e = excursion(apply_d(st, B, d))
            if e is None:
                continue
            k = np.isfinite(e) & np.isfinite(e_m)
            if k.sum() >= 20:
                out.append(e[k] - e_m[k])
        return np.concatenate(out) if out else np.zeros(1)

    return least_squares(resid, x0=np.zeros(3), method="lm", max_nfev=300).x


def score(recs, dmap, cond):
    rows = []
    for r in recs:
        d = dmap.get((r["part"], r["arm"])) if dmap else None
        rec_err = float(np.linalg.norm(d)) if d is not None else np.nan   # |d|, mm
        # markerless side unchanged; the OPTICAL sternum is displaced by d in its body frame
        a = excursion(r["sh_m"])
        b = excursion(r["st"] if d is None else apply_d(r["st"], r["B_o"], d))
        if a is None or b is None:
            continue
        g = agree(a, b)
        if g is None:
            continue
        rmse, corr, bias, peak_d = g
        # the SCALAR the measure tables report: 98th pct of the excursion, one value per trial,
        # correlated ACROSS trials (angle_measures_automq's max_trunk_displacement). The per-frame
        # r above is Table I's quantity and is a much easier question.
        rows.append(dict(cond=cond, part=r["part"], arm=r["arm"], trial=r["trial"],
                         rec_err=rec_err, rmse=rmse, r=corr, bias=bias, peak_diff=peak_d,
                         scal_mmc=float(np.nanpercentile(a[np.isfinite(a)], 98)),
                         scal_omc=float(np.nanpercentile(b[np.isfinite(b)], 98))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "out/scoring/trunk_offset_fit.csv"))
    a = ap.parse_args(argv)

    recs = build()
    groups = {}
    for r in recs:
        groups.setdefault((r["part"], r["arm"]), []).append(r)
    parts = sorted({p for p, _ in groups})
    print(f"\n{len(recs)} trials, {len(groups)} groups, {len(parts)} participants\n", flush=True)

    fit = fit_d
    rows = score(recs, None, "raw")

    # within participant x arm, split-half
    for key, rs in sorted(groups.items()):
        for parity in (0, 1):
            tr = [r for i, r in enumerate(rs) if i % 2 != parity]
            te = [r for i, r in enumerate(rs) if i % 2 == parity]
            if len(tr) < 4 or not te:
                continue
            rows += score(te, {key: fit(tr)}, "within_oos")

    # leave one participant out. The training fold is strided down per group: the fit has 3
    # parameters and the residual walks every training trial, so 700+ trials buys nothing.
    def cap(rs, n=8):
        if len(rs) <= n:
            return rs
        idx = np.linspace(0, len(rs) - 1, n).round().astype(int)
        return [rs[i] for i in sorted(set(idx.tolist()))]

    for p in parts:
        tr = [r for (q, arm), rs in sorted(groups.items()) if q != p for r in cap(rs)]
        te = [r for r in recs if r["part"] == p]
        d = fit(tr)
        rows += score(te, {(q, arm): d for (q, arm) in groups}, "lopo")

    D = pd.DataFrame(rows)
    D.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(D)} rows)")
    print(f"PROCESSING CHECK: per condition " + ", ".join(f"{c} {len(g)}" for c, g in D.groupby("cond")))

    print(f"\n{'condition':12}{'|d| (mm)':>15}{'trunk RMSE':>12}{'r':>8}"
          f"{'bias':>8}{'peak diff':>11}{'n':>6}")
    print(f"{'':12}{'':>15}{'(mm)':>12}{'':>8}{'(mm)':>8}{'(mm)':>11}")
    for c in ("raw", "within_oos", "lopo"):
        g = D[D.cond == c]
        if not len(g):
            continue
        rec = f"{g.rec_err.median():.1f}" if g.rec_err.notna().any() else "--"
        print(f"{c:12}{rec:>15}{g.rmse.median():>12.2f}{g.r.median():>8.3f}"
              f"{g.bias.median():>8.2f}{g.peak_diff.median():>11.2f}{len(g):>6}")
    print(f"\n{'condition':12}{'SCALAR max_trunk_displacement (Table III quantity)':>52}")
    print(f"{'':12}{'r across trials':>20}{'r_av':>10}{'med MMC':>10}{'med OMC':>10}")
    from scipy.stats import pearsonr as _pr
    for c in ("raw", "within_oos", "lopo"):
        g = D[D.cond == c].dropna(subset=["scal_mmc", "scal_omc"])
        if len(g) < 5:
            continue
        av = g.groupby(["part", "arm"])[["scal_mmc", "scal_omc"]].mean()
        rav = _pr(av.scal_mmc, av.scal_omc)[0] if len(av) >= 3 else float("nan")
        print(f"{c:12}{_pr(g.scal_mmc, g.scal_omc)[0]:>20.3f}{rav:>10.3f}"
              f"{g.scal_mmc.median():>10.1f}{g.scal_omc.median():>10.1f}")
    print("\n  peak diff = markerless peak excursion minus optical; the ~14% arc-length gap the "
          "shoulder\n  midpoint carries should shrink toward 0 if the body-frame offset works.")
    print("DONE_TRUNK_OFFSET", flush=True)


if __name__ == "__main__":
    main()
