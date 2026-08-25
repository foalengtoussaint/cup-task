"""Figure-4 ground truth with the anat12/w0 landmark correction applied to the OMC side.

IN-SAMPLE by request: theta is fitted per participant x arm on ALL of that arm's trials and applied
to those same trials -- so the figure shows the corrected relationship, not a prediction. The
split-half OUT-OF-SAMPLE r_s is computed alongside on exactly the same trials and printed, as the
evidence that the correction is a real effect and not curve-fitting.

Corrected here: the four ANGLE-family measures only (flexion, abduction, elbow extension, interjoint).
The wrist-derived panels are left on AutoMQ's stored truth -- under w=0 the wrist offset is the
unidentified axial direction (211mm, split-half noise 143mm), so applying it to velocity panels would
be indefensible.

Reductions match score_vs_automq / omc_matched_angles exactly, using the windows now in the v2 cache:
  flexion, abduction -> max over reaching-start..drinking-end      elbow -> max over reaching..returning
  interjoint         -> corr(flexion, elbow) over reaching

    python fig4_anat12.py    -> out/scoring/score_vs_automq_anat12w0.csv  (+ r_s table)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # anat12 modules live beside this file
import prep_cache2
from anat_frame import apply, NPAR
from score_vs_automq import _planar_body_angles, _elbow_series
MODEL = "anat12"
SCORER = ROOT/"out/scoring/score_vs_automq.csv"
OUT = ROOT/"out/scoring/score_vs_automq_anat12w0.csv"
MEAS = ["max_shoulder_flexion", "max_shoulder_abduction", "max_elbow_angle",
        "interjoint_coordination"]


def _corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 10 or np.std(a[k]) < 1e-9 or np.std(b[k]) < 1e-9: return np.nan
    return float(np.corrcoef(a[k], b[k])[0, 1])


def series(r, th):
    P = apply(r, th, MODEL)
    f, a = _planar_body_angles(P, r["side"], r["other"]); e = _elbow_series(P, r["side"])
    return f, a, e


def resid(recs, th):
    """Angles-only residual (w=0) over the reaching->drinking window, same as every earlier fit."""
    out = []
    for r in recs:
        f, a, e = series(r, th)
        s, t = int(r["rd"][0]), int(r["rd"][1])
        for x, y in ((f, r["flex_mmc"]), (a, r["abd_mmc"]), (e, r["elb_mmc"])):
            k = min(len(x), len(y), t)
            v = x[s:k] - y[s:k]; out.append(v[np.isfinite(v)])
    return np.concatenate(out) if out else np.zeros(1)


def measures(r, th):
    f, a, e = series(r, th)
    rs, re_ = int(r["reach"][0]), int(r["reach"][1])
    ds, de = int(r["rd"][0]), int(r["rd"][1])
    as_, ae = int(r["allw"][0]), int(r["allw"][1])
    return {"max_shoulder_flexion": float(np.nanmax(f[ds:de])),
            "max_shoulder_abduction": float(np.nanmax(a[ds:de])),
            "max_elbow_angle": float(np.nanmax(e[as_:ae])),
            "interjoint_coordination": _corr(f[rs:re_], e[rs:re_])}


if __name__ == "__main__":
    recs = prep_cache2.load_all()
    groups = {}
    for r in recs: groups.setdefault((r["part"], r["arm"]), []).append(r)
    npar = NPAR[MODEL]
    print(f"{len(recs)} trials, {len(groups)} participant x arm groups", flush=True)
    t0 = time.time(); rows = []
    for (p, a), rs in sorted(groups.items()):
        th_all = least_squares(lambda th: resid(rs, th), x0=np.zeros(npar), method="lm",
                               max_nfev=900).x
        ins = {r["trial"]: measures(r, th_all) for r in rs}
        oos = {}
        for parity in (0, 1):
            tr = [r for i, r in enumerate(rs) if i % 2 != parity]
            te = [r for i, r in enumerate(rs) if i % 2 == parity]
            if len(tr) < 4 or not te: continue
            th = least_squares(lambda th: resid(tr, th), x0=np.zeros(npar), method="lm",
                               max_nfev=900).x
            for r in te: oos[r["trial"]] = measures(r, th)
        for r in rs:
            raw = measures(r, np.zeros(npar))
            for m in MEAS:
                rows.append(dict(part=p, arm=a, trial=r["trial"], measure=m,
                                 omc_raw=raw[m], omc_ins=ins[r["trial"]][m],
                                 omc_oos=oos.get(r["trial"], {}).get(m, np.nan)))
        print(f"  [{time.time()-t0:5.0f}s] {p} {a}  {len(rs)} trials  |theta| "
              f"{np.linalg.norm(th_all):6.1f}mm", flush=True)

    D = pd.DataFrame(rows)
    sc = pd.read_csv(SCORER, keep_default_na=False, na_values=[""])
    sc = sc[sc.peak_metric.isin(["n/a", "max"])].copy()
    sc["mmc"] = pd.to_numeric(sc["mmc"], errors="coerce")
    sc["automq"] = pd.to_numeric(sc["automq"], errors="coerce")
    j = sc[sc.variant == "BA+smoothnet"].merge(D, on=["part", "trial", "measure"], how="inner")

    print(f"\nPROCESSING CHECK: cache trials {len(recs)}, rows {len(D)}, joined to scorer {len(j)}, "
          f"OOS missing {int(j.omc_oos.isna().sum())}, non-finite mmc {int(j.mmc.isna().sum())}")
    print(f"\n{'measure':26}{'n':>5}{'raw (AutoMQ)':>14}{'raw (ours)':>12}{'CORRECTED':>11}{'OOS':>9}")
    for m in MEAS:
        g = j[j.measure == m]
        ga = g.dropna(subset=["automq", "mmc"])
        f = lambda x, y: spearmanr(x, y).correlation if len(x) > 3 else np.nan
        go = g.dropna(subset=["omc_oos", "mmc"])
        print(f"{m:26}{len(g):>5}{f(ga.automq, ga.mmc):>14.3f}{f(g.omc_raw, g.mmc):>12.3f}"
              f"{f(g.omc_ins, g.mmc):>11.3f}{f(go.omc_oos, go.mmc):>9.3f}", flush=True)

    # scorer CSV with the corrected (in-sample) OMC swapped into the ground-truth column
    rep = D.set_index(["part", "trial", "measure"]).omc_ins
    out = pd.read_csv(SCORER, keep_default_na=False, na_values=[""])
    idx = pd.MultiIndex.from_frame(out[["part", "trial", "measure"]])
    new = rep.reindex(idx).values
    out["automq"] = np.where(np.isfinite(new), new, pd.to_numeric(out["automq"], errors="coerce"))
    out.to_csv(OUT, index=False)
    D.to_csv(ROOT/"out/scoring/anat12w0_omc_values.csv", index=False)
    print(f"\nwrote {OUT}  ({int(np.isfinite(new).sum())} ground-truth cells replaced)")
    print("DONE", flush=True)
