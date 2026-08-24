"""Leave-one-signal-out evaluation of the three lag estimators.

The three estimators disagree, and each one's own objective is the thing it maximises -- so
scoring them on that objective is circular. This holds ONE signal out, fits every estimator on
the remaining signals only, then scores each by the HELD-OUT signal's correlation at the lag it
chose. A lag that is right for the trial should align a signal the estimator never saw.

  wrist   : H._find_lag on wrist speed alone (the estimator gnn_build_dataset actually applies)
  multi   : argmax over the per-signal best lags (H._find_lag_multi's rule), refit per fold
  stacked : Fisher-z stack of the full correlation curves (lag_stacked's rule), refit per fold

Curves are computed ONCE per trial per signal and reused across folds -- the folds are pure
arithmetic on them.

    python scripts/lag_loso.py                    -> out/scoring/lag_loso.{csv,npz}
    tail -f out/scoring/lag_loso.log
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H      # noqa: E402
import gnn_train as GT                  # noqa: E402
import lag_stacked as LS                # noqa: E402

MAX_LAG = 180
MIN_PEAK = 0.3


def _signals(t, mcup=None, ocup=None):
    """The candidate (name, mmc_sig, omc_sig) list -- same set _find_lag_multi uses."""
    side = t["side"]
    J = list(H.JOINTS)
    mmc = {j: t["mmc"][:, k] for k, j in enumerate(J)}
    omc = {j: t["omc"][:, k] for k, j in enumerate(J)}
    out = []
    for j in ("wrist", "elbow", "shoulder"):
        jn = f"{side}_{j}"
        out.append((f"{j}_speed", H._speed(mmc[jn]), H._speed(omc[jn])))
        out.append((f"{j}_disp", H._disp_from_start(mmc[jn]), H._disp_from_start(omc[jn])))
    if mcup is not None and ocup is not None:
        out.append(("cup_speed", H._speed(mcup), H._speed(ocup)))
        out.append(("cup_disp", H._disp_from_start(mcup), H._disp_from_start(ocup)))
    return out


def _fit(curves, names, lags):
    """(multi_lag, stacked_lag) from the given subset of correlation curves."""
    best_c, best_lag = -np.inf, 0
    Z, W = None, 0.0
    for n in names:
        c = curves[n]
        if not np.isfinite(c).any():
            continue
        pk = np.nanmax(c)
        if pk > best_c:
            best_c, best_lag = pk, int(lags[int(np.nanargmax(c))])
        if pk < MIN_PEAK:
            continue
        z = np.arctanh(np.clip(np.nan_to_num(c, nan=0.0), -0.999, 0.999))
        w = max(pk, 0.0) ** 2
        Z = z * w if Z is None else Z + z * w
        W += w
    st = int(lags[int(np.argmax(Z / max(W, 1e-9)))]) if Z is not None else 0
    return best_lag, st


def main():
    H.use_good_cams()
    import results_v3_delta as R
    trials = GT.load_clean(need_reproj=False)
    N = len(trials)
    print(f"lag_loso: {N} trials, max_lag={MAX_LAG}", flush=True)
    rows = []
    t0 = time.time()
    for i, t in enumerate(trials):
        n = t["mmc"].shape[0]
        try:
            calib = R._calib(t["part"])
            mcup = R._cup_v3(t["part"], t["trial"], calib, n)
            ocup = R._omc_cup(t["part"], t["trial"], n)
            if not np.isfinite(mcup).any() or not np.isfinite(ocup).any():
                mcup = ocup = None
        except Exception:
            mcup = ocup = None
        sigs = _signals(t, mcup, ocup)
        lags, curves = None, {}
        for name, a, b in sigs:
            lg, c = LS._curve(a, b, MAX_LAG)
            lags = lg
            curves[name] = c
        names = [nm for nm in curves if np.isfinite(curves[nm]).any()]
        if len(names) < 3:
            continue
        # the shipped estimator: wrist speed alone, never refit
        wl = int(lags[int(np.nanargmax(curves["wrist_speed"]))]) if "wrist_speed" in curves else 0
        for held in names:
            rest = [nm for nm in names if nm != held]
            ml, sl = _fit(curves, rest, lags)
            ch = curves[held]
            def at(L):
                k = int(np.searchsorted(lags, L))
                return float(ch[k]) if 0 <= k < len(ch) else np.nan
            rows.append(dict(part=t["part"], trial=t["trial"], held=held,
                             n_sig=len(names), best_held=float(np.nanmax(ch)),
                             lag_wrist=wl, lag_multi=ml, lag_stacked=sl,
                             r_wrist=at(wl), r_multi=at(ml), r_stacked=at(sl)))
        if (i + 1) % 20 == 0 or (i + 1) == N:
            d = pd.DataFrame(rows)
            el = time.time() - t0
            print(f"  [{i+1}/{N}] {el:5.0f}s ({el/(i+1):.2f}s/trial)  folds={len(d)}  "
                  f"median held-out r: wrist {d.r_wrist.median():.3f} "
                  f"multi {d.r_multi.median():.3f} stacked {d.r_stacked.median():.3f}", flush=True)
    d = pd.DataFrame(rows)
    out = ROOT / "out/scoring"
    d.to_csv(out / "lag_loso.csv", index=False)
    np.savez(out / "lag_loso.npz", **{c: d[c].values for c in d.columns})
    nz = d[["r_wrist", "r_multi", "r_stacked"]]
    print(f"\nPROCESSING CHECK: {d.trial.nunique()} trials, {len(d)} folds, "
          f"non-finite {int(nz.isna().values.sum())}", flush=True)
    print(f"\nheld-out correlation at each estimator's lag (higher = better aligned):")
    print(f"{'stat':10s} {'wrist':>8s} {'multi':>8s} {'stacked':>8s} {'ceiling':>8s}")
    for nm, f in [("median", np.nanmedian), ("mean", np.nanmean),
                  ("p10", lambda x: np.nanpercentile(x, 10)),
                  ("p25", lambda x: np.nanpercentile(x, 25))]:
        print(f"{nm:10s} {f(d.r_wrist):8.3f} {f(d.r_multi):8.3f} {f(d.r_stacked):8.3f} "
              f"{f(d.best_held):8.3f}", flush=True)
    print(f"\nfolds where estimator beats the others (strictly):")
    w, m, s = d.r_wrist.values, d.r_multi.values, d.r_stacked.values
    print(f"  stacked > multi : {int(np.nansum(s > m)):5d}   multi > stacked : {int(np.nansum(m > s)):5d}"
          f"   tie: {int(np.nansum(s == m)):5d}")
    print(f"  stacked > wrist : {int(np.nansum(s > w)):5d}   wrist > stacked : {int(np.nansum(w > s)):5d}"
          f"   tie: {int(np.nansum(s == w)):5d}")
    print(f"\nper-trial agreement with the held-out signal, by participant:")
    g = d.groupby("part")[["r_wrist", "r_multi", "r_stacked", "best_held"]].median().round(3)
    print(g.to_string())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
