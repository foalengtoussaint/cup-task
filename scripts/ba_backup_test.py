"""DLT-as-BACKUP (user's recipe): keep every finite BA output, revert ONLY the NaN/blown-up frames to
the pipeline point. fallback_mm=inf in refine_trial_ba reverts on non-finite ONLY (moved>inf never
fires), so legit corrections are kept and only true blow-ups fall back to the incumbent.

Expected blend on ALL 1376 cycles: BA (+4.86%) on the 1136 it resolves + pipeline (+4.39%) on the 240
it blows up on -> ~+4.7%, beating pipeline's +6.60%. Live per-40-trial progress (flush=True).
Self-contained (helpers inlined) + __main__-guarded so nothing runs on import.
"""
import sys, time
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T, gnn_refiner as G, ba_refine as BA
import compare_pose_omc_delta as H
JOINTS = G.JOINTS
med = lambda a: float(np.nanmedian([z for z in a if np.isfinite(z)]))


def cycle_peakvels(mmc_wrist, omc_wrist, win=15):
    so = H._lp(H._speed(omc_wrist)); sm = H._lp(H._speed(mmc_wrist))
    if np.isfinite(so).sum() < 20:
        return []
    prom = np.nanmax(so) * 0.25
    pk, _ = find_peaks(np.nan_to_num(so, nan=0.0), prominence=prom, distance=win)
    out = []
    for p in pk:
        if not np.isfinite(so[p]) or so[p] <= 0:
            out.append((p, np.nan)); continue
        seg = sm[max(0, p-win):min(len(sm), p+win+1)]
        pm = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
        out.append((p, (pm - so[p]) / so[p] * 100.0 if np.isfinite(pm) else np.nan))
    return out


def wrist_sig(P, t, apply_savgol):
    if apply_savgol:
        P = T.smooth_baseline(P, t["valid"], kind="savgol")
    side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
    w = P[:, wi].copy(); w[~t["valid"][:, wi]] = np.nan
    return w


def main():
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in ("P07","P08","P15","P17","P19")]
    print(f"trials: {len(trials)}", flush=True)
    configs = [
        ("PIPELINE",              None),
        ("BA unbounded",          dict()),
        ("BA + DLT-backup(NaN)",  dict(fallback_mm=float("inf"))),
    ]
    print(f"\n{'config':22s} {'peak-vel':>9s} {'cycles':>7s} {'reverted':>9s}", flush=True)
    for name, kw in configs:
        t0 = time.time(); pv = []; nfb = 0
        print(f"  [{name}] starting {len(trials)} trials...", flush=True)
        for i, t in enumerate(trials):
            side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
            womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
            if kw is None:
                Xr = t["mmc"]
            else:
                Xr, info = BA.refine_trial_ba(t, 0.0, iters=60, **kw); nfb += info.get("n_fallback", 0)
            wb = wrist_sig(Xr, t, True)
            for f, v in cycle_peakvels(wb, womc):
                if np.isfinite(v):
                    pv.append(v)
            if (i + 1) % 40 == 0:
                print(f"      [{name}] {i+1}/{len(trials)}  running {med(pv):+.2f}%  "
                      f"({len(pv)} cyc, {nfb} reverted, {time.time()-t0:.0f}s)", flush=True)
        print(f"  >> {name:22s} {med(pv):+7.2f}% {len(pv):7d} {nfb:9d}   ({time.time()-t0:.0f}s)", flush=True)
    print("\nRECIPE = BA where finite, pipeline where BA blew up. Beats pipeline if < +6.60%.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
