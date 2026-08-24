"""WHERE does the BA+DLT-backup peak-velocity gain come from, + is the peak TIMING improved too?

Per OMC-defined drink cycle, measure for pipeline vs BA+DLT-backup (both savgol'd):
  - peak AMPLITUDE error %  = (max MMC speed in +-win around OMC peak - OMC peak) / OMC peak
  - peak TIMING error ms    = (frame of that MMC max - OMC peak frame) * 1000/FPS   (NEW: was blind to it)
Saves per-cycle records (npz) + a 4-panel figure (out/gnn/ba_improvement.png):
  A example speed curves (OMC/pipeline/BA) for the cycles BA helps most
  B paired amplitude-error: pipeline vs BA (scatter + medians)
  C peak-timing-error distributions: pipeline vs BA
  D per-participant amplitude-error improvement
tqdm progress. __main__-guarded.
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T, gnn_refiner as G, ba_refine as BA
import compare_pose_omc_delta as H
JOINTS = G.JOINTS
FPS = H.VIDEO_FPS
med = lambda a: float(np.nanmedian([z for z in a if np.isfinite(z)]))


def wrist_speed(P, t, savgol=True):
    if savgol:
        P = T.smooth_baseline(P, t["valid"], kind="savgol")
    side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
    w = P[:, wi].copy(); w[~t["valid"][:, wi]] = np.nan
    return H._lp(H._speed(w))


def peaks_amp_time(sm, so, win=15):
    """For each OMC speed peak p: (amp_err%, time_err_ms) of the matched MMC max in [p-win,p+win]."""
    if np.isfinite(so).sum() < 20:
        return []
    prom = np.nanmax(so) * 0.25
    pk, _ = find_peaks(np.nan_to_num(so, nan=0.0), prominence=prom, distance=win)
    out = []
    for p in pk:
        if not np.isfinite(so[p]) or so[p] <= 0:
            continue
        lo, hi = max(0, p-win), min(len(sm), p+win+1)
        seg = sm[lo:hi]
        if not np.isfinite(seg).any():
            out.append((p, np.nan, np.nan)); continue
        j = np.nanargmax(seg); pm = seg[j]; pf = lo + j
        out.append((p, (pm - so[p]) / so[p] * 100.0, (pf - p) * 1000.0 / FPS))
    return out


def main():
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in ("P07","P08","P15","P17","P19")]
    print(f"trials: {len(trials)}", flush=True)
    recs = []           # (part, omc_peak_speed, pipe_amp%, pipe_time_ms, ba_amp%, ba_time_ms)
    examples = []       # (improvement, part, so, sp, sb, p) for the example panel
    METK = ["wr", "pa", "elb", "sflex", "sabd", "jit", "pve"]   # all score_trial outputs
    mets = {"pipeline": {k: [] for k in METK}, "BA+backup": {k: [] for k in METK}}
    for t in tqdm(trials, mininterval=3, ncols=90, file=sys.stdout):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        # SAVGOL both, so the peak-vel comparison isn't confounded by smoothing
        mmc_s = T.smooth_baseline(t["mmc"], t["valid"], kind="savgol")
        Xr, _ = BA.refine_trial_ba(t, 0.0, iters=60, fallback_mm=float("inf"))
        Xr_s = T.smooth_baseline(Xr, t["valid"], kind="savgol")
        sp = H._lp(H._speed(mmc_s[:, wi]))                              # pipeline (savgol)
        sb = H._lp(H._speed(Xr_s[:, wi]))                              # BA+backup (savgol)
        # full metric panel (NO-regression check across every score_trial output)
        for nm, P in [("pipeline", mmc_s), ("BA+backup", Xr_s)]:
            sc = T.score_trial(P, t["omc"], t["valid"], side)
            for k in METK:
                mets[nm][k].append(sc.get(k, np.nan))
        ap = {p: (a, tm) for p, a, tm in peaks_amp_time(sp, so)}
        ab = {p: (a, tm) for p, a, tm in peaks_amp_time(sb, so)}
        for p in ap:
            if p in ab:
                pa, ptm = ap[p]; ba_, btm = ab[p]
                recs.append((t["part"], so[p], pa, ptm, ba_, btm))
                if np.isfinite(pa) and np.isfinite(ba_) and abs(pa) - abs(ba_) > 8:
                    examples.append((abs(pa)-abs(ba_), t["part"], so, sp, sb, p))

    print("\n=== NO-REGRESSION CHECK: every metric, pipeline -> BA+backup (median, both savgol'd) ===",
          flush=True)
    lbl = {"wr": "wrist err mm", "pa": "arm PA-MPJPE mm", "elb": "elbow ang deg",
           "sflex": "shoulder-flex deg", "sabd": "shoulder-abd deg", "jit": "jitter mm/s^2",
           "pve": "peak-vel %err"}
    for k in METK:
        mp = med(mets["pipeline"][k]); mb = med(mets["BA+backup"][k])
        # for pve, compare |err|; others are already errors/magnitudes where lower=better
        if k == "pve":
            mp, mb = med(np.abs(mets["pipeline"][k])), med(np.abs(mets["BA+backup"][k]))
        arrow = "BETTER" if mb < mp - 1e-9 else ("SAME " if abs(mb-mp) < 0.05*abs(mp)+1e-6 else "WORSE")
        print(f"  {lbl[k]:20s} {mp:8.2f} -> {mb:8.2f}   [{arrow}]", flush=True)
    recs = np.array(recs, dtype=object)
    part = recs[:, 0]; omc_pk = recs[:, 1].astype(float)
    pa = recs[:, 2].astype(float); pt = recs[:, 3].astype(float)
    ba = recs[:, 4].astype(float); bt = recs[:, 5].astype(float)
    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_improvement.npz",
             part=part.astype(str), omc_pk=omc_pk, pipe_amp=pa, pipe_time=pt, ba_amp=ba, ba_time=bt)

    print("\n=== AMPLITUDE err % (peak wrist speed vs OMC) ===", flush=True)
    print(f"  pipeline |err| median {med(np.abs(pa)):.2f}%   signed {med(pa):+.2f}%", flush=True)
    print(f"  BA+backup|err| median {med(np.abs(ba)):.2f}%   signed {med(ba):+.2f}%", flush=True)
    print("=== TIMING err ms (when the peak happens vs OMC) ===", flush=True)
    print(f"  pipeline |err| median {med(np.abs(pt)):.1f}ms  signed {med(pt):+.1f}ms", flush=True)
    print(f"  BA+backup|err| median {med(np.abs(bt)):.1f}ms  signed {med(bt):+.1f}ms", flush=True)
    fin = np.isfinite(pa) & np.isfinite(ba)
    print(f"  cycles where BA better on amplitude: {100*np.mean(np.abs(ba[fin])<np.abs(pa[fin])):.0f}%  "
          f"(n={fin.sum()})", flush=True)

    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    # A: example curves (top improvements)
    examples.sort(key=lambda e: -e[0])
    axA = ax[0, 0]
    for k, (imp, pp, so_, sp_, sb_, p) in enumerate(examples[:3]):
        lo, hi = max(0, p-40), min(len(so_), p+40)
        xs = (np.arange(lo, hi) - p) * 1000 / FPS
        off = k * 0  # overlay
        axA.plot(xs, so_[lo:hi], color="k", lw=2.2, label="OMC" if k == 0 else None)
        axA.plot(xs, sp_[lo:hi], color="tab:red", lw=1.4, alpha=.8, label="pipeline" if k == 0 else None)
        axA.plot(xs, sb_[lo:hi], color="tab:blue", lw=1.4, alpha=.8, label="BA+backup" if k == 0 else None)
    axA.axvline(0, color="gray", ls=":", lw=1)
    axA.set_title("A. Wrist-speed at drink peak (3 cycles BA helps most)")
    axA.set_xlabel("ms from OMC peak"); axA.set_ylabel("speed (mm/s)"); axA.legend()
    # B: paired amplitude
    axB = ax[0, 1]
    axB.scatter(pa[fin], ba[fin], s=8, alpha=.35, c=omc_pk[fin], cmap="viridis")
    lim = np.nanpercentile(np.abs(np.r_[pa[fin], ba[fin]]), 98)
    axB.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    axB.axhline(0, color="gray", lw=.5); axB.axvline(0, color="gray", lw=.5)
    axB.set_xlim(-lim, lim); axB.set_ylim(-lim, lim)
    axB.set_title(f"B. Peak-amp err per cycle (below diag = BA better)\n"
                  f"pipeline {med(pa):+.1f}% -> BA {med(ba):+.1f}%")
    axB.set_xlabel("pipeline amp err %"); axB.set_ylabel("BA+backup amp err %")
    # C: timing hist
    axC = ax[1, 0]
    b = np.linspace(-260, 260, 40)
    axC.hist(pt[np.isfinite(pt)], bins=b, alpha=.5, color="tab:red", label=f"pipeline (|md| {med(np.abs(pt)):.0f}ms)")
    axC.hist(bt[np.isfinite(bt)], bins=b, alpha=.5, color="tab:blue", label=f"BA+backup (|md| {med(np.abs(bt)):.0f}ms)")
    axC.axvline(0, color="k", ls=":"); axC.set_title("C. Peak TIMING error (ms; 0 = same frame as OMC)")
    axC.set_xlabel("MMC peak - OMC peak (ms)"); axC.legend()
    # D: per-participant amplitude improvement
    axD = ax[1, 1]
    parts = sorted(set(part))
    pipe_by = [med(np.abs(pa[part == pp])) for pp in parts]
    ba_by = [med(np.abs(ba[part == pp])) for pp in parts]
    x = np.arange(len(parts))
    axD.bar(x-.2, pipe_by, .4, color="tab:red", label="pipeline")
    axD.bar(x+.2, ba_by, .4, color="tab:blue", label="BA+backup")
    axD.set_xticks(x); axD.set_xticklabels(parts)
    axD.set_title("D. |peak-amp err| by participant"); axD.set_ylabel("median |err| %"); axD.legend()
    plt.tight_layout()
    plt.savefig("/home/imove/Documents/cup-task/out/gnn/ba_improvement.png", dpi=110)
    print("\nwrote out/gnn/ba_improvement.png + ba_improvement.npz", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
