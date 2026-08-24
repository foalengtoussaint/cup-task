"""Full-328 comparison: PIPELINE (real robust-consensus triangulation, NOT plain DLT) vs BA lam=0
(robust reproj) with POST-HOC savgol vs IN-SOLVE smoothness (smooth_w inside the L-BFGS).

Baseline `mmc` = pipeline.triangulate.triangulate_target = inverse-jitter-weighted, 30px-gated,
linear-DLT consensus. BA adds: confidence weighting + soft Huber + DISTORTION-AWARE projection.

Question 1 (user): does BA+smooth hold up on ALL 328 trials, not just the 20-trial spot-check?
Question 2 (user): can BA optimise smoothness IN-SOLVE (smooth_w) and beat post-hoc savgol? In-solve
smoothness competes against the reprojection DATA term, so it only smooths where the data is weak
(gaps/low-conf) and preserves real fast motion where cameras agree -- the peak-preserving property a
blind low-pass lacks. But like any self-consistency term it CAN drift along the depth ray -> judge vs
OMC, not vs the energy.

Metric = honest per-cycle OMC-matched peak-velocity (fk_arm_solver.peak_vel_matched) + wrist err.
"""
import sys, time, argparse, numpy as np
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T
import gnn_refiner as G
import ba_refine as BA
import fk_arm_solver as FK

JOINTS = G.JOINTS
med = lambda x: float(np.nanmedian([z for z in x if np.isfinite(z)]))


def score(P, t, post_smooth=False):
    if post_smooth:
        P = T.smooth_baseline(P, t["valid"], kind="savgol")
    side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
    wmmc = P[:, wi].copy(); wmmc[~t["valid"][:, wi]] = np.nan
    womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
    pv, n = FK.peak_vel_matched(wmmc, womc)
    sc = T.score_trial(P, t["omc"], t["valid"], t["side"])
    return sc["wr"], pv, n, sc["jit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P15", "P17", "P19"])
    ap.add_argument("--smooth-ws", nargs="+", type=float, default=[0.003, 0.01, 0.03])
    a = ap.parse_args()
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in a.parts]
    print(f"trials: {len(trials)}", flush=True)

    def ba_nodist(t):
        """DIAGNOSTIC: BA lam=0 with distortion ZEROED -> isolates whether the distortion-aware
        projection (vs the pipeline's linear DLT) is what earns the win, or if it's Huber/conf."""
        t2 = dict(t); t2["dist"] = np.zeros_like(t["dist"])
        return BA.refine_trial_ba(t2, 0.0, iters=60)[0]

    variants = [
        ("PIPELINE (consensus)",     lambda t: t["mmc"],                              False),
        ("PIPELINE + savgol",        lambda t: t["mmc"],                              True),
        ("BA lam=0",                 lambda t: BA.refine_trial_ba(t, 0.0, iters=60)[0], False),
        ("BA lam=0 + savgol",        lambda t: BA.refine_trial_ba(t, 0.0, iters=60)[0], True),
        ("BA lam=0 NO-DIST",         ba_nodist,                                       False),
        ("BA lam=0 NO-DIST +savgol", ba_nodist,                                       True),
    ]
    for sw in a.smooth_ws:
        variants.append((f"BA in-solve sw={sw}",
                         (lambda sw: lambda t: BA.refine_trial_ba(t, 0.0, iters=60, smooth_w=sw)[0])(sw),
                         False))

    print(f"\n{'variant':24s} {'wrist':>7s} {'peakVel(match)':>15s} {'jitter':>9s}", flush=True)
    for name, fn, ps in variants:
        t0 = time.time(); wr, pv, npk, jt = [], [], [], []
        for i, t in enumerate(trials):
            a1, b1, n1, j1 = score(fn(t), t, post_smooth=ps)
            wr.append(a1); pv.append(b1); npk.append(n1); jt.append(j1)
        print(f"  {name:24s} {med(wr):6.1f}mm {med(pv):+7.1f}% {med(jt):9.0f}   "
              f"({int(np.nansum(npk))} cyc, {time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
