"""LOPO 'static recalibration' baseline: does a per-joint CONSTANT offset match the 2.3M GNN?

Motivated by the finding that the GNN's correction is 99% time-constant (offset_vs_reshape.py). If a
trivial per-joint offset -- learned from the training participants, applied to the held-out one -- fixes
the elbow angle as well as the GNN, the temporal ST-GCN is unnecessary.

HONEST protocol (mirrors the GNN LOPO): the offset is fit on the TRAINING participants only, NOT the
held-out trial (fitting on the target would be cheating). The offset is computed in each trial's
per-trial-aligned frame (arm Kabsch, hips-free), averaged over training trials -> one (J,3) correction,
applied to every held-out frame. Scored with the same score_trial (per-trial aligned, elbow angle).

    python scripts/offset_baseline.py --only-parts P07 P08 P15
"""
import argparse, sys, numpy as np
sys.path.insert(0, "scripts")
import gnn_train as T
import compare_pose_omc_delta as H
J = T.JOINTS; ARM = T.ARM_I


def per_trial_correction(t):
    """The per-joint mean correction (omc - mmc) for ONE trial, in the mmc frame after arm-Kabsch
    maps omc->mmc. Returns (J,3) mean over valid frames, and a per-joint valid count."""
    A, B = [], []
    for f in range(t["mmc"].shape[0]):
        vv = t["valid"][f, ARM]
        for k in range(len(ARM)):
            if vv[k]:
                A.append(t["omc"][f, ARM[k]]); B.append(t["mmc"][f, ARM[k]])
    if not A:
        return None, None
    R, tt, _ = H._kabsch(np.array(A), np.array(B))
    omc_al = t["omc"] @ R.T + tt                 # OMC in mmc frame
    d = omc_al - t["mmc"]                          # target correction per (frame,joint)
    corr = np.full((len(J), 3), np.nan); cnt = np.zeros(len(J))
    for j in range(len(J)):
        v = t["valid"][:, j] & np.isfinite(d[:, j]).all(1)
        if v.sum():
            corr[j] = d[v, j].mean(0); cnt[j] = v.sum()
    return corr, cnt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-parts", nargs="+", default=["P07", "P08", "P15"])
    a = ap.parse_args(argv)
    trials = [t for t in T.load_clean() if t["part"] in a.only_parts]
    parts = sorted({t["part"] for t in trials})
    print(f"offset baseline, {len(trials)} trials, parts {parts}\n")
    print(f"{'fold':6}{'elbow raw':>11}{'elbow OFFSET':>14}{'wrist raw':>11}{'wrist OFFSET':>14}  (deg / mm)")
    agg = {"elb": [[], []], "wr": [[], []]}
    for held in parts:
        train = [t for t in trials if t["part"] != held]
        test = [t for t in trials if t["part"] == held]
        # learn ONE (J,3) offset = weighted mean of per-trial corrections over TRAINING trials only
        corrs, cnts = [], []
        for t in train:
            c, n = per_trial_correction(t)
            if c is not None:
                corrs.append(c); cnts.append(n)
        C = np.stack(corrs); N = np.stack(cnts)                 # (ntrial, J, 3), (ntrial, J)
        offset = np.nansum(C * N[..., None], 0) / (N.sum(0)[..., None] + 1e-9)   # (J,3) weighted mean
        # apply to held-out, score raw vs offset-corrected
        er, eo, wrr, wro = [], [], [], []
        for t in test:
            corrected = t["mmc"] + offset[None]                 # add the fixed offset every frame
            r = T.score_trial(t["mmc"], t["omc"], t["valid"], t["side"])
            o = T.score_trial(corrected, t["omc"], t["valid"], t["side"])
            for arr, key, row in [(er, "elb", r), (eo, "elb", o), (wrr, "wr", r), (wro, "wr", o)]:
                pass
            er.append(r["elb"]); eo.append(o["elb"]); wrr.append(r["wr"]); wro.append(o["wr"])
        med = lambda x: np.nanmedian(x)
        print(f"{held:6}{med(er):10.1f} {med(eo):13.1f} {med(wrr):10.1f} {med(wro):13.1f}")
        agg["elb"][0] += er; agg["elb"][1] += eo; agg["wr"][0] += wrr; agg["wr"][1] += wro
    print(f"\n{'ALL':6}{np.nanmedian(agg['elb'][0]):10.1f} {np.nanmedian(agg['elb'][1]):13.1f} "
          f"{np.nanmedian(agg['wr'][0]):10.1f} {np.nanmedian(agg['wr'][1]):13.1f}")
    print("\ncompare the OFFSET columns to the GNN's LOPO elbow/wrist: if ~equal, the GNN is unnecessary.")


if __name__ == "__main__":
    main()
