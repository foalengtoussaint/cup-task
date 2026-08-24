"""Peak-velocity agreement for cached BA variants, on the FULL AutoMQ-joined cohort.

Uses score_vs_automq's own join + phase path (load_automq -> _find_lag ->
automq_phases_to_video) rather than cache/seg_inputs_26x, so the trial count matches
Table III (697) instead of the seg cache's 636. Operator is peak_velocity_reduce's:
speed of the RAW wrist -> 6 Hz Butterworth -> reduce over the REACHING window.

    python scripts/score_pv_variants.py --tags fix__g150__sn freebone_0.05__g150__sn
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H       # noqa: E402
import gnn_train as T                    # noqa: E402
import gnn_refiner as G                  # noqa: E402
import score_vs_automq as S              # noqa: E402
import re                               # noqa: E402

PAT = re.compile(r"trial_(\d+)_([LR])_")

J = G.JOINTS


def pv(P, wi, w, how):
    sp = S._butter_lowpass(S._hand_speed_mmps(P[:, wi], S.FPS), S.FPS, 6.0, S.DEFAULT_BUTTER_ORDER)
    seg = np.asarray(sp, float)[w[0]:w[1]]
    return S._reduce(seg, how) if np.isfinite(seg).any() else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["fix__g150__sn", "freebone_0.05__g150__sn"])
    ap.add_argument("--how", default="p95", choices=["max", "p95", "p99"])
    a = ap.parse_args()
    H.use_good_cams()
    VAR = {}
    for tg in a.tags:
        z = np.load(ROOT / f"cache/ba_variants/{tg}.npz", allow_pickle=True)
        VAR[tg] = dict(zip(z["ids"], z["traj"]))
    amq = S.load_automq()
    trials = T.load_clean(need_reproj=True)
    rows, n_nomatch, n_nophase = [], 0, 0
    for i, t in enumerate(trials):
        part, trial, side = t["part"], t["trial"], t["side"]
        m = PAT.search(trial)
        if not m:
            n_nomatch += 1; continue
        rec = amq.get(S.automq_key(part, trial))      # block-aware truth row
        if rec is None:
            n_nomatch += 1; continue
        if rec.get("phases") is None:
            n_nophase += 1; continue
        n = t["mmc"].shape[0]
        omc = H._load_omc(part, trial, n)
        wr = f"{side}_wrist"
        lag, _ = H._find_lag(t["mmc"][:, S.GRID_JOINTS.index(wr)], omc[wr])
        ph = S.automq_phases_to_video(rec["phases"], lag, n)
        if ph is None:
            n_nophase += 1; continue
        w = S._win(ph, "reaching")
        if not (w and w[1] - w[0] >= 10):
            n_nophase += 1; continue
        wi = J.index(wr)
        po = pv(t["omc"], wi, w, a.how)
        if not np.isfinite(po) or po <= 0:
            continue
        rec2 = {"part": part, "id": f"{part}/{trial}", "omc": po}
        try:
            rec2["shipped"] = pv(np.load(
                ROOT / f"cache/pose_smoothed/{part}__{trial}.npz", allow_pickle=True)["ba_sn"], wi, w, a.how)
        except Exception:
            rec2["shipped"] = np.nan
        for tg in a.tags:
            P = VAR[tg].get(f"{part}/{trial}")
            rec2[tg] = pv(P, wi, w, a.how) if (P is not None and P.shape[0] == n) else np.nan
        rows.append(rec2)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(trials)}] kept {len(rows)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "out/scoring/pv_variants.csv", index=False)
    cols = ["shipped"] + a.tags
    print(f"\nPROCESSING CHECK: {len(trials)} trials, no-AutoMQ-match {n_nomatch}, "
          f"no/failed-phase {n_nophase}, scored {len(d)}", flush=True)
    print(f"\npeak_velocity ({a.how}), REACHING window, AutoMQ phases\n")
    print(f"{'subset':16s} {'n':>5s} " + " ".join(f"{c[:22]:>24s}" for c in cols))
    for lab, sub in (("ALL", d), ("excl P19", d[d.part != "P19"]), ("P19 only", d[d.part == "P19"])):
        cells = []
        for c in cols:
            s2 = sub.dropna(subset=["omc", c])
            if len(s2) < 5:
                cells.append(f"{'--':>24s}"); continue
            rs = spearmanr(s2.omc, s2[c]).correlation
            e = 100 * (s2[c] - s2.omc) / s2.omc
            cells.append(f"r_s {rs:.3f}  |err| {np.median(np.abs(e)):4.1f}%")
        print(f"{lab:16s} {len(sub):5d} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
