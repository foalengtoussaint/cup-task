"""What does halving the capture rate cost? 60 Hz against a simulated 30 Hz, same trials.

Motivation is latency, not accuracy: the front end needs 28 ms per five-camera frame, against the
16.7 ms budget of 60 Hz capture but comfortably inside the 33.3 ms of 30 Hz. If 30 Hz costs little,
the pipeline runs live on this hardware; if it costs a lot, it does not, and the trade is the result.

SIMULATION. Every markerless channel is decimated to every second frame BEFORE smoothing, so the
30 Hz arm sees exactly what a 30 Hz camera would have delivered:

  * the BA trajectory is decimated, then SmoothNet re-runs on the decimated series -- its window is
    32 samples, so at 30 Hz it spans 1.07 s of movement instead of 0.53 s. Re-running is the whole
    point; decimating an already-smoothed track would hide the change.
  * the segmenter runs on the decimated cup/wrist/nose with fps=30, so its low-pass cutoffs and
    minimum-run constants scale as they would in a 30 Hz deployment.
  * the measures run with FPS=30, so every derivative and every time-valued measure is computed on
    the coarser grid.

The OPTICAL reference is untouched at 60 Hz on its own windows -- the same reference the paper's
Table III uses. So each column answers "how well would a 30 Hz (or 60 Hz) markerless system agree
with the optical ground truth", and the difference between them is the cost of the halved rate.

    python paper/scripts/fps_ablation.py                     # -> paper/table6_fps.{md,csv}
    python paper/scripts/fps_ablation.py --limit 60           # quick look
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import compare_pose_omc_delta as H              # noqa: E402
import results_v3_delta as R                    # noqa: E402
import score_vs_automq as S                     # noqa: E402
import score_own_phases as SOP                  # noqa: E402
from seg_sequential import segment_sequential   # noqa: E402

SEG = ROOT / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs_ship")
SEGB = ROOT / "out" / "scoring" / "seg_boundaries.csv"
GRID = S.GRID_JOINTS
MEAS = [
    ("peak_velocity",                 "PV [mm/s]"),
    ("peak_elbow_angular_velocity",   "Elbow angular PV [deg/s]"),
    ("time_to_peak_velocity",         "Time to PV [s]"),
    ("time_to_first_peak_velocity",   "Time to first PV [s]"),
    ("number_of_movement_units",      "Number of movement units [n]"),
    ("total_movement_time",           "Total movement time [s]"),
    ("interjoint_coordination",       "Interjoint coordination"),
    ("max_trunk_displacement",        "Trunk displacement [mm]"),
    ("max_shoulder_flexion",          "Shoulder flexion [deg]"),
    ("max_shoulder_flexion_drink",    "Shoulder flexion, drinking [deg]"),
    ("max_elbow_angle",               "Elbow extension [deg]"),
    ("max_shoulder_abduction",        "Shoulder abduction [deg]"),
]


def declined():
    """Trials the segmenter declines, excluded here exactly as in Table III and Fig. 4."""
    d = pd.read_csv(SEGB)
    m = (d["seq_mmc_c3kf_grasp"] - d["seq_mmc_c3kf_reach onset"]) == 9
    return set(zip(d.loc[m, "part"], d.loc[m, "trial"]))


def _measures_at(pose, ph, side, fps, fps_aware_lp=True):
    """SOP._measures with every operator's frame rate set to `fps` for the duration of the call.

    `fps_aware_lp` also patches H.VIDEO_FPS, which `H._lp` uses to turn LP_HZ into a normalised
    cutoff. Left at 60 while the data is at 30, the intended 6 Hz becomes 3 Hz and the joint angles
    -- and the elbow angular velocity differentiated from them -- are over-smoothed. That is an
    implementation artefact of running the 60 Hz code at 30 Hz, not a property of 30 Hz capture, so
    the ablation reports it both ways. `peak_velocity_reduce` needs no patch: it passes FPS into its
    own butter already.
    """
    old_s, old_p, old_v = S.FPS, SOP.FPS, H.VIDEO_FPS
    try:
        S.FPS = SOP.FPS = fps
        if fps_aware_lp:
            H.VIDEO_FPS = fps
        return SOP._measures(pose, ph, side, "mmc")
    finally:
        S.FPS, SOP.FPS, H.VIDEO_FPS = old_s, old_p, old_v


def run(limit=0, anat12=None):
    H.use_good_cams()
    theta = {}
    if anat12:
        t = pd.read_csv(anat12)
        cols = [f"th{i}" for i in range(12)]
        theta = {(r["part"], r["arm"]): np.array([r[c] for c in cols], float)
                 for _, r in t.iterrows() if not str(r["part"]).startswith("LOPO_")}
        print(f"anat12: {len(theta)} participant x arm offsets from {anat12}", flush=True)
    import gnn_train as GT
    ba = R._ba_traj_cache()
    bad = declined()
    trials = {f"{t['part']}/{t['trial']}": t for t in GT.load_clean(need_reproj=False)}
    files = sorted(SEG.glob("*.npz"))
    if limit:
        files = files[::max(1, len(files) // limit)][:limit]
    print(f"fps ablation: {len(files)} trials ({SEG.name}), excluding {len(bad)} declined",
          flush=True)

    rows, t0 = [], time.time()
    n_skip = 0
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        part, trial, side = str(z["part"]), str(z["trial"]), str(z["side"])
        if (part, trial) in bad:
            continue
        t = trials.get(f"{part}/{trial}")
        if t is None:
            n_skip += 1; continue
        n = int(z["n"])
        P = ba.get(f"{part}/{trial}")
        if P is None:
            n_skip += 1; continue
        P = np.asarray(P, float)

        # ---- reference: optical pose, optical windows, 60 Hz. Unchanged from Table III. ----
        try:
            ph_omc = segment_sequential(z["cup_omc"], z["wrist_omc"], z["nose_omc"])
            omc_raw = H._load_omc(part, trial, n)
            lag, _, _ = H.find_lag_best({j: t["mmc"][:, k] for k, j in enumerate(GRID)},
                                        omc_raw, side)
            omc_pose = {j: R._shift(omc_raw[j], lag) for j in omc_raw}
            if theta:                      # same landmark match as Table III, so the columns compare
                th = theta.get((part, str(z["arm"])))
                if th is None:
                    n_skip += 1; continue
                other = "right" if side == "left" else "left"
                omc_pose = SOP._apply_anat12(omc_pose, side, other, th)
            ref = _measures_at(omc_pose, ph_omc, side, 60.0)
        except Exception:
            n_skip += 1; continue

        cup = SOP._ship_cup(part, trial, z["cup_mmc"])
        out = {}
        for tag, k, aware in (("60", 1, True), ("30", 2, True), ("30raw", 2, False)):
            fps = 60.0 / k
            try:
                # decimate BEFORE smoothing, then smooth on the coarser grid
                pose = R._smooth_pose({j: P[::k, m] for m, j in enumerate(R._GRID_JOINTS)})
                cu, wr, no = cup[::k], z["wrist_mmc"][::k], z["nose_mmc"][::k]
                m_ = min(len(cu), len(wr), len(no), len(pose[f"{side}_wrist"]))
                ph = segment_sequential(cu[:m_], wr[:m_], no[:m_], fps=fps)
                if not ph:
                    continue
                out[tag] = _measures_at(pose, ph, side, fps, fps_aware_lp=aware)
            except Exception:
                continue
        if "60" not in out or "30" not in out:
            n_skip += 1; continue

        for meas, _label in MEAS:
            rows.append(dict(part=part, trial=trial, arm=str(z["arm"]), measure=meas,
                             omc=ref.get(meas, np.nan),
                             mmc60=out["60"].get(meas, np.nan),
                             mmc30=out["30"].get(meas, np.nan),
                             mmc30raw=out.get("30raw", {}).get(meas, np.nan)))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:5.0f}s  rows {len(rows)} "
                  f"skip {n_skip}", flush=True)
    return pd.DataFrame(rows), n_skip


def report(D, out_stem):
    def rr(g, col):
        g = g.dropna(subset=["omc", col])
        if len(g) < 5:
            return np.nan, np.nan, 0
        av = g.groupby(["part", "arm"])[["omc", col]].mean()
        return (pearsonr(g.omc, g[col])[0],
                pearsonr(av.omc, av[col])[0] if len(av) >= 3 else np.nan, len(g))

    lines = ["# Capture rate ablation: 60 Hz vs simulated 30 Hz", "",
             "Pearson against the same optical reference (60 Hz, optical windows). The markerless "
             "arm is decimated before smoothing; SmoothNet, the segmenter and every measure run at "
             "the stated rate. Trials the segmenter declines are excluded, as in Table III.", "",
             "| Movement quality measure | r_s 60 | r_s 30 | d r_s | r_av 60 | r_av 30 | "
             "med. signed diff 30-60 | as % of 60 | same, 60 Hz cutoff left in | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    recs = []
    for meas, label in MEAS:
        g = D[D.measure == meas]
        r60, a60, n60 = rr(g, "mmc60")
        r30, a30, _ = rr(g, "mmc30")
        p = g.dropna(subset=["mmc60", "mmc30"])
        scale = np.nanmedian(np.abs(p.mmc60)) if len(p) else np.nan
        sd = np.nanmedian(p.mmc30 - p.mmc60) if len(p) else np.nan
        rel = (sd / scale * 100 if len(p) and scale and np.isfinite(scale) and scale > 1e-9
               else np.nan)
        q = g.dropna(subset=["mmc60", "mmc30raw"])
        sdr = np.nanmedian(q.mmc30raw - q.mmc60) if len(q) else np.nan
        relr = (sdr / scale * 100 if len(q) and scale and np.isfinite(scale) and scale > 1e-9
                else np.nan)
        lines.append(f"| {label} | {r60:.2f} | {r30:.2f} | {r30-r60:+.3f} | {a60:.2f} | "
                     f"{a30:.2f} | {sd:+.3f} | {rel:+.1f}% | {relr:+.1f}% | {n60} |")
        recs.append(dict(measure=label, r_s_60=r60, r_s_30=r30, d_r_s=r30 - r60,
                         r_av_60=a60, r_av_30=a30, signed_diff=sd, rel_diff_pct=rel,
                         rel_diff_pct_lp60=relr, n=n60))
    txt = "\n".join(lines)
    Path(f"{out_stem}.md").write_text(txt + "\n")
    pd.DataFrame(recs).to_csv(f"{out_stem}.csv", index=False)
    print("\n" + txt)
    dd = pd.DataFrame(recs)
    print(f"\nr_s change: median {dd.d_r_s.median():+.3f}, worst {dd.d_r_s.min():+.3f} "
          f"({dd.loc[dd.d_r_s.idxmin(), 'measure']}), best {dd.d_r_s.max():+.3f} "
          f"({dd.loc[dd.d_r_s.idxmax(), 'measure']})")
    print(f"measures losing more than 0.02: "
          f"{int((dd.d_r_s < -0.02).sum())}/{len(dd)}; more than 0.05: "
          f"{int((dd.d_r_s < -0.05).sum())}/{len(dd)}")
    print(f"wrote {out_stem}.md and .csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--anat12", default=str(ROOT / "out/scoring/anat12_wv1wa1_theta.csv"),
                    help="apply the fitted landmark offsets to the OPTICAL reference, as Table III "
                         "does; pass '' to compare against the uncorrected optical pose")
    ap.add_argument("--out", default=str(ROOT / "paper" / "table6_fps"))
    a = ap.parse_args(argv)
    D, n_skip = run(a.limit, a.anat12 or None)
    if D.empty:
        print("nothing scored"); return
    D.to_csv(f"{a.out}_pertrial.csv", index=False)
    print(f"\nPROCESSING CHECK: {D.groupby(['part','trial']).ngroups} trials scored, "
          f"{n_skip} skipped, {len(D)} rows")
    report(D, a.out)
    print("DONE_FPS", flush=True)


if __name__ == "__main__":
    main()
