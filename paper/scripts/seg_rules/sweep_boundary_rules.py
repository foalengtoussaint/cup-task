"""Sweep candidate reach-onset and settle rules on cached channels. Emits paper/seg_rule_sweep.csv.

Four families, each thresholded on a quantity that is scale-free except where noted:
  pos-abs   d > X mm from the rest / final reference   (the SHIPPED rule: 30 mm onset, 40 mm settle)
  rel-pos   d > f x L, L = |cup_rest - wrist_rest|     (the scale-free form of the same rule)
  speed     |v_wrist| > f x peak
  proj-fix  v_wrist . u > f x peak, u = unit(cup_rest - wrist_rest) fixed once at rest

Reported per variant. The median |MMC-OMC| is quantised to 16.7 ms frames -- the same shipped rule
reads 17 ms on 750 trials and 33 ms on 631 -- so MEAN, p90 and the fraction beyond 4 frames are the
statistics to read. `|diff|` and `resid_iqr` are against unger2024's boundaries: a constant offset
cancels in every correlation, so the IQR is what matters there, while |diff| is what absolute
comparability to published norms needs.

Settle rows are restricted to settle_observed trials: elsewhere the boundary is a timeout, not a
detection, and timeouts land in the tail that p90 and >4fr are measuring.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from fps_ablation import declined                                    # noqa: E402

SEG = ROOT / "cache" / "seg_inputs_ship"
OUT = ROOT / "paper" / "seg_rule_sweep.csv"
FPS = 60.0
HOLDS = (9, 18)                       # 0.15 s (shipped) and 0.30 s (AutoMQ's). Barely matters.
ONSET = ([("pos-abs", t) for t in (10, 20, 30, 50)] +
         [("rel-pos", f) for f in (0.005, 0.01, 0.015, 0.02)] +
         [("speed", f) for f in (0.03, 0.05, 0.08, 0.10, 0.15)] +
         [("proj-fix", f) for f in (0.01, 0.02, 0.03)])
SETTLE = ([("pos-abs", t) for t in (7, 10, 15, 25, 40)] +
          [("rel-pos", f) for f in (0.016, 0.035, 0.09)] +
          [("speed", f) for f in (0.03, 0.05, 0.08, 0.10, 0.15)] +
          [("proj-fix", f) for f in (0.02, 0.03, 0.05)])


def _sustained(mask, start, hold):
    run = 0
    for i in range(max(int(start), 0), len(mask)):
        if mask[i]:
            run += 1
            if run >= hold:
                return i - hold + 1
        else:
            run = 0
    return np.nan


def _first_below(sig, frm, thr, hold):
    """After the peak past `frm`, the first sustained fall below `thr`."""
    frm = max(int(frm), 1)
    if frm >= len(sig) - 2:
        return np.nan
    seg = sig[frm:]
    if not np.isfinite(seg).any():
        return np.nan
    k = int(np.nanargmax(seg))
    r = _sustained(np.isfinite(seg) & (seg < thr), k, hold)
    return frm + r if np.isfinite(r) else np.nan


def _channels(z, tag):
    """Wrist speed, projected speed, distance-from-rest, distance-from-final, and the reach length."""
    W = np.asarray(z[f"wrist_{tag}"], float)
    C = np.asarray(z[f"cup_{tag}"], float)
    n0 = max(int(0.5 * FPS), 5)
    if np.isfinite(W).all(1).sum() < 30:
        return None
    fin = np.isfinite(C).all(1)
    if not fin.any():
        return None
    rest = np.nanmedian(W[:n0], axis=0)
    endref = np.nanmedian(W[-n0:], axis=0)
    cup0 = np.nanmedian(C[:n0], axis=0)
    if not np.isfinite(cup0).all():
        cup0 = C[np.argmax(fin)]
    L = float(np.linalg.norm(cup0 - rest))
    if not np.isfinite(L) or L < 50:
        return None
    u = (cup0 - rest) / L
    v = np.full(len(W), np.nan); v[1:] = np.linalg.norm(np.diff(W, axis=0), axis=1) * FPS
    vp = np.full(len(W), np.nan); vp[1:] = np.diff(W, axis=0) @ u * FPS
    return dict(v=v, vp=vp, dr=np.linalg.norm(W - rest, axis=1),
                de=np.linalg.norm(W - endref, axis=1), L=L,
                pk=float(np.nanmax(v)), pkp=float(np.nanmax(np.abs(vp))))


def _place(ch, fam, p, which, grasp, rel, hold):
    if which == "onset":
        if fam == "pos-abs":
            k = _sustained(np.isfinite(ch["dr"]) & (ch["dr"] > p), 0, hold)
        elif fam == "rel-pos":
            k = _sustained(np.isfinite(ch["dr"]) & (ch["dr"] > p * ch["L"]), 0, hold)
        elif fam == "speed":
            k = _sustained(np.isfinite(ch["v"]) & (ch["v"] > p * ch["pk"]), 1, hold)
        else:
            k = _sustained(np.isfinite(ch["vp"]) & (ch["vp"] > p * ch["pkp"]), 1, hold)
        if np.isfinite(k) and np.isfinite(grasp) and k > grasp:
            return np.nan
        return k
    if fam == "pos-abs":
        return _sustained(np.isfinite(ch["de"]) & (ch["de"] < p), rel, hold)
    if fam == "rel-pos":
        return _sustained(np.isfinite(ch["de"]) & (ch["de"] < p * ch["L"]), rel, hold)
    if fam == "speed":
        return _first_below(ch["v"], rel, p * ch["pk"], hold)
    return _first_below(np.abs(ch["vp"]), rel, p * ch["pkp"], hold)


def main():
    bad = declined()
    sop = pd.read_csv(ROOT / "out/scoring/score_own_phases_anat12.csv")
    obs = (sop[sop.measure == "total_movement_time"][["part", "trial", "settle_observed"]]
           .drop_duplicates())
    d = pd.read_csv(ROOT / "out/scoring/seg_boundaries.csv")
    d = d[[(p, t) not in bad for p, t in zip(d.part, d.trial)]]
    d = d.merge(obs, on=["part", "trial"], how="left")
    need = ["seq_omc_reach onset", "seq_omc_grasp", "seq_omc_release", "seq_omc_settle",
            "seq_mmc_c3kf_reach onset", "seq_mmc_c3kf_settle"]
    m = np.ones(len(d), bool)
    for c in need:
        m &= d[c].notna()
    d = d[m].reset_index(drop=True)
    print(f"{len(d)} trials", flush=True)

    rec = []
    for _, r in d.iterrows():
        fp = SEG / f"{r.part}__{r.trial}.npz"
        if not fp.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        ch = {t: _channels(z, t) for t in ("omc", "mmc")}
        if ch["omc"] is None or ch["mmc"] is None:
            continue
        row = {"part": r.part, "trial": r.trial, "obs": r.settle_observed,
               "amq_onset": r["amq_reach onset"], "amq_settle": r["amq_settle"],
               "ship_onset_o": r["seq_omc_reach onset"], "ship_onset_m": r["seq_mmc_c3kf_reach onset"],
               "ship_settle_o": r["seq_omc_settle"], "ship_settle_m": r["seq_mmc_c3kf_settle"]}
        for hold in HOLDS:
            for which, variants in (("onset", ONSET), ("settle", SETTLE)):
                for fam, p in variants:
                    for tag in ("omc", "mmc"):
                        k = _place(ch[tag], fam, p, which, r["seq_omc_grasp"],
                                   int(r["seq_omc_release"]), hold)
                        row[f"{which}|{fam}|{p}|{hold}|{tag}"] = k
                        if tag == "omc":
                            c = ch["omc"]
                            row[f"{which}|{fam}|{p}|{hold}|spd"] = (
                                c["v"][int(k)] / c["pk"] * 100
                                if np.isfinite(k) and 0 <= int(k) < len(c["v"]) and c["pk"] > 0
                                else np.nan)
        rec.append(row)
    D = pd.DataFrame(rec)

    f2 = lambda x: x / FPS * 1000
    out = []
    for which, variants, amqc, so, sm in (
            ("onset", ONSET, "amq_onset", "ship_onset_o", "ship_onset_m"),
            ("settle", SETTLE, "amq_settle", "ship_settle_o", "ship_settle_m")):
        sub = D if which == "onset" else D[D.obs == True]           # noqa: E712
        cd = (sub[so] - sub[sm]).abs().dropna()
        sd = (sub[so] - sub[amqc]).dropna()
        out.append(dict(boundary=which, family="SHIPPED", param=np.nan, hold=9, n=len(sub),
                        mean_ms=f2(cd.mean()), median_ms=f2(cd.median()), p90_ms=f2(cd.quantile(.9)),
                        frac_gt_4fr=100 * float((cd > 4).mean()),
                        absdiff_amq_ms=f2(sd.abs().median()),
                        resid_iqr_ms=f2(np.subtract(*np.percentile(sd, [75, 25]))),
                        spd_pct=47.0 if which == "onset" else 52.0, nofire_pct=0.0))
        for hold in HOLDS:
            for fam, p in variants:
                o = sub[f"{which}|{fam}|{p}|{hold}|omc"]
                mm = sub[f"{which}|{fam}|{p}|{hold}|mmc"]
                cd = (o - mm).abs().dropna()
                sd = (o - sub[amqc]).dropna()
                if len(cd) < 20:
                    continue
                out.append(dict(boundary=which, family=fam, param=p, hold=hold, n=len(cd),
                                mean_ms=f2(cd.mean()), median_ms=f2(cd.median()),
                                p90_ms=f2(cd.quantile(.9)),
                                frac_gt_4fr=100 * float((cd > 4).mean()),
                                absdiff_amq_ms=f2(sd.abs().median()),
                                resid_iqr_ms=f2(np.subtract(*np.percentile(sd, [75, 25]))),
                                spd_pct=float(np.nanmedian(sub[f"{which}|{fam}|{p}|{hold}|spd"])),
                                nofire_pct=100 * float(o.isna().mean())))
    R = pd.DataFrame(out)
    # round the METRICS only -- rounding `param` collapses 0.005/0.01/0.015 to 0.0
    R[R.columns.difference(['boundary', 'family', 'param'])] = \
        R[R.columns.difference(['boundary', 'family', 'param'])].round(1)
    R.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(R)} rows)")
    for b in ("onset", "settle"):
        print(f"\n=== {b} (hold 9) ===")
        print(R[(R.boundary == b) & (R.hold == 9)]
              .to_string(index=False, columns=["family", "param", "mean_ms", "p90_ms",
                                               "frac_gt_4fr", "absdiff_amq_ms", "resid_iqr_ms",
                                               "spd_pct", "nofire_pct"]))


if __name__ == "__main__":
    main()
