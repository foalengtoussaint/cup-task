"""Two replacements for the two weakest measures, scored the same way as Table III.

INTERJOINT COORDINATION -> VECTOR CODING. A Pearson correlation between shoulder flexion and elbow
extension collapses a whole coordination pattern into one number, and in a cohort of largely intact
coordination the reference values sit in a narrow band near 1, so the correlation ACROSS trials has
almost no true variance to track (18 of 747 trials carry 94.5 % of it). Vector coding keeps the
pattern: the coupling angle gamma = atan2(d elbow, d flexion) per frame, summarised as a circular
mean, a variability (1 - R), and the four Needham pattern bins. `vcode` is reused verbatim from
scripts/archive/vector_coding.py so the definition is the one already used in this project.

MOVEMENT UNITS -> LOG DIMENSIONLESS JERK. Movement units is a COUNT: on this cohort the median is
1 and the whole distribution spans a handful of integers, so exact agreement and a correlation
measure very different things and the correlation is bounded by the granularity, not by the pose.
LDLJ is the same construct -- smoothness of the reaching speed profile -- on a continuous scale:

    LDLJ = -ln( (T^3 / v_peak^2) * integral (d2v/dt2)^2 dt )        Balasubramanian et al. 2015

Both are computed from the poses and windows the paper already uses, so nothing is re-tracked and
nothing is refitted. The optical side carries the same anat12 landmark offsets as Table III, and the
trials the segmenter declines are excluded, so every number here is comparable with Table III.

    python paper/scripts/alt_measures.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "archive"))

import compare_pose_omc_delta as H              # noqa: E402
import results_v3_delta as R                    # noqa: E402
import score_vs_automq as S                     # noqa: E402
import score_own_phases as SOP                  # noqa: E402
from seg_sequential import segment_sequential   # noqa: E402
from vector_coding import vcode                 # noqa: E402  (the project's own definition)
from fps_ablation import declined               # noqa: E402

SEG = ROOT / "cache" / (__import__("os").environ.get("OT_SEG_INPUTS_DIR") or "seg_inputs_ship")
GRID = S.GRID_JOINTS
FPS = S.FPS

VC_KEYS = ["coup_mean", "coord_var", "in_phase", "anti_phase", "proximal", "distal"]


def ldlj(speed, fps):
    """Log dimensionless jerk of a speed profile (Balasubramanian et al. 2015, velocity form)."""
    v = np.asarray(speed, float)
    v = v[np.isfinite(v)]
    if len(v) < 12:
        return np.nan
    dt = 1.0 / fps
    T = len(v) * dt
    vpk = float(np.max(v))
    if vpk <= 1e-9:
        return np.nan
    d2 = np.gradient(np.gradient(v, dt), dt)
    integ = float(np.sum(d2 ** 2) * dt)
    dlj = (T ** 3 / vpk ** 2) * integ
    return -np.log(dlj) if dlj > 0 else np.nan


def vc_bins_shifted(flex, elb, win, shift_deg):
    """The four Needham bins with every edge rotated by `shift_deg`, plus how many frames sit within
    5 deg of an edge. The bins are hard thresholds on a per-frame angle, so a measure that only
    agrees because both systems bin the same noisy frames the same way is not measuring coordination.
    """
    if not win or win[1] - win[0] < 12:
        return None
    s, e = win
    mg = int(0.1 * (e - s))
    f, l = np.asarray(flex, float)[s + mg:e - mg], np.asarray(elb, float)[s + mg:e - mg]
    df, dl = np.diff(f), np.diff(l)
    ok = np.isfinite(df) & np.isfinite(dl) & ((df ** 2 + dl ** 2) > 1e-6)
    if ok.sum() < 8:
        return None
    g = (np.degrees(np.arctan2(dl[ok], df[ok])) - shift_deg) % 360
    edges = np.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
    near = float(np.mean(np.min(np.abs(((g[:, None] - edges + 180) % 360) - 180), axis=1) < 5.0))
    frac = lambda m: float(np.mean(m))
    return dict(
        prox=frac((g < 22.5) | (g >= 337.5) | ((g >= 157.5) & (g < 202.5))),
        dist=frac(((g >= 67.5) & (g < 112.5)) | ((g >= 247.5) & (g < 292.5))),
        near_edge=near)


def vc_continuous(flex, elb, win):
    """Threshold-FREE versions of the coordination axes the Needham bins carve up.

    The bins split the coupling angle gamma at 22.5 deg boundaries; a quarter of frames sit within
    5 deg of one, so bin membership is partly a coin flip and rotating the edges by +/-5 deg swings
    the agreement by 0.26. The same two axes exist as continuous functions of gamma, with no edges:

        dom   = mean cos 2*gamma    +1 = purely proximal (shoulder moves, elbow does not)
                                    -1 = purely distal,  0 = in- or anti-phase
        phase = mean sin 2*gamma    +1 = in-phase,  -1 = anti-phase

    Each is also computed weighted by the frame's vector magnitude sqrt(dflex^2 + delb^2), which
    demotes the near-stationary frames whose angle is least determined -- the ones the hard bins
    treat exactly like a large, well-determined excursion.
    """
    if not win or win[1] - win[0] < 12:
        return None
    s, e = win
    mg = int(0.1 * (e - s))
    f, l = np.asarray(flex, float)[s + mg:e - mg], np.asarray(elb, float)[s + mg:e - mg]
    df, dl = np.diff(f), np.diff(l)
    ok = np.isfinite(df) & np.isfinite(dl) & ((df ** 2 + dl ** 2) > 1e-6)
    if ok.sum() < 8:
        return None
    g = np.arctan2(dl[ok], df[ok])
    w = np.hypot(df[ok], dl[ok])
    ws = w.sum()
    c2, s2 = np.cos(2 * g), np.sin(2 * g)
    # RECTIFIED, one-sided versions. cos 2*gamma is an AXIS: shoulder-dominant frames are positive,
    # elbow-dominant negative, and a mean over both cancels a participant who drifts proximal against
    # one who drifts distal. If the pathology of interest is specifically excess proximal (shoulder)
    # compensation, only the positive part carries it -- and unlike the Needham proximal bin, this is
    # continuous, so no frame is decided by which side of a 22.5 deg line it fell on.
    pos, neg = np.maximum(c2, 0.0), np.maximum(-c2, 0.0)
    # POOLED forms. cos 2*gamma is itself a per-frame RATIO, so averaging it lets a frame in which
    # neither joint moved contribute a full-magnitude +-1. Aggregating the magnitudes first and
    # dividing once removes that without an explicit weight or a rectifier:
    #   share  = sum|dflex| / (sum|dflex| + sum|delb|)     share of angular TRAVEL at the shoulder
    #   pooled = sum(dflex^2 - delb^2) / sum(dflex^2 + delb^2)   the same in energy terms
    # `share` is one-sided on [0,1] with 0.5 = equal contribution; `pooled` keeps the signed axis but
    # weights each frame by how much actually happened in it.
    af, al = np.abs(df[ok]), np.abs(dl[ok])
    sf, sl = float(af.sum()), float(al.sum())
    q2f, q2l = float((df[ok] ** 2).sum()), float((dl[ok] ** 2).sum())
    return dict(dom=float(np.mean(c2)), phase=float(np.mean(s2)),
                dom_w=float(np.sum(w * c2) / ws) if ws > 0 else np.nan,
                phase_w=float(np.sum(w * s2) / ws) if ws > 0 else np.nan,
                prox=float(np.mean(pos)), dist=float(np.mean(neg)),
                prox_w=float(np.sum(w * pos) / ws) if ws > 0 else np.nan,
                dist_w=float(np.sum(w * neg) / ws) if ws > 0 else np.nan,
                share=(sf / (sf + sl)) if (sf + sl) > 0 else np.nan,
                pooled=((q2f - q2l) / (q2f + q2l)) if (q2f + q2l) > 0 else np.nan)


def alt_for(pose, ph, side, other):
    """Vector coding + LDLJ for one pose under one set of windows."""
    out = {}
    reach = S._win(ph, "reaching")
    try:
        elb = S._elbow_series(pose, side)
        flex, _abd = S._planar_body_angles(pose, side, other)
        vc = vcode(np.asarray(flex, float), np.asarray(elb, float), reach)
        if vc:
            out.update({f"vc_{k}": v for k, v in vc.items()})
        cn = vc_continuous(flex, elb, reach)
        if cn:
            out.update({f"vcc_{k}": v for k, v in cn.items()})
        for sh in (-5.0, 5.0):
            b = vc_bins_shifted(flex, elb, reach, sh)
            if b:
                tag = f"m5" if sh < 0 else "p5"
                out[f"vcs{tag}_proximal"] = b["prox"]
                out[f"vcs{tag}_distal"] = b["dist"]
                out["vc_near_edge"] = b["near_edge"]
    except Exception:
        pass
    try:
        # same operator as peak_velocity_reduce: differentiate the raw wrist, low-pass the SPEED
        sp = S._butter_lowpass(S._hand_speed_mmps(pose[f"{side}_wrist"], FPS), FPS, 6.0,
                               S.DEFAULT_BUTTER_ORDER)
        if reach and reach[1] > reach[0]:
            sp = sp[reach[0]:reach[1]]
        out["ldlj"] = ldlj(sp, FPS)
    except Exception:
        pass
    return out


def run(limit=0, anat12=None):
    H.use_good_cams()
    import gnn_train as GT
    theta = {}
    if anat12:
        t = pd.read_csv(anat12)
        cols = [f"th{i}" for i in range(12)]
        theta = {(r["part"], r["arm"]): np.array([r[c] for c in cols], float)
                 for _, r in t.iterrows() if not str(r["part"]).startswith("LOPO_")}
        print(f"anat12: {len(theta)} offsets from {anat12}", flush=True)
    ba = R._ba_traj_cache()
    bad = declined()
    trials = {f"{t['part']}/{t['trial']}": t for t in GT.load_clean(need_reproj=False)}
    files = sorted(SEG.glob("*.npz"))
    if limit:
        files = files[::max(1, len(files) // limit)][:limit]
    print(f"alt_measures: {len(files)} trials, {len(bad)} declined excluded", flush=True)

    rows, t0, n_skip = [], time.time(), 0
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        part, trial, side = str(z["part"]), str(z["trial"]), str(z["side"])
        if (part, trial) in bad:
            continue
        t = trials.get(f"{part}/{trial}")
        if t is None:
            n_skip += 1; continue
        other = "right" if side == "left" else "left"
        n = int(z["n"])
        try:
            ph_omc = segment_sequential(z["cup_omc"], z["wrist_omc"], z["nose_omc"])
            ph_mmc = segment_sequential(SOP._ship_cup(part, trial, z["cup_mmc"]),
                                        z["wrist_mmc"], z["nose_mmc"])
            omc_raw = H._load_omc(part, trial, n)
            lag, _, _ = H.find_lag_best({j: t["mmc"][:, k] for k, j in enumerate(GRID)},
                                        omc_raw, side)
            omc_pose = {j: R._shift(omc_raw[j], lag) for j in omc_raw}
            if theta:
                th = theta.get((part, str(z["arm"])))
                if th is None:
                    n_skip += 1; continue
                omc_pose = SOP._apply_anat12(omc_pose, side, other, th)
            mmc_pose = S._pose_variant_cached(t, "BA", "smoothnet", ba)
        except Exception:
            n_skip += 1; continue
        if mmc_pose is None or not ph_omc or not ph_mmc:
            n_skip += 1; continue

        # incumbents, from the operators the paper uses
        inc_o = SOP._measures(omc_pose, ph_omc, side, "omc")
        inc_m = SOP._measures(mmc_pose, ph_omc, side, "mmc")
        inc_mm = SOP._measures(mmc_pose, ph_mmc, side, "mmc")
        # alternatives
        a_o = alt_for(omc_pose, ph_omc, side, other)
        a_m = alt_for(mmc_pose, ph_omc, side, other)
        a_mm = alt_for(mmc_pose, ph_mmc, side, other)

        rec = dict(part=part, trial=trial, arm=str(z["arm"]))
        for k in ("interjoint_coordination", "number_of_movement_units"):
            rec[f"{k}__omc"] = inc_o.get(k, np.nan)
            rec[f"{k}__mmc"] = inc_m.get(k, np.nan)
            rec[f"{k}__mmcwin"] = inc_mm.get(k, np.nan)
        for k in ([f"vc_{v}" for v in VC_KEYS] + ["vcc_dom", "vcc_phase", "vcc_dom_w",
              "vcc_phase_w", "vcc_prox", "vcc_prox_w", "vcc_dist", "vcc_dist_w",
              "vcc_share", "vcc_pooled",
              "ldlj", "vc_near_edge",
              "vcsm5_proximal", "vcsp5_proximal", "vcsm5_distal", "vcsp5_distal"]):
            rec[f"{k}__omc"] = a_o.get(k, np.nan)
            rec[f"{k}__mmc"] = a_m.get(k, np.nan)
            rec[f"{k}__mmcwin"] = a_mm.get(k, np.nan)
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:5.0f}s  kept {len(rows)} "
                  f"skip {n_skip}", flush=True)
    return pd.DataFrame(rows), n_skip


def agree(D, key, circular=False):
    """r_s, r_av, n for one measure under optical windows, plus the markerless-window r_s."""
    g = D.dropna(subset=[f"{key}__omc", f"{key}__mmc"])
    if len(g) < 10:
        return None
    a, b = g[f"{key}__omc"].values, g[f"{key}__mmc"].values
    if circular:
        # circular-circular correlation (Jammalamadaka-SenGupta) for the coupling angle
        ar, br = np.radians(a), np.radians(b)
        am = np.arctan2(np.mean(np.sin(ar)), np.mean(np.cos(ar)))
        bm = np.arctan2(np.mean(np.sin(br)), np.mean(np.cos(br)))
        num = np.sum(np.sin(ar - am) * np.sin(br - bm))
        den = np.sqrt(np.sum(np.sin(ar - am) ** 2) * np.sum(np.sin(br - bm) ** 2))
        r_s = float(num / den) if den > 0 else np.nan
        d = np.degrees(np.abs(np.angle(np.exp(1j * (ar - br)))))
        extra = f"median |diff| {np.median(d):.1f} deg"
        r_av = np.nan
    else:
        r_s = pearsonr(a, b)[0]
        av = g.groupby(["part", "arm"])[[f"{key}__omc", f"{key}__mmc"]].mean()
        r_av = pearsonr(av.iloc[:, 0], av.iloc[:, 1])[0] if len(av) >= 3 else np.nan
        extra = ""
    gw = D.dropna(subset=[f"{key}__omc", f"{key}__mmcwin"])
    if circular or len(gw) < 10:
        r_e2e = np.nan
    else:
        r_e2e = pearsonr(gw[f"{key}__omc"], gw[f"{key}__mmcwin"])[0]
    return dict(measure=key, r_s=r_s, r_av=r_av, r_s_mmcwin=r_e2e, n=len(g), note=extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--anat12", default=str(ROOT / "out/scoring/anat12_wv1wa1_theta.csv"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "table8_alt_measures"))
    a = ap.parse_args(argv)

    D, n_skip = run(a.limit, a.anat12 or None)
    if D.empty:
        print("nothing scored"); return
    D.to_csv(f"{a.out}_pertrial.csv", index=False)
    print(f"\nPROCESSING CHECK: {len(D)} trials, {n_skip} skipped")

    order = [("vcc_share", False), ("vcc_pooled", False),
             ("vcc_prox", False), ("vcc_prox_w", False), ("vcc_dist", False),
             ("vcc_dist_w", False),
             ("vcc_dom", False), ("vcc_dom_w", False), ("vcc_phase", False),
             ("vcc_phase_w", False),
             ("vcsm5_proximal", False), ("vcsp5_proximal", False),
             ("vcsm5_distal", False), ("vcsp5_distal", False),
             ("interjoint_coordination", False), ("vc_coup_mean", True),
             ("vc_coord_var", False), ("vc_in_phase", False), ("vc_anti_phase", False),
             ("vc_proximal", False), ("vc_distal", False),
             ("number_of_movement_units", False), ("ldlj", False)]
    recs = [r for r in (agree(D, k, c) for k, c in order) if r]
    T = pd.DataFrame(recs)
    T.to_csv(f"{a.out}.csv", index=False)
    print(f"\n{'measure':28}{'r_s':>8}{'r_av':>8}{'r_s (mmc win)':>15}{'n':>6}  note")
    for r in recs:
        print(f"{r['measure']:28}{r['r_s']:>8.3f}{r['r_av']:>8.3f}"
              f"{r['r_s_mmcwin']:>15.3f}{r['n']:>6}  {r['note']}")

    # ---- is movement units limited by being a COUNT? ----
    g = D.dropna(subset=["number_of_movement_units__omc", "number_of_movement_units__mmc"])
    o = g["number_of_movement_units__omc"].values
    m = g["number_of_movement_units__mmc"].values
    vals, cnt = np.unique(o, return_counts=True)
    print(f"\nMOVEMENT UNITS is a count: optical values "
          f"{ {int(v): int(c) for v, c in zip(vals, cnt)} }")
    print(f"  exact agreement {100*np.mean(o == m):.1f}%   within 1 {100*np.mean(np.abs(o-m) <= 1):.1f}%"
          f"   median |diff| {np.median(np.abs(o-m)):.1f}")
    print(f"  Pearson {pearsonr(o, m)[0]:.3f}   Spearman {spearmanr(o, m)[0]:.3f}"
          f"   ties in the optical column: {100*(1 - len(vals)/len(o)):.1f}%")
    # correlation ceiling: perfect agreement is 1.0, so quantify how much ONE trial's disagreement
    # costs on a distribution this concentrated
    worst = np.argsort(-np.abs(o - m))[:5]
    print(f"  the five largest disagreements: "
          + ", ".join(f"{int(o[i])}v{int(m[i])}" for i in worst))
    ld = D.dropna(subset=["ldlj__omc", "ldlj__mmc"])
    print(f"\nLDLJ spread (optical): median {ld.ldlj__omc.median():.2f}, "
          f"IQR {ld.ldlj__omc.quantile(.25):.2f}-{ld.ldlj__omc.quantile(.75):.2f}, "
          f"{len(np.unique(np.round(ld.ldlj__omc, 3)))} distinct values in {len(ld)} trials")
    print(f"\nwrote {a.out}.csv and _pertrial.csv")
    print("DONE_ALT", flush=True)


if __name__ == "__main__":
    main()
