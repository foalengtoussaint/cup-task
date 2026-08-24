"""Vector coding of shoulder-flexion <-> elbow coordination (instead of a single Pearson interjoint).

Coupling angle per frame: gamma_i = atan2(d elbow, d flexion) in the angle-angle plane. Summarise each
trial's reaching window by:
  - mean coupling angle (circular mean)
  - coordination VARIABILITY  = 1 - R  (R = resultant length; high = inconsistent coupling)
  - coordination PATTERN bins (Needham 2014): in-phase / anti-phase / proximal(shoulder) / distal(elbow)
Then test: (a) does it separate AFFECTED vs UNAFFECTED better than Pearson interjoint? (b) do MMC and OMC
AGREE on it better than on Pearson interjoint? Pose held on OMC phases (paper-style, isolates pose).
"""
from __future__ import annotations
import sys, re, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, mannwhitneyu
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import score_vs_automq as S
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from seg_sequential import segment_sequential
GRID = S.GRID_JOINTS; FPS = S.FPS
JN = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
      "right_wrist", "left_wrist", "right_hip", "left_hip", "nose"]


def _series(pose, side):
    other = "right" if side == "left" else "left"
    elb = S._elbow_series(pose, side)
    try:
        flex, _ = S._planar_body_angles(pose, side, other)
    except Exception:
        flex = np.full(len(elb), np.nan)
    return flex, elb


def _circ_mean(a_deg):
    r = np.radians(a_deg)
    return np.degrees(np.arctan2(np.nanmean(np.sin(r)), np.nanmean(np.cos(r)))) % 360


def vcode(flex, elb, win):
    """Vector-coding summary over [win[0],win[1]] (inner 80%)."""
    if not win or win[1] - win[0] < 12:
        return None
    s, e = win; mg = int(0.1 * (e - s)); s, e = s + mg, e - mg
    f, l = flex[s:e], elb[s:e]
    df, dl = np.diff(f), np.diff(l)
    ok = np.isfinite(df) & np.isfinite(dl) & ((df ** 2 + dl ** 2) > 1e-6)
    if ok.sum() < 8:
        return None
    g = np.degrees(np.arctan2(dl[ok], df[ok])) % 360      # coupling angle per frame
    r = np.radians(g)
    Rlen = np.hypot(np.nanmean(np.cos(r)), np.nanmean(np.sin(r)))
    # 4-bin coordination pattern (Needham 2014)
    def frac(mask):
        return float(np.mean(mask))
    inph = frac(((g >= 22.5) & (g < 67.5)) | ((g >= 202.5) & (g < 247.5)))
    anti = frac(((g >= 112.5) & (g < 157.5)) | ((g >= 292.5) & (g < 337.5)))
    prox = frac((g < 22.5) | (g >= 337.5) | ((g >= 157.5) & (g < 202.5)))
    dist = frac(((g >= 67.5) & (g < 112.5)) | ((g >= 247.5) & (g < 292.5)))
    return dict(coup_mean=_circ_mean(g), coord_var=1 - Rlen,
                in_phase=inph, anti_phase=anti, proximal=prox, distal=dist)


def main():
    S.H.use_good_cams(); amq = S.load_automq(); ba = R._ba_traj_cache()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in S.COHORT_PARTS]
    pat = re.compile(r"trial_(\d+)_([RL])_")
    rows = []; t0 = time.time(); n = 0
    for t in trials:
        part, trial, side = t["part"], t["trial"], t["side"]
        m = pat.search(trial)
        if not m:
            continue
        rec = amq.get((S.automq_part(part), int(m.group(1)), m.group(2)))
        if rec is None or rec.get("phases") is None:
            continue
        nfr = t["mmc"].shape[0]
        pose = S._pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        omc = H._load_omc(part, trial, nfr); wr = f"{side}_wrist"
        if wr not in omc or not np.isfinite(omc[wr]).any() or not all(j in omc for j in JN):
            continue
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
        ocup = R._omc_cup(part, trial, nfr)
        if not np.isfinite(ocup).any():
            continue
        pose_omc = {j: R._shift(omc[j], lag) for j in JN}
        seq_omc = {nm: (s, e) for nm, s, e in segment_sequential(
            R._shift(ocup, lag), pose_omc[wr], pose_omc["nose"])}
        win = seq_omc.get("reaching")
        if win is None:
            continue
        arm = "affected" if rec.get("condition") == "affected" else "unaffected"
        fm, em = _series(pose, side); fo, eo = _series(pose_omc, side)
        vm, vo = vcode(fm, em, win), vcode(fo, eo, win)
        # pearson interjoint for reference (same window)
        def pear(f, l):
            s, e = win; mg = int(0.1 * (e - s)); a, b = f[s + mg:e - mg], l[s + mg:e - mg]
            k = np.isfinite(a) & np.isfinite(b)
            return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 10 and np.std(a[k]) > 1e-6 else np.nan
        if vm and vo:
            row = dict(part=part, trial=trial, arm=arm, pear_mmc=pear(fm, em), pear_omc=pear(fo, eo))
            for k, v in vm.items():
                row[f"{k}_mmc"] = v
            for k, v in vo.items():
                row[f"{k}_omc"] = v
            rows.append(row)
        n += 1
        if n % 100 == 0:
            print(f"[{n}] {time.time()-t0:4.0f}s rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(ROOT / "out/scoring/vector_coding.csv", index=False)
    print(f"\nPROCESSING CHECK: trials {n}, rows {len(df)}", flush=True)

    METS = ["coord_var", "distal", "proximal", "in_phase", "anti_phase", "pear"]
    print("\n== AFFECTED vs UNAFFECTED (OMC truth), and MWU effect ==")
    print(f"{'metric':<12}{'unaff med':>11}{'aff med':>10}{'MWU p':>9}{'rank-biserial':>15}")
    for me in METS:
        col = "pear_omc" if me == "pear" else f"{me}_omc"
        u = df[df.arm == "unaffected"][col].dropna(); a = df[df.arm == "affected"][col].dropna()
        try:
            U, p = mannwhitneyu(u, a); rb = 1 - 2 * U / (len(u) * len(a))
        except Exception:
            p, rb = np.nan, np.nan
        print(f"{me:<12}{u.median():>11.3f}{a.median():>10.3f}{p:>9.1e}{rb:>15.2f}")

    print("\n== MMC-vs-OMC AGREEMENT (Pearson r_av, participant x arm) ==")
    print(f"{'metric':<12}{'r_av':>8}{'r_s(trial)':>12}")
    for me in METS:
        cm = "pear_mmc" if me == "pear" else f"{me}_mmc"
        co = "pear_omc" if me == "pear" else f"{me}_omc"
        g = df[[cm, co, "part", "arm"]].dropna()
        rs = pearsonr(g[co], g[cm])[0] if len(g) > 3 else np.nan
        av = g.groupby(["part", "arm"])[[cm, co]].mean()
        rav = pearsonr(av[co], av[cm])[0] if len(av) >= 3 else np.nan
        print(f"{me:<12}{rav:>8.2f}{rs:>12.2f}")
    print("\nwrote out/scoring/vector_coding.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
