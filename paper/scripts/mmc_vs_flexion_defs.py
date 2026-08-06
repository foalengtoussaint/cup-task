"""Which flexion definition does our MMC actually match: AutoMQ's world-VERTICAL, or a TRUNK-referenced one?

Established (automq_flexion_ref_test.py, OMC-only): AutoMQ ships flexion = arm-vs-world-vertical, and a
trunk-referenced flexion (arm-vs shoulder->hip, using OMC's own hip markers) differs by ~6 deg and
diverges with trunk lean (r=+0.63). Our MMC flexion is hip/trunk-referenced. So: does MMC agree better
with OMC-VERTICAL or OMC-TRUNK? If MMC ~ OMC-trunk, our 'disagreement' with the shipped AutoMQ number is
largely the DEFINITION gap, and MMC computes the more anatomically-correct (trunk-relative) angle.

Per matched trial (BA+SmoothNet pose, reach->drink max):
  mmc_flex        : OUR shipped flexion (per-trial-const hip-down axis)          [from OUR pose]
  omc_vert        : arm(shoulder-elbow) vs global Z, from OMC markers            [AutoMQ shipped def]
  omc_trunk       : arm vs (shoulder_mid - hip_mid), from OMC markers            [trunk-referenced]
  lean            : OMC trunk tilt from vertical (deg)
Reports rs / bias / med|err| of MMC vs each OMC definition, and whether MMC's (mmc - omc_vert) residual
tracks trunk lean the way omc_trunk does. OMC read-only; scoring still uses our pose.
    python paper/scripts/mmc_vs_flexion_defs.py -> paper/mmc_vs_flexion_defs.png + prints
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from compare_pose_omc_delta import _murphy_signals
from score_vs_automq import (load_automq, automq_phases_to_video, _pose_variant_cached,
                             automq_part, _win, AUTOMQ)
GRID = R._GRID_JOINTS
Z = np.array([0.0, 0.0, 1.0])


def _ang(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def _marker(mk, name):
    try:
        return mk.xs(name, level=1)[["x", "y", "z"]].to_numpy()
    except (KeyError, TypeError):
        return None


def _maxseg(a, w):
    if not (w and w[1] > w[0]):
        return np.nan
    s = a[w[0]:w[1]]; s = s[np.isfinite(s)]
    return float(np.max(s)) if len(s) else np.nan


def main():
    H.use_good_cams(); ba = R._ba_traj_cache(); amq = load_automq()
    pat = re.compile(r"trial_(\d+)_([LR])_")
    mk_cache = {}
    rows = []
    for t in GT.load_clean(need_reproj=False):
        m = pat.search(t["trial"])
        if not m:
            continue
        p = t["part"]; tn = int(m.group(1)); sd = m.group(2)
        rec = amq.get((automq_part(p), tn, sd))
        if rec is None or rec.get("phases") is None:
            continue
        pose = _pose_variant_cached(t, "BA", "smoothnet", ba)
        if pose is None:
            continue
        side = t["side"]; other = "right" if side == "left" else "left"
        n = t["mmc"].shape[0]
        # ---- OUR MMC flexion (shipped recipe) ----
        try:
            sig = _murphy_signals(pose, side=side)
            mmc_flex_series = H._lp(sig["shoulder_flexion"])
        except (IndexError, ValueError):
            continue
        # ---- phase window on the MMC (60Hz) timeline for the MMC series ----
        lag, _ = H._find_lag(t["mmc"][:, GRID.index(f"{side}_wrist")], H._load_omc(p, t["trial"], n)[f"{side}_wrist"])
        ph = automq_phases_to_video(rec["phases"], lag, n)
        if not ph:
            continue
        reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
        rd_v = (reach[0], drink[1]) if (reach and drink) else (reach or drink)
        mmc_flex = _maxseg(mmc_flex_series, rd_v)                       # shipped: full-const axis

        # ---- MMC flexion AXIS VARIANTS (all from OUR pose): freeze the HIPS but let the SHOULDER move ----
        m_sh = pose[f"{side}_shoulder"]; m_el = pose[f"{side}_elbow"]
        m_shO = pose[f"{other}_shoulder"]
        m_shmid = (m_sh + m_shO) / 2.0
        m_hipmid = (pose["right_hip"] + pose["left_hip"]) / 2.0
        # arm = elbow - shoulder (points shoulder->elbow, downward-ish) to MATCH the shipped
        # _murphy_signals convention where down = hip - shoulder. (Earlier I used sh-el paired with a
        # hip-sh down axis -> mismatched directions -> ~180 deg garbage. Fixed.)
        m_arm = m_el - m_sh
        def _mmc_flex(down_vec):
            return H._lp(_ang(m_arm, down_vec))                          # down_vec: (T,3) per-frame or (1,3)
        # hip_mid frozen to per-trial median (occluded -> denoise); shoulder_mid LIVE per-frame
        hfin = np.isfinite(m_hipmid).all(1)
        hip_c = np.nanmedian(m_hipmid[hfin], 0) if hfin.any() else m_hipmid[0]
        # down axis = shoulder -> hip (matches shipped: hip_mid - shoulder_mid). Hips frozen (hip_c),
        # shoulder LIVE, so the axis TILTS as the shoulder translates during the reach.
        down_hipC_shLive = hip_c[None, :] - m_shmid                      # shoulder-mid live
        down_hipC_actLive = hip_c[None, :] - m_sh                        # acting-shoulder live
        mmc_flex_hipC_shLive = _maxseg(_mmc_flex(down_hipC_shLive), rd_v)
        mmc_flex_hipC_actLive = _maxseg(_mmc_flex(down_hipC_actLive), rd_v)
        # ---- OMC two definitions from AutoMQ markers (100Hz, native phase frames) ----
        if p not in mk_cache:
            mk_cache[p] = pd.read_pickle(AUTOMQ / automq_part(p) / "combined_data_with_kinematics.pkl")
        cdf = mk_cache[p]
        key = next((fk for fk in cdf.index if int(fk[1]) == tn and fk[2] == sd), None)
        if key is None:
            continue
        mk = cdf.loc[key, "markers"]
        sh = _marker(mk, f"shoulder_{sd}"); el = _marker(mk, f"elbow_{sd}")
        hL = _marker(mk, "hip_L"); hR = _marker(mk, "hip_R")
        shL = _marker(mk, "shoulder_L"); shR = _marker(mk, "shoulder_R")
        if any(x is None for x in (sh, el, hL, hR, shL, shR)):
            continue
        T = min(map(len, (sh, el, hL, hR, shL, shR)))
        sh, el, hL, hR, shL, shR = (a[:T] for a in (sh, el, hL, hR, shL, shR))
        arm = sh - el
        up_trunk = (shL + shR) / 2.0 - (hL + hR) / 2.0                 # PER-FRAME trunk up
        # PER-TRIAL CONSTANT trunk up (median over finite frames) -- IDENTICAL convention to our MMC
        # _murphy_signals (which freezes the down axis). This makes the OMC-trunk comparison apples-to-
        # apples: same operator both sides, so any residual gap is pose/keypoint, not the axis choice.
        un = up_trunk / (np.linalg.norm(up_trunk, axis=1, keepdims=True) + 1e-9)
        ufin = np.isfinite(un).all(1)
        up_const = np.nanmedian(un[ufin], 0) if ufin.any() else un[0]
        up_const = up_const / (np.linalg.norm(up_const) + 1e-9)
        omc_vert_s = _ang(arm, Z[None, :])
        omc_trunk_s = _ang(arm, up_trunk)                              # per-frame axis
        omc_trunkc_s = _ang(arm, up_const[None, :])                    # per-trial-CONSTANT axis
        lean_s = _ang(up_trunk, Z[None, :])
        w100 = None
        rr = rec["phases"].get("Reaching"); dd = rec["phases"].get("Drinking")
        if rr is not None and dd is not None:
            w100 = (int(rr[0]), int(dd[1]))
        rows.append({"part": p, "trial": tn, "side": sd, "mmc_flex": mmc_flex,
                     "mmc_hipC_shLive": mmc_flex_hipC_shLive, "mmc_hipC_actLive": mmc_flex_hipC_actLive,
                     "omc_vert": _maxseg(omc_vert_s, w100), "omc_trunk": _maxseg(omc_trunk_s, w100),
                     "omc_trunkc": _maxseg(omc_trunkc_s, w100),
                     "lean": float(np.nanmedian(lean_s[w100[0]:w100[1]])) if w100 else np.nan})

    df = pd.DataFrame(rows).dropna(subset=["mmc_flex", "mmc_hipC_shLive", "mmc_hipC_actLive",
                                           "omc_vert", "omc_trunk", "omc_trunkc"])
    df.to_csv(PAPER / "mmc_vs_flexion_defs.csv", index=False)
    print(f"\nn = {len(df)} matched trials\n")

    def agree(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return (spearmanr(a[m], b[m]).correlation, float(np.median(b[m] - a[m])), float(np.median(np.abs(b[m] - a[m]))))
    print(f"{'MMC flexion vs ...':34}{'rs':>7}{'bias':>8}{'med|err|':>10}")
    for lab, col in [("OMC world-VERTICAL (AutoMQ)", "omc_vert"),
                     ("OMC TRUNK-ref, PER-FRAME axis", "omc_trunk"),
                     ("OMC TRUNK-ref, PER-TRIAL CONST", "omc_trunkc")]:
        rs, bias, err = agree(df[col].values, df["mmc_flex"].values)
        print(f"{lab:34}{rs:>+7.2f}{bias:>+8.1f}{err:>10.1f}")

    # === AXIS VARIANT TEST: hips frozen, shoulder LIVE (axis tilts as shoulder moves) ===
    # scored vs the matched OMC target (trunk-ref per-frame). Does letting the shoulder move help?
    print(f"\n{'MMC AXIS VARIANT (vs OMC trunk per-frame)':42}{'rs':>7}{'bias':>8}{'med|err|':>10}")
    tgt = df["omc_trunk"].values
    for lab, col in [("full-const (SHIPPED)", "mmc_flex"),
                     ("hips-const, shoulder-MID live", "mmc_hipC_shLive"),
                     ("hips-const, ACTING-shoulder live", "mmc_hipC_actLive")]:
        rs, bias, err = agree(tgt, df[col].values)
        print(f"{lab:42}{rs:>+7.2f}{bias:>+8.1f}{err:>10.1f}")
    # how much does the OMC axis itself MOVE within a trial? (per-frame vs its own constant)
    dd = agree(df["omc_trunkc"].values, df["omc_trunk"].values)
    print(f"\n  OMC per-frame vs OMC per-trial-const (same markers): rs={dd[0]:+.2f} med|err|={dd[2]:.1f} deg")
    print("  (if ~0 diff, the trunk axis barely moves within a trial -> freezing it costs nothing;")
    print("   if large, dynamic lean matters and our constant axis can't follow it)")
    print("\n(if MMC agrees BETTER with TRUNK-referenced, our pipeline computes the trunk-relative angle,")
    print(" and the gap to AutoMQ's shipped number is the DEFINITION difference, not tracking error)")

    # does MMC's residual vs OMC-vertical track lean like the OMC definition gap does?
    res_v = df["mmc_flex"] - df["omc_vert"]
    mlean = np.isfinite(df["lean"]) & np.isfinite(res_v)
    r = pearsonr(df["lean"][mlean], res_v[mlean])[0]
    print(f"\n(MMC - OMC_vertical) vs trunk LEAN:  pearson r = {r:+.2f}  "
          f"(negative => MMC reads LOWER than vertical exactly when the patient leans = trunk-correcting)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    lo = 20; hi = 90
    for a, col, ttl in [(ax[0], "omc_vert", "MMC vs OMC world-vertical"),
                        (ax[1], "omc_trunk", "MMC vs OMC trunk-referenced")]:
        a.scatter(df[col], df["mmc_flex"], s=14, alpha=0.5, color="#c1440e")
        a.plot([lo, hi], [lo, hi], "0.55"); a.set_aspect("equal", "box")
        rs = spearmanr(df[col], df["mmc_flex"]).correlation
        a.set_xlabel(f"OMC flexion [{col.split('_')[1]}] (deg)"); a.set_ylabel("MMC flexion (deg)")
        a.set_title(f"{ttl}\n$r_s$={rs:.2f}"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    ax[2].scatter(df["lean"], res_v, s=14, alpha=0.5, color="#1f77b4")
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].set_xlabel("OMC trunk lean (deg)"); ax[2].set_ylabel("MMC - OMC_vertical flexion (deg)")
    ax[2].set_title(f"MMC trunk-corrects with lean (r={r:+.2f})")
    fig.suptitle("Which flexion definition does MMC match?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PAPER / "mmc_vs_flexion_defs.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {PAPER/'mmc_vs_flexion_defs.png'}\nDONE", flush=True)


if __name__ == "__main__":
    main()
