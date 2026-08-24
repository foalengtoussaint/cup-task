"""THE clinical test: can the mocap-free pipeline detect the AFFECTED vs UNAFFECTED difference?

Everything before this asked "how close is MMC to OMC" (accuracy). This asks the only question a
clinician cares about: **does MMC see the same impairment signal mocap sees?** A measure can carry
a large constant bias and still be perfectly useful if the bias cancels when you compare two arms.

For each measure we compute, on the SAME trials, both sides:
    OMC effect = mean_affected - mean_unaffected   (mocap: the true clinical signal)
    MMC effect = mean_affected - mean_unaffected   (ours)
and ask whether MMC recovers OMC's effect -- in sign, in size, and relative to its own noise.

CAVEAT, stated loudly: in DELTA the affected arm is the LEFT and the unaffected is the RIGHT, so
this compares ACROSS SIDES. The COCO-vs-marker landmark offsets are asymmetric (right_hip 118mm
error vs left_hip 190mm), so left/right offsets do NOT cancel the way a within-side pre/post
comparison would. This is a harder test than the clinical use-case it stands in for.

    python scripts/affected_vs_unaffected_delta.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import ezc3d
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import segment
from pipeline.score import compute_position_measures
from scripts.compare_pose_omc_delta import (_load_mmc, _despike, _resample, _lp, _find_lag,
                                            VIDEO_FPS, C3D_RATE, DELTA)
from scripts.score_omc_delta import _shift, _bonelock_pbd

UNAFF = [f"trial_{i}_R_unaffected" for i in [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
AFF = [f"trial_{i}_L_affected" for i in [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]]
OMC_MAP = {"shoulder": "shoulder_{S}", "elbow": "elbow_{S}", "hip": "hip_{S}",
           "shoulder_o": "shoulder_{O}", "hip_o": "hip_{O}", "head": "head"}


def _omc(trial, side):
    c = ezc3d.c3d(str(DELTA / "P14" / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]
    S = "R" if side == "right" else "L"
    O = "L" if side == "right" else "R"

    def mk(n):
        return P[:3, L.index(n), :].T
    out = {k: mk(v.format(S=S, O=O)) for k, v in OMC_MAP.items()}
    out["wrist"] = np.mean([mk(f"wrist_inner_{S}"), mk(f"wrist_outer_{S}")], axis=0)
    out["cup"] = np.mean([mk(f"cluster_cup_{i}") for i in (1, 2, 3, 4)], axis=0)
    return {k: _despike(_resample(v, C3D_RATE, VIDEO_FPS)) for k, v in out.items()}


def _chain(J):
    """v1 arm bone-lock: hold upperarm/forearm at median length along observed directions."""
    sh, el, wr = J["shoulder"], J["elbow"], J["wrist"]
    Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
    Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
    du = (el - sh) / (np.linalg.norm(el - sh, axis=1, keepdims=True) + 1e-9)
    el2 = sh + du * Lu
    df = (wr - el2) / (np.linalg.norm(wr - el2, axis=1, keepdims=True) + 1e-9)
    J["elbow"], J["wrist"] = el2, el2 + df * Lf
    return J


def _mmc(trial, side, variant="bonelock"):
    """variant: raw | bonelock (v1 chain) | pbd (whole skeleton) | hybrid (PBD torso + chain arm)"""
    M, n = _load_mmc("P14", trial)
    S = side
    O = "left" if side == "right" else "right"
    J = {"shoulder": M[f"{S}_shoulder"], "elbow": M[f"{S}_elbow"], "wrist": M[f"{S}_wrist"],
         "shoulder_o": M[f"{O}_shoulder"], "hip": M[f"{S}_hip"], "hip_o": M[f"{O}_hip"],
         "head": M["nose"]}
    if variant == "raw":
        return J, n
    if variant == "bonelock":
        return _chain(J), n
    TORSO = [("shoulder", "shoulder_o"), ("hip", "hip_o"),
             ("shoulder", "hip"), ("shoulder_o", "hip_o")]
    if variant == "pbd":       # whole skeleton, distributed
        J = _bonelock_pbd(J, bones=TORSO + [("shoulder", "elbow"), ("elbow", "wrist")], iters=8)
        return J, n
    if variant == "hybrid":    # PBD torso + chain arm
        J = _bonelock_pbd(J, bones=TORSO, iters=8)
        return _chain(J), n
    raise ValueError(variant)


def _fill(x):
    x = np.asarray(x, float).copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    return x


def _measures(J, phases, side):
    trunk = (J["shoulder"] + J["shoulder_o"]) / 2
    m = compute_position_measures(_fill(J["wrist"]), _fill(trunk), phases, side,
                                  fps=VIDEO_FPS).to_dict()
    m.pop("side", None)
    # angles
    hip_mid = (J["hip"] + J["hip_o"]) / 2
    down = hip_mid - trunk
    down = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    arm = J["elbow"] - J["shoulder"]
    arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    flex = _lp(np.degrees(np.arccos(np.clip((arm * down).sum(1), -1, 1))))
    u, v = J["shoulder"] - J["elbow"], J["wrist"] - J["elbow"]
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    elb = _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))
    eav = np.abs(np.gradient(elb)) * VIDEO_FPS

    def ph(n):
        for nm, s, e in phases:
            if nm == n:
                return s, e
        return None
    r, fw = ph("reaching"), ph("forward_transport")
    rf = (r[0], fw[1]) if (r and fw) else r
    m["elbow_extension_reaching"] = float(np.nanmax(elb[rf[0]:rf[1]])) if rf else np.nan
    m["shoulder_flexion_reaching"] = float(np.nanmax(flex[r[0]:r[1]])) if r else np.nan
    m["peak_elbow_ang_vel"] = float(np.nanmax(eav[rf[0]:rf[1]])) if rf else np.nan
    return m


def _one(trial, side, src, variant="bonelock"):
    """src='omc'|'mmc'. Phases always from the OMC cup (isolates pose from cup detection)."""
    O = _omc(trial, side)
    if src == "omc":
        J, cup = O, O["cup"]
    else:
        J, n = _mmc(trial, side, variant)
        lag, _ = _find_lag(J["wrist"], O["wrist"][:n] if len(O["wrist"]) >= n else O["wrist"])
        Osh = {k: _shift(v[:n] if len(v) >= n else np.vstack(
            [v, np.full((n - len(v), 3), np.nan)]), lag) for k, v in O.items()}
        cup = Osh["cup"]
    T = min(len(J["wrist"]), len(cup))
    J = {k: v[:T] for k, v in J.items()}
    cup = cup[:T]
    cr = _lp(np.linalg.norm(cup - np.nanmedian(cup[:30], 0), axis=1))
    wd = _lp(np.linalg.norm(J["wrist"] - np.nanmedian(J["wrist"][:30], 0), axis=1))
    lift = np.flatnonzero(cr > 150)
    if not len(lift):
        return None
    both = np.flatnonzero((np.arange(T) > lift[0]) & (cr < 60) & (wd < 60))
    if not len(both):
        return None
    cut = min(both[0] + 20, T)
    J = {k: v[:cut] for k, v in J.items()}
    cup = cup[:cut]
    seg = segment.segment_cup_only(cup, fps=VIDEO_FPS)
    seg = segment.refine_grasp_with_pose(seg, cup, J["wrist"], J["head"], fps=VIDEO_FPS)
    phases = segment.to_murphy_phases(seg, J["wrist"], cup, fps=VIDEO_FPS)
    if not any(n == "reaching" for n, _, _ in phases):
        return None
    return _measures(J, phases, side)


def main():
    data = {("omc", "unaff"): [], ("omc", "aff"): [], ("mmc", "unaff"): [], ("mmc", "aff"): []}
    for src in ("omc", "mmc"):
        for cond, trials, side in [("unaff", UNAFF, "right"), ("aff", AFF, "left")]:
            for t in trials:
                if src == "mmc" and not list((DELTA / "P14" / "dets").glob(f"*{t}*.2.pose.json")):
                    continue
                try:
                    r = _one(t, side, src)
                    if r:
                        data[(src, cond)].append(r)
                except Exception as e:
                    print(f"  skip {src} {t}: {e}", flush=True)
            print(f"{src} {cond}: n={len(data[(src, cond)])}", flush=True)

    keys = ["total_movement_time", "peak_velocity", "number_of_movement_units",
            "max_trunk_displacement", "elbow_extension_reaching",
            "shoulder_flexion_reaching", "peak_elbow_ang_vel"]
    print(f"\n=== CAN MMC SEE THE IMPAIRMENT? (affected - unaffected) ===\n")
    print(f"{'measure':28}{'OMC effect':>22}{'MMC effect':>22}   verdict")
    print("-" * 96)
    for k in keys:
        def stat(src, cond):
            v = np.array([r[k] for r in data[(src, cond)] if np.isfinite(r.get(k, np.nan))])
            return v
        ou, oa = stat("omc", "unaff"), stat("omc", "aff")
        mu, ma = stat("mmc", "unaff"), stat("mmc", "aff")
        if min(len(ou), len(oa), len(mu), len(ma)) < 3:
            continue
        oe, me = oa.mean() - ou.mean(), ma.mean() - mu.mean()
        osd = np.sqrt((ou.var(ddof=1) + oa.var(ddof=1)) / 2)
        msd = np.sqrt((mu.var(ddof=1) + ma.var(ddof=1)) / 2)
        od = oe / osd if osd > 1e-9 else np.nan
        md = me / msd if msd > 1e-9 else np.nan
        agree = "SIGN FLIP" if oe * me < 0 else ("recovers" if abs(md) > 0.8 * abs(od) * 0.5
                                                 else "weak")
        print(f"{k:28}{oe:+11.2f} (d={od:+5.2f}){me:+11.2f} (d={md:+5.2f})   {agree}")
    print("\nd = Cohen's d (effect / pooled SD). OMC d is the ceiling: what mocap can detect.")
    print("MMC d is what we detect. SIGN FLIP = we report the impairment BACKWARDS.")
    print("CAVEAT: affected=LEFT vs unaffected=RIGHT, so left/right landmark asymmetry does NOT")
    print("cancel here -- a harder test than the within-side pre/post use-case.")


if __name__ == "__main__":
    main()
