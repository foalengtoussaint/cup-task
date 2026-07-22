"""How big is a Murphy measure's CLINICAL signal? -- so we can judge whether our MMC error matters.

A raw Δ is meaningless on its own: 5° might be trivial for one measure and swamp another. The
yardstick that matters is **how much the measure moves between AFFECTED and UNAFFECTED arms** --
that IS the clinical signal the measure exists to detect. If our MMC error is small next to it, the
measure survives mocap-free; if comparable, it does not.

Computed from the OMC C3Ds ALONE (P14: 43 L_affected + 47 R_unaffected) -- no video, no
calibration, no sync needed, since everything is in the mocap frame. So this is pure ground truth:
mocap measuring mocap, with the only variable being which arm.

Reports per measure: affected vs unaffected mean, the effect (difference), the within-condition
SD, Cohen's d, and -- given an MMC error -- what fraction of the clinical effect that error is.

    python scripts/omc_effect_size_delta.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import ezc3d
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import segment
from cup_task.score import compute_position_measures
from scripts.compare_pose_omc_delta import _despike, _resample, _lp, VIDEO_FPS, C3D_RATE

C3D_DIR = Path(__file__).resolve().parents[1] / "cache" / "delta" / "P14" / "c3d_all"

# our measured MMC error (|Δ| vs OMC) on P14 sip 1, HYBRID variant -- the thing being judged
MMC_ERR = {
    "total_movement_time": 0.00, "peak_velocity": 1.28, "time_to_peak_velocity": 0.03,
    "number_of_movement_units": 1.0, "max_trunk_displacement": 4.8,
    "elbow_extension_reaching": 0.6, "shoulder_flexion_reaching": 3.30,
    "shoulder_flexion_drinking": 9.0, "shoulder_abduction_drinking": 6.7,
    "peak_elbow_ang_vel": 0.1,
}


def _joints(c3d_path, side):
    c = ezc3d.c3d(str(c3d_path))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]

    def mk(name):
        return P[:3, L.index(name), :].T if name in L else None

    S = "R" if side == "right" else "L"
    O = "L" if side == "right" else "R"
    raw = {
        "wrist": np.mean([mk(f"wrist_inner_{S}"), mk(f"wrist_outer_{S}")], axis=0),
        "elbow": mk(f"elbow_{S}"), "shoulder": mk(f"shoulder_{S}"),
        "shoulder_o": mk(f"shoulder_{O}"), "hip": mk(f"hip_{S}"), "hip_o": mk(f"hip_{O}"),
        "head": mk("head"),
        "cup": np.mean([mk(f"cluster_cup_{i}") for i in (1, 2, 3, 4)], axis=0),
    }
    if any(v is None for v in raw.values()):
        return None
    return {k: _despike(_resample(v, C3D_RATE, VIDEO_FPS)) for k, v in raw.items()}


def _first_cycle(J):
    """Cut to the first complete reach-drink-return: both cup AND wrist back at rest."""
    cup, wr = J["cup"], J["wrist"]
    cr = _lp(np.linalg.norm(cup - np.nanmedian(cup[:30], 0), axis=1))
    wd = _lp(np.linalg.norm(wr - np.nanmedian(wr[:30], 0), axis=1))
    lift = np.flatnonzero(cr > 150)
    if not len(lift):
        return None
    both = np.flatnonzero((np.arange(len(cup)) > lift[0]) & (cr < 60) & (wd < 60))
    if not len(both):
        return None
    cut = min(both[0] + 20, len(cup))
    return {k: v[:cut] for k, v in J.items()}


def _fill(x):
    x = np.asarray(x, float).copy()
    for k in range(3):
        v = np.isfinite(x[:, k])
        if v.sum() >= 2:
            x[:, k] = np.interp(np.arange(len(x)), np.flatnonzero(v), x[v, k])
    return x


def _angles(J, phases):
    sh, el, wr = J["shoulder"], J["elbow"], J["wrist"]
    trunk_mid = (sh + J["shoulder_o"]) / 2
    hip_mid = (J["hip"] + J["hip_o"]) / 2
    down = hip_mid - trunk_mid
    down = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)
    arm = el - sh
    arm = arm / (np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9)
    flex = _lp(np.degrees(np.arccos(np.clip((arm * down).sum(1), -1, 1))))
    side_v = J["shoulder_o"] - sh
    side_v = side_v / (np.linalg.norm(side_v, axis=1, keepdims=True) + 1e-9)
    abd = _lp(np.degrees(np.arcsin(np.clip((arm * side_v).sum(1), -1, 1))))
    u, v = sh - el, wr - el
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    elb = _lp(np.degrees(np.arccos(np.clip(c, -1, 1))))
    eav = np.abs(np.gradient(elb)) * VIDEO_FPS

    def ph(n):
        for nm, s, e in phases:
            if nm == n:
                return s, e
        return None
    r, d = ph("reaching"), ph("drinking")
    out = {}
    # container: MAX of the EXTENSION angle over reaching+forward_transport ("how straight
    # did the arm get"), not min. Same bug fixed in score_omc_delta.py.
    fw = ph("forward_transport")
    rf = (r[0], fw[1]) if (r and fw) else r
    out["elbow_extension_reaching"] = float(np.nanmax(elb[rf[0]:rf[1]])) if rf else np.nan
    out["shoulder_flexion_reaching"] = float(np.nanmax(flex[r[0]:r[1]])) if r else np.nan
    out["shoulder_flexion_drinking"] = float(np.nanmax(flex[d[0]:d[1]])) if d else np.nan
    out["shoulder_abduction_drinking"] = float(np.nanmax(abd[d[0]:d[1]])) if d else np.nan
    out["peak_elbow_ang_vel"] = float(np.nanmax(eav))
    return out


def main():
    rows = {"affected": [], "unaffected": []}
    files = sorted(C3D_DIR.glob("*.c3d"))
    print(f"{len(files)} P14 trials", flush=True)
    for i, f in enumerate(files):
        cond = "affected" if "_L_affected" in f.name else "unaffected"
        side = "left" if cond == "affected" else "right"
        try:
            J = _joints(f, side)
            if J is None:
                continue
            J = _first_cycle(J)
            if J is None:
                continue
            seg = segment.segment_cup_only(J["cup"], fps=VIDEO_FPS)
            seg = segment.refine_grasp_with_pose(seg, J["cup"], J["wrist"], J["head"],
                                                 fps=VIDEO_FPS)
            phases = segment.to_murphy_phases(seg, J["wrist"], J["cup"], fps=VIDEO_FPS)
            if not any(n == "reaching" for n, _, _ in phases):
                continue
            trunk = (J["shoulder"] + J["shoulder_o"]) / 2
            m = compute_position_measures(_fill(J["wrist"]), _fill(trunk), phases, side,
                                          fps=VIDEO_FPS).to_dict()
            m.pop("side", None)
            m.update(_angles(J, phases))
            rows[cond].append(m)
        except Exception as e:
            print(f"  skip {f.name}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    print(f"\nusable: {len(rows['unaffected'])} unaffected, {len(rows['affected'])} affected\n")
    keys = [k for k in rows["unaffected"][0] if k in MMC_ERR]
    print(f"{'measure':30}{'unaff':>9}{'aff':>9}{'EFFECT':>9}{'SD':>7}{'d':>6}"
          f"{'MMCerr':>8}{'err/effect':>11}")
    print("-" * 90)
    for k in keys:
        u = np.array([r[k] for r in rows["unaffected"] if np.isfinite(r.get(k, np.nan))], float)
        a = np.array([r[k] for r in rows["affected"] if np.isfinite(r.get(k, np.nan))], float)
        if len(u) < 5 or len(a) < 5:
            continue
        eff = a.mean() - u.mean()
        sd = np.sqrt((u.std(ddof=1) ** 2 + a.std(ddof=1) ** 2) / 2)
        d = eff / sd if sd > 1e-9 else np.nan
        err = MMC_ERR[k]
        frac = abs(err) / abs(eff) * 100 if abs(eff) > 1e-9 else np.inf
        flag = "  <-- error SWAMPS effect" if frac > 50 else ("  ok" if frac < 20 else "  marginal")
        print(f"{k:30}{u.mean():9.1f}{a.mean():9.1f}{eff:+9.1f}{sd:7.1f}{d:+6.2f}"
              f"{err:8.2f}{frac:10.0f}%{flag}")
    print("\nEFFECT = affected - unaffected (the clinical signal the measure exists to detect)")
    print("err/effect = our MMC error as a % of that signal. <20% ok, >50% the error swamps it.")


if __name__ == "__main__":
    main()
