"""Log the MMC 3D arm to Rerun so the wrist wander / bone-length wobble is visible in orbitable
3D (fixed camera hides it). Logs raw vs KF-only vs KF+RTS vs a fast bone-length-constrained
variant, plus the forearm length over time as a scalar (should be flat for a rigid arm).

    python scripts/rerun_mmc_delta.py            # spawns the viewer
    python scripts/rerun_mmc_delta.py --save out/mmc_P14.rrd   # or write a file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_task import triangulate
from cup_task.triangulate import kf_rts_smooth
from scripts.score_omc_delta import _kf_only
from scripts.compare_pose_omc_delta import _load_calib_mm, _despike, VIDEO_FPS, DELTA

PART, TRIAL = "P14", "trial_1_R_unaffected"
ARM = ["right_shoulder", "right_elbow", "right_wrist"]


def _kp_point(name):
    def fn(fr):
        k = fr.get("kps", {})
        return np.array(k[name][:2], float) if name in k else None
    return fn


def _bone_lock(arm, ref_lo=100, ref_hi=None):
    """Fast skeleton-consistency fix: hold each bone at its MEDIAN length. Keep the shoulder as
    the anchor; re-place elbow along the (shoulder->elbow) DIRECTION at the median upperarm
    length; re-place wrist along (elbow->wrist) direction at the median forearm length. Cheap
    (no optimiser) -- just enforces constant bone length while keeping the observed directions."""
    sh, el, wr = arm["right_shoulder"], arm["right_elbow"], arm["right_wrist"]
    Lu = np.nanmedian(np.linalg.norm(el - sh, axis=1))
    Lf = np.nanmedian(np.linalg.norm(wr - el, axis=1))
    du = el - sh; du = du / (np.linalg.norm(du, axis=1, keepdims=True) + 1e-9)
    el2 = sh + du * Lu
    df = wr - el2; df = df / (np.linalg.norm(df, axis=1, keepdims=True) + 1e-9)
    wr2 = el2 + df * Lf
    return {"right_shoulder": sh, "right_elbow": el2, "right_wrist": wr2}, Lu, Lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path, default=None)
    a = ap.parse_args()
    if a.save:
        rr.init("mmc_delta", spawn=False)
        rr.save(str(a.save))
    else:
        rr.init("mmc_delta", spawn=True)

    cams = _load_calib_mm(PART)
    per_cam = {}
    for pj in sorted((DELTA / PART).glob("dets/*.pose.json")):
        c = pj.name.split(".")[1]
        per_cam[f"cam_{c}"] = json.loads(pj.read_text())["frames"]
    T = max(len(v) for v in per_cam.values())

    raw = {}
    for j in ARM:
        tr = triangulate.triangulate_target(per_cam, cams, _kp_point(j), T)
        raw[j] = _despike(np.array([t["X"] if t.get("X") else [np.nan] * 3 for t in tr]))
    kf = {j: _kf_only(raw[j]) for j in ARM}
    kfrts = {j: kf_rts_smooth(raw[j], fps=VIDEO_FPS) for j in ARM}
    lock, Lu, Lf = _bone_lock(raw)
    print(f"median upperarm {Lu:.0f}mm  forearm {Lf:.0f}mm", flush=True)

    variants = {"raw": (raw, [255, 80, 80]), "kf": (kf, [255, 210, 0]),
                "kfrts": (kfrts, [0, 200, 255]), "bonelock": (lock, [120, 255, 120])}

    def forearm_len(A):
        return np.linalg.norm(A["right_wrist"] - A["right_elbow"], axis=1)

    for f in range(T):
        rr.set_time("frame", sequence=f)
        for name, (A, col) in variants.items():
            pts = np.array([A[j][f] for j in ARM])
            if np.isfinite(pts).all():
                rr.log(f"arm/{name}/bones",
                       rr.LineStrips3D([pts], colors=[col], radii=6))
                rr.log(f"arm/{name}/joints", rr.Points3D(pts, colors=[col], radii=10))
            fl = forearm_len(A)[f]
            if np.isfinite(fl):
                rr.log(f"forearm_len/{name}", rr.Scalars(float(fl)))
    print(f"logged {T} frames"
          + (f" -> {a.save}" if a.save else " (viewer spawned)"), flush=True)


if __name__ == "__main__":
    main()
