"""Fit the body model to the RAW OMC markers (18 slots, not the reduced 9 joints).

The pipeline's t["omc"] averages 46 C3D markers down to 9 joint centres. That discards
wrist_inner/outer (forearm rotation), arm_R/L (humeral roll), chest, hand and thumb.
This feeds the markers directly, so the fit is far better constrained.

OMC only -- MMC stays geometric and fast; this just improves the reference.

    python scripts/biomech/fit_omc_ik.py --limit 6
"""
import argparse
import sys
import time
from pathlib import Path

import ezc3d
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import compare_pose_omc_delta as H       # noqa: E402
import gnn_train as T                    # noqa: E402
import gnn_refiner as G                  # noqa: E402
import imove_ik as IK                    # noqa: E402
from movi_names import movi_joint_names as M   # noqa: E402

XML = HERE / "humanoid_torque_rl_nomesh.xml"
# C3D marker -> model site.  18 of the 22 non-cluster markers; fingers have no site.
MARKER_SITE = {
    "shoulder_R": "RShoulder", "shoulder_L": "LShoulder",
    "elbow_R": "RElbow", "elbow_L": "LElbow",
    "hip_R": "RHip", "hip_L": "LHip",
    "chest": "sternum", "head": "Head",
    "hand_R": "RHand", "hand_L": "LHand",
    "thumb_R": "rthumb", "thumb_L": "lthumb",
    "wrist_inner_R": "rwrithumbside", "wrist_outer_R": "rwripinkieside",
    "wrist_inner_L": "lwrithumbside", "wrist_outer_L": "lwripinkieside",
    "arm_R": "rshom", "arm_L": "lshom",
}


def load_markers(part, trial, n_video):
    """All C3D markers on the video timebase, same resample/despike as _load_omc."""
    c = ezc3d.c3d(str(ROOT / "cache/delta" / part / "c3d" / f"{trial}.c3d"))
    L = [x.strip() for x in c["parameters"]["POINT"]["LABELS"]["value"]]
    P = c["data"]["points"]
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0]) or H.C3D_RATE
    out = {}
    for nm in MARKER_SITE:
        if nm not in L:
            continue
        raw = P[:3, L.index(nm), :].T
        g = H._despike(H._resample(raw, rate, H.VIDEO_FPS))
        if len(g) < n_video:
            g = np.vstack([g, np.full((n_video - len(g), 3), np.nan)])
        out[nm] = g[:n_video]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    a = ap.parse_args()
    model, data, site_names, _ = IK.load_mujoco_model(str(XML))
    sidx = {n: i for i, n in enumerate(site_names) if n}
    trials = T.load_clean(need_reproj=True)[:a.limit]
    print(f"{len(trials)} trials, {len(MARKER_SITE)} marker slots (+mhip)\n", flush=True)

    resid = {v: [] for v in MARKER_SITE.values()}
    t0 = time.time(); nfr = 0; qs = []; keep = []
    for t in trials:
        n = t["mmc"].shape[0]
        mk = load_markers(t["part"], t["trial"], n)
        kp = np.zeros((n, 87, 4))
        for nm, site in MARKER_SITE.items():
            if nm not in mk or site not in M:
                continue
            ok = np.isfinite(mk[nm]).all(-1)
            kp[ok, M.index(site), :3] = mk[nm][ok]; kp[ok, M.index(site), 3] = 1.0
        rh, lh = mk.get("hip_R"), mk.get("hip_L")
        if rh is not None and lh is not None:
            ok = np.isfinite(rh).all(-1) & np.isfinite(lh).all(-1)
            kp[ok, M.index("mhip"), :3] = (rh[ok] + lh[ok]) / 2; kp[ok, M.index("mhip"), 3] = 1.0
        try:
            r = IK.fit_kinematic_classical(kp, xml_path=str(XML), min_visible_joints=5,
                                           lock_legs_seated=True)
        except Exception as e:
            print(f"  [fail] {t['part']}/{t['trial']}: {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        qs.append(r["qpos"]); keep.append(f"{t['part']}/{t['trial']}"); nfr += r["qpos"].shape[0]
        for f in range(0, r["qpos"].shape[0], 10):
            sp = IK.fk_site_positions(model, data, r["qpos"][f])
            for nm, site in MARKER_SITE.items():
                if nm not in mk or site not in sidx or not np.isfinite(mk[nm][f]).all():
                    continue
                resid[site].append(float(np.linalg.norm(sp[sidx[site]] - mk[nm][f])))
    dt = time.time() - t0
    print(f"TIME {dt:.0f}s for {len(keep)} trials = {dt/max(len(keep),1):.1f} s/trial, "
          f"{1000*dt/max(nfr,1):.1f} ms/frame\n", flush=True)
    print(f"{'site':16s} {'n':>7s} {'median mm':>10s} {'p90':>8s}")
    allv = []
    for site, v in resid.items():
        if not v:
            continue
        arr = np.array(v); allv.append(arr)
        print(f"{site:16s} {len(arr):7d} {np.median(arr):10.2f} {np.percentile(arr,90):8.2f}")
    if allv:
        A = np.concatenate(allv)
        print(f"\noverall median {np.median(A):.2f} mm  p90 {np.percentile(A,90):.2f} mm")
        print("(artifacts we want removed: ~7mm length wander, ~19mm shoulder translation)")
    np.savez(ROOT / "out/scoring/omc_ik_subset.npz", ids=np.array(keep),
             qpos=np.array(qs, dtype=object))
    print(f"\nwrote out/scoring/omc_ik_subset.npz", flush=True)


if __name__ == "__main__":
    main()
