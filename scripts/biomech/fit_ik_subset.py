"""Fit the MuJoCo body model to OMC and to MMC through the SAME solver.

Both sides get identical treatment (same model, same settings), so any remaining
disagreement is attributable to the input data rather than to the processing --
which is the justification structure Unger et al. use.

    python scripts/biomech/fit_ik_subset.py --limit 12
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                     # noqa: E402
import gnn_refiner as G                   # noqa: E402
from movi_names import movi_joint_names   # noqa: E402
import imove_ik as IK                     # noqa: E402

XML = HERE / "humanoid_torque_rl_nomesh.xml"
# our 9 joints -> BML-MoVI slot
SLOT = {j: movi_joint_names.index(n) for j, n in enumerate(
    ["RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist", "LWrist", "RHip", "LHip", "Neck"])}


def to_movi(P, valid, jmap):
    """(T,9,3) -> (T,87,4) [x,y,z,conf], zeros elsewhere. Adds mhip for pelvis init."""
    Tn = P.shape[0]
    kp = np.zeros((Tn, 87, 4), np.float64)
    for j, slot in jmap.items():
        ok = valid[:, j] & np.isfinite(P[:, j]).all(-1)
        kp[ok, slot, :3] = P[ok, j]; kp[ok, slot, 3] = 1.0
    rh, lh = G.JOINTS.index("right_hip"), G.JOINTS.index("left_hip")
    ok = valid[:, rh] & valid[:, lh] & np.isfinite(P[:, [rh, lh]]).all(-1).all(-1)
    kp[ok, movi_joint_names.index("mhip"), :3] = (P[ok, rh] + P[ok, lh]) / 2
    kp[ok, movi_joint_names.index("mhip"), 3] = 1.0
    return kp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()
    jmap = {G.JOINTS.index(n): movi_joint_names.index(m) for n, m in [
        ("right_shoulder", "RShoulder"), ("left_shoulder", "LShoulder"),
        ("right_elbow", "RElbow"), ("left_elbow", "LElbow"),
        ("right_wrist", "RWrist"), ("left_wrist", "LWrist"),
        ("right_hip", "RHip"), ("left_hip", "LHip"), ("nose", "Neck")]}
    trials = T.load_clean(need_reproj=True)[:a.limit]
    z = np.load(ROOT / "cache/ba_variants/fix.npz", allow_pickle=True)
    ba = dict(zip(z["ids"], z["traj"]))
    print(f"{len(trials)} trials, model {XML.name}\n", flush=True)
    out = {}
    for src in ("omc", "mmc"):
        res, tt = [], time.time()
        for t in trials:
            tid = f"{t['part']}/{t['trial']}"
            P = t["omc"] if src == "omc" else ba.get(tid, t["mmc"])
            kp = to_movi(P, t["valid"], jmap)
            try:
                r = IK.fit_kinematic_classical(kp, xml_path=str(XML), min_visible_joints=5,
                                               lock_legs_seated=True)
                res.append(r)
            except Exception as e:
                print(f"  [fail] {tid} {src}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                res.append(None)
        dt = time.time() - tt
        ok = [r for r in res if r is not None]
        nfr = sum(r["qpos"].shape[0] for r in ok) if ok else 0
        print(f"{src.upper():4s}  {len(ok)}/{len(trials)} fitted  {dt:6.1f}s  "
              f"({dt/max(len(trials),1):.2f} s/trial, {1000*dt/max(nfr,1):.2f} ms/frame)", flush=True)
        out[src] = res
    np.savez(ROOT / "out/scoring/ik_subset.npz",
             ids=np.array([f"{t['part']}/{t['trial']}" for t in trials]),
             omc=np.array([r["qpos"] if r else None for r in out["omc"]], dtype=object),
             mmc=np.array([r["qpos"] if r else None for r in out["mmc"]], dtype=object))
    print(f"\nwrote out/scoring/ik_subset.npz", flush=True)


if __name__ == "__main__":
    main()
