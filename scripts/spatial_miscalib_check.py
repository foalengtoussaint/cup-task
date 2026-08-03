"""Is a suspect camera's reprojection error a SMOOTH FUNCTION OF 3D POSITION (real miscalibration)
or SPATIALLY INCOHERENT (shuffled cut / random detection noise)?

The definitive miscalibration signature (user's spec): a wrong camera geometry warps its projection
SMOOTHLY, so the reprojection ERROR VECTOR must be
  (1) LOCALLY COHERENT  -- nearby 3D points reproject with similar-direction error vectors, and
  (2) SPATIALLY SYSTEMATIC -- the error vector is a continuous function of 3D position; a low-order
      spatial model (linear in X,Y,Z) EXPLAINS most of it. Rotation/translation/focal errors all
      produce such smooth fields.
A shuffled cut (clip = a DIFFERENT repetition) or random detection noise gives error vectors with
random directions and no spatial structure -- no spatial model fits.

Method: for each frame, RANSAC-consensus 3D of every joint from the FINE reference cams only; project
into the suspect cam; error VECTOR e = proj(X) - detected_px. Then:
  dirR       : resultant length of the UNIT error vectors globally (1=all same dir, 0=random)
  localR     : mean dirR within small 3D neighbourhoods (local coherence -- high even if global
               direction rotates across space, which real miscalib does)
  spatialR2  : R^2 of a LINEAR fit  e_u,e_v ~ [1, X, Y, Z]  (how much of the error vector is
               explained by a smooth function of 3D position). HIGH = systematic geometry.
  median_px  : error magnitude (context)

Verdict: real MISCALIB = high median_px AND high spatialR2 (systematic) AND high localR.
         SHUFFLED/noise = high median_px but LOW spatialR2 and low localR (incoherent).

    python scripts/spatial_miscalib_check.py --part P12 --cams 4 5 --ref 1 2 3
    python scripts/spatial_miscalib_check.py --part P10 --cams 4   --ref 1 2 3 5
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as C  # noqa: E402
from cup_task.kalman_3d import project  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "archive"))
from reaudit_cam_quality import ransac_point, _trials  # noqa: E402

JOINTS = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip"]


def _load_pose(part, trial):
    per = {}
    for pj in sorted(glob.glob(str(C.DELTA / part / "dets" / f"delta_{part}_{trial}.*.pose.json"))):
        cam = f"cam_{Path(pj).name.split('.')[1]}"
        per[cam] = json.loads(Path(pj).read_text())["frames"]
    return per


def _px(frames, f, joint):
    """kps is a DICT keyed by joint name: {'nose':[x,y,conf], ...}."""
    if f >= len(frames):
        return None
    kps = frames[f].get("kps") or frames[f].get("keypoints")
    if not kps or joint not in kps:
        return None
    v = kps[joint]
    x, y = v[0], v[1]
    c = v[2] if len(v) > 2 else 1.0
    if c < 0.3 or (x == 0 and y == 0):
        return None
    return np.array([x, y], float)


def run(part, cam, refs, n_trials=6):
    cams = C._load_calib_mm(part)
    sus = f"cam_{cam}"
    refc = [f"cam_{r}" for r in refs]
    pts3d, errvec = [], []          # (N,3) consensus 3D ; (N,2) error vector in suspect image
    for trial in _trials(part, n_trials):
        per = _load_pose(part, trial)
        if sus not in per or not all(c in per for c in refc):
            continue
        n = min(len(per[c]) for c in [sus] + refc)
        for f in range(n):
            for j in JOINTS:
                obs = {c: _px(per[c], f, j) for c in refc}
                obs = {c: p for c, p in obs.items() if p is not None}
                if len(obs) < 2:
                    continue
                X, inl = ransac_point({c: cams[c] for c in obs}, obs)
                if X is None or len(inl) < 2:
                    continue
                ps = _px(per[sus], f, j)
                if ps is None:
                    continue
                pp, ok = project(cams[sus], X)
                if not ok:
                    continue
                pts3d.append(X); errvec.append(pp - ps)      # error vector = proj - detected
    if len(errvec) < 30:
        print(f"  {part} {sus}: too few points ({len(errvec)})")
        return
    P = np.array(pts3d); E = np.array(errvec)
    mag = np.linalg.norm(E, axis=1)
    med = float(np.median(mag))
    # global direction resultant
    U = E / (mag[:, None] + 1e-9)
    dirR = float(np.linalg.norm(U.mean(0)))
    # local coherence: mean resultant of unit vectors among each point's k nearest 3D neighbours
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    k = min(12, len(P))
    localRs = []
    for i in range(0, len(P), max(1, len(P) // 400)):     # sample ~400 anchors
        _, idx = tree.query(P[i], k=k)
        localRs.append(np.linalg.norm(U[idx].mean(0)))
    localR = float(np.mean(localRs))
    # spatial systematicity: linear model e ~ [1,X,Y,Z] for each of eu,ev; report combined R^2
    A = np.c_[np.ones(len(P)), P]
    r2s = []
    for k2 in range(2):
        y = E[:, k2]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2) + 1e-9
        r2s.append(1 - ss_res / ss_tot)
    spatialR2 = float(np.mean(r2s))
    verdict = ("REAL MISCALIB (smooth spatial field)" if spatialR2 >= 0.4 and localR >= 0.5
               else "INCOHERENT (shuffled/noise -- error not a function of 3D position)"
               if med > 25 else "FINE (low error)")
    print(f"  {part} {sus}: n={len(E)}  median={med:6.1f}px  dirR={dirR:.2f}  localR={localR:.2f}  "
          f"spatialR2={spatialR2:.2f}  -> {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cams", nargs="+", type=int, required=True)
    ap.add_argument("--ref", nargs="+", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=6)
    a = ap.parse_args(argv)
    C.use_good_cams()
    print(f"=== {a.part}: spatial-miscalib check (ref={a.ref}) ===")
    print("  real miscalib = high median AND high spatialR2 AND high localR; "
          "shuffled/noise = high median but LOW spatialR2/localR")
    for cam in a.cams:
        run(a.part, cam, a.ref, a.n_trials)


if __name__ == "__main__":
    main()
