"""Is the residual flexion error a DISPLACED SHOULDER LANDMARK, a TILTED AXIS, or neither?

Geometry: with the arm of length L at flexion phi, displacing the shoulder by a constant d gives
    error(phi) ~= -(d . p_hat(phi)) / L        p_hat = the in-plane perpendicular to the arm
i.e. a SINUSOID in phi with amplitude |d|/L. A tilted down-axis instead gives error = theta, FLAT in
phi. So fit both models per participant x arm and compare:

    landmark : error = A*cos(phi) + B*sin(phi)      (A,B -> d in the sagittal plane, |d| = L*sqrt(A^2+B^2))
    axistilt : error = C                            (constant)
    line     : error = m*phi + c                    (reference/no-mechanism baseline)

Reported: R^2 of each, fitted |d| in mm, and which model wins. Frame-invariant throughout (both
angles are internal). Per-frame data over reach->drink, all trials of a participant x arm pooled.
"""
import sys, re
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
from score_vs_automq import (_pose_variant_cached, _planar_body_angles, load_automq,
                             automq_part, automq_phases_to_video, _win)
PARTS = {"P12","P15","P17","P07","P13"}
JN = ["right_shoulder","left_shoulder","right_elbow","left_elbow","right_wrist","left_wrist",
      "right_hip","left_hip","nose"]
GRID = R._GRID_JOINTS
pat = re.compile(r"trial_(\d+)_([RL])_")
H.use_good_cams(); _ba = R._ba_traj_cache(); _amq = load_automq()

def one(t):
    p, tr, side = t["part"], t["trial"], t["side"]
    m = pat.search(tr)
    if not m: return None
    rec = _amq.get((automq_part(p), int(m.group(1)), m.group(2)))
    if rec is None or rec.get("phases") is None: return None
    n = t["mmc"].shape[0]
    pose = _pose_variant_cached(t, "BA", "smoothnet", _ba)
    if pose is None: return None
    omc = H._load_omc(p, tr, n); wr=f"{side}_wrist"
    if wr not in omc or not np.isfinite(omc[wr]).any() or not all(j in omc for j in JN): return None
    lag,_ = H._find_lag(t["mmc"][:, GRID.index(wr)], omc[wr])
    po = {j: R._shift(omc[j], lag) for j in JN}
    other = "right" if side=="left" else "left"
    try:
        fm,_ = _planar_body_angles(pose, side, other)
        fo,_ = _planar_body_angles(po, side, other)
    except Exception: return None
    ph = automq_phases_to_video(rec["phases"], lag, n)
    if not ph: return None
    reach, drink = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach[0], drink[1]) if (reach and drink) else reach
    if not w: return None
    a, b = fm[w[0]:w[1]], fo[w[0]:w[1]]
    k = min(len(a), len(b)); a, b = a[:k], b[:k]
    ok = np.isfinite(a)&np.isfinite(b)
    if ok.sum()<20: return None
    L = np.linalg.norm(pose[f"{side}_elbow"]-pose[f"{side}_shoulder"], axis=1)
    L = float(np.nanmedian(L[np.isfinite(L)]))
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                phi=b[ok], err=(a[ok]-b[ok]), L=L)

if __name__ == "__main__":
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in PARTS]
    print(f"{len(trials)} trials", flush=True)
    rows=[]
    with Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(one, trials, chunksize=4)):
            if r: rows.append(r)
            if (i+1) % 80 == 0: print(f"  [{i+1}/{len(trials)}] kept {len(rows)}", flush=True)
    print(f"\nPROCESSING CHECK: trials {len(trials)}, kept {len(rows)}\n")
    key = {}
    for r in rows:
        key.setdefault((r["part"], r["arm"]), []).append(r)
    print(f"{'part':6}{'arm':<12}{'n_fr':>7}{'R2_land':>9}{'R2_tilt':>9}{'R2_line':>9}"
          f"{'|d| mm':>9}{'tilt deg':>10}   winner")
    out=[]
    for (p,a), rs in sorted(key.items()):
        phi = np.radians(np.concatenate([r["phi"] for r in rs]))
        err = np.concatenate([r["err"] for r in rs])
        L = float(np.median([r["L"] for r in rs]))
        good = np.isfinite(phi)&np.isfinite(err)
        phi, err = phi[good], err[good]
        if len(phi) < 200: continue
        ss_tot = np.sum((err-err.mean())**2)
        # landmark: err = A cos phi + B sin phi
        M = np.c_[np.cos(phi), np.sin(phi)]
        AB, *_ = np.linalg.lstsq(M, err, rcond=None)
        r2_land = 1 - np.sum((err - M@AB)**2)/ss_tot
        # axis tilt: err = C
        C = err.mean(); r2_tilt = 1 - np.sum((err-C)**2)/ss_tot
        # line baseline
        mm = np.polyfit(phi, err, 1); r2_line = 1 - np.sum((err-np.polyval(mm,phi))**2)/ss_tot
        dmag = L*np.sqrt(AB[0]**2+AB[1]**2)*np.pi/180.0   # AB in deg -> rad
        win = max([("landmark",r2_land),("axistilt",r2_tilt),("line",r2_line)], key=lambda x:x[1])[0]
        print(f"{p:6}{a:<12}{len(phi):>7}{r2_land:>9.3f}{r2_tilt:>9.3f}{r2_line:>9.3f}"
              f"{dmag:>9.0f}{C:>10.1f}   {win}")
        out.append(dict(part=p, arm=a, n=len(phi), r2_land=r2_land, r2_tilt=r2_tilt,
                        r2_line=r2_line, d_mm=dmag, tilt_deg=C, winner=win))
    pd.DataFrame(out).to_csv(ROOT/"out/scoring/fit_landmark_offset.csv", index=False)
    print("\nR2 is vs the MEAN, so axistilt R2=0 by construction; landmark must BEAT 0 to mean anything.")
    print("wrote out/scoring/fit_landmark_offset.csv")
