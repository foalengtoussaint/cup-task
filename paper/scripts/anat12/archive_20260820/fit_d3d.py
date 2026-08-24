"""Can ONE 3D shoulder displacement d explain the flexion, abduction AND elbow differences at once?

Forward model (EXACT -- no small-angle approximation): take the OMC pose, move its acting-side
shoulder by a constant d expressed in that trial's own BODY FRAME (down / fwd / lat, so d is
anatomically meaningful and frame-independent), then recompute all three angles through the SAME
functions the scorer uses (_planar_body_angles, _elbow_series) and compare to our MMC angles.

Fit d (3 params) per participant x arm by least squares over all frames x all three angles.

WHY three angles: flexion+abduction constrain only the arm's DIRECTION, so they cannot see a
displacement ALONG the arm. The elbow angle (shoulder-elbow-wrist) does see it. Together they make d
identifiable in 3D -- which a flexion-only fit never was.

Reports fitted d (down/fwd/lat components, mm), |d|, and per-angle RMSE BEFORE vs AFTER, so a fit
that does not actually explain the differences is visible as "no RMSE improvement".
"""
import sys, re
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import least_squares
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import compare_pose_omc_delta as H, gnn_train as GT, results_v3_delta as R
from score_vs_automq import (_pose_variant_cached, _planar_body_angles, _elbow_series, load_automq,
                             automq_part, automq_phases_to_video, _win)
PARTS = {"P12","P15","P17","P07","P13"}
JN = ["right_shoulder","left_shoulder","right_elbow","left_elbow","right_wrist","left_wrist",
      "right_hip","left_hip","nose"]
GRID = R._GRID_JOINTS
pat = re.compile(r"trial_(\d+)_([RL])_")
H.use_good_cams(); _ba = R._ba_traj_cache(); _amq = load_automq()

def body_basis(P, side, other):
    nn = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
    sh, shO = P[f"{side}_shoulder"], P[f"{other}_shoulder"]
    down = nn((P["right_hip"]+P["left_hip"])/2.0 - (sh+shO)/2.0)
    sline = nn(sh - shO)
    fwd = nn(np.cross(sline, down)); lat = nn(np.cross(down, fwd))
    f = np.isfinite(down).all(1)&np.isfinite(fwd).all(1)
    if not f.any(): return None
    m = lambda v: v/(np.linalg.norm(v)+1e-9)
    return np.stack([m(np.nanmedian(down[f],0)), m(np.nanmedian(fwd[f],0)), m(np.nanmedian(lat[f],0))])

def prep(t):
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
    ph = automq_phases_to_video(rec["phases"], lag, n)
    if not ph: return None
    reach, drink = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach[0], drink[1]) if (reach and drink) else reach
    if not w: return None
    B = body_basis(po, side, other)
    if B is None: return None
    try:
        fm, am = _planar_body_angles(pose, side, other)
        em = _elbow_series(pose, side)
    except Exception: return None
    s_, e_ = w
    sl = slice(s_, e_)
    keep = {j: po[j][sl].copy() for j in JN}
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                side=side, other=other, B=B, omc=keep,
                mmc=(fm[sl], am[sl], em[sl]))

def angles_with_d(rec, d, frame="trunk"):
    """Recompute OMC angles with the acting shoulder displaced by d.
    frame='trunk': d is fixed in the TORSO basis (constant relative to the trunk).
    frame='arm'  : d is fixed in the UPPER-ARM basis, rebuilt PER FRAME, so the offset rotates with
                   the arm -- the right model if the discrepancy is a landmark ON the arm segment."""
    P = {j: v for j, v in rec["omc"].items()}
    side = rec["side"]
    if frame == "trunk":
        d_world = d @ rec["B"]
    else:
        sh, el = P[f"{side}_shoulder"], P[f"{side}_elbow"]
        nn = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
        u1 = nn(el - sh)                                   # along the humerus, per frame
        down = np.broadcast_to(rec["B"][0], u1.shape)
        u2 = nn(np.cross(u1, down)); u3 = nn(np.cross(u1, u2))
        d_world = d[0]*u1 + d[1]*u2 + d[2]*u3
    P[f"{side}_shoulder"] = P[f"{side}_shoulder"] + d_world
    f, a = _planar_body_angles(P, side, rec["other"])
    e = _elbow_series(P, side)
    return f, a, e

def resid(d_body, recs, frame="trunk"):
    out = []
    for r in recs:
        f, a, e = angles_with_d(r, d_body, frame)
        fm, am, em = r["mmc"]
        k = min(len(f), len(fm))
        for x, y in ((f[:k], fm[:k]), (a[:k], am[:k]), (e[:k], em[:k])):
            v = x - y
            out.append(v[np.isfinite(v)])
    return np.concatenate(out) if out else np.zeros(1)

def stats_by_angle(recs, d_body, frame="trunk"):
    """Return per-angle (bias, sd) so a fit that only removes the OFFSET is distinguishable from one
    that also removes orientation-dependent STRUCTURE (which would lower sd too)."""
    acc = [[], [], []]
    for r in recs:
        f, a, e = angles_with_d(r, d_body, frame)
        fm, am, em = r["mmc"]
        k = min(len(f), len(fm))
        for i, (x, y) in enumerate(((f[:k],fm[:k]), (a[:k],am[:k]), (e[:k],em[:k]))):
            v = x - y; acc[i].append(v[np.isfinite(v)])
    return [(float(np.mean(np.concatenate(c))), float(np.std(np.concatenate(c)))) for c in acc]

if __name__ == "__main__":
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in PARTS]
    print(f"preparing {len(trials)} trials", flush=True)
    with Pool(6) as pool:
        prepped = [r for r in pool.map(prep, trials, chunksize=4) if r]
    print(f"prepared {len(prepped)}\n", flush=True)
    groups = {}
    for r in prepped:
        groups.setdefault((r["part"], r["arm"]), []).append(r)
    out=[]
    print("FLEXION: bias and sd BEFORE -> AFTER, for a TRUNK-fixed d and an ARM-fixed d.")
    print("If only the BIAS falls, d is removing a constant offset; if the SD falls too, d is also")
    print("capturing orientation-dependent structure.\n")
    print(f"{'part':6}{'arm':<11}{'|d|trunk':>9}{'|d|arm':>8}   "
          f"{'bias0':>7}{'biasT':>7}{'biasA':>7}   {'sd0':>6}{'sdT':>6}{'sdA':>6}   "
          f"{'elbow sd0/T/A':>18}")
    for (p,a), recs in sorted(groups.items()):
        s0 = stats_by_angle(recs, np.zeros(3))
        solT = least_squares(resid, x0=np.zeros(3), args=(recs,"trunk"), method="lm", max_nfev=200)
        solA = least_squares(resid, x0=np.zeros(3), args=(recs,"arm"), method="lm", max_nfev=200)
        sT = stats_by_angle(recs, solT.x, "trunk"); sA = stats_by_angle(recs, solA.x, "arm")
        print(f"{p:6}{a:<11}{np.linalg.norm(solT.x):>9.0f}{np.linalg.norm(solA.x):>8.0f}   "
              f"{s0[0][0]:>7.1f}{sT[0][0]:>7.1f}{sA[0][0]:>7.1f}   "
              f"{s0[0][1]:>6.1f}{sT[0][1]:>6.1f}{sA[0][1]:>6.1f}   "
              f"{s0[2][1]:>6.1f}{sT[2][1]:>6.1f}{sA[2][1]:>6.1f}", flush=True)
        out.append(dict(part=p, arm=a, n_trials=len(recs),
                        d_trunk=float(np.linalg.norm(solT.x)), d_arm=float(np.linalg.norm(solA.x)),
                        flex_bias0=s0[0][0], flex_biasT=sT[0][0], flex_biasA=sA[0][0],
                        flex_sd0=s0[0][1], flex_sdT=sT[0][1], flex_sdA=sA[0][1],
                        elb_sd0=s0[2][1], elb_sdT=sT[2][1], elb_sdA=sA[2][1],
                        elb_bias0=s0[2][0], elb_biasT=sT[2][0], elb_biasA=sA[2][0]))
    pd.DataFrame(out).to_csv(ROOT/"out/scoring/fit_d3d_bias.csv", index=False)
    print("\nbias0/sd0 = no correction. T = trunk-fixed d. A = arm-fixed d (rotates with the arm).")
    print("wrote out/scoring/fit_d3d_bias.csv")
