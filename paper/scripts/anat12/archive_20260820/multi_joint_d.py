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
    n_reach = max(int(reach[1]-reach[0]), 2) if reach else (e_-s_)
    sl = slice(s_, e_)
    keep = {j: po[j][sl].copy() for j in JN}
    return dict(part=p, arm=("affected" if rec.get("condition")=="affected" else "unaffected"),
                side=side, other=other, B=B, omc=keep, n_reach=n_reach,
                mmc=(fm[sl], am[sl], em[sl]))

def angles_with_d(rec, d, frame="trunk"):
    """Recompute OMC angles with the acting shoulder displaced by d.
    frame='trunk': d is fixed in the TORSO basis (constant relative to the trunk).
    frame='arm'  : d is fixed in the UPPER-ARM basis, rebuilt PER FRAME, so the offset rotates with
                   the arm -- the right model if the discrepancy is a landmark ON the arm segment."""
    P = {j: v for j, v in rec["omc"].items()}
    side = rec["side"]
    th = np.atleast_1d(np.asarray(d, float))
    if th.size >= 9:
        # 9-param: shoulder, elbow AND wrist each get their own constant offset (trunk frame).
        # ⚠GAUGE: a COMMON translation of all three is invisible to both the arm direction and the
        # elbow angle, so only the RELATIVE offsets are identifiable -- report differences.
        B = rec["B"]
        P[f"{side}_shoulder"] = P[f"{side}_shoulder"] + th[0:3] @ B
        P[f"{side}_elbow"]    = P[f"{side}_elbow"]    + th[3:6] @ B
        P[f"{side}_wrist"]    = P[f"{side}_wrist"]    + th[6:9] @ B
        f, a = _planar_body_angles(P, side, rec["other"])
        return f, a, _elbow_series(P, side)
    d = th
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


FPS = 60.0
def pav(v, nr):
    return float(np.nanmax(np.abs(np.diff(v[:nr])))*FPS)

def paired(recs, theta):
    """Per-trial PAIRED differences (corrected OMC - MMC) for the measures a joint offset can move."""
    rows=[]
    for r in recs:
        f, a, e = angles_with_d(r, theta)
        fm, am, em = r["mmc"]
        k = min(len(f), len(fm)); nr = min(r["n_reach"], k)
        if nr < 12: continue
        g = lambda v: float(np.nanmax(v[:k]))
        rows.append(dict(part=r["part"], arm=r["arm"],
            d_flex=g(f)-g(fm), d_abd=g(a)-g(am), d_elb=g(e)-g(em),
            d_pav=pav(e,nr)-pav(em,nr), pav_omc=pav(e,nr), pav_mmc=pav(em,nr)))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in PARTS]
    print(f"preparing {len(trials)} trials", flush=True)
    with Pool(6) as pool:
        prepped = [r for r in pool.map(prep, trials, chunksize=4) if r]
    print(f"prepared {len(prepped)}\n", flush=True)
    groups = {}
    for r in prepped:
        groups.setdefault((r["part"], r["arm"]), []).append(r)
    P0, P3, P9 = [], [], []
    print(f"{'part':6}{'arm':<11}{'|d_sh| 3p':>10}{'|el-sh| 9p':>11}{'|wr-el| 9p':>11}"
          f"{'rmse 3p':>9}{'rmse 9p':>9}   (OOS rmse)")
    for (p,a), recs in sorted(groups.items()):
        P0.append(paired(recs, np.zeros(3)))
        t3 = least_squares(resid, x0=np.zeros(3), args=(recs,), method="lm", max_nfev=300).x
        t9 = least_squares(resid, x0=np.zeros(9), args=(recs,), method="lm", max_nfev=900).x
        P3.append(paired(recs, t3)); P9.append(paired(recs, t9))
        # split-half OOS for both models
        oos = {3: [], 9: []}
        for parity in (0,1):
            tr = [r for i,r in enumerate(recs) if i%2!=parity]
            te = [r for i,r in enumerate(recs) if i%2==parity]
            if len(tr)<4 or not te: continue
            for npar in (3,9):
                th = least_squares(resid, x0=np.zeros(npar), args=(tr,), method="lm",
                                   max_nfev=300*(npar//3)).x
                oos[npar].append(np.sqrt(np.mean(resid(th, te)**2)))
        r3 = np.mean(oos[3]) if oos[3] else np.nan; r9 = np.mean(oos[9]) if oos[9] else np.nan
        print(f"{p:6}{a:<11}{np.linalg.norm(t3):>10.0f}{np.linalg.norm(t9[3:6]-t9[0:3]):>11.0f}"
              f"{np.linalg.norm(t9[6:9]-t9[3:6]):>11.0f}{r3:>9.2f}{r9:>9.2f}", flush=True)
    P0, P3, P9 = pd.concat(P0), pd.concat(P3), pd.concat(P9)
    print(f"\nPAIRED per-trial bias, median (corrected OMC - MMC), and median |error|:")
    print(f"{'measure':22}{'model':8}{'median bias':>13}{'median |err|':>14}   n")
    for nm, c in (("max flexion","d_flex"), ("max abduction","d_abd"), ("max elbow","d_elb"),
                  ("peak elbow ang vel","d_pav")):
        for lab, D in (("raw",P0), ("3-param",P3), ("9-param",P9)):
            v = D[c].dropna()
            print(f"{nm:22}{lab:8}{v.median():>13.2f}{v.abs().median():>14.2f}   {len(v)}")
    print("\n⚠ 9-param has a GAUGE freedom (a common translation is invisible), so judge it by OOS")
    print("rmse vs the 3-param model, not by in-sample fit -- more params always fit better in-sample.")
    P0.to_csv(ROOT/"out/scoring/multi_joint_d_raw.csv", index=False)
    P9.to_csv(ROOT/"out/scoring/multi_joint_d_9p.csv", index=False)
