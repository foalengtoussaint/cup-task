"""7-DOF forward-kinematics arm solver (user's idea) + per-cycle OMC-matched peak-velocity metric.

WHY this differs from the failed bone PENALTY (project_bundle_adjustment_bone_prior.md): there the
3 joints were FREE 3D points and bone length was a soft cost -> the solver could slide a joint along
the unobservable camera ray to fake rigidity. HERE the arm is parameterised as
    q = [shoulder_xyz(3), upperarm_theta, upperarm_phi, forearm_theta, forearm_phi]   # 7 DOF
    X_el = X_sh + L_upper * dir(theta_u, phi_u)      # bone length BAKED IN, cannot stretch
    X_wr = X_el + L_fore  * dir(theta_f, phi_f)
So rigidity is a MANIFOLD we optimise ON, not a penalty we optimise AGAINST -- no depth-slide-for-
rigidity trade exists. This is the "hard constraint done right". Solved per-frame with LM, warm-started
from t-1. L_upper/L_fore = per-trial median RAW triangulated length (self-supervised, no OMC).

Reprojection uses the DISTORTED Brown-Conrady projection (gnn_refiner.project_torch as numpy), NOT a
linear 3x4 P -- our BRIO cams have real distortion in every sidecar.

Also adds peak_vel_matched(): per-cycle peak-velocity error, each MMC peak matched IN TIME to an OMC
speed peak (scipy.find_peaks on the OMC speed, window around each). The whole-trial nanmax pve in
score_trial is coarser (one global max, not time-matched, collapses multi-reach trials).
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T
import gnn_refiner as G
import ba_refine as BA
import compare_pose_omc_delta as H

JOINTS = G.JOINTS
FPS = H.VIDEO_FPS


# --------------------------------------------------------------------------- distorted projection (np)
def project_np(X, K, dist, R, t):
    """Brown-Conrady projection of 3D points X (N,3) into ONE camera -> (N,2) px. Mirrors project_torch."""
    Xc = X @ R.T + t                                   # (N,3) camera frame
    z = np.clip(Xc[:, 2], 1e-3, None)
    xn = Xc[:, 0] / z; yn = Xc[:, 1] / z
    k1, k2, p1, p2, k3 = dist
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return np.stack([u, v], axis=1)


def _dir(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def _fk(q, L_up, L_fo):
    """q(7) -> (X_sh, X_el, X_wr) each (3,). Bone lengths fixed."""
    X_sh = q[0:3]
    X_el = X_sh + L_up * _dir(q[3], q[4])
    X_wr = X_el + L_fo * _dir(q[5], q[6])
    return X_sh, X_el, X_wr


def _residuals(q, calib, obs, conf, L_up, L_fo):
    """Weighted reprojection residuals. calib = list of (K,dist,R,t). obs/conf: (C,3,2)/(C,3)."""
    js = _fk(q, L_up, L_fo)
    out = []
    for c, (K, dist, R, t) in enumerate(calib):
        for j, Xj in enumerate(js):
            w = conf[c, j]
            if w < 0.1 or not np.isfinite(obs[c, j]).all():
                continue
            p = project_np(Xj[None], K, dist, R, t)[0]
            out.extend(np.sqrt(w) * (p - obs[c, j]))
    return np.array(out) if out else np.zeros(1)


def _init_q_from_xyz(sh, el, wr, L_up, L_fo):
    """seed q from a raw triangulated (sh,el,wr) triple."""
    def ang(v):
        v = v / (np.linalg.norm(v) + 1e-9)
        theta = np.arccos(np.clip(v[2], -1, 1))
        phi = np.arctan2(v[1], v[0])
        return theta, phi
    tu, pu = ang(el - sh); tf, pf = ang(wr - el)
    return np.array([sh[0], sh[1], sh[2], tu, pu, tf, pf])


def solve_trial_fk(t):
    """Per-frame 7-DOF FK solve for the AFFECTED arm. Returns refined (T,J,3) (only the 3 arm joints
    on the affected side are updated; other joints copied from raw). Warm-start q from t-1."""
    side = t["side"]; mmc = t["mmc"]; valid = t["valid"]
    si = JOINTS.index(f"{side}_shoulder"); ei = JOINTS.index(f"{side}_elbow"); wi = JOINTS.index(f"{side}_wrist")
    Tn = mmc.shape[0]
    L = BA.trial_bone_lengths(mmc, valid)
    L_up = L.get((wi, ei), L.get((ei, wi)))  # note: LIMBS stores (wrist,elbow)&(elbow,shoulder) pairs
    # recover the two arm bone refs explicitly by joint identity
    def bone_ref(a, b):
        for (u, v), Lv in L.items():
            if {u, v} == {a, b}:
                return Lv
        return None
    L_up = bone_ref(si, ei)   # upper arm (shoulder-elbow)
    L_fo = bone_ref(ei, wi)   # forearm (elbow-wrist)
    out = mmc.copy()
    if L_up is None or L_fo is None:
        return out, {"note": "no bone ref"}
    calib = [(t["K"][c], t["dist"][c], t["R"][c], t["t"][c]) for c in range(t["uv"].shape[1])]
    q_prev = None
    for f in range(Tn):
        obs = t["uv"][f][:, [si, ei, wi]]                       # (C,3,2)
        conf = (t["uv_conf"][f] * t["uv_valid"][f])[:, [si, ei, wi]]  # (C,3)
        if conf.max() < 0.1:
            q_prev = None
            continue
        # seed: warm-start from t-1, else from raw triangulation this frame
        if q_prev is not None:
            q0 = q_prev
        elif valid[f, si] and valid[f, ei] and valid[f, wi]:
            q0 = _init_q_from_xyz(mmc[f, si], mmc[f, ei], mmc[f, wi], L_up, L_fo)
        elif valid[f, si]:
            q0 = np.array([mmc[f, si, 0], mmc[f, si, 1], mmc[f, si, 2], 1.5, 0.0, 1.5, 0.0])
        else:
            q_prev = None
            continue
        try:
            r = least_squares(_residuals, q0, args=(calib, obs, conf, L_up, L_fo),
                              method="lm", max_nfev=60)
            X_sh, X_el, X_wr = _fk(r.x, L_up, L_fo)
            out[f, si] = X_sh; out[f, ei] = X_el; out[f, wi] = X_wr
            q_prev = r.x
        except Exception:
            q_prev = None
    return out, {"L_up": round(L_up, 1), "L_fo": round(L_fo, 1)}


# --------------------------------------------------------------------------- per-cycle matched peak-vel
def peak_vel_matched(mmc_wrist, omc_wrist, min_prom=None, win=15):
    """Per-cycle peak-velocity error, each OMC speed peak matched IN TIME to the nearest MMC peak.

    Find peaks on the OMC low-passed speed (the truth defines the cycles). For each OMC peak at frame p,
    take the MAX MMC speed in [p-win, p+win] (time-matched amplitude). Return median signed %err over
    peaks + the count. Coarse whole-trial nanmax can't do this (one global max, no matching)."""
    so = H._lp(H._speed(omc_wrist)); sm = H._lp(H._speed(mmc_wrist))
    fin = np.isfinite(so)
    if fin.sum() < 20:
        return np.nan, 0
    prom = min_prom if min_prom is not None else np.nanmax(so) * 0.25
    pk, _ = find_peaks(np.nan_to_num(so, nan=0.0), prominence=prom, distance=win)
    errs = []
    for p in pk:
        lo, hi = max(0, p - win), min(len(sm), p + win + 1)
        seg = sm[lo:hi]
        if not np.isfinite(seg).any() or not np.isfinite(so[p]) or so[p] <= 0:
            continue
        pm = np.nanmax(seg)
        errs.append((pm - so[p]) / so[p] * 100.0)
    return (float(np.median(errs)) if errs else np.nan), len(errs)


def wrist_err(P, t):
    return T.score_trial(P, t["omc"], t["valid"], t["side"])["wr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P15", "P17", "P19"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smooth", action="store_true", help="also savgol-smooth each variant before scoring")
    a = ap.parse_args()

    print(f"loading trials (parts={a.parts})...", flush=True)
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in a.parts]
    if a.limit:
        trials = trials[:a.limit]
    print(f"trials: {len(trials)}", flush=True)
    med = lambda x: float(np.nanmedian([z for z in x if np.isfinite(z)]))

    def eval_variant(name, refine_fn):
        t0 = time.time()
        wr, pcm, npk, wtp = [], [], [], []
        for i, t in enumerate(trials):
            P = refine_fn(t)
            if a.smooth:
                P = T.smooth_baseline(P, t["valid"], kind="savgol")
            side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
            wmmc = P[:, wi].copy(); wmmc[~t["valid"][:, wi]] = np.nan
            womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
            e, n = peak_vel_matched(wmmc, womc)
            pcm.append(e); npk.append(n)
            wtp.append(H._speed(wmmc))  # unused agg placeholder
            wr.append(wrist_err(P, t))
            if (i + 1) % 40 == 0:
                print(f"    [{name}] {i+1}/{len(trials)}  ({time.time()-t0:.0f}s)", flush=True)
        tot_pk = int(np.nansum(npk))
        print(f"  {name:22s} wrist {med(wr):5.1f}mm   peakVel(matched) {med(pcm):+6.1f}%  "
              f"(over {tot_pk} cycles)   ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'variant':22s} {'wrist':>8s}   {'per-cycle matched peak-vel err':>20s}", flush=True)
    eval_variant("PIPELINE (consensus)", lambda t: t["mmc"])
    eval_variant("BA lam=0 (robust)", lambda t: BA.refine_trial_ba(t, 0.0, iters=60)[0])
    eval_variant("FK 7DOF (rigid arm)", lambda t: solve_trial_fk(t)[0])
    print("\n=== FK / per-cycle DONE ===", flush=True)


if __name__ == "__main__":
    main()
