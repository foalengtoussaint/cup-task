"""Clean 2x2 to answer 'is SmoothNet's edge over savgol BA-specific or general', all under the SAME
per-cycle-matched metric + SAME 328 trials (the earlier SmoothNet +4% was whole-trial-nanmax on n=18,
NOT comparable). Input: {pipeline mmc, BA-plain from cache} x smoother {savgol, SmoothNet}.
NO BA re-solve (BA from cache/ba_traj/traj_sw0.npz). Watch: tail -f out/gnn/ba_smoother_2x2.log
"""
import sys
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_train as T, gnn_refiner as G
import compare_pose_omc_delta as H
import smoothnet_pose_delta as SN
JOINTS = G.JOINTS
CKPT = "/home/imove/Documents/cup-task/models/smoothnet_h36m_fcn_ckpt_32.pth.tar"; WIN = 32
med = lambda a: float(np.nanmedian([z for z in a if np.isfinite(z)]))


def pv_matched(sm, so, win=15):
    if np.isfinite(so).sum() < 20:
        return []
    pk, _ = find_peaks(np.nan_to_num(so, nan=0.0), prominence=np.nanmax(so) * 0.25, distance=win)
    out = []
    for p in pk:
        if not np.isfinite(so[p]) or so[p] <= 0:
            continue
        seg = sm[max(0, p-win):min(len(sm), p+win+1)]
        pm = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
        out.append((pm - so[p]) / so[p] * 100.0 if np.isfinite(pm) else np.nan)
    return out


def sn_smooth(model, P):
    Tn, J, _ = P.shape
    return SN.smooth_sequence(model, P.reshape(Tn, J * 3), WIN, dim=3).reshape(Tn, J, 3)


def main():
    cache = np.load("/home/imove/Documents/cup-task/cache/ba_traj/traj_sw0.npz", allow_pickle=True)
    cid = {i: tr for i, tr in zip(cache["ids"], cache["traj"])}
    trials = [t for t in T.load_clean(need_reproj=False) if t["part"] in ("P07","P08","P15","P17","P19")]
    trials = [t for t in trials if f"{t['part']}/{t['trial']}" in cid]
    N = len(trials)
    print(f"trials: {N}   (pipeline=mmc, BA from cache, NO re-solve)", flush=True)
    model = SN.load_smoothnet(WIN, CKPT)

    CELLS = ["pipe+savgol", "pipe+smoothnet", "BA+savgol", "BA+smoothnet"]
    pv = {c: [] for c in CELLS}; jt = {c: np.full(N, np.nan) for c in CELLS}
    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        pipe = t["mmc"].astype(np.float64)
        ba = cid[f"{t['part']}/{t['trial']}"].astype(np.float64)
        outs = {
            "pipe+savgol": T.smooth_baseline(pipe, t["valid"], "savgol"),
            "pipe+smoothnet": sn_smooth(model, pipe),
            "BA+savgol": T.smooth_baseline(ba, t["valid"], "savgol"),
            "BA+smoothnet": sn_smooth(model, ba),
        }
        for c, Q in outs.items():
            vfin = t["valid"] & np.isfinite(Q).all(-1)
            try:
                jt[c][i] = T.score_trial(Q, t["omc"], vfin, side)["jit"]
            except Exception:
                pass
            pv[c] += [x for x in pv_matched(H._lp(H._speed(Q[:, wi])), so) if np.isfinite(x)]

    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_smoother_2x2.npz",
             ids=np.array([f"{t['part']}/{t['trial']}" for t in trials]),
             **{f"jit_{c}": jt[c] for c in CELLS}, **{f"pv_{c}": np.array(pv[c]) for c in CELLS})
    print(f"\n=== 2x2 (peak-vel matched + jitter), SAME metric+trials ===", flush=True)
    print(f"  {'cell':16s} {'peak-vel':>9s} {'jitter':>9s}", flush=True)
    for c in CELLS:
        print(f"  {c:16s} {med(pv[c]):+7.2f}% {med(jt[c]):9.0f}", flush=True)
    print("\nSmoothNet edge is GENERAL if it beats savgol on BOTH pipe & BA; BA-specific if only on BA.",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
