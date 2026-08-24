"""SmoothNet vs savgol as the smoother on the BA-refined trajectory. On the ORIGINAL pipeline pose
they TIED at ~+4-5% peak-vel (project_smoothnet_pose / project_pose_refiner_hunt). Does the tie hold
on the BA-plain trajectory, or does one pull ahead?

Reads the TRAJECTORY CACHE (cache/ba_traj/traj_sw0.npz) -> NO BA re-solve (the cache paying off).
Applies to each cached BA-plain trajectory:  raw | savgol | SmoothNet(h36m, w32) | butter (ref).
Metric: peak-vel(matched) + jitter. Saves per-trial data + coverage. Watch:
  tail -f out/gnn/ba_smoothnet_vs_savgol.log
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
CKPT = "/home/imove/Documents/cup-task/models/smoothnet_h36m_fcn_ckpt_32.pth.tar"
WIN = 32
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


def apply_smoothnet(model, P):
    """P (T,9,3) -> SmoothNet-smoothed (T,9,3), per-joint NaN-safe."""
    T_, J, _ = P.shape
    out = SN.smooth_sequence(model, P.reshape(T_, J * 3), WIN, dim=3)
    return out.reshape(T_, J, 3)


def main():
    cache = np.load("/home/imove/Documents/cup-task/cache/ba_traj/traj_sw0.npz", allow_pickle=True)
    cid = {i: tr for i, tr in zip(cache["ids"], cache["traj"])}
    trials = [t for t in T.load_clean(need_reproj=False) if t["part"] in ("P07","P08","P15","P17","P19")]
    trials = [t for t in trials if f"{t['part']}/{t['trial']}" in cid]
    N = len(trials)
    print(f"trials matched to cache: {N}   (NO BA re-solve -- using cache/ba_traj/traj_sw0.npz)", flush=True)
    model = SN.load_smoothnet(WIN, CKPT)
    print(f"SmoothNet loaded: h36m fcn, window {WIN}", flush=True)

    VARS = ["raw", "savgol", "smoothnet", "butter"]
    jit = {v: np.full(N, np.nan) for v in VARS}; pv = {v: [] for v in VARS}; nc = {v: np.zeros(N, int) for v in VARS}
    for i, t in enumerate(tqdm(trials, mininterval=3, ncols=90, file=sys.stdout)):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        P = cid[f"{t['part']}/{t['trial']}"].astype(np.float64)
        outs = {
            "raw": P,
            "savgol": T.smooth_baseline(P, t["valid"], kind="savgol"),
            "smoothnet": apply_smoothnet(model, P),
            "butter": T.smooth_baseline(P, t["valid"], kind="butter"),
        }
        for v, Q in outs.items():
            vfin = t["valid"] & np.isfinite(Q).all(-1)
            try:
                jit[v][i] = T.score_trial(Q, t["omc"], vfin, side)["jit"]
            except Exception:
                pass
            cy = [x for x in pv_matched(H._lp(H._speed(Q[:, wi])), so) if np.isfinite(x)]
            nc[v][i] = len(cy); pv[v] += cy

    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_smoothnet_vs_savgol.npz",
             ids=np.array([f"{t['part']}/{t['trial']}" for t in trials]),
             **{f"jit_{v}": jit[v] for v in VARS}, **{f"pv_{v}": np.array(pv[v]) for v in VARS})
    print(f"\n=== PROCESSING CHECK (N={N}) ===", flush=True)
    for v in VARS:
        print(f"  {v:10s} jitter finite {np.isfinite(jit[v]).sum()}/{N}   cycles {nc[v].sum()}", flush=True)
    print(f"\n=== SmoothNet vs savgol on BA-plain trajectory (peak-vel matched + jitter) ===", flush=True)
    print(f"  {'smoother':10s} {'peak-vel':>9s} {'jitter':>9s}", flush=True)
    for v in VARS:
        print(f"  {v:10s} {med(pv[v]):+7.2f}% {med(jit[v]):9.0f}", flush=True)
    print("\n(raw = BA no smooth. tie ~ same as original pipeline; a winner = it changed on BA output.)",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
