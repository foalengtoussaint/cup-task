"""Honest peak-velocity metric: find peaks INDEPENDENTLY in OMC and MMC speed, then MATCH them by
proximity (not "max of MMC in a window around the OMC peak", which lets MMC reach for its highest
value and hides timing error). Report:
  - |amp err| median/p75/p90  over MATCHED peaks (paired OMC peak <-> nearest MMC peak within tol)
  - timing err ms  (matched MMC peak frame - OMC peak frame)
  - MMC spurious peaks (found in MMC, no OMC peak nearby) and MMC missed (OMC peak, no MMC peak)
Reads cache/ba_traj/traj_sw0.npz (BA) + mmc (pipeline). NO re-solve. Smoothers: savgol, SmoothNet.
Watch: tail -f out/gnn/ba_peaks_independent.log
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
MATCH_TOL = 15   # frames (~250ms): an OMC peak & MMC peak this close are the SAME movement
med = lambda a: float(np.nanmedian(a)) if len(a) else np.nan


def peaks(sp):
    fin = np.isfinite(sp)
    if fin.sum() < 20:
        return np.array([], int)
    pk, _ = find_peaks(np.nan_to_num(sp, nan=0.0), prominence=np.nanmax(sp) * 0.25, distance=15)
    return pk


def match_peaks(so, sm):
    """Independent peaks in each; greedily pair OMC<->MMC within MATCH_TOL by nearest frame.
    Returns (amp_err%_list, timing_ms_list, n_omc, n_mmc, n_matched)."""
    po, pm = peaks(so), peaks(sm)
    used = set(); amp = []; tim = []
    for p in po:
        cand = [(abs(q - p), q) for q in pm if q not in used and abs(q - p) <= MATCH_TOL]
        if not cand:
            continue
        _, q = min(cand); used.add(q)
        if np.isfinite(so[p]) and so[p] > 0 and np.isfinite(sm[q]):
            amp.append((sm[q] - so[p]) / so[p] * 100.0)
            tim.append((q - p) * 1000.0 / H.VIDEO_FPS)
    return amp, tim, len(po), len(pm), len(amp)


def sn_smooth(model, P):
    Tn, J, _ = P.shape
    return SN.smooth_sequence(model, P.reshape(Tn, J * 3), WIN, dim=3).reshape(Tn, J, 3)


def main():
    cache = np.load("/home/imove/Documents/cup-task/cache/ba_traj/traj_sw0.npz", allow_pickle=True)
    cid = {i: tr for i, tr in zip(cache["ids"], cache["traj"])}
    trials = [t for t in T.load_clean(need_reproj=False) if t["part"] in ("P07","P08","P15","P17","P19")]
    trials = [t for t in trials if f"{t['part']}/{t['trial']}" in cid]
    print(f"trials: {len(trials)}   MATCH_TOL={MATCH_TOL}fr  (independent peaks, matched)", flush=True)
    model = SN.load_smoothnet(WIN, CKPT)
    CELLS = ["pipe+savgol", "pipe+smoothnet", "BA+savgol", "BA+smoothnet"]
    amp = {c: [] for c in CELLS}; tim = {c: [] for c in CELLS}
    nomc = {c: 0 for c in CELLS}; nmmc = {c: 0 for c in CELLS}; nmat = {c: 0 for c in CELLS}
    for t in tqdm(trials, mininterval=3, ncols=90, file=sys.stdout):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        pipe = t["mmc"].astype(np.float64); ba = cid[f"{t['part']}/{t['trial']}"].astype(np.float64)
        outs = {"pipe+savgol": T.smooth_baseline(pipe, t["valid"], "savgol"),
                "pipe+smoothnet": sn_smooth(model, pipe),
                "BA+savgol": T.smooth_baseline(ba, t["valid"], "savgol"),
                "BA+smoothnet": sn_smooth(model, ba)}
        for c, Q in outs.items():
            sm = H._lp(H._speed(Q[:, wi]))
            a, ti, no, nm, nmt = match_peaks(so, sm)
            amp[c] += a; tim[c] += ti; nomc[c] += no; nmmc[c] += nm; nmat[c] += nmt

    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_peaks_independent.npz",
             **{f"amp_{c}": np.array(amp[c]) for c in CELLS},
             **{f"tim_{c}": np.array(tim[c]) for c in CELLS})
    print(f"\n=== INDEPENDENT-peak matched metric ===", flush=True)
    print(f"  {'cell':16s} {'|amp|md':>8s} {'|amp|p90':>9s} {'signed':>8s} {'|time|md':>9s} "
          f"{'matched/OMC':>12s} {'MMCspurious':>12s}", flush=True)
    for c in CELLS:
        a = np.abs(amp[c]); spur = nmmc[c] - nmat[c]
        print(f"  {c:16s} {med(a):7.2f}% {np.percentile(a,90):8.2f}% {med(amp[c]):+7.2f}% "
              f"{med(np.abs(tim[c])):8.1f}ms {nmat[c]:5d}/{nomc[c]:<5d} {spur:11d}", flush=True)
    print("\nMMCspurious = MMC speed-peaks with NO matching OMC peak (fake reaches from jitter).", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
