"""Per-PARTICIPANT paired test of the timing (and amplitude) effect. Pooling 1270 cycles overstates n
(cycles cluster within participant/trial). Aggregate to per-participant means, then compare the 4 cells
on the SAME 5 participants (paired). Saves per-cycle (part, amp, timing) so future re-analysis is free.
Reads cache/ba_traj/traj_sw0.npz (BA) + mmc (pipeline) + SmoothNet. NO re-solve.
Watch: tail -f out/gnn/ba_peaks_perpart.log
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
TOL = 15


def peaks(sp):
    if np.isfinite(sp).sum() < 20:
        return np.array([], int)
    pk, _ = find_peaks(np.nan_to_num(sp, nan=0.0), prominence=np.nanmax(sp) * 0.25, distance=15)
    return pk


def match(so, sm):
    po, pm = peaks(so), peaks(sm); used = set(); out = []
    for p in po:
        cand = [(abs(q - p), q) for q in pm if q not in used and abs(q - p) <= TOL]
        if not cand:
            continue
        _, q = min(cand); used.add(q)
        if np.isfinite(so[p]) and so[p] > 0 and np.isfinite(sm[q]):
            out.append(((sm[q] - so[p]) / so[p] * 100.0, (q - p) * 1000.0 / H.VIDEO_FPS))
    return out


def sn(model, P):
    Tn, J, _ = P.shape
    return SN.smooth_sequence(model, P.reshape(Tn, J * 3), WIN, dim=3).reshape(Tn, J, 3)


def main():
    cache = np.load("/home/imove/Documents/cup-task/cache/ba_traj/traj_sw0.npz", allow_pickle=True)
    cid = {i: tr for i, tr in zip(cache["ids"], cache["traj"])}
    trials = [t for t in T.load_clean(need_reproj=False) if t["part"] in ("P07","P08","P15","P17","P19")]
    trials = [t for t in trials if f"{t['part']}/{t['trial']}" in cid]
    print(f"trials: {len(trials)}", flush=True)
    model = SN.load_smoothnet(WIN, CKPT)
    CELLS = ["pipe+savgol", "pipe+smoothnet", "BA+savgol", "BA+smoothnet"]
    rows = {c: [] for c in CELLS}    # list of (part, amp, tim)
    for t in tqdm(trials, mininterval=3, ncols=90, file=sys.stdout):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist"); pp = t["part"]
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        so = H._lp(H._speed(womc))
        pipe = t["mmc"].astype(np.float64); ba = cid[f"{pp}/{t['trial']}"].astype(np.float64)
        outs = {"pipe+savgol": T.smooth_baseline(pipe, t["valid"], "savgol"),
                "pipe+smoothnet": sn(model, pipe),
                "BA+savgol": T.smooth_baseline(ba, t["valid"], "savgol"),
                "BA+smoothnet": sn(model, ba)}
        for c, Q in outs.items():
            for a, ti in match(so, H._lp(H._speed(Q[:, wi]))):
                rows[c].append((pp, a, ti))

    parts = ["P07", "P08", "P15", "P17", "P19"]
    np.savez("/home/imove/Documents/cup-task/out/gnn/ba_peaks_perpart.npz",
             **{f"{c}": np.array(rows[c], dtype=object) for c in CELLS})

    def perpart(c, col):   # col: 1=amp(use |amp|), 2=tim(signed mean)
        arr = rows[c]; out = []
        for p in parts:
            v = [r[col] for r in arr if r[0] == p]
            if col == 1:
                v = [abs(x) for x in v]
            out.append(np.mean(v) if v else np.nan)
        return np.array(out)

    print(f"\n=== PER-PARTICIPANT (n=5). mean over each participant's cycles ===", flush=True)
    print(f"  --- |amp err| % (accuracy) ---", flush=True)
    print(f"  {'part':6s} " + " ".join(f"{c:>14s}" for c in CELLS), flush=True)
    A = {c: perpart(c, 1) for c in CELLS}
    for i, p in enumerate(parts):
        print(f"  {p:6s} " + " ".join(f"{A[c][i]:14.2f}" for c in CELLS), flush=True)
    print(f"  {'MEAN':6s} " + " ".join(f"{np.nanmean(A[c]):14.2f}" for c in CELLS), flush=True)
    print(f"\n  --- signed TIMING mean (ms) ---", flush=True)
    Tm = {c: perpart(c, 2) for c in CELLS}
    for i, p in enumerate(parts):
        print(f"  {p:6s} " + " ".join(f"{Tm[c][i]:+14.1f}" for c in CELLS), flush=True)
    print(f"  {'MEAN':6s} " + " ".join(f"{np.nanmean(Tm[c]):+14.1f}" for c in CELLS), flush=True)
    # paired: pipe+savgol -> BA+smoothnet, per participant
    d_amp = A["pipe+savgol"] - A["BA+smoothnet"]
    d_tim = Tm["pipe+savgol"] - Tm["BA+smoothnet"]
    print(f"\n  PAIRED pipe+savgol - BA+smoothnet, per participant:", flush=True)
    print(f"    |amp| drop:  {np.round(d_amp,2)}  mean {d_amp.mean():+.2f}pp  ({(d_amp>0).sum()}/5 improve)",
          flush=True)
    print(f"    timing drop: {np.round(d_tim,1)}  mean {d_tim.mean():+.1f}ms ({(d_tim>0).sum()}/5 improve)",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
