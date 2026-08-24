"""Graph the position low-pass CUTOFF trade-off for MVGFormer wrist-speed peaks, scored vs UNFILTERED
OMC (never filter ground truth). Two panels:
  (left)  peak-velocity error % vs cutoff — one line per drink reach + mean + global peak. Shows the
          trade-off: the TALLEST peak wants a high cutoff (don't clip it); the smaller peaks want a
          lower cutoff (kill the fuzz inflating them).
  (right) the speed traces themselves at a few cutoffs vs OMC, so you SEE clipping-vs-fuzz.
Model position-lp only; OMC unfiltered. n=1 trial — illustrative, needs multi-trial validation.
Out -> out/mvgformer/lp_cutoff_tradeoff_<part>_<trial>.png
"""
import sys, argparse
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import compare_pose_omc_delta as H

OUT = Path("out/mvgformer"); OUT.mkdir(parents=True, exist_ok=True)


def _shift(v, l):
    o = np.full_like(v, np.nan)
    if l >= 0: o[l:] = v[:len(v)-l] if l else v
    else: o[:l] = v[-l:]
    return o


def lp1(x, hz):
    v = np.isfinite(x)
    if v.sum() < 8: return x
    idx = np.flatnonzero(v); xi = np.interp(np.arange(len(x)), idx, x[idx])
    b, a = butter(2, hz / (H.VIDEO_FPS / 2)); return filtfilt(b, a, xi)


def pspd(A, hz):
    o = A.copy()
    for k in range(3): o[:, k] = lp1(A[:, k], hz)
    return H._speed(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P07"); ap.add_argument("--trial", default="trial_13_L_unaffected")
    args = ap.parse_args()
    H.use_good_cams()
    joint = "left_wrist" if "_L_" in args.trial else "right_wrist"
    npz = np.load(f"cache/mvgformer/{args.part}_{args.trial}.npz")
    mvg = H._despike(npz["wrist"].astype(float)); sc = npz["score"].astype(float); mvg[sc < 0.1] = np.nan
    mmc, n = H._load_mmc(args.part, args.trial); inc = mmc[joint]; omc = H._load_omc(args.part, args.trial, n)[joint]
    m = min(len(mvg), len(inc), len(omc)); mvg, inc, omc = mvg[:m], inc[:m], omc[:m]
    lag, _ = H._find_lag(inc, omc); omc = _shift(omc, lag)
    omc_sp = H._speed(omc)                                   # UNFILTERED OMC truth
    t = np.arange(m) / H.VIDEO_FPS

    # reach windows around the visible OMC speed peaks (auto: 4 tallest local maxima, min 0.6s apart)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(np.nan_to_num(omc_sp), distance=int(0.6*H.VIDEO_FPS), height=200)
    pk = pk[np.argsort(omc_sp[pk])[::-1][:4]]; pk.sort()
    wins = [(max(0, p-int(0.4*H.VIDEO_FPS)), min(m, p+int(0.4*H.VIDEO_FPS))) for p in pk]
    o_rp = [np.nanmax(omc_sp[a:b]) for a, b in wins]
    o_gp = np.nanmax(omc_sp)

    cutoffs = np.arange(2.0, 8.01, 0.5)
    gerr, rerr = [], [[] for _ in wins]
    for hz in cutoffs:
        sp = pspd(mvg, hz)
        gerr.append(abs(np.nanmax(sp) - o_gp) / o_gp * 100)
        for i, (a, b) in enumerate(wins):
            rerr[i].append(abs(np.nanmax(sp[a:b]) - o_rp[i]) / o_rp[i] * 100)
    rerr = np.array(rerr); mean_err = rerr.mean(0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    for i in range(len(wins)):
        axL.plot(cutoffs, rerr[i], marker="o", ms=3, alpha=0.55,
                 label=f"reach {i+1} (peak {o_rp[i]:.0f})")
    axL.plot(cutoffs, gerr, "k--", lw=1.5, marker="s", ms=4, label=f"GLOBAL peak ({o_gp:.0f})")
    axL.plot(cutoffs, mean_err, color="tab:red", lw=2.5, marker="D", ms=5, label="MEAN over reaches")
    best = cutoffs[int(np.argmin(mean_err))]
    axL.axvline(best, color="tab:red", ls=":", alpha=0.6); axL.axvline(6.0, color="gray", ls=":", alpha=0.6)
    axL.text(best, axL.get_ylim()[1]*0.92, f" mean-best {best:g}Hz", color="tab:red", fontsize=9)
    axL.text(6.0, axL.get_ylim()[1]*0.82, " 6Hz (study)", color="gray", fontsize=9)
    axL.set_xlabel("position low-pass cutoff (Hz)"); axL.set_ylabel("peak-velocity error % vs UNFILTERED OMC")
    axL.set_title("Cutoff trade-off: tallest peak wants HIGH cutoff, smaller peaks want LOW", fontsize=10)
    axL.grid(alpha=0.3); axL.legend(fontsize=8)

    for hz, c in [(8, "tab:green"), (6, "tab:blue"), (4, "tab:orange"), (2.5, "tab:purple")]:
        axR.plot(t, pspd(mvg, hz), color=c, lw=1.1, alpha=0.8, label=f"MVGFormer @{hz:g}Hz")
    axR.plot(t, omc_sp, "k", lw=1.8, label="OMC (unfiltered)")
    for a, b in wins: axR.axvspan(t[a], t[min(b, m-1)], color="gray", alpha=0.07)
    axR.set_xlabel("time (s)"); axR.set_ylabel("wrist speed (mm/s)")
    axR.set_title("Speed traces: low cutoff clips the tall peak, high cutoff leaves fuzz", fontsize=10)
    axR.grid(alpha=0.3); axR.legend(fontsize=8)
    fig.suptitle(f"MVGFormer position-lp cutoff vs wrist-speed peak — {args.part} {args.trial}  "
                 f"(n=1 trial, illustrative)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / f"lp_cutoff_tradeoff_{args.part}_{args.trial}.png"
    plt.savefig(out, dpi=140); print("->", out)


if __name__ == "__main__":
    main()
