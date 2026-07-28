"""Plot MVGFormer vs incumbent (YOLO+robust-tri) vs OMC: per-axis displacement + speed over time,
to SEE where MVGFormer's raw variance comes from (uniform wobble vs apex spikes). All estimators
rigid-aligned to OMC (kills the constant rig<->mocap offset; does not touch variance/shape).
Reuses H for calib/omc/sync. Out -> out/mvgformer/<part>_<trial>[_<tag>].png

Usage: python scripts/mvgformer_plot.py --part P07 --trial trial_13_L_unaffected [--tag drop13 --incumbent-cams 1,3]
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
import compare_pose_omc_delta as H

CACHE = Path("/home/imove/Documents/cup-task/cache/mvgformer")
OUT = Path("/home/imove/Documents/cup-task/out/mvgformer"); OUT.mkdir(parents=True, exist_ok=True)


def _shift(v, l):
    out = np.full_like(v, np.nan)
    if l >= 0: out[l:] = v[:len(v)-l] if l else v
    else: out[:l] = v[-l:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P07")
    ap.add_argument("--trial", default="trial_13_L_unaffected")
    ap.add_argument("--tag", default="")
    ap.add_argument("--incumbent-cams", default="")
    ap.add_argument("--lp-hz", type=float, default=6.0, help="position low-pass cutoff (Hz), study "
                    "default 6Hz. Applied to POSITION not speed (speed-domain filtering can't remove "
                    "spike-induced peaks, only smear them: 40-90%% peak overshoot vs ~1%% for pos-lp). "
                    "Do NOT go lower to chase valley noise: vs UNFILTERED OMC, MVGFormer's peak err "
                    "grows 6Hz 1%% -> 2.5Hz 9%% -> 1.5Hz 17%% (the '0%% at 2.5Hz' was an artefact of "
                    "filtering OMC too — never soften ground truth). 6Hz already nails the true peak.")
    args = ap.parse_args()
    H.use_good_cams()
    if args.incumbent_cams:
        H.GOOD_CAMS = {args.part: {f"cam_{x.strip()}" for x in args.incumbent_cams.split(",")}}
    side = "left" if "_L_" in args.trial else "right"
    joint = f"{side}_wrist"

    npz = np.load(CACHE / f"{args.part}_{args.trial}{('_' + args.tag) if args.tag else ''}.npz")
    mvg = npz["wrist"].astype(float); sc = npz["score"].astype(float)
    mvg[sc < 0.1] = np.nan
    mvg_raw = mvg.copy()
    mvg = H._despike(mvg)        # same isolated-teleport removal as the incumbent + scorer
    mmc, n = H._load_mmc(args.part, args.trial); inc = mmc[joint]
    omc = H._load_omc(args.part, args.trial, n)[joint]
    m = min(len(mvg), len(inc), len(omc)); mvg, inc, omc = mvg[:m], inc[:m], omc[:m]

    # sync OMC to video via incumbent lag; align both estimators to OMC (rigid)
    lag, _ = H._find_lag(inc, omc); omc = _shift(omc, lag)
    def align(a):
        R, t, _ = H._kabsch(a, omc); return a @ R.T + t
    mvg_a, inc_a = align(mvg), align(inc)
    t = np.arange(m) / H.VIDEO_FPS
    labels = "XYZ"
    colors = {"OMC": "k", "incumbent": "tab:blue", "MVGFormer": "tab:red"}

    # POSITION low-pass per axis (the same filter the speed row differentiates) — so rows 1-3 show
    # EXACTLY what feeds row 4. Faint line underneath = despiked-but-unfiltered, to show the fuzz.
    def _lp_hz(x, hz):        # H._lp but with an explicit cutoff
        from scipy.signal import butter, filtfilt
        v = np.isfinite(x)
        if v.sum() < 8: return x
        idx = np.flatnonzero(v); xi = np.interp(np.arange(len(x)), idx, x[idx])
        b, a = butter(2, hz / (H.VIDEO_FPS / 2)); return filtfilt(b, a, xi)
    def pos_lp(a):
        out = a.copy()
        for k in range(3): out[:, k] = _lp_hz(a[:, k], args.lp_hz)
        return out
    omc_lp, inc_lp_a, mvg_lp_a = pos_lp(omc), pos_lp(inc_a), pos_lp(mvg_a)

    R_al, t_al, _ = H._kabsch(mvg, omc); mvg_raw_a = mvg_raw @ R_al.T + t_al  # raw in OMC frame
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax, ax_i in zip(axes[:3], range(3)):
        ax.plot(t, mvg_raw_a[:, ax_i], color="tab:red", lw=0.7, alpha=0.28, label="MVGFormer raw")
        ax.plot(t, mvg_a[:, ax_i], color="tab:red", lw=0.8, alpha=0.35, label="MVGFormer despiked")
        for name, a in (("OMC", omc_lp), ("incumbent", inc_lp_a), ("MVGFormer", mvg_lp_a)):
            ax.plot(t, a[:, ax_i], color=colors[name], lw=1.3 if name != "OMC" else 1.7,
                    alpha=0.95, label=(name + " (pos-lp)" if name == "MVGFormer" else name))
        ax.set_ylabel(f"{labels[ax_i]} (mm)"); ax.grid(alpha=0.25)
    axes[0].legend(ncol=5, loc="upper right", fontsize=7.5)

    # speed = derivative of the SAME low-passed position shown above (frame-invariant, no alignment)
    for name, a in (("OMC", omc), ("incumbent", inc), ("MVGFormer", mvg)):
        sp = H._speed(pos_lp(a))
        axes[3].plot(t, sp, color=colors[name], lw=1.3, alpha=0.85,
                     label=(name + " (pos-lp)" if name == "MVGFormer" else name))
    axes[3].legend(ncol=3, loc="upper right", fontsize=8)
    axes[3].set_ylabel("speed (mm/s)"); axes[3].set_xlabel("time (s)"); axes[3].grid(alpha=0.25)

    def jit(a):
        j = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1); return np.nanmedian(j)
    # cam count: npz field if present (dropout runs), else the good-cam set actually triangulated
    ncam = len(npz["cams"]) if "cams" in npz.files else len(
        [c for c in H.GOOD_CAMS.get(args.part, []) ] or "?????")
    fig.suptitle(f"MVGFormer vs incumbent vs OMC — {args.part} {args.trial}"
                 + (f"  [{args.tag}]" if args.tag else "")
                 + f"\nMVGFormer {ncam} cams; raw jitter |d2X| med: MVGFormer {jit(mvg):.1f}mm  "
                 f"incumbent {jit(inc):.1f}mm  (rows 1-3 = per-axis position aligned to OMC; "
                 f"row 4 = speed = d/dt of position low-passed @ {args.lp_hz:g}Hz)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / f"{args.part}_{args.trial}{('_' + args.tag) if args.tag else ''}.png"
    plt.savefig(out, dpi=140); print("->", out)


if __name__ == "__main__":
    main()
