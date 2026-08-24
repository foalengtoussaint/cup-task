"""Pose-model ACCURACY: YOLO vs RTMPose vs BlazePose, triangulated 3D vs OMC, n=18 c3d trials.

Same metrics as the SmoothNet work (the ones we already have):
  jitter        = mean wrist |acceleration| mm/s^2 (lower = smoother)
  OMC speed-corr= low-passed wrist-speed correlation vs OMC (higher = better)
  reproduction  = median Kabsch mm error of the arm+head constellation vs OMC (lower = better)

Each model's per-camera 2D (cached by cache_pose_altmodels.py in dets_<model>/) is triangulated with
the SAME calib + robust triangulation + good-cam whitelist, so the ONLY variable is the pose model.

    python scripts/accuracy_pose_models_delta.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H

FPS = H.VIDEO_FPS
TRIALS = {
    "P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
    "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right"),
    "P13": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
}
ABS_JOINTS = [j for j in H.JOINTS if "hip" not in j]
MODELS = {"yolo": "dets", "rtmpose": "dets_rtmpose", "blazepose": "dets_blazepose"}


def _shift(v, lag):
    out = np.full_like(v, np.nan)
    if lag >= 0:
        out[lag:] = v[:len(v) - lag] if lag else v
    else:
        out[:lag] = v[-lag:]
    return out


def jitter(xyz):
    v = np.isfinite(xyz).all(1)
    a = xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]
    m = v[2:] & v[1:-1] & v[:-2]
    return float(np.mean(np.linalg.norm(a[m], axis=1)) * FPS * FPS) if m.sum() else np.nan


def speed_corr(mmc_w, omc_w):
    sm, so = H._lp(H._speed(mmc_w)), H._lp(H._speed(omc_w))
    m = np.isfinite(sm) & np.isfinite(so)
    return float(np.corrcoef(sm[m], so[m])[0, 1]) if m.sum() > 20 else np.nan


def reproduction(mmc, omc):
    A = np.vstack([omc[j] for j in ABS_JOINTS]); B = np.vstack([mmc[j] for j in ABS_JOINTS])
    _, _, resid = H._kabsch(A, B)
    return float(np.median(resid))


def _complete(part, subdir, trial):
    return len(list((H.DELTA / part / subdir).glob(f"*{trial}*.pose.json"))) >= 3


def eval_model(model_key, only=None):
    H.DETS_SUBDIR = MODELS[model_key]
    rows = []
    for part, (trials, side) in TRIALS.items():
        for trial in trials:
            if only is not None and (part, trial) not in only:
                continue
            try:
                mmc, n = H._load_mmc(part, trial)
            except Exception as e:
                print(f"  {model_key} {part} {trial}: ERR {e}", flush=True); continue
            omc = H._load_omc(part, trial, n)
            wr = f"{side}_wrist"
            lag, _ = H._find_lag(mmc[wr], omc[wr])
            omc = {j: _shift(v, lag) for j, v in omc.items()}
            cov = np.isfinite(mmc[wr]).all(1).mean()
            rows.append((jitter(mmc[wr]), speed_corr(mmc[wr], omc[wr]),
                         reproduction(mmc, omc), cov))
    R = np.array(rows)
    return R


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", action="store_true",
                    help="score only trials ALL models have cached (fair subset), else each model on "
                         "all its complete trials")
    a = ap.parse_args()
    H.use_good_cams()

    # per model, the set of complete trials (>=3 cams cached)
    done = {}
    for k, sub in MODELS.items():
        done[k] = {(p, t) for p, (ts, _) in TRIALS.items() for t in ts if _complete(p, sub, t)}
    matched = set.intersection(*done.values()) if a.matched else None

    print(f"\n{'model':12} {'n':>3} {'jitter':>9} {'OMC corr':>9} {'reprod mm':>10} {'wrist cov':>10}"
          + ("   [matched subset]" if a.matched else ""), flush=True)
    print("-" * 66, flush=True)
    for k in MODELS:
        only = matched if a.matched else done[k]
        if not only:
            print(f"{k:12} (nothing cached yet)", flush=True); continue
        R = eval_model(k, only=only)
        if len(R) == 0:
            print(f"{k:12} (no data)", flush=True); continue
        print(f"{k:12} {len(R):3d} {np.nanmedian(R[:,0]):9.0f} {np.nanmedian(R[:,1]):+9.3f} "
              f"{np.nanmedian(R[:,2]):10.1f} {np.nanmedian(R[:,3])*100:9.0f}%", flush=True)
    print("\njitter mm/s^2 (lower better) | OMC speed-corr (higher) | reprod = Kabsch mm vs OMC (lower) "
          "| cov = frames wrist has 3D", flush=True)
    if not a.matched:
        print("(each model scored on ALL its complete trials; use --matched for the shared subset)",
              flush=True)


if __name__ == "__main__":
    main()
