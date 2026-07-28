"""Build the paired (MMC 3D pose, OMC 3D pose) dataset for the GNN 3D->3D refiner.

For every usable DELTA trial we cache one .npz holding, at 60fps, video-length T:
    mmc   (T, J, 3)  triangulated YOLO pose, mm, cams 1-5 GOOD-ONLY (use_good_cams whitelist)
    omc   (T, J, 3)  mocap markers, mm, resampled 100->60Hz, despiked, LAG-SHIFTED onto the MMC
    valid (T, J)     bool: both mmc & omc finite at that (frame, joint)
    meta  json sidecar: part, trial, side, good_cams, lag, sync_corr, target_wrist_source

J = the 9 harness JOINTS in fixed order. The GNN input is the MMC constellation; the target is the
OMC constellation. Both are in mm. We DO NOT root on the hip (rig-flaky, ~25% detection) -- the GNN
consumes absolute mm and the PA-MPJPE loss Procrustes-aligns per frame, so the root choice is moot.

WHY a cache: triangulating all trials is the slow part (~seconds/trial x hundreds); the trainer
re-reads these .npz with no GPU/triangulation. Cache-first: skips a trial whose .npz already exists
unless --force. Keeps ALL data (never deletes).

Usable pool (cams restricted to 1-5, good-cam audit = cache/delta/cam_quality.json):
    P07 (6, L, 5cams)  P08 (6, R, 5cams)  P15 (90, 5cams)  P17 (83, 4cams)  P19 (94, 3cams, weak)
P19's OMC has ONLY wrist_outer_R (no inner, no left) -> its wrist target is the OUTER marker alone,
flagged in meta as target_wrist_source='outer_only'; kept but separable at train time.

    python scripts/gnn_build_dataset.py --parts P07 P08 P15 P17 P19
    python scripts/gnn_build_dataset.py --parts P15 --force        # rebuild
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import ezc3d
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DELTA = H.DELTA
CACHE = DELTA / "gnn_pairs"                 # one .npz + .json per trial
JOINTS = list(H.JOINTS)                     # 9 joints, fixed order
DEFAULT_PARTS = ["P07", "P08", "P15", "P17", "P19"]

# OMC marker map, per COCO joint -> tuple of candidate mocap markers to average. Tolerant of the
# P15 typo 'writst_outer_L' and P19's missing inner/left markers: we take whatever candidates are
# actually present; if none, that joint is NaN for the whole trial (valid-mask handles it).
OMC_MARKERS = {
    "right_wrist":    ["wrist_inner_R", "wrist_outer_R"],
    "right_elbow":    ["elbow_R"],
    "right_shoulder": ["shoulder_R"],
    "left_wrist":     ["wrist_inner_L", "wrist_outer_L", "writst_outer_L"],  # P15 typo tolerated
    "left_elbow":     ["elbow_L"],
    "left_shoulder":  ["shoulder_L"],
    "right_hip":      ["hip_R"],
    "left_hip":       ["hip_L"],
    "nose":           ["head"],
}


def _trial_side(trial):
    return "right" if "_R_" in trial else "left"


def _load_omc_defensive(part, trial, n_video):
    """Like H._load_omc but tolerant of the P15 typo + missing markers. Returns (dict, wrist_src)
    where wrist_src describes what the (affected-side) wrist target was built from."""
    c = ezc3d.c3d(str(DELTA / part / "c3d" / f"{trial}.c3d"))
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    P = c["data"]["points"]                       # (4, m, T) mm
    lset = set(L)

    def marker(nm):
        return P[:3, L.index(nm), :].T

    out = {}
    wrist_src = {}
    for joint, cand in OMC_MARKERS.items():
        present = [m for m in cand if m in lset]
        if not present:
            out[joint] = np.full((n_video, 3), np.nan)
            continue
        raw = np.mean([marker(m) for m in present], axis=0)
        grid = H._despike(H._resample(raw, H.C3D_RATE, H.VIDEO_FPS))
        if len(grid) < n_video:
            grid = np.vstack([grid, np.full((n_video - len(grid), 3), np.nan)])
        out[joint] = grid[:n_video]
        if "wrist" in joint:
            wrist_src[joint] = present
    side = _trial_side(trial)
    aff = f"{side}_wrist"
    present = wrist_src.get(aff, [])
    src = ("midpoint" if len(present) >= 2 else
           ("outer_only" if present == [f"wrist_outer_{side[0].upper()}"] else
            ("+".join(present) if present else "MISSING")))
    return out, src


def _shift(v, lag):
    out = np.full_like(v, np.nan)
    if lag >= 0:
        out[lag:] = v[:len(v) - lag] if lag else v
    else:
        out[:lag] = v[-lag:]
    return out


def _trials_for(part):
    seen = set()
    for pj in sorted(glob.glob(str(DELTA / part / "c3d" / "*.c3d"))):
        seen.add(Path(pj).name.replace(".c3d", ""))
    return sorted(seen)


def build_trial(part, trial, force=False):
    outnpz = CACHE / part / f"{trial}.npz"
    if outnpz.exists() and not force:
        return "cached"
    side = _trial_side(trial)
    wr = f"{side}_wrist"
    try:
        mmc, n = H._load_mmc(part, trial)             # GOOD cams only (use_good_cams set by caller)
    except Exception as e:
        return f"mmc-err:{type(e).__name__}"
    omc, wrist_src = _load_omc_defensive(part, trial, n)

    # sync on the AFFECTED-side wrist speed (same as the harness), shift OMC onto MMC
    lag, sc = H._find_lag(mmc[wr], omc[wr])
    omc = {j: _shift(v, lag) for j, v in omc.items()}

    M = np.stack([mmc[j] for j in JOINTS], axis=1)    # (T, J, 3)
    O = np.stack([omc[j] for j in JOINTS], axis=1)    # (T, J, 3)
    valid = np.isfinite(M).all(2) & np.isfinite(O).all(2)   # (T, J)

    outnpz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outnpz, mmc=M.astype(np.float32), omc=O.astype(np.float32), valid=valid)
    meta = {
        "part": part, "trial": trial, "side": side, "n": int(n),
        "good_cams": sorted(H.GOOD_CAMS.get(part, []), key=lambda c: int(c.split("_")[1])),
        "lag": int(lag), "sync_corr": float(sc),
        "target_wrist_source": wrist_src,
        "wrist_valid_frac": float(valid[:, JOINTS.index(wr)].mean()),
        "joints": JOINTS,
    }
    outnpz.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return f"built n={n} lag={lag:+d} sync={sc:.3f} wr_src={wrist_src} wr_valid={meta['wrist_valid_frac']*100:.0f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=DEFAULT_PARTS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    H.use_good_cams()   # cams 1-5 GOOD whitelist -- critical, bad cams poison triangulation
    print(f"good-cam whitelist: "
          f"{ {p: sorted(H.GOOD_CAMS[p], key=lambda c:int(c.split('_')[1])) for p in a.parts if p in H.GOOD_CAMS} }",
          flush=True)

    n_ok = n_skip = 0
    for part in a.parts:
        trials = _trials_for(part)
        print(f"\n### {part}  ({len(trials)} trials)", flush=True)
        for i, trial in enumerate(trials):
            r = build_trial(part, trial, force=a.force)
            if r == "cached":
                n_skip += 1
            elif r.startswith("built"):
                n_ok += 1
                if i < 3 or i % 20 == 0:      # sample the log, don't spam hundreds
                    print(f"  [{i+1:3d}/{len(trials)}] {trial[:26]:26} {r}", flush=True)
            else:
                print(f"  [{i+1:3d}/{len(trials)}] {trial[:26]:26} SKIP {r}", flush=True)
        print(f"  {part} done", flush=True)
    print(f"\nbuilt {n_ok} new, {n_skip} already cached -> {CACHE}", flush=True)


if __name__ == "__main__":
    main()
