"""Compare cup-task's phase segmentation against the OMC/MeTRAbs truth, across the cohort.

THE COMPARISON, STATED HONESTLY. Both sides segment the SAME 7 Murphy phases with the SAME
segmenter logic and the SAME refill cup. The ONLY thing that differs is the POSE that feeds
the wrist/mouth signals:
  * OMC side  = MeTRAbs 3D keypoints (biomech npz)      -> cache/omc_phases.json (pre-derived)
  * cup-task  = yolo26s-pose, triangulated multi-cam    -> computed here
So a phase-boundary disagreement is attributable to the pose estimator, not to a cup
difference or a method difference. That is the point: can the mocap-free yolo pose reproduce
the phase boundaries the MeTRAbs pipeline produces?

Cup for BOTH sides is the research refill track (RTS-smoothed), so it is held constant.

Reports, per phase, the onset error (cup-task minus OMC, in ms) and the drink dwell-duration
error -- the number the whole exercise is about.

    conda activate object_tracking
    python scripts/compare_phases_omc.py --calib-root /home/imove/Documents/object_tracking/data/calib
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import pose_keypoints, segment, triangulate
from pipeline.kalman_3d import load_calibration

# ── the iMOVE OMC segmenter, imported by path (its package __init__ needs datajoint) ──
_SEG_PY = Path("/home/imove/Documents/iMOVE/DEV/imove_extensions/imove_extensions"
               "/drink_task_segmentation.py")
_spec = importlib.util.spec_from_file_location("dts", _SEG_PY)
_dts = importlib.util.module_from_spec(_spec)
sys.modules["dts"] = _dts
_spec.loader.exec_module(_dts)
# joint slots segment_drink_task reads in the 87-joint MeTRAbs layout
J_HEAD, J_LWRIST, J_RWRIST = _dts.J_HEAD, _dts.J_LWRIST, _dts.J_RWRIST

ROOT = Path(__file__).resolve().parents[1]
POSE_CACHE = ROOT / "cache" / "pose_models"
OT = Path("/home/imove/Documents/object_tracking/experiments/drink_study/cache")
OMC = OT / "omc_phases.json"
REFILL = OT / "track3d_clean3d_refill"
FPS = 60.0
PHASES = ["rest_pre", "reaching", "forward_transport", "drinking",
          "back_transport", "returning", "rest_post"]


def _kp_fn(name):
    def f(fr):
        k = fr.get("kps", {}).get(name)
        return np.array(k[:2], float) if k else None
    return f


def _norm(stem: str) -> str:
    """P01_P01_drinking_... -> P01_drinking_... (strip duplicate participant prefix)."""
    parts = stem.split("_")
    if len(parts) > 1 and parts[0] == parts[1]:
        return "_".join(parts[1:])
    return stem


def _refill_cup(stem: str, T: int):
    """(T,3) RTS cup for the SAME rep, matched by normalised stem. None if missing."""
    for cand in REFILL.glob("*__clean3d_refill.json"):
        if _norm(cand.name.replace("__clean3d_refill.json", "")) == _norm(stem):
            d = json.loads(cand.read_text())
            cup = np.full((T, 3), np.nan, float)
            for fr in d["frames"]:
                i = fr["fr"]
                if i < T and fr.get("rts"):
                    cup[i] = fr["rts"]
            return cup
    return None


def _intervals_dict(iv):
    return {n: (int(s), int(e)) for n, s, e in iv}


def _tri(per_cam, cams, point_fn, n):
    tr = triangulate.triangulate_target(per_cam, cams, point_fn, n)
    return np.array([t["X"] if t["X"] else [np.nan] * 3 for t in tr], float)


def _refill_cup_kp(stem: str, T: int):
    """(T,1,4) refill cup for segment_drink_task, or None."""
    cup = _refill_cup(stem, T)
    if cup is None:
        return None
    out = np.zeros((T, 1, 4), np.float32)
    valid = np.isfinite(cup).all(1)
    out[valid, 0, :3] = cup[valid]
    out[valid, 0, 3] = 1.0
    return out


def same_seg_phases(stem: str, per_cam: dict, cams: dict):
    """SAME segmenter as OMC (iMOVE segment_drink_task) but on cup-task's YOLO pose.

    Both sides then share identical segmenter code AND the same refill cup, so the only
    variable is the pose (MeTRAbs vs yolo26s). We only need 3 joints -- head(67), wrists
    (77,85) -- so build a sparse (T,87,4) with just those filled. cup-task's head proxy
    (nose/eye centroid) sits a bit differently from MeTRAbs' Head, but the segmenter's
    near-mouth threshold is ADAPTIVE (observed min + pad), so a constant offset cancels."""
    n = max(len(v) for v in per_cam.values())
    cup_kp = _refill_cup_kp(stem, n)
    if cup_kp is None:
        return None, "no refill cup"
    pose87 = np.zeros((n, 87, 4), np.float32)
    for slot, fn in ((J_HEAD, triangulate._mouth_point),
                     (J_LWRIST, lambda fr: triangulate._wrist_point(fr, "left")),
                     (J_RWRIST, lambda fr: triangulate._wrist_point(fr, "right"))):
        X = _tri(per_cam, cams, fn, n)
        v = np.isfinite(X).all(1)
        pose87[v, slot, :3] = X[v]
        pose87[v, slot, 3] = 1.0
    res = _dts.segment_drink_task(cup_kp, pose87, fps=FPS)
    return [(nm, int(s), int(e)) for nm, s, e in res.phase_intervals], None


SMOOTH_POSE = False   # set by --smooth-pose; runs the v2 pose_smooth (SmoothNet) stage


def _smooth_xyz(xyz):
    """Wrap an (T,3) track through the v2 pose_smooth stage (SmoothNet)."""
    from pipeline.pose_smooth import smooth_track
    tr = [{"frame": f, "X": (None if not np.isfinite(p).all() else [float(v) for v in p])}
          for f, p in enumerate(np.asarray(xyz, float))]
    out = smooth_track(tr)
    return np.array([o["X"] if o["X"] is not None else [np.nan] * 3 for o in out])


def cup_task_phases(stem: str, per_cam: dict, cams: dict):
    """Run cup-task's segmenter (its pose + refill cup) -> phase_intervals, or None."""
    n = max(len(v) for v in per_cam.values())
    cup = _refill_cup(stem, n)
    if cup is None:
        return None, "no refill cup"

    def rng(name):
        tr = triangulate.triangulate_target(per_cam, cams, _kp_fn(name), n)
        X = np.array([t["X"] if t["X"] else [np.nan] * 3 for t in tr], float)
        good = X[np.isfinite(X).all(1)]
        return (float(np.linalg.norm(good.max(0) - good.min(0))) if len(good) else 0.0), X

    r_rng, rX = rng("right_wrist")
    l_rng, lX = rng("left_wrist")
    hand = rX if r_rng >= l_rng else lX
    mouth_tr = triangulate.triangulate_target(per_cam, cams, triangulate._mouth_point, n)
    mouth, _ = segment.track_confidence(mouth_tr)

    if SMOOTH_POSE:                       # v2: SmoothNet-refine the pose tracks before segmentation
        hand = _smooth_xyz(hand)
        mouth = _smooth_xyz(mouth)

    seg = segment.segment_cup_only(cup)
    seg = segment.refine_grasp_with_pose(seg, cup, hand, mouth)
    phases = segment.to_murphy_phases(seg, hand, cup)
    return [(n_, int(s), int(e)) for n_, s, e in phases], None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-root", type=Path,
                    default=Path("/home/imove/Documents/object_tracking/data/calib"))
    ap.add_argument("--model", default="s")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "cache" / "phase_compare_omc.json")
    ap.add_argument("--smooth-pose", action="store_true",
                    help="v2: SmoothNet-refine the pose tracks before segmentation")
    a = ap.parse_args(argv)
    global SMOOTH_POSE
    SMOOTH_POSE = a.smooth_pose
    if SMOOTH_POSE:
        print("[v2: pose tracks SmoothNet-refined before segmentation]", flush=True)

    omc = json.loads(OMC.read_text())
    truth = {k: v for k, v in omc.items() if v.get("has_drink")}
    posed = {p.parent.name for p in POSE_CACHE.glob(f"*/yolo26{a.model}.2d.json")}

    # match posed stems to OMC truth by normalised stem
    truth_by_norm = {_norm(k): k for k in truth}
    pairs = [(ps, truth_by_norm[_norm(ps)]) for ps in posed if _norm(ps) in truth_by_norm]
    print(f"posed={len(posed)}  OMC-truth={len(truth)}  comparable={len(pairs)}", flush=True)

    # two variants: "same" isolates the pose (iMOVE segmenter on yolo pose);
    # "e2e" is the whole cup-task product (cup-task segmenter). Both vs the same OMC truth.
    results = {}
    agg = {v: {"onset": {ph: [] for ph in PHASES}, "dwell": []} for v in ("same", "e2e")}
    calib_cache = {}
    for i, (ps, tk) in enumerate(sorted(pairs)):
        part = ps.split("_")[0]
        cfile = a.calib_root / part / "calibration.toml"
        if not cfile.exists():
            print(f"  {ps}: no calib ({cfile})", flush=True); continue
        if part not in calib_cache:
            calib_cache[part] = load_calibration(cfile, target_size=(1920, 1080))
        cams = calib_cache[part]

        per_cam = json.loads((POSE_CACHE / ps / f"yolo26{a.model}.2d.json").read_text())
        omc_iv = _intervals_dict(truth[tk]["phase_intervals"])
        rec = {"omc": truth[tk]["phase_intervals"]}

        for variant, fn in (("same", same_seg_phases), ("e2e", cup_task_phases)):
            iv, err = fn(ps, per_cam, cams)
            if iv is None:
                rec[variant] = None
                print(f"  {ps} [{variant}]: {err}", flush=True)
                continue
            d = _intervals_dict(iv)
            onset = {}
            for ph in PHASES:
                if ph in omc_iv and ph in d:
                    de = (d[ph][0] - omc_iv[ph][0]) / FPS * 1000
                    onset[ph] = round(de, 1)
                    agg[variant]["onset"][ph].append(de)
            r = {"intervals": iv, "onset_ms": onset}
            if "drinking" in omc_iv and "drinking" in d:
                od = (omc_iv["drinking"][1] - omc_iv["drinking"][0]) / FPS * 1000
                cd = (d["drinking"][1] - d["drinking"][0]) / FPS * 1000
                r["dwell_err_ms"] = round(cd - od, 0)
                agg[variant]["dwell"].append(cd - od)
            rec[variant] = r
        results[ps] = rec
        se = rec.get("same") or {}
        e2 = rec.get("e2e") or {}
        print(f"[{i+1}/{len(pairs)}] {ps}  dwell err  same={se.get('dwell_err_ms','?')}ms  "
              f"e2e={e2.get('dwell_err_ms','?')}ms", flush=True)

    a.out.write_text(json.dumps(results, indent=1))
    for variant, label in (("same", "SAME SEGMENTER (isolates POSE: MeTRAbs vs yolo26s)"),
                           ("e2e", "END-TO-END (whole cup-task product)")):
        A = agg[variant]
        n = len(A["dwell"])
        print(f"\n=== {label}  n={n} ===", flush=True)
        print(f"{'phase':18s} {'n':>3} {'onset bias ms':>13} {'onset |med| ms':>15}",
              flush=True)
        for ph in PHASES:
            v = np.array(A["onset"][ph])
            if len(v):
                print(f"{ph:18s} {len(v):>3} {np.median(v):>13.0f} "
                      f"{np.median(np.abs(v)):>15.0f}", flush=True)
        if A["dwell"]:
            dd = np.array(A["dwell"])
            print(f"drink DWELL err (variant - OMC): median {np.median(dd):+.0f}ms  "
                  f"|med| {np.median(np.abs(dd)):.0f}ms", flush=True)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
