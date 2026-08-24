"""Per-trial LATENCY of the shipped pipeline, stage by stage, on real trials.

Methods §II-E defines the quantity as wall-clock from the end of a trial's recording to the
emission of its measures. Two numbers answer that, and this script reports both:

  ONLINE-CAPABLE   decode + YOLO-pose, cup seed, UETrack. These need raw PIXELS and can run
                   while the trial is still being recorded, so a streaming deployment pays
                   none of it after the recording stops.
  OFFLINE TAIL     triangulation, bundle adjustment, SmoothNet, segmentation, measures. These
                   need the whole trial and can only start once it ends. This is the §II-E
                   number under a streaming front end.

Our capture is file-based, so the honest end-to-end figure for THIS implementation is the sum.
Both are printed; do not quote one as the other.

Every stage is timed on a forced cache MISS -- pose is re-detected from the mp4s, the cup seed
is re-scanned, UETrack is re-run into a scratch directory (the shipped cache is never touched),
BA re-solves and SmoothNet re-runs. Model LOAD time is excluded from the per-trial figures and
reported separately as a one-off startup cost, because it is paid once per session, not per trial.

    python scripts/latency_bench.py --trials 20
    python scripts/latency_bench.py --trials 4 --out /tmp/lat.csv     # smoke test
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "archive"))

# `cache_pose_cohort` (the shipped pose-caching path, still in scripts/archive) imports
# pipeline.pose_keypoints, which now lives in pipeline/_archive_20260820. Bind it back under its
# old name so the DETECTION AND PARSE timed here are the ones that produced cache/pose_models,
# rather than a reimplementation that might differ.
def _bind_pose_keypoints():
    import importlib.util
    p = ROOT / "pipeline" / "_archive_20260820" / "pose_keypoints.py"
    if "pipeline.pose_keypoints" in sys.modules or not p.exists():
        return
    spec = importlib.util.spec_from_file_location("pipeline.pose_keypoints", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["pipeline.pose_keypoints"] = m
    spec.loader.exec_module(m)


_bind_pose_keypoints()

import compare_pose_omc_delta as H          # noqa: E402
import gnn_train as GT                      # noqa: E402
import results_v3_delta as R                # noqa: E402
import ba_refine as BA                      # noqa: E402
import cache_uetrack_tracks as CUT          # noqa: E402
import cache_cup_seed26x as SEED            # noqa: E402
from pipeline import cup_track, triangulate  # noqa: E402
from seg_sequential import segment_sequential  # noqa: E402
from score_own_phases import _measures      # noqa: E402
from score_vs_automq import COHORT_PARTS    # noqa: E402

# The shipped BA settings. lam_bone=0.05 is VERIFIED, not read off the source: re-solving six
# trials at 0.00 and 0.05 reproduces cache/ba_traj/traj_sw0_noguard.npz to 0.0000 mm RMS at 0.05
# and misses it by 4-15 mm at 0.00 (scratchpad/ba_lam_check.py, 2026-08-21).
BA_ITERS, BA_SMOOTH_W, BA_FALLBACK, BA_LAM_BONE = 60, 0, None, 0.05
POSE_BATCH = 32
JOINTS = H.JOINTS

STAGES_ONLINE = ["pose", "cup_seed", "cup_track"]
STAGES_OFFLINE = ["triangulate", "ba", "smoothnet", "segment", "measures"]
STAGES = STAGES_ONLINE + STAGES_OFFLINE


class Timer:
    """perf_counter with a CUDA sync, so GPU work is not credited to the next stage."""

    def __init__(self):
        self.t = {}
        try:
            import torch
            self._sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        except Exception:
            self._sync = lambda: None

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self._sync(); self._t0 = time.perf_counter(); return self

    def __exit__(self, *exc):
        self._sync(); self.t[self._name] = time.perf_counter() - self._t0
        return False


def _staged(part, trial, calib):
    """{cam_idx: mp4} for the trial's GOOD cameras that actually have a clip."""
    d = H.DELTA / part / "staged"
    out = {}
    for c in calib:
        n = int(c.split("_")[1])
        p = d / f"delta_{part}_{trial}.{n}.mp4"
        if p.exists():
            out[n] = p
    return out


def _cup_seed_scan(model, vids, calib, cv2, stride=3, max_scan_s=12.0):
    """cache_cup_seed26x's scan, inlined so it can be timed on its own: walk forward until the
    cup-like detections triangulate across >=3 cameras. Returns the seed dict or None."""
    caps = {f"cam_{n}": cv2.VideoCapture(str(v)) for n, v in vids.items()}
    caps = {c: cp for c, cp in caps.items() if c in calib}
    try:
        if len(caps) < SEED.MINC:
            return None
        nfr = int(min(cp.get(cv2.CAP_PROP_FRAME_COUNT) for cp in caps.values()))
        limit = min(nfr, int(max_scan_s * 60))
        for f in range(0, limit, stride):
            imgs = {}
            for c, cp in caps.items():
                cp.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cp.read()
                if ok:
                    imgs[c] = im
            if len(imgs) < SEED.MINC:
                continue
            cams = list(imgs)
            res = model.predict([imgs[c] for c in cams], imgsz=640, conf=SEED.CONF,
                                classes=SEED.CUP_LIKE, device=0, verbose=False)
            obs = {}
            for c, rr in zip(cams, res):
                b = rr.boxes
                if b is None or len(b) == 0:
                    continue
                k = int(np.argmax(b.conf.cpu().numpy()))
                x1, y1, x2, y2 = b.xyxy.cpu().numpy()[k]
                obs[c] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if len(obs) < SEED.MINC:
                continue
            X, kept = SEED.consensus(obs, calib)
            if X is not None:
                return {"frame": f, "n_scanned": f // stride + 1, "n_kept": len(kept)}
        return None
    finally:
        for cp in caps.values():
            cp.release()


def bench_trial(t, models, scratch, cv2):
    """Time every stage for one trial. Returns {stage: seconds} plus a few shape counters."""
    part, trial, side = t["part"], t["trial"], t["side"]
    calib = H._load_calib_mm(part)
    vids = _staged(part, trial, calib)
    if len(vids) < 2:
        return None
    tm = Timer()
    pose_model, seg_model, uet_cls = models

    # ---- 1. decode + YOLO-pose, every camera, every frame (the shipped batched/NVDEC path) ----
    import cache_pose_cohort as CPC
    with tm("pose"):
        per_cam, n_frames = CPC._run_rep(pose_model, vids, POSE_BATCH, 0)
    per_cam = {c: v for c, v in per_cam.items() if c in calib}
    cams = {c: calib[c] for c in per_cam}
    if len(cams) < 2:
        return None

    # ---- 2. cup seed: stock yolo26x-seg, scanned only until it triangulates ----
    with tm("cup_seed"):
        seed = _cup_seed_scan(seg_model, vids, calib, cv2)

    # ---- 3. UETrack, per camera, seed frame -> end. Written to scratch, never the real cache ----
    with tm("cup_track"):
        cf, nfr, ncam, _ = CUT.cache_trial(uet_cls, part, trial, calib,
                                           overwrite=True, seeds26x=True)

    # ---- 4. triangulation: 9 body joints + the cup, from points already in memory ----
    with tm("triangulate"):
        mmc = {}
        for j in JOINTS:
            tr = triangulate.triangulate_target(per_cam, cams, H._kp_point(j), n_frames)
            X = np.array([r["X"] if r.get("X") else [np.nan] * 3 for r in tr])
            mmc[j] = H._despike(X)
        cup3d = None
        if Path(cf).exists():
            ct = cup_track.track_cup_3d_from_cache(cf, calib, refine_ba=True)
            cup3d = np.array([r["X"] if r.get("X") else [np.nan] * 3 for r in ct])

    # ---- 5. bundle adjustment ----
    with tm("ba"):
        P, _info = BA.refine_trial_ba(t, BA_LAM_BONE, iters=BA_ITERS, smooth_w=BA_SMOOTH_W,
                                      fallback_mm=BA_FALLBACK)

    # ---- 6. SmoothNet, one solve per joint ----
    with tm("smoothnet"):
        pose = {j: R._smooth_joint(np.asarray(P, float)[:, k]) for k, j in enumerate(R._GRID_JOINTS)}

    # ---- 7. segmentation from the markerless cup ----
    other = "right" if side == "left" else "left"
    wrist = pose.get(f"{side}_wrist")
    nose = pose.get("nose")
    with tm("segment"):
        ph = None
        if cup3d is not None and wrist is not None and nose is not None:
            m = min(len(cup3d), len(wrist), len(nose))
            ph = segment_sequential(cup3d[:m], wrist[:m], nose[:m])

    # ---- 8. measures ----
    with tm("measures"):
        vals = _measures(pose, ph, side, "mmc") if ph else {}

    out = dict(tm.t)
    out.update(part=part, trial=trial, n_cams=len(cams), n_frames=n_frames,
               seed_frame=(seed or {}).get("frame", -1),
               seed_scanned=(seed or {}).get("n_scanned", -1),
               n_measures=len(vals), segmented=int(bool(ph)))
    return out


def _pick(trials, k):
    """k trials spread evenly over participants, so no single session dominates the median."""
    by = {}
    for t in trials:
        by.setdefault(t["part"], []).append(t)
    parts = sorted(by)
    out, i = [], 0
    while len(out) < k and i < 200:
        for p in parts:
            if i < len(by[p]) and len(out) < k:
                out.append(by[p][i])
        i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "out/scoring/latency_bench.csv"))
    a = ap.parse_args(argv)

    import cv2
    import torch
    from ultralytics import YOLO
    from uetrack_wrap import UETrackBatch

    H.use_good_cams()
    recs = [t for t in GT.load_clean(need_reproj=True) if t["part"] in COHORT_PARTS]
    sel = _pick(recs, a.trials)
    print(f"{len(recs)} admitted trials; timing {len(sel)} of them "
          f"({len({t['part'] for t in sel})} participants)", flush=True)

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {dev}, torch {torch.__version__}", flush=True)

    # ---- one-off model load, reported separately: paid once per session, not per trial ----
    t0 = time.perf_counter()
    pose_model = YOLO(str(ROOT / "models" / "yolo26s-pose.pt")); pose_model.to("cuda:0")
    t_pose_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    seg_model = YOLO(SEED.TEACHER)
    t_seg_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    _warm_uet = UETrackBatch(2)
    t_uet_load = time.perf_counter() - t0
    print(f"model load (one-off): pose {t_pose_load:.1f}s, cup-seg {t_seg_load:.1f}s, "
          f"UETrack {t_uet_load:.1f}s", flush=True)

    # warm every GPU path once so the first trial is not charged for lazy init
    dummy = np.zeros((1080, 1920, 3), np.uint8)
    pose_model.predict([dummy], imgsz=640, device=0, verbose=False)
    seg_model.predict([dummy], imgsz=640, conf=SEED.CONF, classes=SEED.CUP_LIKE,
                      device=0, verbose=False)
    R._smooth_joint(np.tile(np.array([0.0, 0.0, 0.0]), (200, 1)))

    scratch = Path(tempfile.mkdtemp(prefix="lat_tracks_"))
    CUT.OUT26X = scratch                      # never overwrite the shipped tracks
    rows = []
    try:
        for i, t in enumerate(sel):
            try:
                r = bench_trial(t, (pose_model, seg_model, UETrackBatch), scratch, cv2)
            except Exception as e:
                print(f"  [{i+1}/{len(sel)}] {t['part']}/{t['trial']} FAILED "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            if r is None:
                continue
            rows.append(r)
            tot = sum(r[s] for s in STAGES)
            print(f"  [{i+1}/{len(sel)}] {r['part']}/{r['trial']}  {r['n_cams']}cam "
                  f"{r['n_frames']}fr  total {tot:6.1f}s  "
                  + " ".join(f"{s}={r[s]:.1f}" for s in STAGES), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not rows:
        print("no trials timed"); return
    D = pd.DataFrame(rows)
    D["online"] = D[STAGES_ONLINE].sum(axis=1)
    D["offline"] = D[STAGES_OFFLINE].sum(axis=1)
    D["total"] = D["online"] + D["offline"]
    D["trial_s"] = D["n_frames"] / 60.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    print(f"PROCESSING CHECK: {len(D)} trials, "
          f"{int(D.segmented.sum())} segmented, median {D.n_cams.median():.0f} cameras, "
          f"median {D.n_frames.median():.0f} frames ({D.trial_s.median():.1f}s of video)")

    def q(v):
        return f"{np.median(v):7.2f} [{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]"

    print(f"\n{'stage':14}{'s / trial, median [IQR]':>28}{'% of total':>12}  where")
    tot = np.median(D.total)
    for s in STAGES:
        where = "GPU" if s in ("pose", "cup_seed", "cup_track", "ba", "smoothnet") else "CPU"
        print(f"{s:14}{q(D[s].values):>28}{100*np.median(D[s])/tot:11.1f}%  {where}")
    print(f"{'-'*66}")
    print(f"{'ONLINE':14}{q(D.online.values):>28}{100*np.median(D.online)/tot:11.1f}%  "
          f"needs pixels, can overlap recording")
    print(f"{'OFFLINE TAIL':14}{q(D.offline.values):>28}{100*np.median(D.offline)/tot:11.1f}%  "
          f"the §II-E number under a streaming front end")
    print(f"{'TOTAL':14}{q(D.total.values):>28}{100:11.1f}%  this implementation, from files")
    print(f"\nper second of video recorded: total {np.median(D.total/D.trial_s):.2f}x, "
          f"offline tail {np.median(D.offline/D.trial_s):.3f}x")
    print(f"model load, once per session: "
          f"{t_pose_load + t_seg_load + t_uet_load:.1f}s")
    print("DONE_LATENCY", flush=True)


if __name__ == "__main__":
    main()
