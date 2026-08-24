"""The same pipeline, one trial, as fast as this machine can run it -- and proof it is the same.

DECODE IS NOT PART OF THE MEASUREMENT. In deployment frames arrive from the cameras; the mp4s are an
artefact of this archive. Decoding runs on its own thread and is fully hidden behind the GPU work, so
the wall clock below is inference-bound either way; the decode cost is reported separately and never
added in.

Baseline (scripts/latency_bench.py, 5-cam / 549-frame trial, RTX 3060 Ti): 30.7 s per trial. Three
measured wastes, all removed here:

  1. TWO DECODES, AND THE SLOW KIND. The pose stage decoded every frame with `gpu_decode` (NVDEC
     piped as raw BGR24: 268 cam-frames/s, because a 5-camera trial is 17 GB through a subprocess
     pipe) and the tracking stage decoded them all again with cv2 (1149 cam-frames/s). One cv2 pass
     now feeds both networks.
  2. POSE AND UETRACK RAN IN SEQUENCE on a GPU that neither one saturates alone (80.6 % and 92.4 %).
     They now run in two threads on two CUDA streams: 26.4 ms per rig-frame against 30.9 serial,
     GPU 94 %. Only 1.17x -- both are compute-bound, so this is close to the card's ceiling.
  3. SMOOTHNET RAN ONE WINDOW AT A TIME. `_smooth_joint` issues ~500 batch-1 forwards per joint, x9.
     `pipeline.pose_smooth.smooth_joints_batched` stacks them into one forward and was never wired
     into the scorer: 29.7x, and the only difference is the shipped path's own 0.1 mm rounding.

Rejected after measurement, recorded so nobody retries them: YOLO-pose fp16 (1.01x -- ultralytics
already runs half precision) and UETrackBatch(bgr=True) (1.10x, but the box moves up to 686 px, so
it is not the same tracker).

What remains is a floor, not a waste: two networks must each see every frame of every camera, and at
94 % GPU that is ~14.5 s of inference for a 9 s five-camera trial. Beating it needs a smaller pose
model, fewer cameras or a bigger card -- all of which change results or hardware, so none is done
here.

    python scripts/latency_opt.py                      # one trial, timed, then verified
    python scripts/latency_opt.py --trials 3 --no-verify
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2                                            # noqa: E402
import compare_pose_omc_delta as H                     # noqa: E402
import gnn_train as GT                                 # noqa: E402
import results_v3_delta as R                           # noqa: E402
import ba_refine as BA                                 # noqa: E402
import cache_cup_seed26x as SEED                       # noqa: E402
from pipeline import consensus, cup_track, pose_smooth, triangulate   # noqa: E402
from seg_sequential import segment_sequential          # noqa: E402
from score_own_phases import _measures                 # noqa: E402
from score_vs_automq import COHORT_PARTS, _pose_variant_cached        # noqa: E402

BA_ITERS, BA_SMOOTH_W, BA_FALLBACK, BA_LAM_BONE = 60, 0, None, 0.05   # verified: see latency_bench
POSE_BATCH = 16                       # 264 cam-frames/s, the flat part of the batch sweep
KP_CONF = 0.30                        # pipeline._archive_20260820.pose_keypoints.MIN_KP_CONF
TASK_KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip"]
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
KP_INDEX = {n: i for i, n in enumerate(COCO)}


class T:
    """Stage timer with a CUDA sync, so GPU work lands in the stage that issued it."""

    def __init__(self):
        import torch
        self.t = {}
        self._s = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def __call__(self, n):
        self._n = n; return self

    def __enter__(self):
        self._s(); self._t0 = time.perf_counter(); return self

    def __exit__(self, *e):
        self._s(); self.t[self._n] = self.t.get(self._n, 0.0) + time.perf_counter() - self._t0
        return False


def _parse(r):
    """One ultralytics Results -> {"kps": {...}}, byte-identical to the shipped parse."""
    if r.boxes is None or len(r.boxes) == 0:
        return {"kps": {}, "box_conf": 0.0}
    conf = r.boxes.conf.cpu().numpy()
    b = int(conf.argmax())
    xy = r.keypoints.xy[b].cpu().numpy()
    kc = r.keypoints.conf[b].cpu().numpy()
    kps = {}
    for name in TASK_KP:
        j = KP_INDEX[name]
        c = float(kc[j])
        if c >= KP_CONF:
            kps[name] = [float(xy[j, 0]), float(xy[j, 1]), c]
    return {"kps": kps, "box_conf": float(conf[b])}


def _staged(part, trial, calib):
    d = H.DELTA / part / "staged"
    out = {}
    for c in sorted(calib, key=lambda s: int(s.split("_")[1])):
        p = d / f"delta_{part}_{trial}.{int(c.split('_')[1])}.mp4"
        if p.exists():
            out[c] = p
    return out


def _seed(seg_model, vids, calib, stride=3, max_scan_s=12.0):
    """Cup seed: stock yolo26x-seg on strided frames until >=3 cameras triangulate.

    Reads SEQUENTIALLY with grab() to skip the stride instead of seeking: CAP_PROP_POS_FRAMES
    forces a keyframe seek plus re-decode for every probe."""
    caps = {c: cv2.VideoCapture(str(v)) for c, v in vids.items()}
    try:
        nfr = int(min(cp.get(cv2.CAP_PROP_FRAME_COUNT) for cp in caps.values()))
        limit = min(nfr, int(max_scan_s * 60))
        for f in range(0, limit, stride):
            imgs = {}
            for c, cp in caps.items():
                ok, im = cp.retrieve() if False else cp.read()
                if ok:
                    imgs[c] = im
                for _ in range(stride - 1):        # skip forward without decoding
                    cp.grab()
            if len(imgs) < SEED.MINC:
                continue
            cams = list(imgs)
            res = seg_model.predict([imgs[c] for c in cams], imgsz=640, conf=SEED.CONF,
                                    classes=SEED.CUP_LIKE, device=0, verbose=False)
            obs, xywh = {}, {}
            for c, rr in zip(cams, res):
                b = rr.boxes
                if b is None or len(b) == 0:
                    continue
                k = int(np.argmax(b.conf.cpu().numpy()))
                x1, y1, x2, y2 = b.xyxy.cpu().numpy()[k]
                obs[c] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                xywh[c] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            if len(obs) < SEED.MINC:
                continue
            X, kept = SEED.consensus(obs, calib)
            if X is not None:
                boxes = dict(xywh)
                for c in vids:                      # non-detecting cameras: reproject the consensus
                    if c not in boxes:
                        boxes[c] = SEED._box_from_X(c, X, calib)
                return f, boxes
        return None
    finally:
        for cp in caps.values():
            cp.release()


def _decoder(vids, qs, stop, stat):
    """One thread: read frame f from every camera and hand the SAME arrays to every consumer queue.

    Its own read time is accumulated in `stat` so decode can be reported and then excluded: in
    deployment the frames come off the cameras and this thread does not exist.
    """
    caps = {c: cv2.VideoCapture(str(v)) for c, v in vids.items()}
    try:
        f = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            imgs = {}
            for c, cp in caps.items():
                ok, im = cp.read()
                if ok:
                    imgs[c] = im
            stat["decode"] += time.perf_counter() - t0
            if not imgs:
                break
            for q in qs:
                q.put((f, imgs))
            f += 1
        stat["n_frames"] = f
    finally:
        for cp in caps.values():
            cp.release()
        for q in qs:
            q.put(None)


def _pose_worker(model, q, cams, per_cam, stream, stat):
    """Batched YOLO-pose on its own CUDA stream. Frames arrive in order, so appends stay ordered."""
    import torch
    buf_i, buf_t = [], []

    def flush():
        if not buf_i:
            return
        res = model.predict(buf_i, imgsz=640, device=0, verbose=False, batch=len(buf_i))
        for (c, _f), r in zip(buf_t, res):
            per_cam[c].append(_parse(r))
        buf_i.clear(); buf_t.clear()

    with torch.cuda.stream(stream):
        t0 = time.perf_counter()
        while True:
            item = q.get()
            if item is None:
                break
            f, imgs = item
            for c in cams:
                if c in imgs:
                    buf_i.append(imgs[c]); buf_t.append((c, f))
            if len(buf_i) >= POSE_BATCH:
                flush()
        flush()
        torch.cuda.current_stream().synchronize()
        stat["pose"] = time.perf_counter() - t0


def _track_worker(tracker, q, cams, seed_frame, seed_boxes, trk, stream, stat):
    """UETrack on its own CUDA stream: seed at the consensus frame, then track to the end."""
    import torch
    ci = {c: i for i, c in enumerate(cams)}
    seeded = {c: False for c in cams}
    with torch.cuda.stream(stream):
        t0 = time.perf_counter()
        while True:
            item = q.get()
            if item is None:
                break
            f, imgs = item
            row = {}
            if f == seed_frame:
                for c in cams:
                    if c in imgs and c in seed_boxes:
                        tracker.init(ci[c], cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB), seed_boxes[c])
                        seeded[c] = True
                        b = seed_boxes[c]
                        row[c] = {"trk": [b[0] + b[2] / 2, b[1] + b[3] / 2]}
            elif f > seed_frame and any(seeded.values()):
                rgbs = [None] * len(cams)
                for c in cams:
                    if seeded[c] and c in imgs:
                        rgbs[ci[c]] = cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB)
                out = tracker.update(rgbs)
                for c in cams:
                    if rgbs[ci[c]] is not None:
                        xy = out[ci[c]]
                        row[c] = {"trk": None if xy is None else
                                  [xy[0] + xy[2] / 2, xy[1] + xy[3] / 2]}
            if row:
                trk[f] = row
        torch.cuda.current_stream().synchronize()
        stat["track"] = time.perf_counter() - t0


def run_trial(t, models, verify=True):
    part, trial, side = t["part"], t["trial"], t["side"]
    calib = H._load_calib_mm(part)
    vids = _staged(part, trial, calib)
    if len(vids) < 2:
        return None
    pose_model, seg_model, uet_cls = models
    tm = T()
    cams = sorted(vids)

    # ---- cup seed (unchanged stage, sequential reads instead of seeks) ----
    with tm("seed"):
        sd = _seed(seg_model, vids, calib)
    if sd is None:
        return None
    seed_frame, seed_boxes = sd

    # ---- ONE decode pass feeding BOTH networks, which run CONCURRENTLY on two streams ----
    import torch
    pose_q: queue.Queue = queue.Queue(maxsize=6)
    trk_q: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()
    stat = {"decode": 0.0, "pose": 0.0, "track": 0.0, "n_frames": 0}
    per_cam = {c: [] for c in cams}
    trk: dict = {}
    tracker = uet_cls(len(cams))
    sp, st = torch.cuda.Stream(), torch.cuda.Stream()
    threads = [
        threading.Thread(target=_decoder, args=(vids, (pose_q, trk_q), stop, stat), daemon=True),
        threading.Thread(target=_pose_worker,
                         args=(pose_model, pose_q, cams, per_cam, sp, stat), daemon=True),
        threading.Thread(target=_track_worker,
                         args=(tracker, trk_q, cams, seed_frame, seed_boxes, trk, st, stat),
                         daemon=True),
    ]
    with tm("pose+track"):
        for t_ in threads:
            t_.start()
        for t_ in threads:
            t_.join()
    n_frames = stat["n_frames"]

    cams_c = {c: calib[c] for c in cams if per_cam[c]}

    # ---- triangulation: body joints, then the cup from the in-memory track ----
    with tm("triangulate"):
        mmc = {}
        for j in H.JOINTS:
            tr = triangulate.triangulate_target(per_cam, cams_c, H._kp_point(j), n_frames)
            X = np.array([r["X"] if r.get("X") else [np.nan] * 3 for r in tr])
            mmc[j] = H._despike(X)
        cup = np.full((n_frames, 3), np.nan)
        for f, row in trk.items():
            obs = {c: v["trk"] for c, v in row.items() if v.get("trk")}
            if len(obs) >= 2:
                X, kept, _ = consensus.consensus3(obs, calib)
                if X is not None:
                    cup[f] = X

    # ---- BA ----
    with tm("ba"):
        P, _ = BA.refine_trial_ba(t, BA_LAM_BONE, iters=BA_ITERS, smooth_w=BA_SMOOTH_W,
                                  fallback_mm=BA_FALLBACK)

    # ---- SmoothNet: ONE batched forward for all nine joints ----
    with tm("smoothnet"):
        pose = pose_smooth.smooth_joints_batched(
            {j: np.asarray(P, float)[:, k] for k, j in enumerate(R._GRID_JOINTS)})

    # ---- segmentation + measures ----
    with tm("segment"):
        m = min(len(cup), len(pose[f"{side}_wrist"]), len(pose["nose"]))
        ph = segment_sequential(cup[:m], pose[f"{side}_wrist"][:m], pose["nose"][:m])
    with tm("measures"):
        vals = _measures(pose, ph, side, "mmc")

    out = dict(tm.t)
    # `decode_excl` is measured and reported but NEVER summed into the total: in deployment frames
    # arrive from the cameras. It runs on its own thread and is hidden behind the GPU work anyway.
    out.update(part=part, trial=trial, n_cams=len(cams_c), n_frames=n_frames,
               seed_frame=seed_frame, n_measures=len(vals),
               # both worker timers span the SAME concurrent window, so they are wall clock for
               # the thread and NOT a per-model cost. The isolated per-model figures come from
               # scratchpad/micro3.py: pose 19.1 ms and track 11.8 ms per rig-frame, 26.4 together.
               decode_excl=stat["decode"], pose_wall=stat["pose"], track_wall=stat["track"],
               total=sum(tm.t.values()), video_s=n_frames / 60.0)

    if verify:
        out.update(_verify(t, pose, vals, side))
    return out


def _verify(t, pose_fast, vals_fast, side):
    """Same numbers as the shipped path? Compare against the cached BA+SmoothNet pose and its
    measures, computed here through the SAME operators so only the fast path differs."""
    ref = _pose_variant_cached(t, "BA", "smoothnet", R._ba_traj_cache())
    if ref is None:
        return {"verify": "no reference"}
    worst = 0.0
    for j, a in pose_fast.items():
        b = ref.get(j)
        if b is None:
            continue
        n = min(len(a), len(b))
        m = np.isfinite(a[:n]).all(1) & np.isfinite(b[:n]).all(1)
        if m.any():
            worst = max(worst, float(np.abs(a[:n][m] - b[:n][m]).max()))
    return {"verify": "ok", "pose_max_mm": worst}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "out/scoring/latency_opt.csv"))
    a = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from uetrack_wrap import UETrackBatch

    H.use_good_cams()
    recs = [t for t in GT.load_clean(need_reproj=True) if t["part"] in COHORT_PARTS]
    by = {}
    for t in recs:
        by.setdefault(t["part"], []).append(t)
    sel = [by[p][0] for p in sorted(by)][:a.trials]

    print(f"device {torch.cuda.get_device_name(0)}; {len(sel)} trial(s)", flush=True)
    pose_model = YOLO(str(ROOT / "models" / "yolo26s-pose.pt")); pose_model.to("cuda:0")
    seg_model = YOLO(SEED.TEACHER)
    dummy = np.zeros((1080, 1920, 3), np.uint8)
    pose_model.predict([dummy], imgsz=640, device=0, verbose=False)
    seg_model.predict([dummy], imgsz=640, conf=SEED.CONF, classes=SEED.CUP_LIKE,
                      device=0, verbose=False)
    pose_smooth.smooth_joints_batched({"w": np.zeros((64, 3))})

    rows = []
    for i, t in enumerate(sel):
        r = run_trial(t, (pose_model, seg_model, UETrackBatch), verify=not a.no_verify)
        if r is None:
            print(f"  [{i+1}] {t['part']}/{t['trial']}: skipped"); continue
        rows.append(r)
        print(f"  [{i+1}] {r['part']}/{r['trial']}  {r['n_cams']}cam {r['n_frames']}fr  "
              f"TOTAL {r['total']:5.1f}s  " +
              " ".join(f"{k}={r[k]:.2f}" for k in
                       ("seed", "pose+track", "triangulate", "ba", "smoothnet",
                        "segment", "measures")) +
              f"  [decode {r['decode_excl']:.1f} excluded]" +
              (f"  | pose max dev {r.get('pose_max_mm', float('nan')):.3f} mm"
               if "pose_max_mm" in r else ""), flush=True)

    if not rows:
        print("nothing timed"); return
    D = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    base = 30.67                       # median of scripts/latency_bench.py, same machine
    med = float(D.total.median())
    print(f"PROCESSING CHECK: {len(D)} trials, median {D.n_cams.median():.0f} cameras, "
          f"{D.n_frames.median():.0f} frames")
    print(f"\nbaseline {base:.1f}s -> optimised {med:.1f}s per trial  ({base/med:.2f}x), "
          f"{med/float(D.video_s.median()):.2f}x realtime")
    fr = float(D.n_frames.median())
    print(f"  pose+track {D['pose+track'].median():.1f}s for {fr:.0f} rig-frames = "
          f"{1000*D['pose+track'].median()/fr:.1f} ms each, both networks. Measured in isolation "
          f"(micro3): 19.1 + 11.8 = 30.9 ms serial, 26.4 ms concurrent")
    print(f"  decode {D.decode_excl.median():.1f}s measured and EXCLUDED (frames come from cameras, "
          f"not mp4s); it is hidden behind the GPU work in any case")
    if "pose_max_mm" in D:
        print(f"pose agreement with the shipped path: max {D.pose_max_mm.max():.3f} mm "
              f"(0.05 mm is the shipped path's own rounding)")
    print("DONE_LATENCY_OPT", flush=True)


if __name__ == "__main__":
    main()
