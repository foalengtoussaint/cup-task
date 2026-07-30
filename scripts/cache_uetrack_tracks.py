"""Cache detect-once UETrack cup tracks for the v3 pipeline (the cup that drives markerless phases).

WHY. The Murphy consolidation grid scores every pose/smoother/flow variant END-TO-END: each MMC arm
segments with its OWN markerless cup track (v3 = detect-once UETrack + greedy consensus). Those tracks
were cached once for P07/P08/P13 but never for the other OMC participants (P15/P17/P19), so the full
grid could only ever score P07/P08. This builds the missing tracks so the grid runs on all 5.

WHAT. For each trial, for each GOOD camera: seed UETrack from that camera's FIRST cached YOLO cup box
(detect-once), then track every staged-clip frame. Cameras with no cup detection anywhere are dropped.
Output matches cache_tracks.py / bench_v3.py exactly:

    cache/tracks_uetrack/<part>__<trial>__uetrack__fs1.json
      {frame: {cam: {"yolo": [x,y,w,h]|null, "trk": [cx,cy]|null, "seeded": bool}}}

The consumer (cup_task.cup_track.track_cup_3d_from_cache) only reads `trk` per cam per frame and runs
consensus3 over whatever cameras have a point that frame -- so a camera that seeds late (its first
detection at frame 46) simply contributes from frame 46 on. No GPU re-detection: seeds come from the
already-cached <clip>.<cam>.cup.json boxes.

Cost (RTX 3060 Ti, measured): ~22 rig-fr/s incl. 1080p decode -> ~21s / 5-cam trial. ~50-60 min for
the P15/P17/P19 set (159 trials). ONE model load per process (~8s). Idempotent: skips existing files.

    python scripts/cache_uetrack_tracks.py --parts P15 P17 P19          # the grid's missing set
    python scripts/cache_uetrack_tracks.py --parts P15 --limit 1        # smoke test one trial
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H  # noqa: E402

OUT = ROOT / "cache" / "tracks_uetrack"
DETS = ROOT / "cache" / "delta"          # <part>/dets/<clip>.<cam>.cup.json
STAGED = ROOT / "cache" / "delta"        # <part>/staged/<clip>.<cam>.mp4


def _cam_num(cam: str) -> int:
    return int(cam.split("_")[1])


def _clip_stem(part: str, trial: str) -> str:
    """Staged clips are named delta_<part>_<trial>.<camnum>.mp4."""
    return f"delta_{part}_{trial}"


def _staged_dir(part: str) -> Path:
    d = STAGED / part / "staged"
    return d if d.is_dir() else STAGED / part / "recut"


def _seed_box_xywh(part: str, trial: str, cam: str):
    """First cached YOLO cup box for this cam, as (seed_frame, [x,y,w,h]) or None.

    The .cup.json box is xyxy; UETrack.init wants xywh (top-left + size)."""
    stem = _clip_stem(part, trial)
    f = DETS / part / "dets" / f"{stem}.{_cam_num(cam)}.cup.json"
    if not f.exists():
        return None
    frames = json.loads(f.read_text()).get("frames", [])
    for fr in frames:
        b = fr.get("box")
        if b:
            x1, y1, x2, y2 = b
            return int(fr["frame"]), [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
    return None


def _center(xywh):
    x, y, w, h = xywh
    return [x + w / 2.0, y + h / 2.0]


def cache_trial(batch_cls, part, trial, calib, overwrite=False):
    """Build one trial's per-camera UETrack track file. Returns (path, n_frames, n_cams_used)."""
    cf = OUT / f"{part}__{trial}__uetrack__fs1.json"
    if cf.exists() and not overwrite:
        return cf, None, None, True

    stem = _clip_stem(part, trial)
    sd = _staged_dir(part)
    # cameras we can actually use: good-cam AND has a clip AND has a seed box
    seeds, vids = {}, {}
    for cam in calib:
        v = sd / f"{stem}.{_cam_num(cam)}.mp4"
        s = _seed_box_xywh(part, trial, cam)
        if v.exists() and s is not None:
            vids[cam] = v
            seeds[cam] = s                       # (seed_frame, xywh)
    if len(vids) < 2:
        return cf, 0, len(vids), False           # can't triangulate; write nothing

    cams = sorted(vids, key=_cam_num)
    caps = {c: cv2.VideoCapture(str(vids[c])) for c in cams}
    nframes = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in cams)

    b = batch_cls(len(cams))
    cam_idx = {c: i for i, c in enumerate(cams)}
    seeded = {c: False for c in cams}
    rec: dict[str, dict] = {}

    for f in range(nframes):
        imgs = {}
        for c in cams:
            ok, im = caps[c].read()
            if ok:
                imgs[c] = im                     # BGR
        row = {}
        # 1) seed any camera whose detect-once frame is now (RGB for the tracker)
        for c in cams:
            if not seeded[c] and c in imgs and seeds[c][0] == f:
                b.init(cam_idx[c], imgs[c][:, :, ::-1].copy(), seeds[c][1])
                seeded[c] = True
                row[c] = {"yolo": seeds[c][1], "trk": _center(seeds[c][1]), "seeded": True}
        # 2) batched update for all already-seeded cameras with a frame this step
        rgbs = [None] * len(cams)
        for c in cams:
            if seeded[c] and c in imgs and c not in row:
                rgbs[cam_idx[c]] = imgs[c][:, :, ::-1].copy()
        if any(r is not None for r in rgbs):
            out = b.update(rgbs)                  # list per camera, xywh|None
            for c in cams:
                if rgbs[cam_idx[c]] is not None:
                    xywh = out[cam_idx[c]]
                    row[c] = {"yolo": None,
                              "trk": (None if xywh is None else _center(xywh)),
                              "seeded": False}
        # cameras not yet seeded (or no frame) get an explicit null so the schema is dense
        for c in cams:
            if c not in row:
                row[c] = {"yolo": None, "trk": None, "seeded": False}
        rec[str(f)] = row

    for c in caps.values():
        c.release()
    OUT.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(rec))
    return cf, nframes, len(cams), False


def _trials_for(part):
    """Trials to build = the load_clean survivors for this participant (what the grid will score)."""
    import gnn_train as GT
    return sorted({t["trial"] for t in GT.load_clean(need_reproj=False) if t["part"] == part})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["P15", "P17", "P19"])
    ap.add_argument("--limit", type=int, default=None, help="cap trials/participant (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    H.use_good_cams()
    from uetrack_wrap import UETrackBatch

    # count work up front for a real ETA in the log
    work = []
    for p in a.parts:
        trials = _trials_for(p)
        if a.limit:
            trials = trials[:a.limit]
        work += [(p, t) for t in trials]
    print(f"cache_uetrack_tracks: {len(work)} trials over {a.parts} -> {OUT}", flush=True)

    t_start = time.perf_counter()
    done = skipped = empty = 0
    for i, (part, trial) in enumerate(work):
        calib = H._load_calib_mm(part)
        if part in H.GOOD_CAMS:
            calib = {c: v for c, v in calib.items() if c in H.GOOD_CAMS[part]}
        t0 = time.perf_counter()
        cf, nfr, ncam, existed = cache_trial(UETrackBatch, part, trial, calib,
                                              overwrite=a.overwrite)
        dt = time.perf_counter() - t0
        if existed:
            skipped += 1
            tag = "SKIP (exists)"
        elif nfr == 0:
            empty += 1
            tag = f"EMPTY (only {ncam} usable cams)"
        else:
            done += 1
            tag = f"{nfr}fr x{ncam}cam  {dt:.1f}s"
        el = time.perf_counter() - t_start
        eta = el / (i + 1) * (len(work) - i - 1)
        print(f"  [{i+1}/{len(work)}] {part}/{trial:32s} {tag}   "
              f"(elapsed {el/60:.1f}m, eta {eta/60:.1f}m)", flush=True)

    print(f"\nPROCESSING CHECK: {done} built, {skipped} skipped-existing, {empty} empty "
          f"(too few cams), {len(work)} total", flush=True)
    print(f"wrote to {OUT}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
