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


def _boxes_at(part, trial, cam, frame):
    """The YOLO cup box (xyxy) for this cam at a specific frame, or None."""
    stem = _clip_stem(part, trial)
    f = DETS / part / "dets" / f"{stem}.{_cam_num(cam)}.cup.json"
    if not f.exists():
        return None
    frames = json.loads(f.read_text()).get("frames", [])
    if frame < len(frames):
        fr = frames[frame]
        if fr.get("frame") == frame and fr.get("box"):
            return fr["box"]                                  # xyxy
    for fr in frames:                                         # fallback: search
        if fr.get("frame") == frame:
            return fr.get("box")
    return None


def _consensus_seed(part, trial, calib, cams, max_frame=180):
    """Find the EARLIEST frame where >=2 detecting cameras AGREE (consensus3 passes the reprojection
    gate), and return (seed_frame, X_mm, {cam: xywh_box}, kept_cams). This is the 'only seed if the
    cameras agree' gate you asked for -- a reprojected seed is only placed off a 3D the detecting
    cameras actually concur on. None if no agreeing frame in the first max_frame frames.

    Reprojection into the NON-detecting cameras is done by the caller (needs X + calib)."""
    from cup_task import consensus
    # per-cam first-detection frame, so we know where boxes start appearing
    first = {c: _seed_box_xywh(part, trial, c) for c in cams}
    first = {c: v for c, v in first.items() if v is not None}
    if len(first) < 2:
        return None
    lo = min(v[0] for v in first.values())
    for f in range(lo, min(lo + max_frame, 10_000)):
        obs = {}
        boxes = {}
        for c in cams:
            b = _boxes_at(part, trial, c, f)                 # xyxy
            if b:
                x1, y1, x2, y2 = b
                obs[c] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # box centre for consensus
                boxes[c] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # xywh
        if len(obs) < 2:
            continue
        X, kept, _ = consensus.consensus3({c: obs[c] for c in obs}, calib)
        if X is not None and len(kept) >= 2:
            return f, np.asarray(X, float), boxes, set(kept)
    return None


def _existing_cams(cf):
    """Cameras that the existing track file actually tracks (have any non-null trk)."""
    if not cf.exists():
        return set()
    try:
        d = json.loads(cf.read_text())
    except Exception:
        return set()
    cams = set()
    for fr in d.values():
        cams |= {c for c, v in fr.items() if v.get("trk") is not None}
    return cams


def _reproj_box(X, cam, ref_wh=(46.0, 46.0)):
    """Reproject world 3D X into `cam` -> an xywh box centred there, sized like a cup (ref_wh px).
    None if the point is behind the camera or off-image is unknown (caller checks bounds)."""
    from cup_task.kalman_3d import project
    uv, ok = project(cam, np.asarray(X, float))
    if not ok:
        return None
    w, h = ref_wh
    return [float(uv[0] - w / 2), float(uv[1] - h / 2), float(w), float(h)]


def cache_trial(batch_cls, part, trial, calib, overwrite=False, reproject=True):
    """Build one trial's per-camera UETrack track file. Returns (path, n_frames, n_cams_used, existed).

    reproject=True: seed cameras that have NO direct YOLO detection by reprojecting the CONSENSUS 3D
    (only when >=2 detecting cameras AGREE -- _consensus_seed gates on the reprojection consensus).
    Reprojected seeds are tagged `seeded_by:"reproject"` (direct ones `"yolo"`) so a downstream reader
    can drop the reprojected cameras. SKIP rule: if a track already exists and reproject-seed would add
    NO new camera, keep it (don't waste GPU) -- unless overwrite."""
    cf = OUT / f"{part}__{trial}__uetrack__fs1.json"
    stem = _clip_stem(part, trial)
    sd = _staged_dir(part)

    # all cameras with a clip (reprojected cams need a slot even with no detection)
    vids = {c: sd / f"{stem}.{_cam_num(c)}.mp4" for c in calib
            if (sd / f"{stem}.{_cam_num(c)}.mp4").exists()}
    if len(vids) < 2:
        return cf, 0, len(vids), False

    # direct-YOLO per-cam first-detection seeds
    direct = {c: _seed_box_xywh(part, trial, c) for c in vids}
    direct = {c: v for c, v in direct.items() if v is not None}       # (frame, xywh)

    # consensus seed (for reprojection into non-detecting cams)
    seed3 = _consensus_seed(part, trial, calib, list(vids)) if reproject else None
    reproj_cams = set()
    if seed3 is not None:
        sf, X, cboxes, kept = seed3
        for c in vids:
            if c not in direct:                                       # no direct detection -> reproject
                box = _reproj_box(X, calib[c])
                if box is not None:
                    reproj_cams.add(c)

    # cameras this build will track = direct-seeded + reproject-seeded
    build_cams = set(direct) | reproj_cams
    if len(build_cams) < 2:
        return cf, 0, len(build_cams), False

    # ONLY track cameras that don't already have a track. Existing (direct-YOLO) cameras are kept
    # verbatim from the old file -- never re-run (wastes GPU + would perturb good tracks). We track
    # ONLY the newly-reprojected cameras and MERGE them into the existing JSON.
    have = _existing_cams(cf) if not overwrite else set()
    new_cams = sorted(build_cams - have, key=_cam_num)
    if not new_cams:
        return cf, None, None, True                                  # nothing new -> keep as-is

    existing = {}
    if cf.exists():
        try:
            existing = json.loads(cf.read_text())
        except Exception:
            existing = {}
    merge = bool(existing) and not overwrite                          # merge new cams into old file

    cams = new_cams if merge else sorted(build_cams, key=_cam_num)
    caps = {c: cv2.VideoCapture(str(vids[c])) for c in cams}
    nframes = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in cams)

    # per-cam seed plan: (frame, xywh, origin)
    plan = {}
    for c in cams:
        if c in direct:
            plan[c] = (direct[c][0], direct[c][1], "yolo")
        elif c in reproj_cams:
            plan[c] = (seed3[0], _reproj_box(seed3[1], calib[c]), "reproject")

    b = batch_cls(len(cams))
    cam_idx = {c: i for i, c in enumerate(cams)}
    seeded = {c: False for c in cams}
    rec: dict[str, dict] = {}

    for f in range(nframes):
        imgs = {}
        for c in cams:
            ok, im = caps[c].read()
            if ok:
                imgs[c] = im
        row = {}
        for c in cams:
            if not seeded[c] and c in imgs and plan[c][0] == f:
                b.init(cam_idx[c], imgs[c][:, :, ::-1].copy(), plan[c][1])
                seeded[c] = True
                row[c] = {"yolo": plan[c][1] if plan[c][2] == "yolo" else None,
                          "trk": _center(plan[c][1]), "seeded": True,
                          "seeded_by": plan[c][2]}
        rgbs = [None] * len(cams)
        for c in cams:
            if seeded[c] and c in imgs and c not in row:
                rgbs[cam_idx[c]] = imgs[c][:, :, ::-1].copy()
        if any(r is not None for r in rgbs):
            out = b.update(rgbs)
            for c in cams:
                if rgbs[cam_idx[c]] is not None:
                    xywh = out[cam_idx[c]]
                    row[c] = {"yolo": None, "trk": (None if xywh is None else _center(xywh)),
                              "seeded": False, "seeded_by": plan[c][2]}
        for c in cams:
            if c not in row:
                row[c] = {"yolo": None, "trk": None, "seeded": False, "seeded_by": plan[c][2]}
        rec[str(f)] = row

    for c in caps.values():
        c.release()

    if merge:
        # graft the newly-tracked cameras' entries into the existing per-frame rows (keep old cams)
        for f in range(nframes):
            k = str(f)
            base = existing.get(k, {})
            base.update(rec.get(k, {}))                               # new cams override/add
            existing[k] = base
        out_rec = existing
        n_out_cams = len(_existing_cams(cf)) + len(cams)
    else:
        out_rec = rec
        n_out_cams = len(cams)

    OUT.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out_rec))
    return cf, nframes, n_out_cams, False


def _trials_for(part):
    """Trials to build for this participant.

    Prefer the load_clean survivors (the trials the grid will score). But a freshly-imported
    participant has no gnn_pairs cache yet (that MMC/OMC pairing is a downstream stage), so
    load_clean returns nothing for it. Fall back to the trials that have DETECTIONS pulled
    (<part>/dets/*.cup.json) -- that is exactly the set UETrack can build a cup track for."""
    import gnn_train as GT
    trials = sorted({t["trial"] for t in GT.load_clean(need_reproj=False) if t["part"] == part})
    if trials:
        return trials
    # fallback: derive from detection files  delta_<part>_<trial>.<cam>.cup.json
    import re, glob
    dets = glob.glob(str(ROOT / "cache" / "delta" / part / "dets" / f"delta_{part}_*.cup.json"))
    pat = re.compile(rf"delta_{re.escape(part)}_(.+)\.\d+\.cup\.json$")
    out = set()
    for f in dets:
        m = pat.search(Path(f).name)
        if m:
            out.add(m.group(1))
    return sorted(out)


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
