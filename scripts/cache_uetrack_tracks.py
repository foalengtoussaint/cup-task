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
# --seeds26x: seed from cache/cup_seed26x (stock COCO yolo26x-seg, scanned forward to the FIRST frame
# that triangulates >=3-cam/30px) instead of each camera's first clean3d_refill box. WORKLOG:927 --
# the finetune gets 8.2% >=3-cam consensus on P14 vs the COCO teacher's 77%; measured cohort-wide it
# detects 6-54% of frames and is near-blind in most views, so its per-camera first box can seed
# different objects in different cameras. Tracks go to a separate dir; the old cache is untouched.
SEED26X = ROOT / "cache" / "cup_seed26x"
OUT26X = ROOT / "cache" / "tracks_uetrack_26x"
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


def _consensus_seed(part, trial, calib, cams):
    """Find the EARLIEST frame where >=2 detecting cameras AGREE (consensus3 passes the reprojection
    gate), and return (seed_frame, X_mm, {cam: xywh_box}, kept_cams). This is the 'only seed if the
    cameras agree' gate you asked for -- a reprojected seed is only placed off a 3D the detecting
    cameras actually concur on. None if no frame ever has >=2 agreeing cameras.

    Scans the frames that actually HAVE >=2 detections (not a fixed early window), so it works for a
    cut clip AND for an uncut recording where the cup only enters late (e.g. cam co-detect at frame
    827). Reprojection into the NON-detecting cameras is done by the caller (needs X + calib)."""
    from cup_task import consensus
    # per-cam set of detection frames + the box at each
    det = {}
    for c in cams:
        stem = _clip_stem(part, trial)
        fp = DETS / part / "dets" / f"{stem}.{_cam_num(c)}.cup.json"
        if not fp.exists():
            continue
        boxes = {}
        for fr in json.loads(fp.read_text()).get("frames", []):
            if fr.get("box"):
                boxes[int(fr["frame"])] = fr["box"]          # xyxy
        if boxes:
            det[c] = boxes
    if len(det) < 2:
        return None
    # candidate frames = those where >=2 cams detect, earliest first
    from collections import Counter
    cnt = Counter()
    for c, boxes in det.items():
        for f in boxes:
            cnt[f] += 1
    cand = sorted(f for f, k in cnt.items() if k >= 2)
    for f in cand:
        obs, xywh = {}, {}
        for c in det:
            if f in det[c]:
                x1, y1, x2, y2 = det[c][f]
                obs[c] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                xywh[c] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        X, kept, _ = consensus.consensus3(obs, calib)
        if X is not None and len(kept) >= 2:
            return f, np.asarray(X, float), xywh, set(kept)
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


def _seed26x(part, trial):
    """(frame, {cam: xywh}, {cam: origin}) from the yolo26x-seg consensus seed, or None."""
    f = SEED26X / f"{part}__{trial}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if not d.get("boxes"):
        return None
    return int(d["frame"]), d["boxes"], d["origin"]


def cache_trial(batch_cls, part, trial, calib, overwrite=False, reproject=True, seeds26x=False):
    """Build one trial's per-camera UETrack track file. Returns (path, n_frames, n_cams_used, existed).

    reproject=True: seed cameras that have NO direct YOLO detection by reprojecting the CONSENSUS 3D
    (only when >=2 detecting cameras AGREE -- _consensus_seed gates on the reprojection consensus).
    Reprojected seeds are tagged `seeded_by:"reproject"` (direct ones `"yolo"`) so a downstream reader
    can drop the reprojected cameras. SKIP rule: if a track already exists and reproject-seed would add
    NO new camera, keep it (don't waste GPU) -- unless overwrite."""
    cf = (OUT26X if seeds26x else OUT) / f"{part}__{trial}__uetrack__fs1.json"
    cf.parent.mkdir(parents=True, exist_ok=True)
    stem = _clip_stem(part, trial)
    sd = _staged_dir(part)

    # all cameras with a clip (reprojected cams need a slot even with no detection)
    vids = {c: sd / f"{stem}.{_cam_num(c)}.mp4" for c in calib
            if (sd / f"{stem}.{_cam_num(c)}.mp4").exists()}
    if len(vids) < 2:
        return cf, 0, len(vids), False

    # direct-YOLO per-cam first-detection seeds
    if seeds26x:
        s26 = _seed26x(part, trial)
        if s26 is None:
            return cf, 0, 0, False                                    # no consensus seed -> no track
        sfr, sboxes, sorig = s26
        direct = {c: (sfr, b) for c, b in sboxes.items() if c in vids}
        seed3 = None; reproj_cams = set()
    else:
        direct = {c: _seed_box_xywh(part, trial, c) for c in vids}
        direct = {c: v for c, v in direct.items() if v is not None}   # (frame, xywh)

    # consensus seed (for reprojection into non-detecting cams)
    if not seeds26x:
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
            plan[c] = (direct[c][0], direct[c][1],
                       sorig.get(c, "yolo") if seeds26x else "yolo")
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
                b.init(cam_idx[c], cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB), plan[c][1])
                seeded[c] = True
                row[c] = {"yolo": plan[c][1] if plan[c][2] == "yolo" else None,
                          "trk": _center(plan[c][1]), "seeded": True,
                          "seeded_by": plan[c][2]}
        # cv2.cvtColor, NOT `[:, :, ::-1].copy()`. The numpy strided flip costs 32.5ms per 5-cam
        # rig-frame at 1080p vs 6.3ms (1 thread) / 3.1ms (4 threads) for cvtColor -- MEASURED, and
        # byte-identical output. It was ~25ms of the ~35ms this stage cost, which made UETrack LOOK
        # like the bottleneck when its own step is 11.3ms (out/speed/, 2026-08-12). Cheaper still:
        # UETrackBatch.update(..., bgr=True) converts the 224x224 CROPS instead (~0.05ms).
        rgbs = [None] * len(cams)
        for c in cams:
            if seeded[c] and c in imgs and c not in row:
                rgbs[cam_idx[c]] = cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB)
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

    cf.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--seeds26x", action="store_true",
                    help="seed from cache/cup_seed26x (stock COCO yolo26x-seg consensus seed); "
                         "tracks are written to cache/tracks_uetrack_26x")
    ap.add_argument("--audit-clean", action="store_true",
                    help="only build trials that PASS the clip/OMC audit (cache/delta/clip_omc_audit.json)"
                         " -- excludes uncut clips + trials whose OMC time-window doesn't map to the video")
    a = ap.parse_args(argv)

    H.use_good_cams()
    from uetrack_wrap import UETrackBatch

    audit = {}
    if a.audit_clean:
        ap_ = ROOT / "cache" / "delta" / "clip_omc_audit.json"
        audit = json.loads(ap_.read_text()) if ap_.exists() else {}

    # count work up front for a real ETA in the log
    work = []
    for p in a.parts:
        trials = _trials_for(p)
        if a.audit_clean and p in audit:
            clean = set(audit[p]["clean"])
            before = len(trials)
            trials = [t for t in trials if t in clean]
            print(f"  {p}: audit-clean keeps {len(trials)}/{before} trials "
                  f"(dropped {before-len(trials)} broken-clip/OMC-mismatch)", flush=True)
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
                                              overwrite=a.overwrite, seeds26x=a.seeds26x)
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
