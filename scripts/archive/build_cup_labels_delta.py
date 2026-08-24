"""Build a DELTA cup training set with the CONVERGED object_tracking filter: REJECT-THEN-FILL.

Port of experiments/drink_study/cache_scripts/run_clean3d_fill.py (CFG=pscale_1_clean3d_refill),
which beat every variant on BRIO (mean recall 0.803, cam10 0.74, 3D-prec 2.99px).

TEACHER = generic COCO `yolo26x-seg.pt` + CUP_LIKE_CLASSES, NOT our `cup_clean3d_refill.pt`.
Measured on P14 (the WORST participant): the COCO teacher gets 45-78%/cam and 77% >=3-cam
consensus, while our BRIO finetune gets 4-22% and 8.2%. The finetune is over-specialised to the
BRIO cup by construction -- the "DELTA domain gap" was our own model, not the data.

The three-way label decision per (frame, camera):
  detected & passes 30px consensus gate  -> KEEP real detection
  detected & FAILS the gate              -> SUPPRESS it, REFILL from consensus reprojection
                                            (this is the bit that beat reject-only: the wrong
                                             detection is replaced, not merely dropped)
  not detected, but consensus exists     -> FILL from consensus reprojection
  no >=3-cam consensus                   -> DROP the frame for that camera
Labels are YOLO-SEG polygons (the student is a seg model, so a box is not a valid label).

DELTA specifics vs BRIO:
  * calib translations are in METRES -> scaled x1000 (triangulate rounds X to 1dp: 0.1mm in
    BRIO's mm world, a catastrophic 100mm in metres).
  * frames are 1920x1080; cam1 was upscaled from 720p at import time.

    python scripts/build_cup_labels_delta.py --part P14 --out data/delta_cup
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.kalman_3d import load_calibration, project, triangulate_dlt  # noqa: E402
from gpu_decode import frames as gpu_frames  # noqa: E402  (NVDEC decode, cv2 CPU fallback)

ROOT = Path(__file__).resolve().parents[1]
DELTA = ROOT / "cache" / "delta"
TEACHER = "/home/imove/Documents/object_tracking/data/pretrained/yolo26x-seg.pt"
CUP_LIKE_CLASSES = [39, 40, 41, 45, 75]   # bottle, wine glass, cup, bowl, vase
RES = (1920, 1080)
CONF = 0.25
THR = 30.0      # px, consensus gate
MINC = 3        # cameras required for a consensus
CUP_R = 35.0    # mm, cup apparent radius


def load_calib_mm(part, usable_only=False):
    cams = load_calibration(str(DELTA / part / "calib" / f"{part}_calibration.toml"),
                            target_size=RES)
    for c in cams.values():
        c.t = c.t * 1000.0          # METRES -> mm
    if usable_only:
        keep = usable_cams(part)
        cams = {k: v for k, v in cams.items() if k in keep}
        if not cams:
            raise SystemExit(f"{part}: no usable cams in {DELTA/'cam_quality.json'}")
    return cams


def usable_cams(part):
    """Cameras that pass BOTH tests: sync (r>=0.65) AND calibration (reproj<=30px).

    cam_quality.json's `good` field is SYNC-ONLY -- for P14 it lists 9 cams, but cam4/cam5 are
    miscalibrated (reproj 30/60px). Building the cup labels off those would (a) reproject fills
    to the wrong pixel and (b) REFILL -- i.e. DELETE a miscalibrated camera's correct in-image
    detection and replace it with a geometry-derived wrong one. Both re-inject the exact poison
    the good-cam fix removed. Fill/consensus must use the sync+calib-clean set.
    """
    q = json.loads((DELTA / "cam_quality.json").read_text())[part]
    s, r = q["sync"], q["reproj"]
    return {c for c in s if s[c] >= 0.65 and r.get(c, 1e9) <= 30.0}


def teacher_dets(part, stem, cams, cache_dir, clips_dir, vid_stride=1):
    """Per-camera cup centroids from the COCO teacher. Cached: the teacher is the slow part.
    vid_stride>1 runs the teacher on only every Nth frame -- the teacher is the GPU cost, so this
    is what makes a lean pool FAST (seq[i] corresponds to video frame i*vid_stride)."""
    cf = cache_dir / f"{stem}__teacher_coco__c{CONF}__vs{vid_stride}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return {c: [tuple(x) if x else None for x in v] for c, v in d.items()}
    from ultralytics import YOLO
    model = YOLO(TEACHER)
    out = {}
    for cam in cams:
        k = int(cam.split("_")[1])
        v = clips_dir / f"delta_{part}_{stem}.{k}.mp4"
        if not v.exists():
            continue
        seq = []
        for res in model.predict(str(v), imgsz=640, conf=CONF, classes=CUP_LIKE_CLASSES,
                                 device=0, verbose=False, stream=True, vid_stride=vid_stride):
            b = res.boxes
            if b is None or len(b) == 0:
                seq.append(None)
                continue
            i = int(np.argmax(b.conf.cpu().numpy()))       # highest-conf cup-like
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[i]
            # float() not float32: json.dumps chokes on numpy scalars, and the cache write
            # happens AFTER all 10 cams -> a TypeError here throws away ~3min of GPU work.
            seq.append((float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)))
        out[cam] = seq
        print(f"    {cam}: {sum(x is not None for x in seq)}/{len(seq)}", flush=True)
    cf.write_text(json.dumps(out))
    return out


def consensus(obs, calib):
    """>=3-cam, <=30px gated consensus (iteratively eject the worst-reprojecting camera)."""
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= THR:
            break
        del cur[w]
    if len(cur) < MINC:
        return None, set()
    return triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur]), set(cur)


def apparent_radius_px(cam, X, calib):
    c0 = project(calib[cam], X)[0]
    # Offset CUP_R mm PERPENDICULAR to the viewing ray, not along a fixed world axis. The old
    # `Xo[0] += CUP_R` foreshortens for any camera whose optical axis points along world-X
    # (DELTA cam3: axis.X=1.00 -> radius collapsed to ~2px, floored to 6, vs a true ~25px), which
    # starved that camera's labels below the stride-8 grid and killed its training. R[0] is the
    # camera's image-horizontal axis in world coords (perpendicular to the optical axis by
    # construction), so this gives fx*CUP_R/Z regardless of the cup's world orientation. (world mm)
    Rx = np.asarray(calib[cam].R)[0]
    Xo = X + CUP_R * Rx
    r = float(np.hypot(*(project(calib[cam], Xo)[0] - c0)))
    return c0, max(r, 6.0)


def poly_line(cx, cy, r):
    W, H = RES
    if not (0 < cx < W and 0 < cy < H):
        return None
    pts = []
    for x, y in [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]:
        pts += [min(max(x / W, 0.0), 1.0), min(max(y / H, 0.0), 1.0)]
    return "0 " + " ".join(f"{v:.6f}" for v in pts) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P14")
    ap.add_argument("--out", default=str(ROOT / "data" / "delta_cup"))
    ap.add_argument("--max-trials", type=int, default=0, help="0 = all")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame (dedup)")
    ap.add_argument("--usable-cams", action="store_true",
                    help="restrict consensus/fill/images to sync+calib-clean cams "
                         "(cam_quality.json). Without this the fill/refill steps poison labels on "
                         "the desynced/miscalibrated cameras. Strongly recommended on DELTA.")
    ap.add_argument("--clips-dir", default=None,
                    help="dir holding delta_<P>_<stem>.<cam>.mp4 (default: <P>/staged). "
                         "For the validated 5-good-camera cohort pass <P>/work/clips.")
    ap.add_argument("--cams", nargs="+", type=int, default=None,
                    help="explicit camera list (overrides --usable-cams filtering); e.g. 1 2 3 4 5")
    ap.add_argument("--vid-stride", type=int, default=1,
                    help="run the TEACHER on every Nth frame -> N-fold less GPU + N-fold fewer "
                         "labels (a rigid cup needs few frames; use ~60 for a ~3k-image pool)")
    ap.add_argument("--only-trials", nargs="+", default=None,
                    help="restrict to these exact trial stems (for the 1-trial experiment)")
    a = ap.parse_args(argv)

    calib = load_calib_mm(a.part, usable_only=a.usable_cams)
    if a.cams:
        calib = {f"cam_{c}": calib[f"cam_{c}"] for c in a.cams if f"cam_{c}" in calib}
    if a.usable_cams or a.cams:
        print(f"[cams] {a.part}: {sorted(calib, key=lambda z:int(z.split('_')[1]))}", flush=True)
    staged = Path(a.clips_dir) if a.clips_dir else DELTA / a.part / "staged"
    stems = sorted({f.name.rsplit(".", 2)[0].replace(f"delta_{a.part}_", "")
                    for f in staged.glob("*.mp4")})
    if a.only_trials:
        stems = [s for s in stems if s in set(a.only_trials)]
    if a.max_trials:
        stems = stems[:a.max_trials]
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    # teacher cache MUST be tied to the clip source (work clips are re-cut/scaled -> different
    # pixels than the old staged clips; reusing the staged cache would mislabel).
    tcache = (staged.parent / "teacher") if a.clips_dir else (DELTA / a.part / "teacher")
    tcache.mkdir(exist_ok=True, parents=True)

    stats = {"kept_real": 0, "filled": 0, "refilled": 0, "dropped": 0}
    per_cam = {c: {"real": 0, "fill": 0, "refill": 0, "drop": 0} for c in calib}
    print(f"### {a.part}: {len(stems)} trials, cams={sorted(calib)}", flush=True)

    for si, stem in enumerate(stems):
        print(f"  [{si+1}/{len(stems)}] {stem}", flush=True)
        dets = teacher_dets(a.part, stem, sorted(calib), tcache, staged, a.vid_stride)
        dets = {c: v for c, v in dets.items() if c in calib}
        if len(dets) < MINC:
            continue
        ndec = min(len(v) for v in dets.values())   # decimated length (teacher frames)
        VS = a.vid_stride
        # NVDEC (GPU) decode via gpu_decode.frames() -- one generator per camera, stepped in
        # lockstep. Offloads the CPU-bound H.264 decode that bottlenecked this loop (we still read
        # every full frame to stay index-aligned, but only PROCESS decimated indices di=full//VS).
        gens = {c: gpu_frames(staged / f"delta_{a.part}_{stem}.{int(c.split('_')[1])}.mp4")
                for c in dets}
        full = -1
        while True:
            full += 1
            frames = {}
            anyok = False
            for c, g in gens.items():
                img = next(g, None)
                if img is not None:
                    anyok = True
                    frames[c] = img
            if not anyok:
                break
            if full % VS:
                continue
            fi = full // VS                          # index into the (decimated) teacher dets
            if fi >= ndec or (fi % a.stride):
                if fi >= ndec:
                    break
                continue
            obs = {c: dets[c][fi] for c in dets if dets[c][fi]}
            if len(obs) < MINC:
                for c in dets:
                    per_cam[c]["drop"] += 1
                stats["dropped"] += len(dets)
                continue
            X, kept = consensus(obs, calib)
            if X is None:
                for c in dets:
                    per_cam[c]["drop"] += 1
                stats["dropped"] += len(dets)
                continue
            for c in dets:
                if c not in frames:
                    continue
                if c in kept:
                    cx, cy = obs[c]
                    _, r = apparent_radius_px(c, X, calib)
                    kind = "real"
                else:
                    c0, r = apparent_radius_px(c, X, calib)
                    cx, cy = c0
                    kind = "refill" if c in obs else "fill"
                line = poly_line(cx, cy, r)
                if line is None:
                    per_cam[c]["drop"] += 1
                    stats["dropped"] += 1
                    continue
                tag = f"{a.part}_{stem}_{c}_f{fi:05d}"
                cv2.imwrite(str(out / "images" / f"{tag}.jpg"), frames[c])
                (out / "labels" / f"{tag}.txt").write_text(line)
                per_cam[c][kind] += 1
                stats[{"real": "kept_real", "fill": "filled", "refill": "refilled"}[kind]] += 1
        for g in gens.values():
            g.close()          # stop the ffmpeg NVDEC subprocess even if we broke early
        print(f"      running: {stats}", flush=True)

    (out / "stats.json").write_text(json.dumps({"stats": stats, "per_cam": per_cam}, indent=1))
    print("\n=== LABEL POOL ===", flush=True)
    print(stats)
    print(f"{'cam':10}{'real':>8}{'fill':>8}{'refill':>8}{'drop':>8}")
    for c in sorted(per_cam, key=lambda k: int(k.split("_")[1])):
        d = per_cam[c]
        print(f"{c:10}{d['real']:8d}{d['fill']:8d}{d['refill']:8d}{d['drop']:8d}")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
