"""Find the cup SEED with the stock COCO teacher (yolo26x-seg), scanning only until it triangulates.

WHY. The DELTA cup detections in cache/delta/*/dets/*.cup.json were produced by
models/cup_clean3d_refill.pt -- the BRIO finetune that WORKLOG.md:927 already recorded as the wrong
model for this cohort ("COCO teacher yolo26x-seg gets 77% >=3-cam consensus on P14, our BRIO finetune
8.2%"). Measured now over the whole cohort, that finetune detects the cup on 6-54% of frames and is
near-blind in most cameras (P08 cam4 0.02, P10 cam5 0.00), while ONE camera carries 0.85-0.98. Since
detect-once seeds from the FIRST box in ~437 frames, a near-blind camera still gets seeded -- off
whatever that single frame fired on -- and UETrack then tracks the wrong thing confidently for the
whole trial. That is the "tracker has a point everywhere, cameras never agree" pattern.

WHAT. Detect-once done properly: walk forward from frame 0 and stop at the FIRST frame whose
cup-like detections actually triangulate (>=3 cameras agreeing within 30px, the pipeline's own gate).
No full-video detection -- the teacher only runs until the seed exists, which is the expensive part
avoided. Cameras that did not detect at that frame are seeded by REPROJECTING the consensus 3D, so
every camera starts on the same physical object.

Writes one seed per trial; the tracking pass reads these instead of _seed_box_xywh.

    python scripts/cache_cup_seed26x.py [--stride 3] [--max-scan-s 12] [--part P13]
    -> cache/cup_seed26x/<part>__<trial>.json   {frame, X, boxes:{cam:[x,y,w,h]}, origin:{cam:...}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cup_task.kalman_3d import project, triangulate_dlt          # noqa: E402

TEACHER = "/home/imove/Documents/object_tracking/data/pretrained/yolo26x-seg.pt"
CUP_LIKE = [39, 40, 41, 45, 75]      # bottle, wine glass, cup, bowl, vase
CONF = 0.25
THR = 30.0                            # px, the pipeline's consensus gate
MINC = 3                              # the >=3 agreeing-camera floor
CUP_R = 35.0                          # mm, for the reprojected seed box
OUT = ROOT / "cache" / "cup_seed26x"


def consensus(obs, calib):
    """>=3-cam, <=30px gated consensus (iteratively eject the worst reprojector)."""
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


def _box_from_X(cam, X, calib):
    """Seed box for a camera with no detection: reproject the consensus, size it by apparent radius.
    The offset is along the camera's IMAGE-HORIZONTAL axis (calib.R[0]) -- offsetting along a world
    axis foreshortens to ~0 for a camera looking down that axis (the cam3 apparent_radius bug)."""
    c0 = project(calib[cam], X)[0]
    Rx = np.asarray(calib[cam].R)[0]
    r = max(float(np.hypot(*(project(calib[cam], X + CUP_R * Rx)[0] - c0))), 6.0)
    return [float(c0[0] - r), float(c0[1] - r), float(2 * r), float(2 * r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3, help="scan every Nth frame")
    ap.add_argument("--max-scan-s", type=float, default=12.0, help="give up after this many seconds")
    ap.add_argument("--part", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    import compare_pose_omc_delta as H
    H.use_good_cams()
    import cache_seg_inputs as CSI
    import results_v3_delta as R
    OUT.mkdir(parents=True, exist_ok=True)

    recs = CSI.load_all()
    if a.part:
        recs = [r for r in recs if r["part"] == a.part]
    if a.limit:
        recs = recs[:a.limit]
    model = YOLO(TEACHER)
    print(f"{len(recs)} trials, teacher {Path(TEACHER).name}, stride {a.stride}, "
          f"max scan {a.max_scan_s}s", flush=True)

    t0 = time.time(); n_ok = n_skip = n_fail = 0; scanned = []
    for i, r in enumerate(recs):
        part, trial = r["part"], r["trial"]
        out = OUT / f"{part}__{trial}.json"
        if out.exists():
            n_skip += 1
            continue
        calib = R._calib(part)
        staged = H.DELTA / part / "staged"
        vids = {c: staged / f"delta_{part}_{trial}.{int(c.split('_')[1])}.mp4" for c in calib}
        vids = {c: v for c, v in vids.items() if v.exists()}
        if len(vids) < MINC:
            n_fail += 1
            out.write_text(json.dumps({"seed": None, "why": "too few clips"}))
            continue
        caps = {c: cv2.VideoCapture(str(v)) for c, v in vids.items()}
        nfr = int(min(cp.get(cv2.CAP_PROP_FRAME_COUNT) for cp in caps.values()))
        limit = min(nfr, int(a.max_scan_s * 60))
        found = None
        for f in range(0, limit, a.stride):
            imgs = {}
            for c, cp in caps.items():
                cp.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cp.read()
                if ok:
                    imgs[c] = im
            if len(imgs) < MINC:
                continue
            cams = list(imgs)
            res = model.predict([imgs[c] for c in cams], imgsz=640, conf=CONF,
                                classes=CUP_LIKE, device=0, verbose=False)
            obs = {}
            for c, rr in zip(cams, res):
                b = rr.boxes
                if b is None or len(b) == 0:
                    continue
                k = int(np.argmax(b.conf.cpu().numpy()))
                x1, y1, x2, y2 = b.xyxy.cpu().numpy()[k]
                obs[c] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if len(obs) < MINC:
                continue
            X, kept = consensus(obs, calib)
            if X is not None:
                boxes, origin = {}, {}
                for c, rr in zip(cams, res):
                    if c in kept:
                        b = rr.boxes
                        k = int(np.argmax(b.conf.cpu().numpy()))
                        x1, y1, x2, y2 = [float(v) for v in b.xyxy.cpu().numpy()[k]]
                        boxes[c] = [x1, y1, x2 - x1, y2 - y1]; origin[c] = "yolo"
                    else:
                        boxes[c] = _box_from_X(c, X, calib); origin[c] = "reproject"
                found = dict(frame=f, X=[float(v) for v in X], boxes=boxes, origin=origin,
                             n_agree=len(kept), scanned_frames=f // a.stride + 1)
                break
        for cp in caps.values():
            cp.release()
        if found:
            out.write_text(json.dumps(found)); n_ok += 1; scanned.append(found["scanned_frames"])
        else:
            out.write_text(json.dumps({"seed": None, "why": "no consensus within scan window"}))
            n_fail += 1
        if (i + 1) % 10 == 0:
            med = int(np.median(scanned)) if scanned else -1
            print(f"  [{i+1}/{len(recs)}] {time.time()-t0:5.0f}s  seeded {n_ok}  failed {n_fail}  "
                  f"skip {n_skip}  median frames scanned {med}", flush=True)

    print(f"\nPROCESSING CHECK: trials {len(recs)}, seeded {n_ok}, no-consensus {n_fail}, "
          f"already-cached {n_skip}", flush=True)
    if scanned:
        s = np.array(scanned)
        print(f"detector calls per seeded trial: median {np.median(s):.0f} frames "
              f"(x{len(vids)} cams), p90 {np.percentile(s, 90):.0f}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
