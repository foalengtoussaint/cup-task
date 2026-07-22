"""Per-CAMERA drift for detect-once+track, plus a YOLO per-frame detection CACHE.

Two things:

1. CACHE. The baseline (YOLO on every frame) is the expensive part and it does not change unless
   the model does. Cached to cache/yolo_dets/<model-hash>__<part>__<trial>__fs<N>.json so every
   tracker variant (P07/P13, ViT vs DaSiamRPN, periodic re-anchor) reads the same detections
   instead of re-inferring. Matches the project rule: never re-run GPU inference when a cached
   detection JSON exists.

2. PER-CAMERA drift. Pooled drift (median 6-8px, p90 35, max 554) is consistent with ONE camera
   failing while the rest are fine -- consensus needs only 3 of 5, so a bad camera gets outvoted
   and the aggregate still reads 100%. Pooling cannot tell "all five track well" from "four track
   well and one blew up". This reports drift PER CAMERA and flags WHEN each one exceeds the gate,
   so a failure is localised to a camera and a time window (i.e. a re-anchor trigger).

    python scripts/siam_percam_drift.py --part P08 --model <yolo.pt> --tracker dasiamrpn
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_cup_3d_delta as E  # noqa: E402
import compare_pose_omc_delta as C  # noqa: E402
from siam_gap_bridge import make_tracker, yolo_box, ctr  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache" / "yolo_dets"


def model_hash(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def yolo_boxes_cached(model, mh, part, stem, work, calib, fstride):
    """-> {frame_idx: {cam: xywh or None}}, cached to disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{mh}__{part}__{stem}__fs{fstride}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return {int(k): {c: (tuple(v) if v else None) for c, v in fr.items()}
                for k, fr in d.items()}, True
    caps = {}
    for c in calib:
        v = work / f"delta_{part}_{stem}.{int(c.split('_')[1])}.mp4"
        if v.exists():
            caps[c] = cv2.VideoCapture(str(v))
    out, fi = {}, -1
    while True:
        fi += 1
        got = {}
        for c, cap in caps.items():
            ok, img = cap.read()
            if ok:
                got[c] = img
        if len(got) < len(caps):
            break
        if fi % fstride:
            continue
        out[fi] = {c: yolo_box(model, im) for c, im in got.items()}
    for cap in caps.values():
        cap.release()
    cf.write_text(json.dumps({str(k): {c: (list(v) if v else None) for c, v in fr.items()}
                              for k, fr in out.items()}))
    return out, False


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracker", default="dasiamrpn", choices=["dasiamrpn", "vit"])
    ap.add_argument("--max-trials", type=int, default=4)
    ap.add_argument("--fstride", type=int, default=4)
    a = ap.parse_args(argv)

    m = YOLO(a.model)
    mh = model_hash(a.model)
    calib = C._load_calib_mm(a.part)
    work = C.DELTA / a.part / "work" / "clips"
    stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{a.part}_", "")
                    for p in glob.glob(str(work / "*.mp4"))})[:a.max_trials]

    percam = {c: [] for c in calib}          # (trial, frame, drift_px)
    B = K = T = 0
    for stem in stems:
        ybx, hit = yolo_boxes_cached(m, mh, a.part, stem, work, calib, a.fstride)
        print(f"  {stem:28} YOLO dets {'CACHED' if hit else 'computed+cached'}", flush=True)
        caps = {}
        for c in calib:
            v = work / f"delta_{a.part}_{stem}.{int(c.split('_')[1])}.mp4"
            if v.exists():
                caps[c] = cv2.VideoCapture(str(v))
        trk = {c: None for c in caps}
        fi = -1
        while True:
            fi += 1
            got = {}
            for c, cap in caps.items():
                ok, img = cap.read()
                if ok:
                    got[c] = img
            if len(got) < len(caps):
                break
            if fi % a.fstride or fi not in ybx:
                continue
            T += 1
            bx = ybx[fi]
            ydet = {c: ctr(b) for c, b in bx.items() if b is not None}
            Xb = E.consensus(ydet, calib)[0] if len(ydet) >= 2 else None
            B += Xb is not None
            tdet = {}
            for c, im in got.items():
                if trk[c] is None:
                    if bx.get(c) is not None:
                        t = make_tracker(a.tracker)
                        t.init(im, tuple(int(v) for v in bx[c]))
                        trk[c] = t
                        tdet[c] = ctr(bx[c])
                    continue
                ok2, bb = trk[c].update(im)
                if ok2:
                    tdet[c] = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
            Xt = E.consensus(tdet, calib)[0] if len(tdet) >= 2 else None
            K += Xt is not None
            if Xb is not None:
                for c, p in tdet.items():
                    e = float(np.hypot(*(E.project(calib[c], Xb)[0] - np.array(p))))
                    percam[c].append((stem, fi, e))
        for cap in caps.values():
            cap.release()

    print(f"\n=== {a.part}  tracker={a.tracker}  (n={T}) ===")
    print(f"  YOLO every frame    : {B/T*100:5.1f}%")
    print(f"  detect-once + track : {K/T*100:5.1f}%   ({(K-B)/T*100:+.1f} pts)")
    print(f"\n  PER-CAMERA drift vs consensus reprojection:")
    print(f"  {'cam':7}{'n':>6}{'median':>9}{'p90':>8}{'max':>9}{'>30px':>8}  worst window")
    for c in sorted(percam, key=lambda z: int(z.split('_')[1])):
        v = percam[c]
        if not v:
            print(f"  {c:7}{'--':>6}")
            continue
        d = np.array([e for _, _, e in v])
        bad = [(s, f, e) for s, f, e in v if e > 30]
        worst = max(v, key=lambda z: z[2])
        print(f"  {c:7}{len(d):6d}{np.median(d):8.0f}px{np.percentile(d,90):7.0f}{d.max():8.0f}"
              f"{len(bad)/len(d)*100:7.0f}%  {worst[0][:18]} f{worst[1]} ({worst[2]:.0f}px)")


if __name__ == "__main__":
    main()
