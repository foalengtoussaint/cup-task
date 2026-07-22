"""Geometry-GATED Siamese gap bridging: recover cameras that lose the cup, without inviting drift.

THE PROBLEM (measured, see WORKLOG 2026-07-20): cup loss is not detector flicker, it is 5-7
BLACKOUTS of 1.3-2.7s at the drink apex. A frame is lost when fewer than 3 cameras detect. Often
2 cameras still see it -- one more would restore consensus.

THE IDEA: where YOLO detects, use YOLO (unchanged). Where a camera LOSES the cup, run a Siamese
tracker (DaSiamRPN / ViT) seeded from that camera's last confident YOLO box, and propose a point.

THE GATE (why this cannot make things worse): a tracker proposal is ADMITTED only if it reprojects
within THR px of the >=3-cam consensus built from the cameras that DID detect. A drifting tracker
that has locked onto the hand/face reprojects far away and is REJECTED. Geometry vetoes appearance.
So bridging can only ever ADD a camera that already agrees with the others.

HONEST LIMITS:
  * needs >=2 other cameras detecting, so a consensus exists to gate against -> it CANNOT recover
    the P07/P13 frames where <3 cameras have any view of the cup (that is a rig problem).
  * needs a template from before the occlusion -> bridges INTO a gap, not across a cold start.

    python scripts/siam_gap_bridge.py --part P15 --model <yolo.pt> --tracker dasiamrpn \
        --max-trials 4 --fstride 4
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_cup_3d_delta as E  # noqa: E402
import compare_pose_omc_delta as C  # noqa: E402

TDIR = Path(__file__).resolve().parents[1] / "models" / "trackers"


def make_tracker(kind):
    if kind == "dasiamrpn":
        p = cv2.TrackerDaSiamRPN_Params()
        p.model = str(TDIR / "dasiamrpn_model.onnx")
        p.kernel_r1 = str(TDIR / "dasiamrpn_kernel_r1.onnx")
        p.kernel_cls1 = str(TDIR / "dasiamrpn_kernel_cls1.onnx")
        return cv2.TrackerDaSiamRPN_create(p)
    if kind == "vit":
        p = cv2.TrackerVit_Params()
        p.net = str(TDIR / "vittrack.onnx")
        return cv2.TrackerVit_create(p)
    raise SystemExit(f"unknown tracker {kind}")


def yolo_box(model, img):
    r = model.predict(img, imgsz=640, conf=E.CONF, verbose=False, device=0)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return None
    i = int(np.argmax(b.conf.cpu().numpy()))
    x1, y1, x2, y2 = b.xyxy.cpu().numpy()[i]
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))   # xywh


def ctr(bb):
    return None if bb is None else (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)


def run_part(model, part, kind, max_trials, fstride, thr):
    calib = C._load_calib_mm(part)
    work = C.DELTA / part / "work" / "clips"
    stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{part}_", "")
                    for p in glob.glob(str(work / "*.mp4"))})[:max_trials]
    base_good = brid_good = tot = 0
    admitted = rejected = 0
    for stem in stems:
        caps = {}
        for c in calib:
            v = work / f"delta_{part}_{stem}.{int(c.split('_')[1])}.mp4"
            if v.exists():
                caps[c] = cv2.VideoCapture(str(v))
        if len(caps) < E.MINC:
            continue
        trk = {c: None for c in caps}          # live tracker per camera
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
            if fi % fstride:
                continue
            tot += 1
            boxes = {c: yolo_box(model, im) for c, im in got.items()}
            det = {c: ctr(b) for c, b in boxes.items() if b is not None}

            # BASELINE: YOLO only
            Xb = E.consensus(det, calib)[0] if len(det) >= 2 else None
            base_good += Xb is not None

            # keep each detecting camera's tracker seeded with the fresh YOLO box
            for c, b in boxes.items():
                if b is not None:
                    t = make_tracker(kind)
                    t.init(got[c], tuple(int(v) for v in b))
                    trk[c] = t

            # BRIDGE: for cameras with no detection, propose from the tracker, then GATE
            if Xb is not None:                      # need a consensus to gate against
                extra = {}
                for c in got:
                    if boxes[c] is not None or trk[c] is None:
                        continue
                    ok, bb = trk[c].update(got[c])
                    if not ok:
                        continue
                    p = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
                    # GATE: does the proposal agree with the geometry the others already fixed?
                    err = float(np.hypot(*(E.project(calib[c], Xb)[0] - np.array(p))))
                    if err <= thr:
                        extra[c] = p
                        admitted += 1
                    else:
                        rejected += 1
                merged = {**det, **extra}
                Xr = E.consensus(merged, calib)[0] if len(merged) >= 2 else None
            else:
                Xr = None
            brid_good += Xr is not None
        for cap in caps.values():
            cap.release()
    return dict(tot=tot, base=base_good, brid=brid_good, admitted=admitted, rejected=rejected)


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracker", default="dasiamrpn", choices=["dasiamrpn", "vit"])
    ap.add_argument("--max-trials", type=int, default=4)
    ap.add_argument("--fstride", type=int, default=4)
    ap.add_argument("--thr", type=float, default=E.THR, help="px gate vs consensus reprojection")
    a = ap.parse_args(argv)
    m = YOLO(a.model)
    r = run_part(m, a.part, a.tracker, a.max_trials, a.fstride, a.thr)
    b, g, n = r["base"], r["brid"], r["tot"]
    print(f"\n=== {a.part}  tracker={a.tracker}  gate={a.thr}px  (n={n} frames) ===")
    print(f"  YOLO only        : {b/n*100:5.1f}% good")
    print(f"  + gated bridging : {g/n*100:5.1f}% good   ({(g-b)/n*100:+.1f} pts)")
    print(f"  tracker proposals: {r['admitted']} admitted / {r['rejected']} REJECTED by the gate")


if __name__ == "__main__":
    main()
