"""Cup-TRACKING accuracy for a DELTA cup model, using the object_tracking 3D metrics (ported from
drink_study run_clean3d_fill.precision_3d / _consensus_for). This measures the detector as a 3D
TRACKER -- what actually matters downstream -- instead of mAP against noisy teacher labels:

  precision_3d      : of all student cup detections, the fraction that land in the >=3-cam <=30px
                      3D consensus (i.e. the real cup, not a glass/FP). "when it fires, is it right"
  good_frame_frac   : fraction of frames that reach a usable >=3-cam consensus (a trackable cup)
  median_px         : median reprojection error of the consensus cup (3D precision)
  per_cam_detrate   : fraction of frames each camera emits any cup box (NOT recall -- no GT)

Runs the student on a participant's work clips (correctly-synced 5-cam), triangulates per frame
with the DELTA calib (metres x1000).

    python scripts/eval_cup_3d_delta.py --model runs/.../best.pt --parts P07 P08 P12 P13 P15
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
import compare_pose_omc_delta as C  # noqa: E402
from pipeline.kalman_3d import project, triangulate_dlt  # noqa: E402

THR = 30.0     # px consensus gate (object_tracking value)
MINC = 3       # >=3 cams for a trustworthy 3D point
CONF = 0.25


def consensus(obs, calib):
    """>=3-cam, <=THR px gated 3D cup: iteratively eject the worst reprojector.
    Returns (X, kept_set) or (None, set())."""
    cur = dict(obs)
    while len(cur) >= 2:
        X = triangulate_dlt([calib[c] for c in cur], [np.array(cur[c]) for c in cur])
        e = {c: float(np.hypot(*(project(calib[c], X)[0] - np.array(cur[c])))) for c in cur}
        w = max(e, key=e.get)
        if e[w] <= THR:
            if len(cur) >= MINC:
                return X, set(cur), max(e.values())
            return None, set(), None
        del cur[w]
    return None, set(), None


def cup_center(model, img):
    r = model.predict(img, imgsz=640, conf=CONF, verbose=False, device=0)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return None
    i = int(np.argmax(b.conf.cpu().numpy()))
    x1, y1, x2, y2 = b.xyxy.cpu().numpy()[i]
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def eval_part(model, part, max_trials=4, fstride=4):
    calib = C._load_calib_mm(part)
    clipdir = C.DELTA / part / "work" / "clips"
    trials = sorted({Path(f).name.split(".")[0].replace(f"delta_{part}_", "")
                     for f in glob.glob(str(clipdir / "*.mp4"))})
    # metrics are aggregate rates -> a few trials + strided frames is statistically equivalent
    # and ~20x cheaper than running the model over every frame of every clip.
    if max_trials:
        trials = trials[:max_trials]
    tp = fp = 0
    good = tot = 0
    pxs = []
    detrate = {}
    for t in trials:
        caps = {}
        for f in glob.glob(str(clipdir / f"delta_{part}_{t}.*.mp4")):
            c = "cam_" + Path(f).name.split(".")[1]
            if c in calib:
                caps[c] = cv2.VideoCapture(f)
        if len(caps) < MINC:
            for cap in caps.values():
                cap.release()
            continue
        # per-frame detections -- run the model only every fstride-th frame (decode is cheap,
        # the model is the cost); cams stay aligned since all read the same frames.
        seqs = {c: [] for c in caps}
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
            for c, img in got.items():
                seqs[c].append(cup_center(model, img))
        for cap in caps.values():
            cap.release()
        n = min(len(v) for v in seqs.values()) if seqs else 0
        for c in seqs:
            d = detrate.setdefault(c, [0, 0])
            d[0] += sum(x is not None for x in seqs[c][:n]); d[1] += n
        for fr in range(n):
            obs = {c: seqs[c][fr] for c in seqs if seqs[c][fr] is not None}
            tot += 1
            if len(obs) < 2:
                continue
            X, kept, worst = consensus(obs, calib)
            if X is not None:
                good += 1
                pxs.append(worst)
                for c in obs:
                    if c in kept:
                        tp += 1
                    else:
                        fp += 1
            else:
                fp += len(obs)   # no consensus -> all detections unverified
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    gff = good / tot if tot else 0.0
    med = float(np.median(pxs)) if pxs else float("nan")
    dr = {c: (d[0] / d[1] if d[1] else 0) for c, d in detrate.items()}
    return prec, gff, med, dr, tot


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P12", "P13", "P15"])
    ap.add_argument("--max-trials", type=int, default=4)
    ap.add_argument("--fstride", type=int, default=4)
    a = ap.parse_args(argv)
    model = YOLO(a.model)
    print(f"model {a.model}  (sample: {a.max_trials} trials/part, every {a.fstride}th frame)")
    print(f"{'part':6}{'precision_3d':>13}{'good_frame%':>13}{'median_px':>11}   per-cam det-rate")
    P = G = 0.0; nn = 0
    for part in a.parts:
        prec, gff, med, dr, tot = eval_part(model, part, a.max_trials, a.fstride)
        drs = " ".join(f"c{c.split('_')[1]}:{dr[c]*100:.0f}" for c in sorted(dr, key=lambda z: int(z.split('_')[1])))
        print(f"{part:6}{prec*100:>12.1f}%{gff*100:>12.1f}%{med:>10.1f}px   {drs}", flush=True)
        P += prec; G += gff; nn += 1
    print(f"\nMEAN precision_3d {P/nn*100:.1f}%   good_frame {G/nn*100:.1f}%")


if __name__ == "__main__":
    main()
