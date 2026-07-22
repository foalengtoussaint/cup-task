"""Render what the Siamese tracker ACTUALLY does while a camera has lost the cup.

The gated-bridging experiment gave +0.0, and I claimed the tracker drifts during the apex
occlusion. That is an assertion until someone looks. This finds a camera-trial where YOLO loses
the cup for a run of frames, seeds the tracker from the last confident YOLO box, and draws what
it proposes every frame after:

  YELLOW = last YOLO box (the template the tracker was seeded with)
  CYAN   = tracker proposal this frame
  GREEN  = YOLO box, on frames where YOLO does detect
  MAGENTA= where the >=3-cam consensus says the cup is, reprojected into this camera (truth-ish)

If CYAN follows MAGENTA -> the tracker is genuinely bridging and the gate anchor is the problem.
If CYAN wanders off (hand/face/background) -> the template has nothing to match and the idea is dead.

    python scripts/viz_tracker_during_gap.py --part P08 --model <yolo.pt> --cam cam_4 \
        --tracker dasiamrpn --out /tmp/tracker_gap.png
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
from siam_gap_bridge import make_tracker, yolo_box, ctr  # noqa: E402


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cam", default=None, help="camera to watch; default = the one that loses most")
    ap.add_argument("--tracker", default="dasiamrpn", choices=["dasiamrpn", "vit"])
    ap.add_argument("--trial", default=None)
    ap.add_argument("--fstride", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=8, help="gap frames to render")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    m = YOLO(a.model)
    calib = C._load_calib_mm(a.part)
    work = C.DELTA / a.part / "work" / "clips"
    stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{a.part}_", "")
                    for p in glob.glob(str(work / "*.mp4"))})
    stem = a.trial or stems[0]
    caps = {}
    for c in calib:
        v = work / f"delta_{a.part}_{stem}.{int(c.split('_')[1])}.mp4"
        if v.exists():
            caps[c] = cv2.VideoCapture(str(v))

    # pass 1: per-frame YOLO boxes for every camera
    seq = []
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
        if fi % a.fstride:
            continue
        seq.append((fi, {c: yolo_box(m, im) for c, im in got.items()}))
    for cap in caps.values():
        cap.release()

    # pick the camera with the longest run of misses (or the one asked for)
    if a.cam:
        cam = a.cam
    else:
        best, cam = -1, None
        for c in caps:
            run = mx = 0
            for _, bx in seq:
                run = 0 if bx[c] is not None else run + 1
                mx = max(mx, run)
            if mx > best:
                best, cam = mx, c
        print(f"camera losing most: {cam} (longest miss run {best} sampled frames)", flush=True)

    # find a gap on that camera preceded by a detection (need a template)
    start = None
    for i in range(1, len(seq)):
        if seq[i][1][cam] is None and seq[i - 1][1][cam] is not None:
            start = i
            break
    if start is None:
        print("no gap with a preceding detection on that camera", flush=True)
        return
    print(f"gap starts at sampled index {start} (frame {seq[start][0]})", flush=True)

    # pass 2: re-read, seed tracker at start-1, draw the following frames
    cap = cv2.VideoCapture(str(work / f"delta_{a.part}_{stem}.{int(cam.split('_')[1])}.mp4"))
    others = {c: cv2.VideoCapture(str(work / f"delta_{a.part}_{stem}.{int(c.split('_')[1])}.mp4"))
              for c in caps if c != cam}
    frames = {}
    fi = -1
    want = {seq[j][0] for j in range(start - 1, min(start - 1 + a.max_frames + 1, len(seq)))}
    while True:
        fi += 1
        ok, img = cap.read()
        oth = {}
        for c, cc in others.items():
            o, im = cc.read()
            if o:
                oth[c] = im
        if not ok:
            break
        if fi in want:
            frames[fi] = (img, oth)
    cap.release()
    [c.release() for c in others.values()]

    trk = make_tracker(a.tracker)
    seed_fi = seq[start - 1][0]
    seed_box = seq[start - 1][1][cam]
    trk.init(frames[seed_fi][0], tuple(int(v) for v in seed_box))
    tiles = []
    for j in range(start - 1, min(start - 1 + a.max_frames + 1, len(seq))):
        f, bx = seq[j]
        if f not in frames:
            continue
        img, oth = frames[f]
        vis = img.copy()
        # magenta: where consensus (other cams) says the cup is
        det = {c: ctr(bx[c]) for c in bx if c != cam and bx[c] is not None}
        if len(det) >= 2:
            X = E.consensus(det, calib)[0]
            if X is not None:
                p = E.project(calib[cam], X)[0]
                cv2.circle(vis, (int(p[0]), int(p[1])), 16, (255, 0, 255), 3)
        if j == start - 1:
            x, y, w, h = [int(v) for v in seed_box]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 3)   # yellow template
        else:
            ok2, bb = trk.update(img)
            if ok2:
                x, y, w, h = [int(v) for v in bb]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 3)  # cyan proposal
        if bx[cam] is not None:
            x, y, w, h = [int(v) for v in bx[cam]]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)     # green YOLO
        tag = "SEED" if j == start - 1 else f"+{j-(start-1)}"
        cv2.putText(vis, f"{cam} f{f} {tag}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 255), 2)
        tiles.append(cv2.resize(vis, (480, 270)))
    if not tiles:
        print("nothing rendered", flush=True)
        return
    cols = 3
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"wrote {a.out}  YELLOW=template CYAN=tracker GREEN=YOLO MAGENTA=consensus-reproj",
          flush=True)


if __name__ == "__main__":
    main()
