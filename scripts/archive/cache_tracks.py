"""ONE GPU pass that caches EVERYTHING downstream analysis needs: per-frame, per-camera YOLO boxes
AND DaSiamRPN tracker points.

WHY. Every tracker experiment so far (detect-once, per-camera drift, drift-vs-time,
consensus-vs-consensus) re-ran the identical DaSiamRPN pass and differed only in which summary
statistic it printed. That is 4 GPU runs to answer 4 questions about the same trajectories. Worse,
the summaries are all COLLAPSED OVER TIME (median/p90/%>gate), which is exactly the structure the
interesting findings live in -- the gap result and the discrete-capture-event result both only
became visible when binned by time.

So: dump the raw trajectories once. After this, drift timelines, recovery windows, re-anchor
simulations (re-init vs re-position), and consensus-vs-consensus are all OFFLINE computations over
a few thousand points -- seconds, no GPU.

Cache: cache/tracks/<model-hash>__<part>__<trial>__<tracker>__fs<N>.json
  {frame: {cam: {"yolo": [x,y,w,h]|null, "trk": [cx,cy]|null, "seeded": bool}}}

    python scripts/cache_tracks.py --parts P07 P08 P15 P13 --tracker dasiamrpn --fstride 4
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from siam_gap_bridge import make_tracker, ctr  # noqa: E402
from siam_percam_drift import yolo_boxes_cached, model_hash  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "cache" / "tracks"


def cache_trial(model, mh, part, stem, work, calib, kind, fstride):
    OUT.mkdir(parents=True, exist_ok=True)
    cf = OUT / f"{mh}__{part}__{stem}__{kind}__fs{fstride}.json"
    if cf.exists():
        return cf, True
    ybx, _ = yolo_boxes_cached(model, mh, part, stem, work, calib, fstride)
    caps = {}
    for c in calib:
        v = work / f"delta_{part}_{stem}.{int(c.split('_')[1])}.mp4"
        if v.exists():
            caps[c] = cv2.VideoCapture(str(v))
    trk = {c: None for c in caps}
    rec = {}
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
        if fi % fstride or fi not in ybx:
            continue
        bx = ybx[fi]
        row = {}
        for c, im in got.items():
            seeded = False
            tp = None
            if trk[c] is None:
                if bx.get(c) is not None:
                    t = make_tracker(kind)
                    t.init(im, tuple(int(v) for v in bx[c]))
                    trk[c] = t
                    tp = ctr(bx[c])
                    seeded = True
            else:
                ok2, bb = trk[c].update(im)
                if ok2:
                    tp = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
            row[c] = {"yolo": (list(bx[c]) if bx.get(c) is not None else None),
                      "trk": (list(tp) if tp is not None else None), "seeded": seeded}
        rec[fi] = row
    for cap in caps.values():
        cap.release()
    cf.write_text(json.dumps(rec))
    return cf, False


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P15", "P13"])
    ap.add_argument("--tracker", default="dasiamrpn", choices=["dasiamrpn", "vit"])
    ap.add_argument("--max-trials", type=int, default=4)
    ap.add_argument("--fstride", type=int, default=4)
    a = ap.parse_args(argv)
    for part in a.parts:
        mp = f"runs/segment/runs/cup_seg/cup_yolo_1k_{part}/weights/best.pt"
        m = YOLO(mp)
        mh = model_hash(mp)
        calib = C._load_calib_mm(part)
        work = C.DELTA / part / "work" / "clips"
        stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{part}_", "")
                        for p in glob.glob(str(work / "*.mp4"))})[:a.max_trials]
        for stem in stems:
            cf, hit = cache_trial(m, mh, part, stem, work, calib, a.tracker, a.fstride)
            print(f"  {part} {stem:28} {'CACHED' if hit else 'written'} -> {cf.name}", flush=True)
    print(f"\nall trajectories in {OUT}", flush=True)


if __name__ == "__main__":
    main()
