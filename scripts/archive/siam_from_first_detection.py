"""DETECT ONCE, THEN TRACK: YOLO finds the cup on the first frame it can, a Siamese tracker
carries it for the rest of the trial. No further detection.

This is the actual "SiamFC with YOLO detecting once at the start" experiment. Per CAMERA:
  * run YOLO until it first finds the cup -> seed the tracker with that box
  * every frame after, the tracker alone produces the box (YOLO is NOT run again)
Then feed the per-camera tracked points into the SAME >=3-cam consensus as everything else and
compare to the YOLO-every-frame baseline.

What it tests that gap-bridging could not: can appearance tracking hold the cup through the whole
drink cycle -- including the apex occlusion -- without re-detection? And it is the version that
could be FAST, since YOLO runs once instead of every frame.

Also reports the failure mode that matters for detect-once: DRIFT over time (distance from the
consensus point, by phase of trial) and whether a camera ever RECOVERS after occlusion.

    python scripts/siam_from_first_detection.py --part P08 --model <yolo.pt> \
        --tracker dasiamrpn --max-trials 4 --fstride 4
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


def run_trial(model, calib, work, part, stem, kind, fstride):
    caps = {}
    for c in calib:
        v = work / f"delta_{part}_{stem}.{int(c.split('_')[1])}.mp4"
        if v.exists():
            caps[c] = cv2.VideoCapture(str(v))
    if len(caps) < E.MINC:
        return None
    trk = {c: None for c in caps}      # tracker per camera, seeded once
    seeded_at = {c: None for c in caps}
    base_good = trk_good = tot = 0
    drift = []                          # (phase 0-1, px distance tracked-vs-consensus)
    rows = []
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
        ybox = {c: yolo_box(model, im) for c, im in got.items()}      # baseline (every frame)
        ydet = {c: ctr(b) for c, b in ybox.items() if b is not None}
        Xb = E.consensus(ydet, calib)[0] if len(ydet) >= 2 else None
        base_good += Xb is not None

        # DETECT-ONCE: seed each camera's tracker at its FIRST detection, then track only
        tdet = {}
        for c, im in got.items():
            if trk[c] is None:
                if ybox[c] is not None:
                    t = make_tracker(kind)
                    t.init(im, tuple(int(v) for v in ybox[c]))
                    trk[c] = t
                    seeded_at[c] = fi
                    tdet[c] = ctr(ybox[c])
                continue
            ok2, bb = trk[c].update(im)
            if ok2:
                tdet[c] = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        Xt = E.consensus(tdet, calib)[0] if len(tdet) >= 2 else None
        trk_good += Xt is not None
        rows.append((fi, len(ydet), len(tdet), Xb is not None, Xt is not None))
        # drift: where both exist, how far is the tracked point from the YOLO consensus?
        if Xb is not None:
            for c, p in tdet.items():
                if trk[c] is not None and seeded_at[c] is not None and fi > seeded_at[c]:
                    e = float(np.hypot(*(E.project(calib[c], Xb)[0] - np.array(p))))
                    drift.append((fi, e))
    for cap in caps.values():
        cap.release()
    return dict(tot=tot, base=base_good, trk=trk_good, drift=drift, rows=rows,
                nframes=rows[-1][0] if rows else 0)


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
    calib = C._load_calib_mm(a.part)
    work = C.DELTA / a.part / "work" / "clips"
    stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{a.part}_", "")
                    for p in glob.glob(str(work / "*.mp4"))})[:a.max_trials]

    T = B = K = 0
    alld = []
    for stem in stems:
        r = run_trial(m, calib, work, a.part, stem, a.tracker, a.fstride)
        if r is None:
            continue
        T += r["tot"]; B += r["base"]; K += r["trk"]; alld += r["drift"]
        print(f"  {stem:28} YOLO-every {r['base']/r['tot']*100:5.1f}%  "
              f"detect-once+track {r['trk']/r['tot']*100:5.1f}%", flush=True)

    print(f"\n=== {a.part}  tracker={a.tracker}  (n={T} frames, {len(stems)} trials) ===")
    print(f"  YOLO every frame      : {B/T*100:5.1f}% good")
    print(f"  detect-once + track   : {K/T*100:5.1f}% good   ({(K-B)/T*100:+.1f} pts)")
    if alld:
        d = np.array([e for _, e in alld])
        fr = np.array([f for f, _ in alld], float)
        q = np.quantile(fr, [0.25, 0.5, 0.75])
        print(f"\n  tracked-vs-consensus drift (px): median {np.median(d):.0f}  "
              f"p90 {np.percentile(d,90):.0f}  max {d.max():.0f}")
        for lo, hi, lab in [(fr.min(), q[0], "first quarter"), (q[0], q[1], "second"),
                            (q[1], q[2], "third"), (q[2], fr.max(), "last quarter")]:
            s = d[(fr >= lo) & (fr <= hi)]
            if len(s):
                print(f"    {lab:14}: median {np.median(s):6.0f}px   "
                      f"(within 30px gate: {(s<=30).mean()*100:4.0f}%)")


if __name__ == "__main__":
    main()
