"""How LONG does the cup get lost, not just how often.

good_frame% alone can't distinguish two very different failure shapes:
  * 72% good as scattered single-frame misses  -> harmless, a KF/interp fills them
  * 72% good as a few long blackouts           -> the track is genuinely gone for ~a second
This measures the RUN LENGTHS of consecutive frames with no >=3-cam consensus, and reports how
much of the total loss lives in short (fillable) vs long (unrecoverable) gaps.

Uses the SAME consensus/gate as eval_cup_3d_delta (imported, not reimplemented), so the good/bad
per-frame sequence is exactly the one behind the reported good_frame%.

    python scripts/gap_analysis_cup.py --model <best.pt> --parts P07 P08 P15 P13 \
        --max-trials 4 --fstride 4
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_cup_3d_delta as E  # noqa: E402
import compare_pose_omc_delta as C  # noqa: E402

SRC_FPS = 60.0


def good_sequence(model, part, max_trials, fstride):
    """-> list of per-trial boolean arrays: True = frame reached >=3-cam consensus."""
    calib = C._load_calib_mm(part)
    work = C.DELTA / part / "work" / "clips"
    stems = sorted({Path(p).name.split(".")[0].replace(f"delta_{part}_", "")
                    for p in glob.glob(str(work / "*.mp4"))})[:max_trials]
    out = []
    for stem in stems:
        caps = {}
        for c in calib:
            k = int(c.split("_")[1])
            v = work / f"delta_{part}_{stem}.{k}.mp4"
            if v.exists():
                caps[c] = cv2.VideoCapture(str(v))
        if len(caps) < E.MINC:
            continue
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
                seqs[c].append(E.cup_center(model, img))
        for cap in caps.values():
            cap.release()
        n = min((len(v) for v in seqs.values()), default=0)
        good = np.zeros(n, bool)
        for fr in range(n):
            obs = {c: seqs[c][fr] for c in seqs if seqs[c][fr] is not None}
            if len(obs) >= 2:
                X, _, _ = E.consensus(obs, calib)
                good[fr] = X is not None
        if n:
            out.append(good)
    return out


def gap_stats(seqs, fstride):
    """Run lengths of consecutive BAD frames, in source-frame units and ms."""
    gaps = []
    tot = ngood = 0
    for g in seqs:
        tot += len(g)
        ngood += int(g.sum())
        run = 0
        for v in g:
            if v:
                if run:
                    gaps.append(run)
                run = 0
            else:
                run += 1
        if run:
            gaps.append(run)
    if not gaps:
        return dict(good=ngood / tot if tot else 0, n_gaps=0)
    gaps = np.array(gaps)
    ms = gaps * fstride / SRC_FPS * 1000.0     # each sampled step = fstride source frames
    lost = gaps.sum()
    return dict(
        good=ngood / tot, n_gaps=len(gaps),
        med_ms=float(np.median(ms)), p90_ms=float(np.percentile(ms, 90)), max_ms=float(ms.max()),
        # where does the LOST TIME live? short gaps are interpolatable, long ones are not
        frac_lost_in_short=float(gaps[ms <= 100].sum() / lost),
        frac_lost_in_long=float(gaps[ms > 500].sum() / lost),
    )


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P15", "P13"])
    ap.add_argument("--max-trials", type=int, default=4)
    ap.add_argument("--fstride", type=int, default=4)
    a = ap.parse_args(argv)
    model = YOLO(a.model)
    print(f"model {a.model}  (fstride {a.fstride} -> one sample per {a.fstride} src frames "
          f"= {a.fstride/SRC_FPS*1000:.0f}ms)", flush=True)
    print(f"{'part':6}{'good%':>7}{'gaps':>6}{'med':>8}{'p90':>8}{'max':>9}"
          f"{'lost<=100ms':>13}{'lost>500ms':>12}", flush=True)
    for part in a.parts:
        s = gap_stats(good_sequence(model, part, a.max_trials, a.fstride), a.fstride)
        if not s.get("n_gaps"):
            print(f"{part:6}{s['good']*100:6.1f}%   (no gaps)", flush=True)
            continue
        print(f"{part:6}{s['good']*100:6.1f}%{s['n_gaps']:6d}{s['med_ms']:7.0f}ms"
              f"{s['p90_ms']:7.0f}ms{s['max_ms']:8.0f}ms"
              f"{s['frac_lost_in_short']*100:12.0f}%{s['frac_lost_in_long']*100:11.0f}%", flush=True)


if __name__ == "__main__":
    main()
