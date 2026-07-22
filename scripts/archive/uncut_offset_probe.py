"""Measure a camera's CUT-LEVEL desync by searching the UNCUT with reprojection as the score.

Why the earlier tools failed:
  * lag_probe_delta shifts detections WITHIN the existing cut clip -- if the clip was mis-cut, the
    correct action lies OUTSIDE that window, so no within-clip shift can ever align it (and the cam
    gets wrongly called MISCALIB).
  * per-trial / per-clip motion-energy align is defeated by the repetitive drink motion (many
    near-equal peaks -> inconsistent, low-quality offsets).

This probe cuts a WIDE +-PAD window from the (local) uncut around the current cut position, detects
the wrist once, then sweeps a CONSTANT frame offset O, scoring each O by the median reprojection of
the suspect wrist onto the consensus 3D wrist built from the FINE reference cameras. Reprojection is
geometry, so it is immune to periodicity, and searching the uncut reaches content outside the clip.

  min reproj low at a single consistent O across trials  -> DESYNC by O frames (re-cut fixes it)
  min reproj high at EVERY O                              -> MISCALIB (geometry wrong)

    python scripts/uncut_offset_probe.py --part P12 --cam 4 --ref 1 2 3 --trials trial_10_L_unaffected ...
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from cup_task.kalman_3d import project  # noqa: E402
from reaudit_cam_quality import ransac_point, _side  # noqa: E402
from delta_recut import local_uncut, locate, coarse_thumbs, fps_of, duration, SHARE  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
               "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/uncut_probe")
POSE_MODEL = "models/yolo26s-pose.pt"
CUP_MODEL = "models/cup_clean3d_refill.pt"
LOW = 15.0


def _detect_wide(part, cam, trial, uncut, thumbs, fps, pad):
    """Cut a +-pad window around the current cut of (part,cam,trial) from uncut, detect wrist.
    Returns (frames, base_wide_idx) where clip-frame 0 of the ORIGINAL cut sits at base_wide_idx."""
    vd = Path(SHARE) / part / "01_Measurement" / "04_Video"
    cut = vd / "03_Cut" / "drinking" / f"cam{cam}" / f"{trial}.mp4"
    dur = duration(cut)
    t0, conf = locate(cut, thumbs, uncut)           # where the existing cut sits in the uncut
    ss = max(t0 - pad, 0.0)
    base = int(round((t0 - ss) * fps))              # wide-index of clip-frame 0
    SCRATCH.mkdir(parents=True, exist_ok=True)
    wide = SCRATCH / f"delta_{part}_{trial}__wide.{cam}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.3f}", "-i", str(uncut),
                    "-t", f"{dur + 2 * pad:.3f}", "-vf", "scale=1920:1080", "-c:v", "libx264",
                    "-crf", "20", "-preset", "veryfast", "-an", str(wide)], check=True)
    subprocess.run([sys.executable, "scripts/detect_rep_batched.py", str(SCRATCH),
                    f"delta_{part}_{trial}__wide", "-o", str(SCRATCH), "--pose-model", POSE_MODEL,
                    "--cup-model", CUP_MODEL, "--batch", "32", "--device", "0"],
                   check=True, capture_output=True)
    pj = SCRATCH / f"delta_{part}_{trial}__wide.{cam}.pose.json"
    frames = json.loads(pj.read_text())["frames"]
    return frames, base, dur, conf


def probe(part, cam, refs, trials, pad, maxoff):
    side = _side(part)
    fn = C._kp_point(f"{side}_wrist")
    cams = C._load_calib_mm(part)
    scam = f"cam_{cam}"
    uncut = local_uncut(part, cam)
    if not uncut.exists():
        raise SystemExit(f"no local uncut for {part} cam{cam}: {uncut}")
    fps = fps_of(uncut)
    thumbs = coarse_thumbs(uncut, f"{part}_cam{cam}")

    # accumulate reproj(O) across trials
    offs = np.arange(-maxoff, maxoff + 1, 1)
    per_off = {int(o): [] for o in offs}
    for t in trials:
        wide, base, dur, conf = _detect_wide(part, cam, t, uncut, thumbs, fps, pad)
        nclip = int(round(dur * fps))
        # consensus per clip-frame from ref cams' EXISTING cut dets
        refd = {}
        for r in refs:
            p = C.DELTA / part / "dets" / f"delta_{part}_{t}.{r}.pose.json"
            if p.exists():
                refd[f"cam_{r}"] = json.loads(p.read_text())["frames"]
        if len(refd) < 2:
            print(f"  {t}: <2 ref cams, skip", flush=True); continue
        nref = min(len(v) for v in refd.values())
        X3 = [None] * nref
        for f in range(nref):
            pts = {c: fn(refd[c][f]) for c in refd if fn(refd[c][f]) is not None}
            if len(pts) >= 2:
                X3[f], _ = ransac_point({c: cams[c] for c in pts if c in cams}, pts)
        # sweep offset
        best_o, best_e = None, 1e9
        for o in offs:
            errs = []
            for f in range(min(nref, nclip)):
                if X3[f] is None:
                    continue
                wi = base + f + int(o)
                if 0 <= wi < len(wide):
                    p = fn(wide[wi])
                    if p is not None and scam in cams:
                        errs.append(float(np.hypot(*(project(cams[scam], X3[f])[0] - p))))
            if len(errs) >= 10:
                e = float(np.median(errs))
                per_off[int(o)].append(e)
                if e < best_e:
                    best_e, best_o = e, int(o)
        print(f"  {t}: located conf {conf:.2f}  best O={best_o:+d} -> {best_e:.0f}px", flush=True)

    # collapse across trials
    curve = {o: (np.median(v) if v else float("nan")) for o, v in per_off.items()}
    finite = {o: e for o, e in curve.items() if np.isfinite(e)}
    if not finite:
        print(f"\n{part} cam{cam}: no measurements"); return
    bestO = min(finite, key=finite.get)
    e0 = curve.get(0, float("nan"))
    print(f"\n{part} cam{cam}  (ref={refs}, {len(trials)} trials, +-{pad:.0f}s window)")
    show = sorted(set([0, bestO] + [bestO + d for d in (-60, -30, -5, 5, 30, 60)]) & set(finite))
    print("  offset(frames) -> reproj(px):  " + "  ".join(f"{o:+d}:{curve[o]:.0f}" for o in show))
    print(f"  reproj @O=0 : {e0:.0f}px      best O={bestO:+d} : {finite[bestO]:.0f}px")
    if finite[bestO] <= LOW and abs(bestO) >= 2:
        v = f"DESYNC -> constant re-cut by {bestO:+d} frames ({bestO/fps*1000:+.0f} ms); reproj {e0:.0f}->{finite[bestO]:.0f}px"
    elif finite[bestO] <= LOW:
        v = "FINE (already aligned)"
    elif finite[bestO] > 2 * LOW:
        v = f"MISCALIB (best {finite[bestO]:.0f}px at O={bestO:+d} -- no offset rescues geometry)"
    else:
        v = f"AMBIGUOUS (best {finite[bestO]:.0f}px at O={bestO:+d})"
    print(f"  => {v}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--ref", nargs="+", type=int, required=True, help="FINE reference cams")
    ap.add_argument("--trials", nargs="+", default=None)
    ap.add_argument("--auto-trials", type=int, default=0,
                    help="if set, auto-pick this many trials that have both the suspect cut clip "
                         "and >=2 ref-cam dets cached")
    ap.add_argument("--pad", type=float, default=8.0, help="+-seconds window around current cut")
    ap.add_argument("--maxoff", type=int, default=480, help="max |offset| to test (frames)")
    a = ap.parse_args(argv)
    trials = a.trials
    if not trials and a.auto_trials:
        vd = Path(SHARE) / a.part / "01_Measurement" / "04_Video"
        cutdir = vd / "03_Cut" / "drinking" / f"cam{a.cam}"
        have_cut = {p.stem for p in cutdir.glob("*.mp4")}
        # need >=2 ref-cam dets cached for the same trial
        cand = []
        for t in sorted(have_cut):
            nref = sum((C.DELTA / a.part / "dets" / f"delta_{a.part}_{t}.{r}.pose.json").exists()
                       for r in a.ref)
            if nref >= 2:
                cand.append(t)
        trials = cand[:a.auto_trials]
        print(f"auto-picked {len(trials)} trials: {trials}", flush=True)
    if not trials:
        raise SystemExit("no trials (pass --trials or --auto-trials with cached ref dets)")
    probe(a.part, a.cam, a.ref, trials, a.pad, min(a.maxoff, int(a.pad * 60)))


if __name__ == "__main__":
    main()
