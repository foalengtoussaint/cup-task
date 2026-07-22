"""Re-cut mis-cut trials at the CORRECT source position, using the reference camera's placement.

For each trial the correct content sits at t_ref (located in the ref uncut) + session_offset in
the suspect's uncut. We use that as a GUESS and refine to frame accuracy with a NARROW-window
motion-energy align (narrow = periodicity-safe, unlike the +-10s align that landed on random
repetitions earlier). Writes re-cut clips to cache/delta/<P>/recut/ (NO deletes of originals).

Validate mode (--check) re-detects the re-cut + a few ref clips and reports the wrist reprojection
vs the FINE-cam consensus: a correct re-cut collapses reproj to ~consensus level (P13-style ~9px);
a still-high value means the re-cut is wrong (or the camera is also miscalibrated).

    python scripts/recut_from_audit.py --part P13 --cam 2 --ref 3 --trials trial_10_L_unaffected ...
    python scripts/recut_from_audit.py --part P12 --cam 5 --ref 3 --all-trials
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from delta_recut import (coarse_thumbs, motion_signal, align_trial, fps_of, duration,  # noqa
                         SHARE, CT)
from cut_placement_audit import resolve_uncut, session_offset, locate2

OUT_FPS_ARGS = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an"]


def recut(part, cam, ref, trials, search=2.0, dry=False):
    vd = Path(SHARE) / part / "01_Measurement" / "04_Video"
    uref = resolve_uncut(part, ref)
    usus = resolve_uncut(part, cam)
    if uref is None or usus is None:
        raise SystemExit(f"missing uncut: ref={uref} sus={usus}")
    if str(CT) not in str(usus):
        raise SystemExit(f"suspect uncut not LOCAL ({usus}); download first")
    fps = fps_of(usus)
    tref = coarse_thumbs(uref, f"{part}_cam{ref}")
    off, r = session_offset(motion_signal(coarse_thumbs(uref, f"{part}_cam{ref}")),
                            motion_signal(coarse_thumbs(usus, f"{part}_cam{cam}")))
    print(f"{part} cam{cam}: session offset {off:+.1f}s vs cam{ref} (r={r:.3f}), fps {fps:.2f}",
          flush=True)
    outdir = CT / "cache" / "delta" / part / "recut"
    outdir.mkdir(parents=True, exist_ok=True)
    done = []
    for t in trials:
        rclip = vd / "03_Cut" / "drinking" / f"cam{ref}" / f"{t}.mp4"
        if not rclip.exists():
            print(f"  {t}: no ref clip, skip", flush=True); continue
        dur = duration(rclip)
        t_ref, f_ref, _ = locate2(rclip, tref, uref)
        if t_ref is None or f_ref < 0.99:
            print(f"  {t}: ref locate unsure (ncc {f_ref:.3f}), skip", flush=True); continue
        guess = t_ref + off
        t_correct, q = align_trial(rclip, usus, guess, dur, fps, search=search)
        dst = outdir / f"delta_{part}_{t}.{cam}.mp4"
        tag = "" if q >= 1.05 else "  [LOW-Q]"
        print(f"  {t}: ref@{t_ref:.2f}s guess {guess:.2f}s -> align {t_correct:.2f}s (q={q:.2f})"
              f"{tag} -> {dst.name}", flush=True)
        if not dry:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t_correct:.3f}",
                            "-i", str(usus), "-t", f"{dur:.3f}", "-vf", "scale=1920:1080",
                            *OUT_FPS_ARGS, str(dst)], check=True)
            done.append((t, dst))
    print(f"re-cut {len(done)} clips to {outdir}", flush=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--ref", type=int, default=3)
    ap.add_argument("--trials", nargs="+", default=None)
    ap.add_argument("--all-trials", action="store_true")
    ap.add_argument("--search", type=float, default=2.0, help="+-s align window around guess")
    ap.add_argument("--dry", action="store_true", help="print positions, don't write clips")
    a = ap.parse_args(argv)
    trials = a.trials
    if a.all_trials:
        cd = Path(SHARE) / a.part / "01_Measurement" / "04_Video" / "03_Cut" / "drinking" / f"cam{a.cam}"
        trials = sorted(p.stem for p in cd.glob("*.mp4"))
    if not trials:
        raise SystemExit("pass --trials or --all-trials")
    recut(a.part, a.cam, a.ref, trials, a.search, a.dry)


if __name__ == "__main__":
    main()
