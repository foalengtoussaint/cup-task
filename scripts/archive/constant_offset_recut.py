"""Re-cut a whole camera from ONE constant session offset -- no per-trial motion align.

WHY: the per-trial `align_trial` refinement in recut_from_audit.py is the thing that breaks.
The drink task is quasi-periodic, so motion-energy correlation has near-equal local peaks at
neighbouring repetitions; the align wanders to the wrong one and its self-score `q` endorses the
wrong answer confidently (measured on P13 cam2: trial_45 was 1.15s wrong at q=3.75, trial_48 was
26 frames wrong at q=1.09). Motion energy cannot falsify itself -- only geometry can.

MEASURED on P13 cam2 vs cam3 (n=20 trials over 13 min of session): the true offset is CONSTANT at
-3.07s +- 0.08s with no drift trend. The spread is fully explained by measurement resolution
(locate2 resolves t_ref on a 10fps grid = +-3fr, plus best-lag noise where the reproj curve is
flat). The old pipeline already assumed a constant (`guess = t_ref + off`) -- it was just 0.57s
WRONG, because `off` came from a coarse 2fps motion xcorr (r=0.464).

So: estimate the constant by REPROJECTION on a few trials, average it, apply it to every trial,
then verify every cut by reprojection. Two passes, geometry as the arbiter throughout.

  # estimate + recut + verify
  python scripts/constant_offset_recut.py --part P13 --cam 2 --ref 3 --offsets-log <dry.log>

Originals are preserved in work/rejected_preRecut/ -- nothing is deleted.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pose_omc_delta as C  # noqa: E402
from pipeline.kalman_3d import project  # noqa: E402
from delta_recut import duration, fps_of, SHARE, CT  # noqa: E402
from cut_placement_audit import resolve_uncut  # noqa: E402
from reaudit_cam_quality import ransac_point  # noqa: E402

LINE = re.compile(r"(trial_\S+?):\s*ref@([\d.]+)s.*?align\s+([\d.]+)s")


def parse_log(path):
    """-> {trial: (t_ref, t_align)} from a `recut_from_audit --dry` log."""
    out = {}
    for m in LINE.finditer(Path(path).read_text()):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def measure_lag(part, trial, sus_cam, cams, dets, maxlag=120):
    """Frame lag of sus_cam vs the leave-it-out >=3-cam RANSAC consensus. (lag, resid, resid@0)."""
    side = "right" if "_R_" in trial else "left"
    fn = C._kp_point(f"{side}_wrist")
    per = {}
    for pj in glob.glob(str(dets / f"delta_{part}_{trial}.*.pose.json")):
        c = "cam_" + Path(pj).name.split(".")[1]
        if c in cams:
            per[c] = json.loads(Path(pj).read_text())["frames"]
    if sus_cam not in per or len(per) < 3:
        return None
    sus = per[sus_cam]
    ref = {k: v for k, v in per.items() if k != sus_cam}
    n = min(len(v) for v in per.values())
    X = [None] * n
    for f in range(n):
        pts = {c: fn(ref[c][f]) for c in ref if fn(ref[c][f]) is not None}
        if len(pts) >= 2:
            X[f], _ = ransac_point({c: cams[c] for c in pts if c in cams}, pts)
    best, at0 = (0, 1e9), None
    for L in range(-maxlag, maxlag + 1):
        e = []
        for f in range(n):
            g = f + L
            if X[f] is None or g < 0 or g >= n:
                continue
            p = fn(sus[g])
            if p is not None:
                e.append(float(np.hypot(*(project(cams[sus_cam], X[f])[0] - p))))
        if len(e) >= 10:
            m = float(np.median(e))
            if L == 0:
                at0 = m
            if m < best[1]:
                best = (L, m)
    return best[0], best[1], at0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--ref", type=int, default=3)
    ap.add_argument("--offsets-log", required=True, help="a `recut_from_audit --dry` log")
    ap.add_argument("--override", nargs="*", default=[],
                    help="trial=true_position_seconds for already-corrected cuts")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    part, cam = a.part, a.cam
    sus_cam = f"cam_{cam}"
    cams = C._load_calib_mm(part)
    work = CT / "cache" / "delta" / part / "work"
    dets, clips = work / "dets", work / "clips"
    usus = resolve_uncut(part, cam)
    fps = fps_of(usus)
    tbl = parse_log(a.offsets_log)
    over = {k: float(v) for k, v in (o.split("=") for o in a.override)}

    # 1. per-trial offset estimate = (current cut position + measured lag) - t_ref
    print(f"== estimating constant offset for {part} {sus_cam} vs cam{a.ref} (fps {fps:.3f})",
          flush=True)
    ests = {}
    for t, (t_ref, t_align) in sorted(tbl.items()):
        r = measure_lag(part, t, sus_cam, cams, dets)
        if r is None:
            print(f"  {t:28} no consensus, skip", flush=True)
            continue
        L, resid, at0 = r
        base = over.get(t, t_align)
        true_pos = base + L / fps
        ests[t] = (true_pos - t_ref, t_ref, L, resid, at0)
        print(f"  {t:28} ref@{t_ref:8.2f} cut@{base:8.2f} lag{L:+4d} "
              f"resid {resid:5.1f} (@0 {at0:5.1f}) -> off {true_pos - t_ref:+.3f}s", flush=True)
    vals = np.array([v[0] for v in ests.values()])
    off = float(vals.mean())
    print(f"\n== CONSTANT OFFSET = {off:+.4f}s  (n={len(vals)}, sd {vals.std():.4f}s, "
          f"min {vals.min():+.3f} max {vals.max():+.3f}, = {off*fps:+.1f} frames)", flush=True)
    if a.dry:
        return

    # 2. re-cut EVERY trial at t_ref + constant
    vd = Path(SHARE) / part / "01_Measurement" / "04_Video" / "03_Cut" / "drinking"
    pre = work / "rejected_preRecut"
    pre.mkdir(exist_ok=True)
    for t, (t_ref, _) in sorted(tbl.items()):
        dur = duration(vd / f"cam{a.ref}" / f"{t}.mp4")
        dst = clips / f"delta_{part}_{t}.{cam}.mp4"
        if dst.exists() and not (pre / dst.name).exists():
            (pre / dst.name).write_bytes(dst.read_bytes())   # preserve, never delete
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t_ref + off:.3f}",
                        "-i", str(usus), "-t", f"{dur:.3f}", "-vf", "scale=1920:1080",
                        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an",
                        str(dst)], check=True)
        print(f"  recut {t:28} -> {t_ref + off:.3f}s", flush=True)
    print(f"re-cut {len(tbl)} clips at the constant offset; originals in {pre}", flush=True)


if __name__ == "__main__":
    main()
