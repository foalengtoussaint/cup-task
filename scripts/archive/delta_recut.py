"""Re-cut a desynced DELTA camera from its UNCUT video to recover FULL coverage.

WHY re-cut and not reindex: the desync is a constant per-camera frame offset (measured: P13
cam2 = +139, SD 2). Reindexing the already-cut clip loses ~offset frames at the edge (the clip
doesn't contain them) and a half-aligned camera HURTS robust triangulation -- verified: P13
reindex gave 46% vs 99.7% for just dropping cam2. Re-cutting from the uncut video recovers the
missing frames, so the camera becomes genuinely good -- needed because the CUP is occluded at the
apex and needs all 5 cameras for consistent >=3-cam consensus.

Method (offset already known from scripts/sync_fix_delta.py):
  1. NCC-locate the (wrong) cut clip inside the camera's uncut video. Same-source frames match
     ~1.0 at 160x90, so the true position is unambiguous.  [ported from fix_slicing.locate]
  2. The wrong clip's action is delayed by `offset` frames, so the CORRECT window in the uncut
     starts `offset` frames later. Re-cut uncut[S+offset : S+offset+dur].
  3. cam1 is 720p on the share -> upscale the re-cut to 1080p (calib expects 1080p).

    python scripts/delta_recut.py --part P13 --cam 2 --offset 139 --trials trial_10_L_unaffected
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

SHARE = ("/run/user/1000/gvfs/smb-share:server=nslliappl01.lli.local,"
         "share=research_analyzed_dataset/DELTA/DELTA/DATA/data_newStruc")
CT = Path("/home/imove/Documents/cup-task")
W, H = 160, 90
COARSE_FPS = 2.0


def gray_frames(path, fps=None, ss=0.0, t=None):
    cmd = ["ffmpeg", "-v", "error"]
    if ss:
        cmd += ["-ss", f"{ss:.3f}"]
    cmd += ["-i", str(path)]
    if t:
        cmd += ["-t", f"{t:.3f}"]
    vf = (f"fps={fps}," if fps else "") + f"scale={W}:{H},format=gray"
    cmd += ["-vf", vf, "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, check=True, capture_output=True)
    n = len(p.stdout) // (W * H)
    return np.frombuffer(p.stdout[:n * W * H], np.uint8).reshape(n, W * H).astype(np.float32)


def local_uncut(part, cam):
    """Prefer a locally-downloaded uncut (fast full-fps decode); fall back to the SMB share."""
    loc = CT / "cache" / "delta" / part / "uncut" / f"cam{cam}.mp4"
    if loc.exists():
        return loc
    return Path(SHARE) / part / "01_Measurement" / "04_Video" / "01_Uncut" / f"cam{cam}.mp4"


def coarse_thumbs(uncut_path, cache_key):
    """Full-session thumbnails at COARSE_FPS, cached to .npy so each uncut is decoded once."""
    cache = CT / "cache" / "delta" / "_thumbs"
    cache.mkdir(parents=True, exist_ok=True)
    npy = cache / f"{cache_key}_{COARSE_FPS:g}fps.npy"
    if npy.exists():
        return np.load(npy)
    thumbs = gray_frames(uncut_path, fps=COARSE_FPS)
    np.save(npy, thumbs)
    return thumbs


def _norm(f):
    f = f - f.mean(1, keepdims=True)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-6)


def ncc_scan(uncut, clip):
    U, C = _norm(uncut), _norm(clip)
    n, m = len(U), len(C)
    if n < m:
        return np.array([-1.0])
    sims = U @ C.T
    out = np.zeros(n - m + 1, np.float32)
    for i in range(m):
        out += sims[i:i + n - m + 1, i]
    return out / m


def fps_of(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
                       check=True, capture_output=True)
    num, den = p.stdout.decode().strip().split("/")
    return float(num) / float(den)


def duration(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], check=True, capture_output=True)
    return float(p.stdout.strip())


def motion_signal(thumbs):
    """Per-frame motion energy (view-invariant): the pattern of movement bursts lines up across
    viewpoints even though the images don't. [ported from fix_slicing.motion_signal]"""
    m = np.abs(np.diff(thumbs, axis=0)).mean(axis=1)
    m = m - np.median(m)
    mad = np.median(np.abs(m)) or 1.0
    return (m / mad).astype(np.float32)


def align_trial(ref_clip, uncut_path, guess_s, dur, fps, search=10.0):
    """Find the window in the bad uncut whose MOTION matches the reference clip -> (t_start, q).

    Slides the reference clip's motion signature over the bad uncut in +-search s around guess_s,
    scored by NORMALIZED cross-correlation (zero-mean, unit-norm per window -- a raw dot favours
    whatever window has the most motion). View-invariant, so it aligns a different camera directly.
    [ported from fix_slicing.align_trial, run at FULL fps for frame-accurate precision rather than
    the original's 10fps -- feasible now that the uncut is local. Precision = 1 frame not ~6.]
    quality = best vs best-outside-+-1s; drink trials repeat ~10s so q~=1 needs visual QC.
    """
    mr = motion_signal(gray_frames(ref_clip))          # full fps
    mr = mr - mr.mean()
    mr /= max(float(np.linalg.norm(mr)), 1e-9)
    ss = max(guess_s - search, 0.0)
    mb = motion_signal(gray_frames(uncut_path, ss=ss, t=dur + 2 * search))
    n = len(mb) - len(mr) + 1
    if n < 2:
        return guess_s, 0.0
    # sliding normalized xcorr via cumulative sums (n can be ~1300 at 60fps -> vectorize)
    m = len(mr)
    sc = np.full(n, -1.0, np.float64)
    for i in range(n):
        w = mb[i:i + m]
        w = w - w.mean()
        d = float(np.linalg.norm(w))
        if d > 1e-9:
            sc[i] = float(np.dot(mr, w / d))
    i0 = int(np.argmax(sc))
    guard = int(fps)                                   # +-1s
    masked = sc.copy()
    masked[max(i0 - guard, 0): i0 + guard] = -np.inf
    rest = float(masked.max())
    q = sc[i0] / rest if rest > 0 else 9.9
    return ss + i0 / fps, float(q)


def locate(clip_path, uncut_thumbs, uncut_path):
    """(t_start_seconds, confidence) of the clip inside the uncut video."""
    c = gray_frames(clip_path, fps=COARSE_FPS)
    score = ncc_scan(uncut_thumbs, c)
    s0 = int(np.argmax(score))
    guard = int(60 * COARSE_FPS)
    masked = score.copy()
    masked[max(s0 - guard, 0): s0 + guard] = -1
    conf = float(score[s0] / max(float(masked.max()), 1e-6))
    t0 = s0 / COARSE_FPS
    # full-fps refine within +-1.5s
    fps = fps_of(uncut_path)
    seg_ss = max(t0 - 1.5, 0.0)
    seg = gray_frames(uncut_path, ss=seg_ss, t=3.0 + len(c) / COARSE_FPS)
    cf = gray_frames(clip_path)
    step = max(1, round(fps / 10))
    fine = ncc_scan(seg[::step], cf[::step][: max(len(seg[::step]) - 1, 1)])
    tf = seg_ss + int(np.argmax(fine)) * step / fps
    return tf, conf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="[legacy shortcut] frame shift; superseded by --align")
    ap.add_argument("--align", action="store_true",
                    help="motion-energy align to --ref-cam (robust; the fix_slicing method)")
    ap.add_argument("--ref-cam", type=int, default=3, help="reference (synced) camera for --align")
    ap.add_argument("--trials", nargs="+", default=None)
    ap.add_argument("--all-trials", action="store_true",
                    help="re-cut every trial this participant has cut clips for this camera")
    ap.add_argument("--min-q", type=float, default=1.05,
                    help="flag (don't trust) alignments below this quality for QC")
    ap.add_argument("--out", default=None, help="dir for re-cut clips (default: cache/delta/<P>/recut)")
    a = ap.parse_args(argv)

    vd = Path(SHARE) / a.part / "01_Measurement" / "04_Video"
    uncut = local_uncut(a.part, a.cam)
    if not uncut.exists():
        raise SystemExit(f"no uncut: {uncut}")
    is_local = str(CT) in str(uncut)
    out = Path(a.out) if a.out else CT / "cache" / "delta" / a.part / "recut"
    out.mkdir(parents=True, exist_ok=True)

    fps = fps_of(uncut)
    print(f"uncut cam{a.cam}: {'LOCAL' if is_local else 'SMB'}  ({duration(uncut)/60:.0f}min)  "
          f"coarse thumbs @ {COARSE_FPS}fps (cached)...", flush=True)
    thumbs = coarse_thumbs(uncut, f"{a.part}_cam{a.cam}")
    print(f"  {len(thumbs)} coarse frames", flush=True)

    trials = a.trials
    if a.all_trials:
        cdir = vd / "03_Cut" / "drinking" / f"cam{a.cam}"
        trials = sorted(p.stem for p in cdir.glob("*.mp4"))
    if not trials:
        raise SystemExit("no trials: pass --trials or --all-trials")

    flagged = []
    for t in trials:
        cut = vd / "03_Cut" / "drinking" / f"cam{a.cam}" / f"{t}.mp4"
        if not cut.exists():
            print(f"  {t}: no cut clip, skip", flush=True)
            continue
        dur = duration(cut)
        t0, conf = locate(cut, thumbs, uncut)
        if a.align:
            # robust: match the reference camera's action MOTION directly inside cam's uncut
            ref_clip = vd / "03_Cut" / "drinking" / f"cam{a.ref_cam}" / f"{t}.mp4"
            t_correct, q = align_trial(ref_clip, uncut, t0, dur, fps)
            tag = "" if q >= a.min_q else "  [LOW-Q, needs QC]"
            if q < a.min_q:
                flagged.append((t, q))
            print(f"  {t}: located {t0:.2f}s (conf {conf:.2f}); motion-align to cam{a.ref_cam}"
                  f" -> {t_correct:.2f}s (q={q:.2f}){tag}", flush=True)
        else:
            t_correct = t0 + a.offset / fps
            print(f"  {t}: located {t0:.2f}s (conf {conf:.2f}) -> +{a.offset}fr shortcut "
                  f"-> {t_correct:.2f}s", flush=True)
        dst = out / f"delta_{a.part}_{t}.{a.cam}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t_correct:.3f}", "-i", str(uncut),
               "-t", f"{dur:.3f}", "-vf", "scale=1920:1080", "-c:v", "libx264", "-crf", "18",
               "-preset", "veryfast", "-an", str(dst)]
        subprocess.run(cmd, check=True)
    print(f"wrote re-cut clips to {out}", flush=True)
    if flagged:
        print(f"LOW-Q (needs QC, {len(flagged)}): "
              + ", ".join(f"{t}(q{q:.2f})" for t, q in flagged), flush=True)


if __name__ == "__main__":
    main()
