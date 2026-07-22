"""Audit WHERE each camera's cut clips actually sit in its uncut session video.

The P12 discovery: cam4's "trial_10" cut is footage from a DIFFERENT repetition (+154s away),
and the misplacement varies per trial and per camera. Such shuffled cuts masquerade as
"miscalibration" in every reprojection test (a different repetition of the same task produces
plausibly-wrong wrist positions, so no time-shift and no calibration ever fits).

Method, per participant, all calibration-free:
  1. Whole-session motion-energy xcorr suspect-uncut vs ref-uncut -> the true session offset
     (PC clock difference; drift-free constant, +-30 s search at 2 fps).
  2. Per trial: locate() each camera's cut clip inside ITS OWN uncut (same-camera NCC template
     match -- the clip is literally a re-encoded excerpt of that uncut, verified pixel-exact,
     sequence-NCC .998 vs .986 background floor).
  3. misplacement = (t_suspect - t_ref) - session_offset. Zero for a correct cut.

Verdicts per camera:
  |mis| <= 1 s on all trials       -> CUTS-OK      (any reproj error is real miscalibration)
  constant mis beyond 1 s          -> CONST-OFFSET (single re-cut shift fixes all trials)
  varying mis                      -> SHUFFLED     (per-trial re-cut at t_ref + session_offset)

    python scripts/cut_placement_audit.py --part P14 --ref 3 --cams 2 5
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from delta_recut import (coarse_thumbs, motion_signal, locate, ncc_scan, gray_frames,  # noqa: E402
                         fps_of, SHARE, CT)

THUMB_FPS = 2.0
FINE_FPS = 10.0    # disambiguation pass
FINE_T = 4.0       # seconds of clip used for fine scoring
EXACT = 0.995      # fine NCC at the TRUE source position (pixel-exact) vs ~0.986 elsewhere


def _top_peaks(score, k=5, min_sep=8.0):
    """Indices of the k best coarse-NCC peaks, greedily separated by min_sep seconds."""
    order = np.argsort(score)[::-1]
    picks = []
    for i in order:
        if all(abs(i - j) >= min_sep * THUMB_FPS for j in picks):
            picks.append(int(i))
            if len(picks) == k:
                break
    return picks


def locate2(clip_path, uncut_thumbs, uncut_path, k=5):
    """locate() upgraded: the coarse 2fps argmax lands on the wrong REPETITION when trials sit
    ~8s apart (P13 put two different trials at the same spot -- impossible). Fix: keep the top-k
    coarse peaks and let PIXEL-EXACTNESS decide -- the cut clip is a re-encode of its own uncut,
    so a fine sequence-NCC at the true source is ~0.998 while other repetitions floor at ~0.986
    (measured on P12). Returns (t_seconds, fine_best, fine_runnerup)."""
    c2 = gray_frames(clip_path, fps=THUMB_FPS)
    score = ncc_scan(uncut_thumbs, c2)
    cands = _top_peaks(score, k=k)
    cf = gray_frames(clip_path, fps=FINE_FPS, t=FINE_T)
    best = (None, -9.0)
    scores = []
    for i in cands:
        t0 = i / THUMB_FPS
        ss = max(t0 - 2.0, 0.0)
        seg = gray_frames(uncut_path, fps=FINE_FPS, ss=ss, t=FINE_T + 4.0)
        if len(seg) < len(cf) + 1:
            continue
        sc = ncc_scan(seg, cf)
        j = int(np.argmax(sc))
        scores.append(float(sc[j]))
        if sc[j] > best[1]:
            best = (ss + j / FINE_FPS, float(sc[j]))
    scores.sort(reverse=True)
    runner = scores[1] if len(scores) > 1 else 0.0
    return best[0], best[1], runner


def resolve_uncut(part, cam):
    """Local cache first; then flat share layout; then pc*/ subdirs with timestamped names."""
    loc = CT / "cache" / "delta" / part / "uncut" / f"cam{cam}.mp4"
    if loc.exists():
        return loc
    base = Path(SHARE) / part / "01_Measurement" / "04_Video" / "01_Uncut"
    flat = base / f"cam{cam}.mp4"
    if flat.exists():
        return flat
    for pat in (f"pc*/cam{cam}_*.mp4", f"pc*/cam{cam}_*.mkv", f"pc*/cam{cam}*.mp4"):
        hits = sorted(base.glob(pat))
        if hits:
            return hits[0]
    return None


def session_offset(sig_ref, sig_sus, maxlag=60):
    """t_sus(event) - t_ref(event) in seconds, from whole-session motion xcorr at THUMB_FPS."""
    n = min(len(sig_ref), len(sig_sus))
    a, b = sig_ref[:n], sig_sus[:n]
    best = (0, -9.0)
    for L in range(-maxlag, maxlag + 1):
        x, y = (a[L:], b[:n - L]) if L >= 0 else (a[:n + L], b[-L:])
        x = x - x.mean(); y = y - y.mean()
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if c > best[1]:
            best = (L, c)
    # a[i+L] pairs with b[i] -> event at index i in sus == index i+L in ref
    return -best[0] / THUMB_FPS, best[1]


def audit_part(part, ref, cams, n_trials):
    vd = Path(SHARE) / part / "01_Measurement" / "04_Video"
    uref = resolve_uncut(part, ref)
    if uref is None:
        print(f"{part}: no uncut for ref cam{ref} -- skip"); return
    print(f"\n===== {part} (ref cam{ref}) =====", flush=True)
    print(f"  thumbs ref cam{ref} ({uref.name})...", flush=True)
    tref = coarse_thumbs(uref, f"{part}_cam{ref}")
    sref = motion_signal(tref)

    # trials present in both cut dirs
    for cam in cams:
        usus = resolve_uncut(part, cam)
        if usus is None:
            print(f"  cam{cam}: no uncut found -- skip"); continue
        print(f"  thumbs cam{cam} ({usus.name})...", flush=True)
        tsus = coarse_thumbs(usus, f"{part}_cam{cam}")
        ssus = motion_signal(tsus)
        off, r = session_offset(sref, ssus)
        print(f"  cam{cam}: session offset {off:+.1f}s vs cam{ref} (r={r:.3f})", flush=True)

        rdir = vd / "03_Cut" / "drinking" / f"cam{ref}"
        sdir = vd / "03_Cut" / "drinking" / f"cam{cam}"
        trials = sorted(({p.stem for p in rdir.glob("*.mp4")} &
                         {p.stem for p in sdir.glob("*.mp4")}))[:n_trials]
        mis = []
        for t in trials:
            t_r, f_r, r_r = locate2(rdir / f"{t}.mp4", tref, uref)
            t_s, f_s, r_s = locate2(sdir / f"{t}.mp4", tsus, usus)
            if t_r is None or t_s is None:
                print(f"    {t}: locate2 failed, skip", flush=True); continue
            sure_r = "" if f_r >= EXACT else "  [ref UNSURE]"
            sure_s = "" if f_s >= EXACT else f"  [cam{cam} UNSURE]"
            m = (t_s - t_r) - off
            if f_r >= EXACT and f_s >= EXACT:
                mis.append(m)
            print(f"    {t}: ref@{t_r:8.2f}s(ncc {f_r:.3f})  cam{cam}@{t_s:8.2f}s(ncc {f_s:.3f})"
                  f"  misplacement {m:+8.2f}s{sure_r}{sure_s}", flush=True)
        if not mis:
            print(f"  cam{cam}: no shared trials"); continue
        mis = np.array(mis)
        mx, med, sd = float(np.max(np.abs(mis))), float(np.median(mis)), float(np.std(mis))
        if mx <= 1.0:
            v = "CUTS-OK (reproj errors here = real miscalibration)"
        elif sd <= 0.5:
            v = f"CONST-OFFSET {med:+.2f}s (one shift re-cuts everything)"
        else:
            v = f"SHUFFLED (per-trial re-cut needed; spread {mis.min():+.1f}..{mis.max():+.1f}s)"
        print(f"  => cam{cam}: {v}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--ref", type=int, required=True)
    ap.add_argument("--cams", nargs="+", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=6)
    a = ap.parse_args(argv)
    audit_part(a.part, a.ref, a.cams, a.n_trials)


if __name__ == "__main__":
    main()
