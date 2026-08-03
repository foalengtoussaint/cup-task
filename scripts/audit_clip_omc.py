"""Per-trial audit: is the VIDEO cut correctly, and does its OMC c3d MAP to the same time window?

Three lengths must agree for a trial to be usable:
  n_video : staged clip frame count (cam1)
  n_det   : detection json n_frames (must == n_video; else dets were run on a different clip)
  n_omc   : c3d samples resampled to video fps (should be ~= n_video if the c3d is the CUT trial)

Flags:
  UNCUT      : n_video > 800  (a ~8s drink is ~500 frames @60fps; thousands = uncut recording)
  TINY       : n_video < 120  (broken/empty clip, e.g. the 1-frame P252 one)
  DET_MISMATCH : n_det != n_video (detections ran on a different clip than the staged video)
  OMC_MISMATCH : |n_omc - n_video| / n_video > 0.25  (c3d is a different-length window ->
                 the OMC-to-camera time mapping is BROKEN for this trial; validation would be garbage)

Prints a per-participant tally + the flagged trials. No GPU. This is the gate BEFORE reproject-seed
and before any OMC comparison.

    python scripts/audit_clip_omc.py --parts P10 P12 P13 P14 P251 P252
"""
from __future__ import annotations
import sys, argparse, json, glob, re
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H

C3D_RATE = 100.0
FPS = H.VIDEO_FPS


def _c3d_len_video_frames(part, trial):
    """c3d sample count resampled to video fps, or None if no c3d."""
    import ezc3d
    f = H.DELTA / part / "c3d" / f"{trial}.c3d"
    if not f.exists():
        return None
    try:
        c = ezc3d.c3d(str(f))
        n = c["data"]["points"].shape[2]
        rate = c["parameters"]["POINT"]["RATE"]["value"][0]
        return int(round(n / rate * FPS))
    except Exception:
        return None


def audit_trial(part, trial):
    sd = H.DELTA / part / "staged"
    v = sd / f"delta_{part}_{trial}.1.mp4"
    if not v.exists():
        return None
    cap = cv2.VideoCapture(str(v)); nv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    dj = H.DELTA / part / "dets" / f"delta_{part}_{trial}.1.cup.json"
    nd = json.loads(dj.read_text()).get("n_frames") if dj.exists() else None
    no = _c3d_len_video_frames(part, trial)
    flags = []
    if nv > 800: flags.append("UNCUT")
    if nv < 120: flags.append("TINY")
    if nd is not None and nd != nv: flags.append("DET_MISMATCH")
    if no is not None and nv > 0 and abs(no - nv) / nv > 0.25: flags.append("OMC_MISMATCH")
    if no is None: flags.append("NO_OMC")
    return dict(n_video=nv, n_det=nd, n_omc=no, flags=flags, ok=(len(flags) == 0))


def _trials(part):
    ids = set()
    for f in glob.glob(str(H.DELTA / part / "staged" / f"delta_{part}_*.1.mp4")):
        m = re.match(rf"delta_{part}_(.+)\.1\.mp4$", Path(f).name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P10", "P12", "P13", "P14", "P251", "P252"])
    a = ap.parse_args(argv)
    H.use_good_cams()

    print(f"{'part':6s} {'trials':>6s} {'ok':>5s} {'UNCUT':>6s} {'TINY':>5s} "
          f"{'DET_MM':>7s} {'OMC_MM':>7s} {'NO_OMC':>7s}", flush=True)
    allflag = []
    for p in a.parts:
        rows = []
        for t in _trials(p):
            r = audit_trial(p, t)
            if r is None:
                continue
            rows.append((t, r))
            if not r["ok"]:
                allflag.append((p, t, r))
        if not rows:
            print(f"{p:6s}  (no trials)"); continue
        def cnt(flag): return sum(1 for _, r in rows if flag in r["flags"])
        print(f"{p:6s} {len(rows):>6d} {sum(1 for _,r in rows if r['ok']):>5d} "
              f"{cnt('UNCUT'):>6d} {cnt('TINY'):>5d} {cnt('DET_MISMATCH'):>7d} "
              f"{cnt('OMC_MISMATCH'):>7d} {cnt('NO_OMC'):>7d}", flush=True)

    print(f"\n=== {len(allflag)} flagged trials ===", flush=True)
    for p, t, r in allflag[:80]:
        print(f"  {p} {t:30s} nv={r['n_video']} nd={r['n_det']} nomc={r['n_omc']}  {','.join(r['flags'])}",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
