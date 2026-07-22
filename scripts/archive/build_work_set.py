"""Assemble a correctly-timed 10L+10R x 5-cam WORK SET per participant.

The 5-good-camera cohort (all of cams 1-5 geometrically sound): P07, P08, P15, P12, P13.
For P12/P13 some cams are MIS-CUT (clip = wrong repetition) -> re-cut from the uncut; the rest
come straight from the study's cut clips. Output is self-contained in cache/delta/<P>/work/:
  clips/ delta_<P>_<trial>.<cam>.mp4   (correctly-timed, all 5 cams)
  dets/  delta_<P>_<trial>.<cam>.{pose,cup}.json
so downstream (cup labels, pose validation) reads one clean dir. NO deletes of originals.

Trial pick: first N _L_ and first N _R_ by trial number.

    python scripts/build_work_set.py --part P13 --n 10
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from delta_recut import SHARE, CT
from recut_from_audit import recut

# per participant: cams sourced from the study CUT clips (good geometry, correctly placed)
# and cams that must be RE-CUT (mis-cut) with their reference cam.
COHORT = {
    "P07": {"good": [1, 2, 3, 4, 5], "recut": {}},
    "P08": {"good": [1, 2, 3, 4, 5], "recut": {}},
    "P15": {"good": [1, 2, 3, 4, 5], "recut": {}},
    "P12": {"good": [1, 2, 3],       "recut": {4: 3, 5: 3}},
    "P13": {"good": [1, 3, 4, 5],    "recut": {2: 3}},
}
POSE_MODEL = "models/yolo26s-pose.pt"
CUP_MODEL = "models/cup_clean3d_refill.pt"


def _tnum(stem):
    m = re.search(r"trial_(\d+)", stem)
    return int(m.group(1)) if m else 1 << 30


def pick_trials(part, n):
    vd = Path(SHARE) / part / "01_Measurement" / "04_Video" / "03_Cut" / "drinking"
    cdir = next((vd / f"cam{c}" for c in range(1, 11) if (vd / f"cam{c}").is_dir()), None)
    stems = sorted((p.stem for p in cdir.glob("*.mp4")), key=_tnum)
    L = [s for s in stems if "_L_" in s][:n]
    R = [s for s in stems if "_R_" in s][:n]
    return L + R


def build(part, n, redetect):
    cfg = COHORT[part]
    trials = pick_trials(part, n)
    work = CT / "cache" / "delta" / part / "work"
    clips = work / "clips"; dets = work / "dets"
    clips.mkdir(parents=True, exist_ok=True); dets.mkdir(parents=True, exist_ok=True)
    vd = Path(SHARE) / part / "01_Measurement" / "04_Video" / "03_Cut" / "drinking"
    print(f"{part}: {len(trials)} trials ({sum('_L_' in t for t in trials)}L/"
          f"{sum('_R_' in t for t in trials)}R), good={cfg['good']} recut={cfg['recut']}", flush=True)

    # 1. good cams: stage study cut clips, SCALED to the calibration resolution 1920x1080.
    # (Plain copy is WRONG for the 10-cam rigs whose cam1 records 1280x720 -> detections land in
    # 720p coords and reproject ~350px against the 1080p calib. The re-cut path already scales.)
    for t in trials:
        for c in cfg["good"]:
            src = vd / f"cam{c}" / f"{t}.mp4"
            dst = clips / f"delta_{part}_{t}.{c}.mp4"
            if src.exists() and not dst.exists():
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf",
                                "scale=1920:1080", "-c:v", "libx264", "-crf", "18",
                                "-preset", "veryfast", "-an", str(dst)], check=True)
    print(f"  staged good-cam clips (scaled 1920x1080)", flush=True)

    # 2. recut cams: re-cut from uncut into work/clips
    for cam, ref in cfg["recut"].items():
        done = recut(part, cam, ref, trials, search=2.0)
        for t, dst in done:
            tgt = clips / f"delta_{part}_{t}.{cam}.mp4"
            if dst.resolve() != tgt.resolve():
                shutil.copy(dst, tgt)
    print(f"  recut cams done", flush=True)

    # 3. detect all cams per trial (one batched call per trial does all 5)
    for i, t in enumerate(trials):
        have = glob.glob(str(dets / f"delta_{part}_{t}.*.pose.json"))
        if have and not redetect:
            continue
        subprocess.run([sys.executable, "scripts/detect_rep_batched.py", str(clips),
                        f"delta_{part}_{t}", "-o", str(dets), "--pose-model", POSE_MODEL,
                        "--cup-model", CUP_MODEL, "--batch", "32", "--device", "0"],
                       check=True, capture_output=True)
        print(f"  detected {i+1}/{len(trials)}: {t}", flush=True)
    # 4. MANDATORY reprojection gate: validate every camera-trial vs leave-it-out consensus,
    #    QUARANTINE (not delete) any >GATE px. Never trust align-q; reproj is the truth.
    reproj_gate(part)
    ncam = len(cfg["good"]) + len(cfg["recut"])
    npose = len(glob.glob(str(dets / "*.pose.json")))
    print(f"{part}: WORK SET READY -- {npose} pose jsons (~{len(trials)}x{ncam}) in {dets}",
          flush=True)


GATE_PX = 20.0


def reproj_gate(part, gate_px=GATE_PX):
    """Per (camera, trial): median wrist reproj vs the leave-that-cam-out >=3-cam RANSAC consensus.
    >gate_px => the clip is mis-cut/desynced/miscalibrated -> QUARANTINE its clip+dets into
    work/rejected/ (kept, not deleted) so it can't poison labels/consensus. Writes gate_report.json.
    This is the check I should have run on every re-cut from the start."""
    import numpy as np
    import compare_pose_omc_delta as _C
    from cup_task.kalman_3d import project as _project
    from reaudit_cam_quality import ransac_point as _ransac
    work = CT / "cache" / "delta" / part / "work"
    dets = work / "dets"; clips = work / "clips"; rej = work / "rejected"
    rej.mkdir(exist_ok=True)
    # RESTORE any previously-quarantined files first, so re-gating at a new threshold re-examines
    # everything (idempotent). .mp4 -> clips, .json -> dets.
    for f in list(rej.glob("*")):
        (clips if f.suffix == ".mp4" else dets).joinpath(f.name).write_bytes(f.read_bytes())
        f.unlink()
    cams = _C._load_calib_mm(part)
    trials = sorted({Path(p).name.split(".")[0].replace(f"delta_{part}_", "")
                     for p in glob.glob(str(dets / "*.pose.json"))})
    report = {}
    n_drop = 0
    for t in trials:
        side = "right" if "_R_" in t else "left"
        fn = _C._kp_point(f"{side}_wrist")
        per = {}
        for pj in glob.glob(str(dets / f"delta_{part}_{t}.*.pose.json")):
            c = "cam_" + Path(pj).name.split(".")[1]
            if c in cams:
                per[c] = json.loads(Path(pj).read_text())["frames"]
        n = min((len(v) for v in per.values()), default=0)
        for c in list(per):
            refks = [k for k in per if k != c]
            if len(refks) < 2:
                continue
            errs = []
            for f in range(n):
                pts = {k: fn(per[k][f]) for k in refks if fn(per[k][f]) is not None}
                if len(pts) < 2:
                    continue
                X, _ = _ransac({k: cams[k] for k in pts if k in cams}, pts)
                if X is None:
                    continue
                p = fn(per[c][f])
                if p is not None:
                    errs.append(float(np.hypot(*(_project(cams[c], X)[0] - p))))
            if not errs:
                continue
            med = float(np.median(errs))
            report.setdefault(t, {})[c] = round(med, 1)
            if med > gate_px:
                kc = c.split("_")[1]
                for suf in (f".{kc}.mp4", f".{kc}.pose.json", f".{kc}.cup.json"):
                    src = (clips if suf.endswith("mp4") else dets) / f"delta_{part}_{t}{suf}"
                    if src.exists():
                        src.rename(rej / src.name)
                n_drop += 1
                print(f"  GATE DROP {t} {c}: {med:.0f}px > {gate_px}", flush=True)
    (work / "gate_report.json").write_text(json.dumps(report, indent=1))
    print(f"  reproj gate: dropped {n_drop} cam-trials (>{gate_px}px), report -> gate_report.json",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(COHORT))
    ap.add_argument("--n", type=int, default=10, help="trials per hand (L and R)")
    ap.add_argument("--redetect", action="store_true")
    ap.add_argument("--gate-only", action="store_true",
                    help="skip build; just run the reprojection gate on the existing work set")
    ap.add_argument("--gate-px", type=float, default=GATE_PX,
                    help="quarantine a camera-trial if median wrist reproj > this. 20 is too strict "
                         "(drops merely-NOISY cams like the 2.5Mbps cam5); use ~30-35 to drop only "
                         "BROKEN cams (desync/miscalib) and let per-frame RANSAC handle noise.")
    a = ap.parse_args(argv)
    if a.gate_only:
        reproj_gate(a.part, a.gate_px)
    else:
        build(a.part, a.n, a.redetect)


if __name__ == "__main__":
    main()
