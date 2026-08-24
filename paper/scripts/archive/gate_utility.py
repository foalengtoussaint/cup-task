"""Is the reprojection gate dropping BAD detections (useful, interpolatable) or GOOD detections whose
cameras disagree from miscalibration (harmful)? Per (participant, trial, arm joint, frame): does the
gate fire (<MIN_CAMS within REPROJ_PX), what's the 2D CONFIDENCE of the dropped detections, and are the
drops SPORADIC (isolated frames -> interpolatable) or SYSTEMATIC (long runs / whole joint)?

If gate-fired frames have LOW conf + sporadic -> gate catches bad detections (useful).
If gate-fired frames have HIGH conf + systematic -> gate drops good detections (miscalib, harmful).
Compare a clean participant (P07) vs the miscalibrated one (P19).
    python paper/scripts/gate_utility.py
"""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
from pipeline import triangulate as TR


def _percam(part, trial):
    d = H.DELTA / part; cams = H._load_calib_mm(part); pc = {}
    for pj in sorted(glob.glob(str(d / H.DETS_SUBDIR / f"*{trial}*.pose.json"))):
        cam = Path(pj).name.split(".")[1]; pc[f"cam_{cam}"] = json.loads(Path(pj).read_text())["frames"]
    if H.GOOD_CAMS and part in H.GOOD_CAMS:
        keep = H.GOOD_CAMS[part]; pc = {c: v for c, v in pc.items() if c in keep}; cams = {c: v for c, v in cams.items() if c in keep}
    return pc, cams


def _kp_conf(fr, joint):
    kps = fr.get("kps", {}); v = kps.get(joint)
    return float(v[2]) if v is not None and len(v) > 2 else np.nan


def _runs(mask):
    """max consecutive-True run length in a bool array (systematic-ness)."""
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0; best = max(best, cur)
    return best


def analyze(part, ntrials=12):
    H.use_good_cams()
    trials = GT.load_clean(parts=[part])[:ntrials]
    fail_conf, pass_conf = [], []
    fail_frac_per_jt, maxrun_per_jt = [], []
    for t in trials:
        side = t["side"]
        per_cam, cams = _percam(part, t["trial"]); n = max(len(v) for v in per_cam.values())
        for joint in [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]:
            pf = H._kp_point(joint); firemask = []
            for f in range(n):
                uc, up, cf = [], [], []
                for ck, frames in per_cam.items():
                    if f < len(frames):
                        p = pf(frames[f])
                        if p is not None:
                            uc.append(cams[ck]); up.append(p); cf.append(_kp_conf(frames[f], joint))
                if len(uc) < 2:
                    firemask.append(False); continue
                X = TR.triangulate_dlt(uc, up)
                errs = [np.linalg.norm(TR.project(c, X)[0] - p) for c, p in zip(uc, up)]
                fired = sum(e <= TR.REPROJ_PX for e in errs) < TR.MIN_CAMS   # gate would DROP this frame
                firemask.append(fired)
                (fail_conf if fired else pass_conf).extend([c for c in cf if np.isfinite(c)])
            firemask = np.array(firemask)
            if firemask.size:
                fail_frac_per_jt.append(firemask.mean()); maxrun_per_jt.append(_runs(firemask) / max(firemask.size, 1))
    fc = np.array(fail_conf); pc = np.array(pass_conf)
    ff = np.array(fail_frac_per_jt); mr = np.array(maxrun_per_jt)
    print(f"\n=== {part} (arm joints, {len(trials)} trials) ===")
    print(f"  gate-DROP frames: 2D conf median {np.median(fc) if len(fc) else float('nan'):.2f}  (n={len(fc)})")
    print(f"  gate-KEEP frames: 2D conf median {np.median(pc) if len(pc) else float('nan'):.2f}  (n={len(pc)})")
    print(f"  per joint-trial: gate-drop frac median {np.median(ff):.2f}  |  frac of joint-trials with >50% dropped: {(ff>0.5).mean():.2f}")
    print(f"  SYSTEMATIC-ness: median longest consecutive-drop run / trial length = {np.median(mr):.2f}")
    print(f"    (high conf on dropped + high systematic run => gate drops GOOD detections, not bad ones)")


def main():
    for p in ["P07", "P19"]:
        analyze(p)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
