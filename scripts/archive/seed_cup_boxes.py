"""Seed-only cup detection: for trials that have NO cached cup boxes, run the cup YOLO on only the
FIRST ~90 frames of each good camera's staged clip (the cup sits on the table at clip start, so a
consensus seed is available early) and write the standard <stem>.<cam>.cup.json. This is all
cache_uetrack_tracks.py needs -- UETrack is detect-ONCE, so it only wants a seed box per camera and
tracks the rest. Barely slower than tracking alone (a fraction of a clip of detection, not the whole).

Idempotent: skips a (trial,cam) whose .cup.json already exists. One model load for the whole run.

    python scripts/seed_cup_boxes.py --parts P07 P08            # then re-run cache_uetrack_tracks.py
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import gnn_train as GT
import results_v3_delta as R
from pipeline.cup_detect import DEFAULT_MODEL, DEFAULT_IMGSZ, MIN_CONF, _pick_cup, CUP_CLASS

DETS = ROOT / "cache" / "delta"
MAX_SEED_FR = 90                      # detect only this many frames per clip (enough for a seed)


def _staged_dir(part):
    d = DETS / part / "staged"
    return d if d.is_dir() else DETS / part / "recut"


def _cam_num(cam):
    return cam.split("_")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--max-frames", type=int, default=MAX_SEED_FR)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    H.use_good_cams()
    from ultralytics import YOLO
    model = YOLO(DEFAULT_MODEL)
    print(f"model {DEFAULT_MODEL} loaded; seed <= {args.max_frames} fr/clip", flush=True)

    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in args.parts]
    t0 = time.time(); n_clip = 0; n_seed = 0; n_skip = 0; n_notrack = 0
    for t in trials:
        part, trial = t["part"], t["trial"]
        # skip trials that already have a full cup track cached
        if (R.TRACKS / f"{part}__{trial}__uetrack__fs1.json").exists():
            continue
        good = sorted(H.GOOD_CAMS.get(part, []))
        sd = _staged_dir(part)
        stem = f"delta_{part}_{trial}"
        for cam in good:
            outp = DETS / part / "dets" / f"{stem}.{_cam_num(cam)}.cup.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            if outp.exists():
                n_skip += 1; continue
            clip = sd / f"{stem}.{_cam_num(cam)}.mp4"
            if not clip.exists():
                n_notrack += 1; continue
            frames = []
            res = model.predict(source=str(clip), stream=True, device=args.device,
                                imgsz=DEFAULT_IMGSZ, conf=MIN_CONF, verbose=False)
            for i, r in enumerate(res):
                picked = _pick_cup(r)
                if picked is None:
                    frames.append({"frame": i, "conf": 0.0, "box": None, "center": None})
                else:
                    box, conf = picked
                    frames.append({"frame": i, "conf": conf, "box": box,
                                   "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})
                if i + 1 >= args.max_frames:
                    break
            res.close() if hasattr(res, "close") else None
            outp.write_text(json.dumps({"clip": str(clip), "model": DEFAULT_MODEL,
                                        "cup_class": CUP_CLASS, "min_conf": MIN_CONF,
                                        "n_frames": len(frames), "frames": frames}))
            n_clip += 1
            if sum(1 for f in frames if f["box"]) > 0:
                n_seed += 1
        if (n_clip) and n_clip % 40 == 0:
            print(f"  {n_clip} clips seeded ({n_seed} with a box), {time.time()-t0:4.0f}s", flush=True)

    print(f"\nPROCESSING CHECK: clips detected {n_clip}, with>=1 box {n_seed}, "
          f"already-had {n_skip}, no-clip {n_notrack}, {time.time()-t0:.0f}s", flush=True)
    print("DONE -- now run: python scripts/cache_uetrack_tracks.py --parts " + " ".join(args.parts),
          flush=True)


if __name__ == "__main__":
    main()
