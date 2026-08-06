"""Re-detect all reps of a participant after a recut (picks up the new cam clips) -- SINGLE PROCESS.

detect_rep_batched works per-rep. Spawning a subprocess PER rep pays the model-load + CUDA-init
(~6-7s) every time (~8s/rep). This loads the cup+pose models ONCE and loops _run_rep over every
rep in-process -> ~1-2 min for 80 reps instead of ~11. Byte-identical output (same _run_rep + payloads).

    python scripts/redetect_part.py --part P12 [--only-with-cam 4]
Watch:  tail -f out/recut/redetect_P12.log
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import detect_rep_batched as D
from detect_rep_batched import _run_rep, cup_payload, pose_payload
import gpu_decode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--only-with-cam", type=int, default=None,
                    help="only reps that have this cam clip (e.g. 4) -- skips reps missing the recut cam")
    ap.add_argument("--cams", nargs="*", type=int, default=None,
                    help="ONLY detect these cams per rep (e.g. 4 5). Skips re-decoding unchanged good "
                         "cams whose dets are already valid -- decode is the bottleneck (full-res raw "
                         "frames piped to CPU), so 2/5 cams = ~2.5x faster. Omit = all cams.")
    ap.add_argument("--cup-model", default="models/cup_clean3d_refill.pt")
    ap.add_argument("--pose-model", default="models/yolo26s-pose.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=0)
    a = ap.parse_args(argv)

    clipsdir = ROOT / "cache" / "delta" / a.part / "work" / "clips"
    dets = ROOT / "cache" / "delta" / a.part / "work" / "dets"
    dets.mkdir(parents=True, exist_ok=True)
    reps = sorted({m.group(1) for p in clipsdir.glob("*.mp4")
                   if (m := re.match(rf"(delta_{a.part}_\S+?)\.\d+\.mp4$", p.name))})
    if a.only_with_cam is not None:
        reps = [r for r in reps if (clipsdir / f"{r}.{a.only_with_cam}.mp4").exists()]
    print(f"{a.part}: {len(reps)} reps -> {dets}  nvdec={gpu_decode.gpu_available()}", flush=True)

    # load models ONCE
    from ultralytics import YOLO
    cup_model = YOLO(str(ROOT / a.cup_model)); cup_model.to(f"cuda:{a.device}")
    pose_model = YOLO(str(ROOT / a.pose_model)); pose_model.to(f"cuda:{a.device}")
    print("models loaded", flush=True)

    t0 = time.time(); ok = fail = 0
    for i, rep in enumerate(reps):
        clips = {}
        for mp4 in sorted(clipsdir.glob(f"{rep}.*.mp4")):
            m = re.match(rf"{re.escape(rep)}\.(\d+)\.mp4$", mp4.name)
            if m and (a.cams is None or int(m.group(1)) in a.cams):
                clips[int(m.group(1))] = mp4
        if not clips:
            fail += 1; continue
        try:
            cup_dense, pose_dense = _run_rep(cup_model, pose_model, clips, a.batch, a.device)
            for cam, clip in sorted(clips.items()):
                (dets / f"{clip.stem}.cup.json").write_text(
                    json.dumps(cup_payload(clip, cup_dense[cam], str(ROOT / a.cup_model))))
                (dets / f"{clip.stem}.pose.json").write_text(
                    json.dumps(pose_payload(clip, pose_dense[cam], str(ROOT / a.pose_model))))
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {rep}: {type(e).__name__}: {str(e)[-160:]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(reps)}] {time.time()-t0:.0f}s  ok {ok} fail {fail}", flush=True)

    print(f"\nPROCESSING CHECK: {ok} reps detected, {fail} failed ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
