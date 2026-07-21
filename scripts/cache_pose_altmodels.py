"""Run RTMPose or BlazePose per-camera on the DELTA clips, write our .pose.json format so the EXISTING
compare_pose_omc_delta harness triangulates + scores them exactly like YOLO. Accuracy question:
does an alt pose model beat YOLO on jitter / OMC speed-corr / reproduction?

Both are top-down/single-person: RTMPose needs a person bbox (use full frame -- one subject per clip);
BlazePose finds the person itself. Keypoints mapped to our COCO-11 joint names (full coverage, verified).

Cache: cache/delta/<P>/dets_<model>/delta_<P>_<trial>.<cam>.pose.json  (same schema as yolo pose json)

    python scripts/cache_pose_altmodels.py --model rtmpose --parts P07 P08 P13
    python scripts/cache_pose_altmodels.py --model blazepose --parts P07 P08 P13
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
DELTA = ROOT / "cache" / "delta"
RTM_CK = Path.home() / ".cache/rtmlib/hub/checkpoints"
SP = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
          "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad")

OURS = ["nose", "left_eye", "left_ear", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"]
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
BLAZE = {0: "nose", 2: "left_eye", 7: "left_ear", 5: "right_eye", 8: "right_ear",
         11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
         15: "left_wrist", 16: "right_wrist", 23: "left_hip", 24: "right_hip"}


class RTM:
    def __init__(self, use_det=False):
        from rtmlib import RTMPose
        ck = sorted(glob.glob(str(RTM_CK / "rtmpose-*.onnx")))[0]
        self.m = RTMPose(onnx_model=ck, model_input_size=(288, 384),
                         backend="onnxruntime", device="cpu")
        self.det = None
        if use_det:
            from rtmlib import YOLOX
            dck = sorted(glob.glob(str(RTM_CK / "yolox*.onnx")))[0]
            self.det = YOLOX(onnx_model=dck, model_input_size=(640, 640),
                             backend="onnxruntime", device="cpu")

    def __call__(self, bgr, box=None):
        h, w = bgr.shape[:2]
        if box is not None:
            bb = [list(box)]                           # YOLO-keypoint-derived person box
        elif self.det is not None:
            boxes = self.det(bgr)
            bb = [list(boxes[0])] if len(boxes) else [[0, 0, w, h]]
        else:
            bb = [[0, 0, w, h]]                        # one person = full frame
        kp, sc = self.m(bgr, bboxes=bb)
        kp = np.asarray(kp)[0]; sc = np.asarray(sc)[0]   # (17,2),(17,)
        out = {}
        for i, name in enumerate(COCO17):
            if name in OURS:
                out[name] = [float(kp[i][0]), float(kp[i][1]), float(sc[i])]
        return out


class Blaze:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        task = SP / "blazepose" / "pose_landmarker_full.task"
        opts = vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(task)),
            running_mode=vision.RunningMode.IMAGE)
        self.lm = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect(img)
        out = {}
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            for idx, name in BLAZE.items():
                if name in OURS and idx < len(lms):
                    l = lms[idx]
                    out[name] = [float(l.x * w), float(l.y * h), float(l.visibility)]
        return out


def _yolo_boxes(clip: Path):
    """Per-frame person box derived from the CACHED YOLO keypoints (the already-detected clips):
    min/max of the 11 joints + 15% margin. Reuses YOLO's person, no re-detection. Returns {frame:[x,y,w,h]}."""
    yj = clip.parent.parent / "dets" / clip.name.replace(".mp4", ".pose.json")
    if not yj.exists():
        return {}
    d = json.loads(yj.read_text())
    out = {}
    for fr in d["frames"]:
        pts = [v[:2] for v in fr.get("kps", {}).values() if v and v[2] > 0.3]
        if len(pts) < 3:
            continue
        p = np.array(pts)
        x0, y0 = p.min(0); x1, y1 = p.max(0)
        mx, my = (x1 - x0) * 0.15 + 20, (y1 - y0) * 0.15 + 20
        out[fr["frame"]] = [x0 - mx, y0 - my, (x1 - x0) + 2 * mx, (y1 - y0) + 2 * my]
    return out


def cache_clip(model, clip: Path, out: Path, yolo_box=False):
    if out.exists():
        return "cached"
    boxes = _yolo_boxes(clip) if yolo_box else {}
    cap = cv2.VideoCapture(str(clip))
    frames = []
    fi = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        kps = model(im, box=boxes.get(fi)) if yolo_box else model(im)
        frames.append({"frame": fi, "box_conf": 1.0, "kps": kps})
        fi += 1
    cap.release()
    out.write_text(json.dumps({"clip": clip.name, "model": out.parent.name,
                               "n_frames": len(frames), "frames": frames}))
    return f"{len(frames)}fr"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["rtmpose", "blazepose"], required=True)
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P13"])
    ap.add_argument("--det", action="store_true", help="rtmpose: use YOLOX person detector (proper "
                    "top-down) instead of full-frame bbox")
    ap.add_argument("--suffix", default="", help="dets dir suffix, e.g. _det")
    a = ap.parse_args(argv)
    model = RTM(use_det=a.det) if a.model == "rtmpose" else Blaze()
    print(f"[{a.model}] loaded", flush=True)
    clips = []
    for p in a.parts:
        clips += sorted(glob.glob(str(DELTA / p / "staged" / "*.mp4")))
    print(f"{len(clips)} clips to process", flush=True)
    t0 = time.time()
    for i, c in enumerate(clips):
        cp = Path(c)
        part = cp.name.split("_")[1]
        outdir = DELTA / part / f"dets_{a.model}{a.suffix}"
        outdir.mkdir(exist_ok=True)
        out = outdir / cp.name.replace(".mp4", ".pose.json")
        r = cache_clip(model, cp, out)
        el = time.time() - t0
        eta = el / (i + 1) * (len(clips) - i - 1)
        print(f"[{i+1}/{len(clips)}] {cp.name}: {r}  (elapsed {el:.0f}s, ETA {eta:.0f}s)", flush=True)
    print(f"DONE {a.model} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
