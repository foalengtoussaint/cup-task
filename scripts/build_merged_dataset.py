"""Build a 2-class pose dataset: person(17 kpts) + cup(box, 17 invisible slots).

Source is the existing cup-only dataset (segmentation-polygon labels, class 0 =
cup). We keep the cup box (converted polygon->bbox, relabeled class 1) and
AUTO-LABEL the person keypoints on each image with a pretrained YOLO-pose model
(distillation: the pose model's prediction is the person ground truth, since no
hand-labeled person keypoints exist).

YOLO-pose label format, one row per object:
    cls  cx cy w h  (x y v)*17          # normalized; v: 0=absent,1=occluded,2=vis
Cup rows carry 17 keypoint triplets all set to (0 0 0) — every class must share
kpt_shape, and the cup simply has no keypoints.

    python scripts/build_merged_dataset.py SRC_DATASET_DIR OUT_DIR [--pose-model ...]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PERSON_CLASS = 0
CUP_CLASS = 1
N_KPT = 17
MIN_KP_CONF = 0.30


def _poly_to_bbox(vals: list[float]) -> tuple[float, float, float, float]:
    """4-point (or N-point) normalized polygon -> (cx, cy, w, h) normalized."""
    xs = vals[0::2]
    ys = vals[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def _cup_rows(src_label: Path) -> list[str]:
    """Convert cup-only polygon labels -> class-1 box rows with 17 empty kpts."""
    if not src_label.exists():
        return []
    rows = []
    empty_kpt = " ".join(["0 0 0"] * N_KPT)
    for line in src_label.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        vals = [float(v) for v in parts[1:]]  # drop original class (all 0=cup)
        if len(vals) == 4:                    # already a box
            cx, cy, w, h = vals
        else:                                 # polygon -> bbox
            cx, cy, w, h = _poly_to_bbox(vals)
        rows.append(f"{CUP_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {empty_kpt}")
    return rows


def _person_rows(model, img_path: Path, W: int, H: int,
                 person_conf: float = 0.5) -> list[str]:
    """Run pose model -> ONE class-0 row (the subject) with 17 keypoints.

    The task has exactly one subject, but the pose model emits extra boxes per
    frame (a ~0.3 reflection/marker-cluster FP, and occasionally a second >0.5
    box splitting the mocap-suited body). Auto-labeling any of those would teach
    the merged model to hallucinate people, so we keep only the single
    highest-confidence person (if it clears person_conf).
    """
    r = model.predict(str(img_path), imgsz=1280, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    confs = r.boxes.conf.cpu().numpy()
    bi = int(confs.argmax())                 # top-1 person only
    if confs[bi] < person_conf:
        return []
    xywh = r.boxes.xywh[bi].cpu().numpy()     # pixel
    cx, cy, w, h = xywh / np.array([W, H, W, H])
    kxy = r.keypoints.xy[bi].cpu().numpy()
    kcf = r.keypoints.conf[bi].cpu().numpy()
    trip = []
    for j in range(N_KPT):
        x, y = kxy[j] / np.array([W, H])
        v = 2 if kcf[j] >= MIN_KP_CONF else 0
        if v == 0:
            x = y = 0.0
        trip.append(f"{x:.6f} {y:.6f} {v}")
    return [f"{PERSON_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(trip)]


def main(argv=None) -> int:
    import cv2
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="cup-only dataset dir (images/ labels/)")
    ap.add_argument("out", type=Path, help="output merged dataset dir")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--person-conf", type=float, default=0.5,
                    help="min person confidence to auto-label (drops FP reflections)")
    ap.add_argument("--limit", type=int, default=0, help="cap images (0=all) for a quick test")
    args = ap.parse_args(argv)

    model = YOLO(args.pose_model)
    for split in ("train", "val"):
        img_dir = args.src / "images" / split
        lbl_dir = args.src / "labels" / split
        out_img = args.out / "images" / split
        out_lbl = args.out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        imgs = sorted(img_dir.glob("*.jpg"))
        if args.limit:
            imgs = imgs[:args.limit]
        print(f"[{split}] {len(imgs)} images", flush=True)
        n_person = n_cup = 0
        for i, img in enumerate(imgs):
            H, W = cv2.imread(str(img)).shape[:2]
            prows = _person_rows(model, img, W, H, person_conf=args.person_conf)
            crows = _cup_rows(lbl_dir / f"{img.stem}.txt")
            n_person += len(prows); n_cup += len(crows)
            (out_lbl / f"{img.stem}.txt").write_text("\n".join(prows + crows) + "\n")
            # symlink image rather than copy (saves 3450x disk)
            dst = out_img / img.name
            if not dst.exists():
                dst.symlink_to(img.resolve())
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(imgs)}  (person rows {n_person}, cup {n_cup})", flush=True)
        print(f"[{split}] done: {n_person} person, {n_cup} cup objects", flush=True)

    # data.yaml — kpt_shape [17,3], flip_idx for the standard COCO L/R swap
    flip = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\n"
        "kpt_shape: [17, 3]\n"
        f"flip_idx: {flip}\n"
        "nc: 2\nnames:\n  0: person\n  1: cup\n"
    )
    print(f"wrote {args.out/'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
