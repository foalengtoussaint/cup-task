"""Convert a YOLO-seg cup dataset split (train.txt/val.txt image lists) to RF-DETR COCO format.

RF-DETR (roboflow) trains on a COCO dir:
    <out>/train/*.jpg + _annotations.coco.json
    <out>/valid/*.jpg + _annotations.coco.json
Our labels are single-class ("cup") YOLO-seg polygons; we take each polygon's axis-aligned bbox
(the label is already a square, so bbox == the square). One category id=1 "cup" (COCO is 1-indexed;
id 0 reserved). Images are symlinked, not copied.

    python scripts/yolo_to_coco_cup.py --pool data/delta_cup_mix4_fix_P07 \
        --train-list onetrial_train.txt --val-list onetrial_val.txt --out data/rfdetr_P07
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def load_list(pool, name):
    return [Path(p) for p in (pool / name).read_text().split() if p.strip()]


def yolo_seg_to_bbox(txt, W, H):
    """YOLO-seg line '0 x1 y1 x2 y2 ...' (normalized) -> COCO bbox [x,y,w,h] in pixels."""
    v = txt.split()
    if len(v) < 5:
        return None
    xs = [float(v[i]) * W for i in range(1, len(v), 2)]
    ys = [float(v[i]) * H for i in range(2, len(v), 2)]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def build_split(imgs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "cup", "supercategory": "none"}]}
    ann_id = 1
    for img_id, ip in enumerate(imgs, 1):
        ip = ip.resolve()
        W, H = Image.open(ip).size
        link = out_dir / ip.name
        if not link.exists():
            link.symlink_to(ip)
        coco["images"].append({"id": img_id, "file_name": ip.name, "width": W, "height": H})
        lp = ip.parent.parent / "labels" / (ip.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                bb = yolo_seg_to_bbox(line, W, H)
                if bb and bb[2] > 0 and bb[3] > 0:
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": 1,
                        "bbox": [round(x, 2) for x in bb], "area": round(bb[2] * bb[3], 2),
                        "iscrowd": 0, "segmentation": []})
                    ann_id += 1
    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    return len(coco["images"]), len(coco["annotations"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--train-list", default="onetrial_train.txt")
    ap.add_argument("--val-list", default="onetrial_val.txt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pool = Path(a.pool)
    out = Path(a.out)
    ni, na = build_split(load_list(pool, a.train_list), out / "train")
    print(f"train: {ni} imgs / {na} anns -> {out/'train'}", flush=True)
    ni, na = build_split(load_list(pool, a.val_list), out / "valid")
    print(f"valid: {ni} imgs / {na} anns -> {out/'valid'}", flush=True)


if __name__ == "__main__":
    main()
