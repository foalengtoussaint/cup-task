"""Fine-tune yolo26s-seg into the DELTA cup detector on the reject-then-fill label pool.

imgsz=640 is NOT negotiable: cup-task ran YOLO at 1280 by mistake and cup recall went 64%@640 ->
8%@1280 -> 0%@1920 (upsampling past the training size is off-distribution). Single class (cup),
seg polygons. Start from COCO yolo26s-seg so generic object priors transfer.

    python scripts/train_cup_seg.py --data data/delta_cup_cohort/cup.yaml --epochs 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

BASE = "/home/imove/Documents/object_tracking/data/pretrained/yolo26s-seg.pt"


def main(argv=None):
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/delta_cup_cohort/cup.yaml")
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)   # MUST match inference size
    ap.add_argument("--batch", type=int, default=8)   # 7.6GB 3060 Ti OOMs at 16 for s-seg@640
    ap.add_argument("--name", default="cup_seg_delta_cohort")
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                name=a.name, project="runs/cup_seg", patience=20, close_mosaic=10, seed=0)
    print(f"done -> runs/cup_seg/{a.name}/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
