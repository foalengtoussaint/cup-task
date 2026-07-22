"""Fine-tune yolo11n-pose into a 2-class person+cup pose model.

Starts from COCO pose weights (so the person/keypoint knowledge transfers) and
adds the cup as class 1. The head is reshaped for nc=2; keypoints stay 17.

    python scripts/train_merged.py --data data/merged_person_cup/data.yaml \
        --epochs 60 --name merged_person_cup
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default="data/merged_person_cup/data.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="merged_person_cup")
    ap.add_argument("--device", default=0)
    args = ap.parse_args(argv)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs",
        patience=20,
        # cup box vs person keypoints can compete; keep pose loss weighted so
        # the person keypoints don't degrade while learning the cup class.
        pose=12.0,
        plots=True,
        verbose=True,
    )
    print(f"done -> runs/{args.name}/weights/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
