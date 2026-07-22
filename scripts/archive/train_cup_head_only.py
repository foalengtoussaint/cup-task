"""Train ONLY the cup detection onto a frozen pose model — zero keypoint cost.

The insight: keypoints are regressed from backbone features by the pose head's
keypoint branch (cv4). If BOTH the backbone/neck (modules 0-22) AND cv4 are
frozen, the keypoint output is mathematically identical to the base
yolo11n-pose, no matter what else trains. We leave only the box+class branch
(cv2/cv3/dfl) trainable so the head learns to fire on the new `cup` class (and
re-confirm person) without disturbing the keypoints at all.

This is the fix for the observed problem: full fine-tuning DEGRADED the limbs
(~20px) because it moved the backbone; freezing prevents that by construction.

    python scripts/train_cup_head_only.py --data data/merged_person_cup/data.yaml \
        --epochs 60 --name cup_head_frozen
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/merged_person_cup/data.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=300)  # ceiling; patience stops early
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="cup_head_frozen")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop if val mAP doesn't improve for this many epochs")
    ap.add_argument("--device", default=0)
    args = ap.parse_args(argv)

    model = YOLO(args.model)

    # Freeze the keypoint branch (cv4) of the Pose head IN ADDITION to the
    # backbone/neck that --freeze handles. A callback re-asserts requires_grad
    # = False on every cv4 param at train start, so gradients never touch it.
    def freeze_kpt_branch(trainer):
        pose_head = trainer.model.model[-1]          # the Pose module
        frozen = 0
        for name, p in pose_head.named_parameters():
            if name.startswith("cv4"):               # keypoint branch only
                p.requires_grad_(False)
                frozen += 1
        print(f"[freeze] locked {frozen} keypoint-branch (cv4) params "
              f"-> keypoints stay identical to base", flush=True)

    model.add_callback("on_train_start", freeze_kpt_branch)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs",
        freeze=23,          # freeze modules 0..22 (backbone + neck)
        patience=args.patience,
        plots=True,
        verbose=True,
    )
    print(f"done -> runs/pose/{args.name}/weights/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
