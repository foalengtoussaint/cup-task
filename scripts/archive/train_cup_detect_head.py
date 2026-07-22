"""Train a standalone 1-class cup detector whose backbone == the pose model's.

The cup detector is a normal YOLO detection model (own Detect head, native
Ultralytics loss). Its backbone/neck weights are transplanted from
yolo11n-pose.pt and FROZEN, so they are byte-identical to what the pose model
uses. That lets us later fuse the two: one shared backbone pass -> pose head
(person+keypoints, unchanged) + this cup head. Only the cup Detect head trains.

Dataset: a 1-class cup detection view of the merged data (we reuse the cup boxes;
person rows are ignored by a cup-only data.yaml).

    python scripts/train_cup_detect_head.py --data data/cup_only/data.yaml \
        --epochs 300 --patience 5 --name cup_detect_head
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _transplant_backbone(detect_model, pose_model):
    """Copy backbone/neck (modules 0..N-2) weights from pose into detect.

    Both are yolo11n with identical backbone+neck; only the head (last module)
    differs (Detect vs Pose). We copy every matching-shape param outside the
    head so the shared features are identical.
    """
    dsd = detect_model.model.state_dict()
    psd = pose_model.model.state_dict()
    n_head = len(detect_model.model.model) - 1     # last index = head
    head_prefix = f"model.{n_head}."
    copied = 0
    for k, v in psd.items():
        if k.startswith(head_prefix):
            continue                                # skip head (Pose != Detect)
        if k in dsd and dsd[k].shape == v.shape:
            dsd[k] = v.clone(); copied += 1
    detect_model.model.load_state_dict(dsd, strict=False)
    return copied


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/cup_only/data.yaml")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--detect-model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="cup_detect_head")
    ap.add_argument("--device", default=0)
    args = ap.parse_args(argv)

    pose = YOLO(args.pose_model)
    det = YOLO(args.detect_model)

    n = _transplant_backbone(det, pose)
    print(f"[arch] transplanted {n} backbone/neck params from pose -> detect "
          f"(shared features)", flush=True)

    # Freeze the transplanted backbone/neck (modules 0..22); only the Detect
    # head trains. This keeps the shared features byte-identical to the pose
    # model's so the two heads can be fused on one backbone pass.
    model = det
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, name=args.name, project="runs",
        freeze=23, patience=args.patience, plots=True, verbose=True,
    )
    print(f"done -> runs/detect/{args.name}/weights/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
