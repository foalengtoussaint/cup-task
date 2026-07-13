"""Give the detector a DEEPER, dedicated classification head, train only it.

Motivation: the shared class branch (cv3) is what decides person-vs-cup-vs-
background. The frozen-backbone cup detector plateaus with false positives on
background clutter (shelf, etc.) because cv3 lacks the capacity to separate a
real cup from a cup-shaped blob. Here we DEEPEN cv3 (insert an extra conv block
per scale) so the class branch has more capacity, and train ONLY cv3 — the
backbone, box branch (cv2) and keypoint branch (cv4) stay frozen, so keypoints
remain bit-identical to base (Delta=0) and the person box is untouched.

This is the "new/deeper head" experiment: does extra class-branch capacity let
the model reject the background cup FPs?

    python scripts/train_deep_cls_head.py --data data/merged_person_cup/data.yaml \
        --epochs 300 --patience 5 --name cup_deep_cls
"""
from __future__ import annotations

import argparse
import copy

import torch.nn as nn


def _deepen_cv3(head):
    """Insert one extra Conv block into each scale's cv3, preserving shapes.

    cv3[i] is Sequential(block0, block1, final_1x1conv). We clone block1 (a
    channel-preserving Sequential(DWConv, Conv)) and insert it before the final
    1x1, so the class branch is one block deeper. New block starts from a copy
    of trained weights (warm start) rather than random.
    """
    added = 0
    for i in range(len(head.cv3)):
        seq = head.cv3[i]
        *blocks, final = list(seq.children())
        extra = copy.deepcopy(blocks[-1])          # clone the last conv block
        head.cv3[i] = nn.Sequential(*blocks, extra, final)
        added += 1
    return added


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/merged_person_cup/data.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="cup_deep_cls")
    ap.add_argument("--device", default=0)
    args = ap.parse_args(argv)

    model = YOLO(args.model)

    # 1) deepen cv3 BEFORE training so the optimizer sees the new params.
    added = _deepen_cv3(model.model.model[-1])
    print(f"[arch] deepened cv3 class branch on {added} scales (+1 conv block each)",
          flush=True)

    # 2) freeze everything except cv3 (the class branch). A callback locks
    #    requires_grad on the whole model, then re-enables ONLY cv3 params.
    def freeze_all_but_cv3(trainer):
        m = trainer.model
        for p in m.parameters():
            p.requires_grad_(False)
        head = m.model[-1]
        on = 0
        for name, p in head.named_parameters():
            if name.startswith("cv3"):
                p.requires_grad_(True); on += 1
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[freeze] only cv3 trainable ({on} tensors, {trainable:,} params); "
              f"backbone/cv2/cv4 frozen -> keypoints & person-box unchanged", flush=True)

    model.add_callback("on_train_start", freeze_all_but_cv3)

    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, name=args.name, project="runs",
        patience=args.patience, plots=True, verbose=True,
    )
    print(f"done -> runs/pose/{args.name}/weights/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
