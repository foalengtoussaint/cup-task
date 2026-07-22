"""Finetune RF-DETR on a COCO cup dataset (from yolo_to_coco_cup.py). One class: cup.

RF-DETR (roboflow) wants a dataset dir with train/ valid/ test/ subdirs each holding images +
_annotations.coco.json. We only build train/ + valid/, so symlink test -> valid.

    python scripts/train_rfdetr_cup.py --data data/rfdetr_P07 --epochs 10 --batch 4 \
        --out runs/rfdetr/P07
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    import rfdetr
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="COCO dataset dir (train/ valid/)")
    ap.add_argument("--variant", default="RFDETRNano",
                    help="RF-DETR size. Nano/Small are closest to our yolo26s-seg (~12M); "
                         "Base is ~29M (unfair size advantage).")
    ap.add_argument("--epochs", type=int, default=20,
                    help="DETR-family finetunes slower than YOLO (which used 3); default 20.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch*grad_accum")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    data = Path(a.data)
    test = data / "test"
    if not test.exists():
        test.symlink_to("valid")          # RF-DETR expects a test split
    Path(a.out).mkdir(parents=True, exist_ok=True)

    Variant = getattr(rfdetr, a.variant)
    print(f"RF-DETR {a.variant} finetune: {a.data}  epochs={a.epochs} "
          f"batch={a.batch}x{a.grad_accum} -> {a.out}", flush=True)
    model = Variant()
    model.train(dataset_dir=str(data), epochs=a.epochs, batch_size=a.batch,
                grad_accum_steps=a.grad_accum, output_dir=a.out)
    print(f"DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
