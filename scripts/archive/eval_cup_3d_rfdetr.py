"""3D cup-tracking eval for an RF-DETR checkpoint -- SAME metrics/consensus as eval_cup_3d_delta,
just a different detector. Monkeypatches eval_cup_3d_delta.cup_center with an RF-DETR predictor so
precision_3d / good_frame% / median_px / per-cam det-rate are directly comparable to the YOLO runs.

    python scripts/eval_cup_3d_rfdetr.py --model runs/rfdetr_P07/checkpoint_best_total.pth \
        --parts P07 --max-trials 4 --fstride 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_cup_3d_delta as E  # noqa: E402


def make_rfdetr_center(model):
    def rfdetr_center(_model, img):
        # img is BGR (cv2); RF-DETR expects RGB. ascontiguousarray: the ::-1 flip yields NEGATIVE
        # strides and torch refuses those ("tensors with negative strides are not supported").
        dets = model.predict(np.ascontiguousarray(img[:, :, ::-1]), threshold=E.CONF)
        if dets is None or len(dets) == 0:
            return None
        i = int(np.argmax(dets.confidence))
        x1, y1, x2, y2 = dets.xyxy[i]
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))
    return rfdetr_center


def main(argv=None):
    import rfdetr
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="RF-DETR checkpoint .pth")
    ap.add_argument("--variant", default="RFDETRNano",
                    help="MUST match the variant the checkpoint was trained with -- Nano/Small use "
                         "patch_size=16, Base uses 14, and a mismatch errors out on load.")
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P15", "P13"])
    ap.add_argument("--max-trials", type=int, default=4)
    ap.add_argument("--fstride", type=int, default=4)
    a = ap.parse_args(argv)

    model = getattr(rfdetr, a.variant)(pretrain_weights=a.model)
    E.cup_center = make_rfdetr_center(model)   # swap detector; eval_part looks it up globally

    print(f"model {a.model}  (RF-DETR; sample {a.max_trials} trials/part, every {a.fstride}th frame)",
          flush=True)
    print(f"{'part':6}{'precision_3d':>13}{'good_frame%':>13}{'median_px':>11}   per-cam det-rate",
          flush=True)
    P = G = nn = 0.0
    for part in a.parts:
        prec, gff, med, dr, tot = E.eval_part(None, part, a.max_trials, a.fstride)
        drs = "  ".join(f"c{c.split('_')[1]}:{int(v*100)}" for c, v in sorted(dr.items()))
        print(f"{part:6}{prec*100:12.1f}%{gff*100:12.1f}%{med:10.1f}px   {drs}", flush=True)
        P += prec; G += gff; nn += 1
    print(f"\nMEAN precision_3d {P/nn*100:.1f}%   good_frame {G/nn*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
