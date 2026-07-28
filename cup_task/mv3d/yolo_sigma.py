"""Expose YOLO Pose26's per-axis keypoint sigma at INFERENCE.

YOLO26-pose trains an RLE (Residual Log-likelihood Estimation) sigma head: `cv4_sigma` /
`one2one_cv4_sigma`, a 1x1 conv per FPN level emitting 34 = 17 kpts x (sx, sy). Calibrated by the
RLELoss (utils/loss.py:893): `sigma = sigmoid(raw)`, and it is the scale of `(pred - gt)` in
*grid-cell* units (before `*stride`) -> a per-axis, per-keypoint reliability. Ultralytics gates it
behind `if self.training` (deploy only needs coords), so a stock model drops it at inference.

This module re-exposes it, aligned to the SAME winning anchor as each decoded keypoint, by riding
the sigma channels through the exact decode+topk path the coordinates use (no grid_sample, no
train-mode BN corruption). Result per detection: kpts (nk) + sigma (34), same order.

We patch by METHOD OVERRIDE on the live Pose26 instance (monkeypatch of bound methods), basing every
line on ultralytics' own Pose/Pose26 code so behaviour matches the stock decode exactly. Nothing is
retrained; we only stop discarding a head that was already trained.
"""
from __future__ import annotations
import types
import torch


def enable_sigma(yolo_model) -> None:
    """Patch a loaded ultralytics YOLO (pose26) so its raw output carries per-detection sigma.

    After this, model.model(imgs) inference tensor last-dim = [4 box, nc cls, nk kpts, 34 sigma].
    Idempotent. Operates on the Detect/Pose26 head instance in-place.
    """
    head = yolo_model.model.model[-1]
    if getattr(head, "_sigma_enabled", False):
        return
    assert type(head).__name__ == "Pose26", f"need Pose26 head, got {type(head).__name__}"
    assert head.cv4_sigma is not None, "cv4_sigma missing (fused checkpoint?) — reload unfused"

    nk_sigma = head.nk_sigma  # 34

    # -- 1. forward_head: always emit kpts_sigma (drop the `if self.training` gate) ----------------
    # Faithful copy of Pose26.forward_head (head.py:723) with the training gate removed. Uses the
    # SAME sigma head (one2many cv4_sigma or one2one one2one_cv4_sigma) that forward() passes in.
    def forward_head(self, x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head):
        from ultralytics.nn.modules.head import Detect
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat(
                [kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
            preds["kpts_sigma"] = torch.cat(
                [kpts_sigma_head[i](features[i]).view(bs, nk_sigma, -1) for i in range(self.nl)], 2
            )
        return preds

    # -- 2. _inference: concat sigma channels after kpts so they ride the same decode path ---------
    # Pose._inference (head.py:597) = super()._inference(x) then cat kpts_decode(x["kpts"]).
    # kpts_sigma is anchor-grid-aligned like kpts; sigma needs NO anchor/stride offset (it is a
    # scale, not a coordinate), so we append the raw sigma logits unchanged (activation applied at
    # readout). Shapes: preds (bs, 4+nc, A); kpts_decode (bs, nk, A); sigma (bs, 34, A).
    def _inference(self, x):
        from ultralytics.nn.modules.head import Pose
        base = Pose._inference(self, x)                     # (bs, 4+nc+nk, A)
        return torch.cat([base, x["kpts_sigma"]], dim=1)   # (bs, 4+nc+nk+34, A)

    # -- 3. postprocess: split off sigma alongside kpts via the SAME topk idx ----------------------
    # Faithful copy of Pose.postprocess (head.py:612) extended to gather the 34 sigma channels with
    # the identical winning-anchor idx, guaranteeing sigma[d] belongs to kpt[d].
    def postprocess(self, preds):
        boxes, scores, kpts, sigma = preds.split([4, self.nc, self.nk, nk_sigma], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        sigma = sigma.gather(dim=1, index=idx.repeat(1, 1, nk_sigma))
        return torch.cat([boxes, scores, conf, kpts, sigma], dim=-1)

    head.forward_head = types.MethodType(forward_head, head)
    head._inference = types.MethodType(_inference, head)
    head.postprocess = types.MethodType(postprocess, head)
    head._sigma_enabled = True
    head._sigma_nk = nk_sigma


def split_sigma(raw_det: torch.Tensor, nk: int, nk_sigma: int = 34):
    """Split a sigma-enabled raw detection row into (box+score+cls, kpts, sigma).

    raw_det: (..., 4+2+nk+nk_sigma) after ultralytics NMS keeps [box(4), conf, cls, kpts(nk), sigma].
    Returns kpts (..., nk) and sigma (..., nk_sigma) with sigma = sigmoid(raw) in (0,1),
    grid-cell units, higher = less reliable. Per-axis: sigma.view(..., nk//2 or 17, 2) = (sx, sy).
    """
    kpts = raw_det[..., 6 : 6 + nk]
    sigma = torch.sigmoid(raw_det[..., 6 + nk : 6 + nk + nk_sigma])
    return kpts, sigma
