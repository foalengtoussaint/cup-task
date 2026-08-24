"""A separate cup detection head bolted onto a frozen yolo-pose backbone.

Architecture (single forward pass, one shared backbone):

    backbone/neck ─┬─ [FROZEN] Pose head  -> person box + 17 keypoints  (Δ=0)
                   └─ [NEW]    Detect head -> cup box                    (trainable)

The Pose head (person + keypoints) is never touched — its weights are the base
model's, frozen, so keypoints stay bit-identical. The cup gets its OWN detection
sub-network (own box regressor cv2, own class conv cv3, own DFL) reading the same
neck features. This is the true two-heads-on-one-backbone design: nothing about
person/keypoint detection is shared at the head level, only backbone features.

Because Ultralytics' Pose training loop/loss/NMS assume a single Pose head, this
module is used for INFERENCE (predict) after the cup head is trained separately.
The cup head is trained as a standalone 1-class Detect on the same features via
train_cup_detect_head.py; here we just fuse the two heads for a single-pass
person+keypoints+cup prediction.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class DualHeadPose(nn.Module):
    """Wrap a yolo-pose model + a separate cup Detect head sharing the backbone.

    forward(x) returns (pose_out, cup_out): the untouched Pose head output and
    the cup Detect head output, both computed from one backbone pass.
    """

    def __init__(self, pose_model: nn.Module, cup_head: nn.Module,
                 head_index: int = -1):
        super().__init__()
        self.pose_model = pose_model          # full DetectionModel (backbone+neck+Pose)
        self.cup_head = cup_head              # a Detect(nc=1) reading the same 3 scales
        self.head_index = head_index
        # capture the neck feature maps that feed the Pose head via a hook
        self._feats: list[torch.Tensor] | None = None

    def _neck_features(self, x):
        """Run the backbone/neck exactly as the base model does, returning the
        list of feature maps that the final head consumes."""
        m = self.pose_model.model
        y, feats = [], None
        for i, mod in enumerate(m):
            if mod.f != -1:  # gather inputs from earlier layers
                x = (y[mod.f] if isinstance(mod.f, int)
                     else [x if j == -1 else y[j] for j in mod.f])
            if i == len(m) - 1:               # the head: capture its inputs
                feats = x if isinstance(x, list) else [x]
                break
            x = mod(x)
            y.append(x if mod.i in getattr(self.pose_model, "save", []) else None)
        return feats

    def forward(self, x):
        feats = self._neck_features(x)
        pose_out = self.pose_model.model[self.head_index](list(feats))
        cup_out = self.cup_head(list(feats))
        return pose_out, cup_out


def build_cup_head(pose_model: nn.Module):
    """Create a fresh Detect(nc=1) head matching the pose model's neck channels."""
    from ultralytics.nn.modules import Detect

    head = pose_model.model[-1]
    ch = tuple(head.cv2[i][0].conv.in_channels for i in range(head.nl))
    cup = Detect(nc=1, ch=ch)
    cup.stride = head.stride                  # same strides as the pose head
    return cup
