"""CSPDarknet backbone wrapper — the per-view encoder that REPLACES CDRNet's ResNet152.

YOLO-pose's CSPDarknet neck produces multi-scale feature maps; CDRNet's canonical fusion consumes
feature maps. So we tap the neck at layers 16/19/22 = P3/P4/P5 and expose them, unfreezable.

P3: 128ch, stride-8, 80x80 @640    (finest — primary sampling/fusion scale)
P4: 256ch, stride-16, 40x40
P5: 512ch, stride-32, 20x20

We run the model forward with hooks and read the taps. YOLO's graph computes P5 by layer 22, so a
full forward is needed to populate all three taps (the detection head after them is cheap relative
to the neck); we just ignore the head output. Gradients reach every neck/backbone weight when
`set_trainable(True)`.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO

TAPS = (16, 19, 22)                 # P3, P4, P5
TAP_CH = (128, 256, 512)
TAP_STRIDE = (8, 16, 32)
DEFAULT_WEIGHTS = 'models/yolo26s-pose.pt'


class CSPDarknetEncoder(nn.Module):
    """image (N,3,640,640) -> {8: P3(N,128,80,80), 16: P4(N,256,40,40), 32: P5(N,512,20,20)}."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, imgsz: int = 640):
        super().__init__()
        self.imgsz = imgsz
        yolo = YOLO(weights)
        self.net = yolo.model                       # DetectionModel (nn.Module)
        self._feats = {}
        self._handles = []
        self._hooks_ok = False
        self._ensure_hooks()
        self.set_trainable(False)

    def __deepcopy__(self, memo):
        """YOLO's ModelEMA deepcopies the model. Forward hooks (RemovableHandle) don't survive a
        deepcopy meaningfully — the copy's hooks would point at the ORIGINAL layers. So copy WITHOUT
        the hook state, then re-register on the copy's own layers."""
        import copy as _copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ('_handles',):                  # don't copy stale handles
                continue
            if k == '_feats':
                new._feats = {}
                continue
            if k == '_hooks_ok':
                new._hooks_ok = False
                continue
            setattr(new, k, _copy.deepcopy(v, memo))
        new._handles = []
        new._ensure_hooks()                         # register on the COPY's own layers
        return new

    def set_trainable(self, flag: bool):
        """Freeze/unfreeze the whole backbone+neck. BN eval when frozen (stable stats)."""
        for p in self.net.parameters():
            p.requires_grad_(flag)
        self.net.train(flag)                        # BN train iff trainable
        self._trainable = flag

    def _ensure_hooks(self):
        """(Re)register tap hooks. deepcopy (YOLO ModelEMA) duplicates the module but the original
        hooks point at the ORIGINAL layers and never fire on the copy -> _feats stays empty. Detect
        that (no live handles for THIS module's layers) and re-register on self.net."""
        if getattr(self, '_hooks_ok', False) and self._handles:
            return
        self._feats = getattr(self, '_feats', {})
        self._handles = []
        for i in TAPS:
            h = self.net.model[i].register_forward_hook(
                lambda m, inp, out, i=i: self._feats.__setitem__(i, out))
            self._handles.append(h)
        self._hooks_ok = True

    def forward(self, imgs: torch.Tensor) -> dict:
        """imgs (N,3,H,W) in [0,1] RGB. Returns {stride: feature_map} for strides 8/16/32."""
        self._ensure_hooks()
        self._feats.clear()
        ctx = torch.enable_grad() if self._trainable else torch.no_grad()
        with ctx:
            _ = self.net(imgs)                      # populates hooks (head output ignored)
        out = {TAP_STRIDE[k]: self._feats[TAPS[k]] for k in range(3)}
        return out

    def __del__(self):
        for h in getattr(self, '_handles', []):
            try:
                h.remove()
            except Exception:
                pass
