"""Minimal UETrack-B wrapper (CVPR2026 unified tracker), RGB-only path via GOT10K dataset context.

Same interface as litetrack_wrap.LiteTrackB4:
    t = UETrackB(); t.init(rgb,(x,y,w,h)); (x,y,w,h)=t.update(rgb)
GPU, object_tracking env. UETrack-B is multi-modal (RGB/D/T/language); GOT10K context disables the
depth/thermal/language branches for plain RGB tracking.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

UE = Path(__file__).resolve().parents[1] / "external" / "UETrack"
# Weights live in the REPO (models/trackers/uetrack/), not a scratchpad: an earlier version pointed
# at a session temp dir which was cleaned up, silently breaking every tracker run. Re-fetch with
# huggingface_hub from "kangben258/UETrack" if these are ever missing.
_MODELS = Path(__file__).resolve().parents[1] / "models" / "trackers" / "uetrack"
CKPT = _MODELS / "checkpoints" / "uetrack_base.tar"
PRETRAIN = _MODELS / "pretrained_models" / "fast_itpn_tiny_1600e_1k.pt"


class UETrackB:
    def __init__(self, cfg_name="uetrack_base", checkpoint=None, dataset="got10k"):
        sys.path.insert(0, str(UE))
        # torch>=2.6 defaults weights_only=True; UETrack's checkpoint pickles a training helper
        # (AverageMeter). The ckpt is from the official HF repo, so force the legacy full unpickle.
        import torch as _t
        _orig = _t.load
        _t.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        from lib.config.uetrack.config import cfg, update_config_from_file
        update_config_from_file(str(UE / "experiments" / "uetrack" / f"{cfg_name}.yaml"))
        # Point the backbone at the repo copy regardless of what the YAML says -- the config ships
        # an absolute path, so a moved/cleaned checkpoint dir would otherwise fail deep inside the
        # model builder with a bare FileNotFoundError.
        if PRETRAIN.exists():
            cfg.MODEL.ENCODER.PRETRAIN_TYPE = str(PRETRAIN)
        from lib.test.tracker.uetrack import UETrack
        p = SimpleNamespace(cfg=cfg, checkpoint=str(checkpoint or CKPT),
                            template_factor=cfg.TEST.TEMPLATE_FACTOR,
                            template_size=cfg.TEST.TEMPLATE_SIZE,
                            search_factor=cfg.TEST.SEARCH_FACTOR,
                            search_size=cfg.TEST.SEARCH_SIZE,
                            debug=0, save_all_boxes=False, yaml_name=cfg_name)
        self._trk = UETrack(p, dataset)

    def init(self, rgb, box):
        self._trk.initialize(rgb, {"init_bbox": [float(v) for v in box]})

    def update(self, rgb):
        out = self._trk.track(rgb)
        return tuple(float(v) for v in out["target_bbox"])


class UETrackBatch:
    """N-camera batched UETrack: ONE model, N per-camera states, ONE batched forward per rig-frame.

    The single-camera wrapper runs the backbone once PER camera (sequential) -- the bottleneck at
    5-10 cams. Here each camera keeps its own state (template/anno/text/prev-box), and a rig-frame
    stacks all cameras' search crops into one (N,C,H,W) forward. Measured: 5cam 23->95 fps, 10cam
    11->53 fps (4-5x). Output is per-camera (cx,cy,w,h), identical math to the sequential tracker.

    Usage:
        b = UETrackBatch(n_cams)
        for c in range(n_cams): b.init(c, rgb_c, box_c)   # seed each camera (from its YOLO box)
        boxes = b.update([rgb_0, ..., rgb_{n-1}])          # one batched step -> list of xywh|None
    """

    def __init__(self, n_cams, cfg_name="uetrack_base", checkpoint=None, dataset="got10k"):
        base = UETrackB(cfg_name, checkpoint, dataset)
        self._t = base._trk                      # the shared UETrack instance (model + cfg + preproc)
        self.net = self._t.network
        self.n = n_cams
        self.states = [None] * n_cams            # per-camera dict: state/template_list/anno/text/task
        # sample_target lives under UETrack's lib (path set by UETrackB above)
        from lib.test.tracker.utils import sample_target as _st
        self._sample_target = _st

    def init(self, cam, rgb, box):
        """Seed camera `cam` from its first YOLO box -- reuses the single-cam initialize to build the
        template exactly, then snapshots that per-camera state."""
        self._t.initialize(rgb, {"init_bbox": [float(v) for v in box]})
        self.states[cam] = {
            "state": list(self._t.state),
            "template_list": [t.clone() for t in self._t.template_list],
            "template_anno_list": [a.clone() for a in self._t.template_anno_list],
            "text_src": None if self._t.text_src is None else self._t.text_src.clone(),
            "task": self._t.task_index_batch,
        }

    def update(self, rgbs, bgr=False):
        """One batched step across all seeded cameras. rgbs = list of N RGB frames (None to skip a
        camera this frame). Returns list of (cx-derived xywh) | None per camera.

        bgr=True: the frames are BGR and the RGB conversion is done on the CROPS instead of the full
        frames. sample_target is pure numpy slicing + one resize, i.e. colour-agnostic, so cropping
        first is IDENTICAL math on ~100x fewer pixels. MEASURED at 5 cams 1080p per rig-frame:
        full-frame cv2.cvtColor 3.1ms (and the numpy `[:, :, ::-1].copy()` it replaced, 32.5ms) vs
        ~0.05ms on the crops. Lets the caller skip the colour conversion entirely.
        """
        import cv2
        import torch
        from lib.utils.box_ops import clip_box
        active = [c for c in range(self.n) if self.states[c] is not None and rgbs[c] is not None]
        if not active:
            return [None] * self.n
        import numpy as np
        crops, rf, tmpl_b, anno_b, text_b = [], {}, [], [], []
        for c in active:
            st = self.states[c]
            # crop only (CPU numpy + one cv2.resize) -- cheap; the GPU upload is batched below
            xp, r = self._sample_target(rgbs[c], st["state"], self._t.params.search_factor,
                                        output_sz=self._t.params.search_size)
            crops.append(xp); rf[c] = r
            tmpl_b.append(st["template_list"]); anno_b.append(st["template_anno_list"])
            text_b.append(st["text_src"])
        # BATCHED preprocess: ONE cuda upload + normalize for all N crops (not N small transfers).
        # Replicates preprocessor.process exactly, but on the stacked (N,H,W,C) array.
        pp = self._t.preprocessor
        arr = np.stack(crops, 0)                                      # (N,Hs,Ws,3) uint8
        sb = torch.from_numpy(arr).cuda().float().permute(0, 3, 1, 2)  # (N,3,Hs,Ws)
        mean, std = (pp.mean, pp.std)
        sb = ((sb / 255.0) - mean) / std
        if self._t.multi_modal_vision and sb.size(1) == 3:            # duplicate RGB->6ch (RGB+RGB)
            sb = torch.cat((sb, sb), axis=1)
        n_tmpl = len(tmpl_b[0])
        tmpl_cat = [torch.cat([tmpl_b[i][k] for i in range(len(active))], 0) for k in range(n_tmpl)]
        anno_cat = [torch.cat([anno_b[i][k] for i in range(len(active))], 0)
                    for k in range(len(anno_b[0]))]
        text_cat = (None if text_b[0] is None
                    else torch.cat([t for t in text_b], 0))
        task = self.states[active[0]]["task"]
        task_cat = None if task is None else task.repeat(len(active))
        with torch.no_grad():
            enc, _, _ = self.net.forward_encoder(tmpl_cat, [sb], anno_cat, text_cat, task_cat)
            out = self.net.forward_decoder(feature=enc)
        resp = out["score_map"]
        if self._t.cfg.TEST.WINDOW:
            resp = self._t.output_window * resp
        if "size_map" in out:
            boxes, _ = self.net.decoder.cal_bbox(resp, out["size_map"], out["offset_map"],
                                                 return_score=True)
        else:
            boxes, _ = self.net.decoder.cal_bbox(resp, out["offset_map"], return_score=True)
        boxes = boxes.view(len(active), 4)                            # (N,4) normalized
        result = [None] * self.n
        H, W = rgbs[active[0]].shape[:2]
        # ONE GPU->CPU transfer for the whole batch. A per-camera .tolist() inside the loop below
        # costs N separate device syncs, each stalling the pipeline right after the batched forward
        # -- i.e. it re-serialises the batch we just built. Profiling put that single line at ~50%
        # of UETrack's per-frame CPU time (118ms of 230ms over 20 rig-frames at 5 cams). Scale by
        # search_size here too, so the loop is pure Python arithmetic with no device traffic.
        scaled = (boxes * self._t.params.search_size).tolist()        # [[cx,cy,w,h], ...]
        for i, c in enumerate(active):
            st = self.states[c]
            pred = [v / rf[c] for v in scaled[i]]                     # cx,cy,w,h
            # map back with THIS camera's own prev state (map_box_back reads self.state)
            self._t.state = st["state"]
            new = clip_box(self._t.map_box_back(pred, rf[c]), H, W, margin=10)
            st["state"] = new
            result[c] = tuple(float(v) for v in new)
        return result
