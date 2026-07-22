"""Minimal LiteTrack-B4 wrapper -- isolates the fragile external import chain in one place.

Usage:
    from litetrack_wrap import LiteTrackB4
    t = LiteTrackB4()
    t.init(rgb_img, (x,y,w,h))
    (x,y,w,h) = t.update(rgb_img)     # xywh, same convention as cv2 trackers / our YOLO boxes

Runs on GPU (object_tracking env). See project_litetrack_setup memory note for the setup gotchas.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

LT = Path(__file__).resolve().parents[1] / "external" / "LiteTrack"
CKPT = (Path(__file__).resolve().parents[1] / "external" / "LiteTrack" / "ckpts" /
        "B4_cae_center_all_ep300" / "LiteTrack_ep0300.pth.tar")
# fall back to the scratchpad download location if not copied in yet
if not CKPT.exists():
    CKPT = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
                "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/litetrack_ckpts/train/"
                "litetrack/B4_cae_center_all_ep300/LiteTrack_ep0300.pth.tar")


class LiteTrackB4:
    def __init__(self, cfg_name="B4_cae_center_all_ep300", checkpoint=None):
        sys.path.insert(0, str(LT))
        from lib.config.litetrack.config import cfg, update_config_from_file
        update_config_from_file(str(LT / "experiments" / "litetrack" / f"{cfg_name}.yaml"))
        from lib.test.tracker.litetrack import LiteTrack
        p = SimpleNamespace(cfg=cfg, checkpoint=str(checkpoint or CKPT),
                            template_factor=cfg.TEST.TEMPLATE_FACTOR,
                            template_size=cfg.TEST.TEMPLATE_SIZE,
                            search_factor=cfg.TEST.SEARCH_FACTOR,
                            search_size=cfg.TEST.SEARCH_SIZE,
                            debug=0, save_all_boxes=False)
        self._trk = LiteTrack(p, "got10k")

    def init(self, rgb, box):
        self._trk.initialize(rgb, {"init_bbox": [float(v) for v in box]})

    def update(self, rgb):
        out = self._trk.track(rgb)
        return tuple(float(v) for v in out["target_bbox"])   # xywh
