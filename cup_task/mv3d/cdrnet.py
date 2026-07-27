"""CDRNet Canonical-Fusion model on a CSPDarknet encoder (single-person multi-view lifter).

Faithful to scratchpad/CDRnet/model_define.py, with two substitutions:
  * encoder = CSPDarknet neck (P3/P4/P5) instead of ResNet152  (the whole point of the port)
  * multi-scale fusion: we run canonical fusion at the P3 grid (finest), after pulling P4/P5 up to
    the P3 resolution and concatenating — so the fused canonical space sees coarse+fine context.

Pipeline (single forward pass, NO iterative refine loop — that was the wrong mental model):
  per view v:  CSPDarknet -> {P3,P4,P5}; upsample P4,P5 to P3 grid; concat -> 1x1 conv to Cenc
     -> FTL_inv(P⁺_v) lift to CANONICAL latent (Cenc//3*4 ch)
  fuse:        concat canonical latents over views -> 1x1 convs (shared canonical rep)
  per view v:  FTL(P_v) project canonical back -> decoder (up to heatmap grid) -> K heatmaps
     -> soft-argmax -> 2D kpts (feature-grid px) -> scale to native px
  triangulate: SII differentiable DLT across views -> 3D (K,3)

Outputs BOTH per-view 2D (the primary supervised signal, distilled from YOLO) and the 3D recon.
P must be provided at NATIVE image resolution; we rescale internally to the P3/heatmap grid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPDarknetEncoder, TAP_CH
from .ftl import FTLInv, FTL, rescale_P, soft_argmax_2d
from .dlt import sii_triangulate, weighted_dlt


class CanonicalFusionCDR(nn.Module):
    def __init__(self, n_kpts: int = 17, enc_ch: int = 300, fuse_ch: int = 400,
                 weights: str = 'models/yolo26s-pose.pt', imgsz: int = 640,
                 hm_stride: int = 8, use_sii: bool = True):
        """
        n_kpts: keypoints to lift (COCO-17; we mainly score the wrist).
        enc_ch: per-view channels fed to FTL_inv (must be %3==0; 300 like CDRNet).
        fuse_ch: canonical fusion channels (=enc_ch//3*4 = 400 for enc_ch=300).
        hm_stride: heatmaps live on the P3 grid (stride 8 -> 80x80 @640).
        use_sii: True = CDRNet's SII DLT; False = Hartley SVD DLT (stable fallback).
        """
        super().__init__()
        self.encoder = CSPDarknetEncoder(weights, imgsz)
        self.imgsz, self.hm_stride, self.n_kpts, self.use_sii = imgsz, hm_stride, n_kpts, use_sii
        self.hm_size = imgsz // hm_stride                       # 80
        assert enc_ch % 3 == 0 and fuse_ch % 4 == 0

        # per-view: concat(P3, up P4, up P5) = 128+256+512=896 -> enc_ch (shared across views)
        self.reduce = nn.Sequential(
            nn.Conv2d(sum(TAP_CH), enc_ch, 1), nn.BatchNorm2d(enc_ch), nn.ReLU(inplace=True))
        self.ftl_inv = FTLInv()
        self.ftl = FTL()
        canon_ch = enc_ch // 3 * 4                              # 400
        # fusion over views happens by concatenating canonical latents; a shared 1x1 stack fuses.
        # We fuse with a conv that maps (V*canon_ch)->fuse_ch, but V varies (3..5 cams). To stay
        # view-count-agnostic we fuse by MEAN-pool over views in canonical space, then convs.
        self.fuse = nn.Sequential(
            nn.Conv2d(canon_ch, fuse_ch, 1), nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, fuse_ch, 1), nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True))
        self.canon_ch = canon_ch
        # decoder: FTL back gives fuse_ch//4*3 per-view channels -> heatmaps
        dec_in = fuse_ch // 4 * 3
        self.decoder = nn.Sequential(
            nn.Conv2d(dec_in, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, n_kpts, 1))

    def set_trainable_backbone(self, flag: bool):
        self.encoder.set_trainable(flag)

    def _per_view_maps(self, imgs):
        f = self.encoder(imgs)                                  # {8:P3,16:P4,32:P5}
        p3 = f[8]
        p4 = F.interpolate(f[16], size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p5 = F.interpolate(f[32], size=p3.shape[-2:], mode='bilinear', align_corners=False)
        return self.reduce(torch.cat([p3, p4, p5], 1))          # (V,enc_ch,80,80)

    def forward(self, imgs, P_native, Pinv_grid=None, native_size=None):
        """imgs (V,3,H,W) RGB[0,1] (resized to imgsz, may be non-uniform for non-square native);
        P_native (V,3,4) at NATIVE res; native_size (V,2)=(W,H) native per view (defaults to square
        imgsz if None — only valid when native is already square==imgsz).
        Returns dict: kpts2d (V,K,2) 640-img px, X3d (K,3), heatmaps (V,K,hm,hm)."""
        V = imgs.shape[0]
        dev = imgs.device
        # P is at NATIVE res, but the image was resized to imgsz (NON-UNIFORMLY for non-square native,
        # e.g. DELTA 1920x1080->640). So map native -> imgsz-image -> grid with SEPARATE x/y scales:
        #   sx = (imgsz/native_W)/hm_stride,  sy = (imgsz/native_H)/hm_stride.
        # (Earlier bug: rescale_P(1/hm_stride) assumed native==imgsz -> off by native/imgsz, and
        # NON-UNIFORMLY for non-square native -> froze the decoder on DELTA. Square SynBody hid it.)
        if native_size is None:
            nw = nh = torch.full((V,), float(self.imgsz), device=dev)
        else:
            ns = native_size.float().to(dev); nw, nh = ns[:, 0], ns[:, 1]
        sx = (self.imgsz / nw) / self.hm_stride                 # (V,)
        sy = (self.imgsz / nh) / self.hm_stride
        with torch.autocast(device_type=('cuda' if dev.type == 'cuda' else 'cpu'), enabled=False):
            P_grid = torch.stack([rescale_P(P_native[v].float(), float(sx[v]), float(sy[v]))
                                  for v in range(V)])            # (V,3,4) fp32, correct per-axis
            if Pinv_grid is None:
                Pinv_grid = torch.linalg.pinv(P_grid)           # (V,4,3) fp32

        enc = self._per_view_maps(imgs)                         # (V,enc_ch,80,80) may be fp16
        # FTL_inv per view -> canonical latents, then FUSE (mean over views: view-count agnostic)
        with torch.autocast(device_type=('cuda' if dev.type == 'cuda' else 'cpu'), enabled=False):
            enc32 = enc.float()
            canon = torch.stack([self.ftl_inv(enc32[v:v+1], Pinv_grid[v:v+1]) for v in range(V)], 0)
            canon = canon[:, 0]                                 # (V,canon,80,80) fp32
        fused = self.fuse(canon.mean(0, keepdim=True))          # (1,fuse_ch,80,80)
        hms, kpts_grid = [], []
        for v in range(V):
            with torch.autocast(device_type=('cuda' if dev.type == 'cuda' else 'cpu'), enabled=False):
                back = self.ftl(fused.float(), P_grid[v:v+1])   # (1,dec_in,80,80) fp32
            hm = torch.sigmoid(self.decoder(back))              # (1,K,80,80)
            hms.append(hm[0])
            with torch.autocast(device_type=('cuda' if dev.type == 'cuda' else 'cpu'), enabled=False):
                kp_grid = soft_argmax_2d(hm.float())[0]         # (K,2) GRID px (0..hm_size)
            kpts_grid.append(kp_grid)
        kpts2d_grid = torch.stack(kpts_grid, 0)                 # (V,K,2) GRID px
        # report 2D in NATIVE px (match YOLO/GT targets): native = grid / (sx,sy) per axis per view
        scale_xy = torch.stack([sx, sy], -1)[:, None, :]        # (V,1,2)
        kpts2d = kpts2d_grid / scale_xy                         # (V,K,2) native px
        heatmaps = torch.stack(hms, 0)                          # (V,K,80,80)
        # triangulate in a CONSISTENT space: GRID-space 2D with GRID-space P (P_grid). The earlier
        # bug triangulated 640-space 2D against P_NATIVE -> geometrically inconsistent (fine only if
        # native==640; SynBody 1024 partly absorbed it, DELTA 1920x1080 collapsed it). DLT is
        # scale-covariant, so grid+P_grid gives the correct 3D.
        with torch.autocast(device_type=('cuda' if dev.type == 'cuda' else 'cpu'), enabled=False):
            tri = sii_triangulate if self.use_sii else (
                lambda uv, P: weighted_dlt(uv, torch.ones(uv.shape[0], device=uv.device), P))
            X = torch.stack([tri(kpts2d_grid[:, k, :].float(), P_grid.float())
                             for k in range(self.n_kpts)], 0)  # (K,3)
        return {'kpts2d': kpts2d, 'X3d': X, 'heatmaps': heatmaps}
