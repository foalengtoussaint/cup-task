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

    def forward(self, imgs, P_native, Pinv_grid=None):
        """imgs (V,3,H,W) RGB[0,1]; P_native (V,3,4) at native res.
        Returns dict: kpts2d (V,K,2) native px, X3d (K,3), heatmaps (V,K,80,80)."""
        V = imgs.shape[0]
        dev = imgs.device
        # P rescaled to the heatmap/feature grid (stride hm_stride), + its pinv
        s = 1.0 / self.hm_stride
        P_grid = rescale_P(P_native, s, s)                      # (V,3,4)
        if Pinv_grid is None:
            Pinv_grid = torch.linalg.pinv(P_grid)               # (V,4,3)

        enc = self._per_view_maps(imgs)                         # (V,enc_ch,80,80)
        # FTL_inv per view -> canonical latents, then FUSE (mean over views: view-count agnostic)
        canon = torch.stack([self.ftl_inv(enc[v:v+1], Pinv_grid[v:v+1]) for v in range(V)], 0)  # (V,1,canon,80,80)
        canon = canon[:, 0]                                     # (V,canon,80,80)
        fused = self.fuse(canon.mean(0, keepdim=True))          # (1,fuse_ch,80,80)
        # project canonical fusion back to each view -> decode -> heatmaps
        hms, kpts = [], []
        for v in range(V):
            back = self.ftl(fused, P_grid[v:v+1])               # (1,dec_in,80,80)
            hm = torch.sigmoid(self.decoder(back))              # (1,K,80,80)
            hms.append(hm[0])
            kp_grid = soft_argmax_2d(hm)[0]                     # (K,2) grid px
            kpts.append(kp_grid / s)                            # -> native px (divide by scale)
        kpts2d = torch.stack(kpts, 0)                           # (V,K,2) native px
        heatmaps = torch.stack(hms, 0)                          # (V,K,80,80)
        # triangulate each kpt across views (native-res P)
        tri = sii_triangulate if self.use_sii else (lambda uv, P: weighted_dlt(uv, torch.ones(uv.shape[0], device=uv.device), P))
        X = torch.stack([tri(kpts2d[:, k, :], P_native) for k in range(self.n_kpts)], 0)  # (K,3)
        return {'kpts2d': kpts2d, 'X3d': X, 'heatmaps': heatmaps}
