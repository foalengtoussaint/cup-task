"""Feature Transform Layers (FTL / FTL_inv) — CDRNet's canonical-fusion core, ported to torch.

Faithful to scratchpad/CDRnet/model_define.py:
  FTL_inv(P⁺, z): reshape per-view features to (...,3,1), multiply by the 4x3 pseudo-inverse
    projection P⁺ → lift each view's 2D-ish features into a shared CANONICAL 3D-consistent latent
    (channels grouped in 3s → 4s). This is the "camera disentanglement".
  FTL(P, z):      reshape fused canonical features to (...,4,1), multiply by the 3x4 projection P
    → project the canonical latent back into a specific view (channels grouped in 4s → 3s).

Channel bookkeeping (matches reference, generalized): FTL_inv takes C channels (C%3==0), emits
C//3*4 canonical channels. FTL takes canonical channels (C%4==0), emits C//4*3 per-view channels.

P here must be the projection matrix RESCALED to the FEATURE-MAP resolution (see rescale_P), NOT
the native-image P — because FTL operates on the feature grid, exactly like CDRNet rescales P to
the 64x64 heatmap grid before pinv (train_and_test.py L106-124).
"""
import torch
import torch.nn as nn


def rescale_P(P: torch.Tensor, sx: float, sy: float, ox: float = 0.0, oy: float = 0.0) -> torch.Tensor:
    """Rescale a native-image projection matrix to a downsampled feature-grid.

    A pixel (u,v) maps to feature coords (u*sx+ox, v*sy+oy). Left-multiply P by that affine.
    P (...,3,4) -> (...,3,4). For CDRNet's 256->64 (factor 1/4) with half-pixel align, use
    sx=sy=2**-2, ox=oy=2**-3-0.5 (their resize_mat).
    """
    S = torch.tensor([[sx, 0.0, ox], [0.0, sy, oy], [0.0, 0.0, 1.0]],
                     dtype=P.dtype, device=P.device)
    return S @ P


class FTLInv(nn.Module):
    """z (N,C,H,W), P_inv (N,4,3) -> (N, C//3*4, H,W). Lift each view to canonical latent."""
    def forward(self, z: torch.Tensor, P_inv: torch.Tensor) -> torch.Tensor:
        N, C, H, W = z.shape
        assert C % 3 == 0, f"FTL_inv needs C%3==0, got {C}"
        g = C // 3
        # (N,C,H,W) -> (N,H,W,g,3,1)
        zt = z.permute(0, 2, 3, 1).reshape(N, H, W, g, 3, 1)
        Pi = P_inv.reshape(N, 1, 1, 1, 4, 3)              # (N,1,1,1,4,3)
        out = Pi @ zt                                     # (N,H,W,g,4,1)
        return out.reshape(N, H, W, g * 4).permute(0, 3, 1, 2).contiguous()


class FTL(nn.Module):
    """z (N,C,H,W) canonical, P (N,3,4) -> (N, C//4*3, H,W). Project canonical back to a view."""
    def forward(self, z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        N, C, H, W = z.shape
        assert C % 4 == 0, f"FTL needs C%4==0, got {C}"
        g = C // 4
        zt = z.permute(0, 2, 3, 1).reshape(N, H, W, g, 4, 1)
        Pm = P.reshape(N, 1, 1, 1, 3, 4)
        out = Pm @ zt                                     # (N,H,W,g,3,1)
        return out.reshape(N, H, W, g * 3).permute(0, 3, 1, 2).contiguous()


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """Center-of-mass keypoint from heatmaps, CDRNet-style (_calculate_heatmap_keypoints).

    heatmaps (N,K,H,W) with non-negative weights -> (N,K,2) (x,y) in feature-grid pixel coords.
    Uses a 1..H linspace and subtracts 1 to match the reference's 0-based convention.
    """
    N, K, H, W = heatmaps.shape
    eps = 1e-9
    tot = heatmaps.sum(dim=(-2, -1)) + eps                # (N,K)
    ys = torch.arange(1, H + 1, dtype=heatmaps.dtype, device=heatmaps.device)
    xs = torch.arange(1, W + 1, dtype=heatmaps.dtype, device=heatmaps.device)
    h_y = (ys[None, None, :] * heatmaps.sum(-1)).sum(-1) / tot   # (N,K)
    h_x = (xs[None, None, :] * heatmaps.sum(-2)).sum(-1) / tot
    return torch.stack([h_x - 1, h_y - 1], dim=-1)
