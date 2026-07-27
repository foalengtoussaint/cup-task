"""Numerically-stable differentiable weighted DLT triangulation.

This is the fusion CORE, validated in the smoke test (scripts/mv3d_smoke_fusion.py). CDRNet's own
author disabled the differentiable DLT loss for "serious instability"; our fix makes it stable:
  * per-row Hartley normalization (scale each DLT equation row to unit norm)
  * float64 SVD solve
  * guarded homogeneous divide
Clean-data DLT then recovers 3D to ~1e-14 mm; the earlier 1e8 blowup was ill-conditioning from
wild views, not the algorithm.

One point (the tracked joint) at a time, C contributing views. Grads flow to `uv` and `w`.
"""
import torch


def weighted_dlt(uv: torch.Tensor, w: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Triangulate one 3D point from C views.

    Args:
        uv: (C, 2) pixel coords (native px), per view.
        w:  (C,) per-view weights in [0, inf) (e.g. sigmoid confidences).
        P:  (C, 3, 4) projection matrices K[R|t] matching `uv`'s native px.
    Returns:
        (3,) triangulated point (float32), differentiable wrt uv and w.
    """
    P64, uv64, w64 = P.double(), uv.double(), w.double()
    p0, p1, p2 = P64[:, 0, :], P64[:, 1, :], P64[:, 2, :]          # (C,4) each
    A0 = uv64[:, 0:1] * p2 - p0                                     # (C,4)  x-eqn
    A1 = uv64[:, 1:2] * p2 - p1                                     # (C,4)  y-eqn
    A = torch.cat([A0, A1], 0)                                      # (2C,4)
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-9)                   # Hartley row-normalize
    wr = torch.cat([w64, w64], 0).clamp(min=1e-4).sqrt()[:, None]   # (2C,1)
    A = A * wr
    _, _, V = torch.linalg.svd(A, full_matrices=False)
    Xh = V[-1]                                                      # (4,) homogeneous
    denom = Xh[3]
    if denom.abs() < 1e-6:
        denom = torch.tensor(1e-6, dtype=Xh.dtype, device=Xh.device)
    return (Xh[:3] / denom).float()


def weighted_dlt_batch(uv: torch.Tensor, w: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Batched over J joints. uv (C,J,2), w (C,J), P (C,3,4) -> (J,3)."""
    C, J = uv.shape[:2]
    P64 = P[:, None].expand(C, J, 3, 4).double()
    uv64, w64 = uv.double(), w.double()
    p0, p1, p2 = P64[..., 0, :], P64[..., 1, :], P64[..., 2, :]
    A = torch.cat([uv64[..., 0:1] * p2 - p0, uv64[..., 1:2] * p2 - p1], 0)   # (2C,J,4)
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-9)
    wr = torch.cat([w64, w64], 0).clamp(min=1e-4).sqrt()[..., None]
    A = (A * wr).permute(1, 0, 2)                                            # (J,2C,4)
    _, _, V = torch.linalg.svd(A, full_matrices=False)
    Xh = V[..., -1, :]
    denom = Xh[..., 3:4]
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    return (Xh[..., :3] / denom).float()


def sii_triangulate(uv: torch.Tensor, P: torch.Tensor, num_iters: int = 2,
                    alpha: float = 1e-3) -> torch.Tensor:
    """CDRNet's differentiable DLT = Shifted Inverse Iterations (their SII layer, ported to torch).

    Solves A x = 0 (A built from uv,P) via inverse-iteration on B = AᵀA − αI:
    repeat  X = solve(B, X); X = X/|X|. This is the EXACT triangulator CDRNet uses (model_define.py
    SII). Differentiable. Kept faithful to the reference; for a STABLE production solve prefer
    `weighted_dlt` (Hartley-normalized SVD). uv (C,2), P (C,3,4) -> (3,).

    Note CDRNet inits X randomly ~N(0.5,0.5); here we init at [0,0,0,1] (world-centre homogeneous)
    for determinism — inverse iteration converges to the same smallest-singular-vector regardless.
    """
    C = uv.shape[0]
    P64, uv64 = P.double(), uv.double()
    # A: (2C,4) — two rows per view: u*P[2]-P[0], v*P[2]-P[1]
    A = (uv64[:, :, None] * P64[:, 2:3, :] - P64[:, 0:2, :]).reshape(2 * C, 4)
    B = A.T @ A - alpha * torch.eye(4, dtype=A.dtype, device=A.device)
    X = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=A.dtype, device=A.device)[:, None]  # (4,1)
    for _ in range(num_iters):
        X = torch.linalg.solve(B, X)
        X = X / (X.norm() + 1e-12)
    X = X[:, 0]
    return (X[:3] / X[3].clamp(min=1e-6) if X[3].abs() > 1e-6 else X[:3]).float()


def reproject(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project 3D point(s) into views. X (...,3), P (C,3,4) -> (C,...,2) pixel coords."""
    Xh = torch.cat([X, torch.ones_like(X[..., :1])], -1)        # (...,4)
    uvw = torch.einsum('cij,...j->c...i', P, Xh)                # (C,...,3)
    return uvw[..., :2] / uvw[..., 2:3].clamp(min=1e-6)


def reproj_residual(X: torch.Tensor, uv: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Mean per-view reprojection error (px) of triangulated X vs the 2D used. Frame-invariant.

    X (3,), uv (C,2), P (C,3,4) -> scalar mean pixel residual.
    """
    proj = reproject(X, P)                    # (C,2)
    return (proj - uv).norm(dim=-1).mean()
