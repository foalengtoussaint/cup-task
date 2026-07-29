"""ST-GCN 3D->3D pose refiner for the DELTA MMC track, trained with a PA-MPJPE loss.

WHY THIS EXISTS (see project_pose_refiner_hunt): no PRETRAINED graph model ingests 3D -- every
"pose refinement" repo is a 2D->3D lifter (st_gcn in_channels=2) or an action classifier. So to get
a graph 3D->3D refiner we TRAIN one here, in_channels=3, on our own cache:
    input  = triangulated YOLO 3D pose (mm), 9 joints, cams 1-5 good-only
    target = OMC mocap markers (mm), same 9 joints, lag-synced
The refiner predicts a RESIDUAL (delta added to the input), so it starts at identity and only learns
the systematic MMC->OMC distortion -- measured PA-MPJPE gap ~49mm arm+head / ~31mm affected wrist.

MODEL: a compact spatio-temporal GCN.
  * spatial: learnable adjacency over the 9-joint skeleton graph (A + learned residual), graph conv.
  * temporal: 1D conv over a sliding window (odd kernel, 'same' padding) -- captures the trajectory.
  * stacked ST blocks, residual, BN, then a linear head to (J,3) residual.
Input is a WINDOW (T_win, J, 3) centred on each frame; output is the refined centre frame.

LOSS: PA-MPJPE. Per frame, Procrustes-align (rot+trans, NO scale) the predicted J-joint constellation
to the OMC target, then mean L2, masked by the valid mask. Rationale (user's choice): downstream
scoring already does a rigid Kabsch, so penalising global placement wastes capacity; PA isolates the
POSE-SHAPE error the GNN can actually fix. NO hip root (rig-flaky, ~25% detected) -- PA's per-frame
alignment removes the global DOF anyway, so absolute mm in, absolute mm out.

  training/eval driven by scripts/gnn_train.py (LOPO). This file = model + loss + a windowed dataset.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# 9 joints in the cache's fixed order
JOINTS = ["right_wrist", "right_elbow", "right_shoulder",
          "left_wrist", "left_elbow", "left_shoulder",
          "right_hip", "left_hip", "nose"]
NJ = len(JOINTS)
_JI = {j: i for i, j in enumerate(JOINTS)}

# skeleton edges (undirected) -> adjacency. Arms + shoulder girdle + trunk + head.
_EDGES = [
    ("right_wrist", "right_elbow"), ("right_elbow", "right_shoulder"),
    ("left_wrist", "left_elbow"), ("left_elbow", "left_shoulder"),
    ("right_shoulder", "left_shoulder"),
    ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip"),
    ("right_hip", "left_hip"),
    ("right_shoulder", "nose"), ("left_shoulder", "nose"),
]


def build_adjacency():
    """Symmetric normalised adjacency (with self-loops), (J,J) float32."""
    A = np.eye(NJ, dtype=np.float32)
    for a, b in _EDGES:
        A[_JI[a], _JI[b]] = 1.0
        A[_JI[b], _JI[a]] = 1.0
    d = A.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(d))
    return (Dinv @ A @ Dinv).astype(np.float32)


# ---------------------------------------------------------------------------- model
class GraphConv(nn.Module):
    """x (B, C, T, J) -> (B, C', T, J). Fixed skeleton A + a learnable residual adjacency."""
    def __init__(self, cin, cout, A):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A))
        self.A_res = nn.Parameter(torch.zeros_like(self.A))   # learned graph correction
        self.lin = nn.Conv2d(cin, cout, kernel_size=1)

    def forward(self, x):
        A = self.A + self.A_res
        x = torch.einsum("bctv,vw->bctw", x, A)   # aggregate over joints
        return self.lin(x)


class STBlock(nn.Module):
    """Graph conv over joints + temporal conv over time, residual."""
    def __init__(self, cin, cout, A, t_kernel=5, dropout=0.1):
        super().__init__()
        self.gcn = GraphConv(cin, cout, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (t_kernel, 1), padding=(t_kernel // 2, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout),
        )
        self.res = (nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.res(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.act(x + r)


class GNNRefiner(nn.Module):
    """Windowed ST-GCN residual refiner. in (B, T, J, 3) -> out refined (B, T, J, 3).

    Predicts a residual added to the input, so an untrained net ~= identity (safe). Input is
    per-window mean-centred (translation removed) for a stable scale; the centre is added back so
    the output stays in absolute mm (the PA loss is translation-invariant anyway)."""
    def __init__(self, hidden=64, blocks=3, t_kernel=5, dropout=0.1):
        super().__init__()
        A = build_adjacency()
        self.in_bn = nn.BatchNorm1d(3 * NJ)
        chans = [3] + [hidden] * blocks
        self.blocks = nn.ModuleList(
            STBlock(chans[i], chans[i + 1], A, t_kernel, dropout) for i in range(blocks))
        self.head = nn.Conv2d(hidden, 3, kernel_size=1)
        nn.init.zeros_(self.head.weight)   # start at identity (residual 0)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x (B, T, J, 3)
        B, T, J, _ = x.shape
        centre = x.mean(dim=(1, 2), keepdim=True)          # (B,1,1,3) per-window centroid
        xc = x - centre
        h = xc.permute(0, 3, 1, 2).contiguous()            # (B,3,T,J)
        # input BN over flattened joints*coords (stabilises scale across trials)
        hbn = self.in_bn(h.reshape(B, 3 * J, T).transpose(1, 2).reshape(B * T, 3 * J))
        h = hbn.reshape(B, T, 3 * J).transpose(1, 2).reshape(B, 3, J, T).permute(0, 1, 3, 2)
        for blk in self.blocks:
            h = blk(h)
        delta = self.head(h).permute(0, 2, 3, 1)           # (B,T,J,3)
        return x + delta                                    # residual, absolute mm


# ---------------------------------------------------------------------------- PA-MPJPE loss
def procrustes_align(pred, targ, mask):
    """Batched per-frame Procrustes (rot+trans, no scale): align pred -> targ over the MASKED joints.

    pred,targ (B,J,3); mask (B,J) bool. Returns pred aligned into targ's frame (B,J,3). Frames with
    <3 valid joints are returned unchanged (their loss is masked out anyway). Differentiable via
    torch.linalg.svd."""
    B, J, _ = pred.shape
    dev = pred.device
    eye = torch.eye(3, device=dev).unsqueeze(0)
    m = mask.unsqueeze(-1).float()                         # (B,J,1)
    n = mask.sum(1).clamp(min=1).view(B, 1, 1)             # valid count
    muP = (pred * m).sum(1, keepdim=True) / n
    muT = (targ * m).sum(1, keepdim=True) / n
    P0 = (pred - muP) * m
    T0 = (targ - muT) * m
    H = P0.transpose(1, 2) @ T0                            # (B,3,3)
    # A frame with <3 valid joints CANNOT define a rotation; svd_backward on its ~0 (degenerate)
    # H produces NaN gradients (measured). Replace those H with the identity so SVD is well-posed
    # (their loss weight is 0 anyway) and add a scale-relative ridge to the rest for conditioning.
    # The optimal rotation R is a GEOMETRIC alignment, not a learned quantity. Backpropagating
    # through torch.linalg.svd is unstable whenever singular values coincide -- which masked
    # (zeroed) joint rows guarantee -> NaN grads (measured on any <full frame). Standard fix for a
    # Procrustes LOSS: compute R under no_grad (detached) and let the gradient flow only through the
    # aligned coordinates P0 @ R.T. R being constant-per-step is exactly right: we penalise the
    # residual shape error AT the optimal alignment, we don't move the alignment itself.
    with torch.no_grad():
        ok = (mask.sum(1) >= 3).view(B, 1, 1).float()
        scale = torch.linalg.matrix_norm(H, keepdim=True).clamp(min=1e-6)
        Hs = ok * (H + eye * 1e-4 * scale) + (1 - ok) * eye
        U, _, Vt = torch.linalg.svd(Hs)
        d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))   # (B,)
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)     # (B,3,3) rotation pred->targ, detached
    aligned = (P0 @ R.transpose(1, 2)) + muT               # gradient flows through P0 (=pred) only
    return aligned


def pa_mpjpe_loss(pred, targ, mask, joint_weights=None):
    """Mean PA-MPJPE over valid (frame, joint). pred,targ (B,J,3); mask (B,J).

    joint_weights (J,) optional -- e.g. up-weight the affected wrist. Loss is in mm (inputs mm)."""
    aligned = procrustes_align(pred, targ, mask)
    err = torch.linalg.norm(aligned - targ, dim=-1)        # (B,J)
    w = mask.float()
    if joint_weights is not None:
        w = w * joint_weights.to(pred.device).unsqueeze(0)
    denom = w.sum().clamp(min=1.0)
    return (err * w).sum() / denom


def window_aligned_loss(pred, targ, mask, joint_weights=None):
    """PA loss with ONE alignment per WINDOW (not per frame). pred,targ (B,T,J,3); mask (B,T,J).

    WHY (user's point): per-FRAME Procrustes re-fits the rotation EVERY frame, so its alignment jumps
    frame-to-frame -- it judges each frame's shape under a different transform and thus has NO notion
    of temporal consistency (it can't penalise jitter/temporal drift; it re-centres it away). A fixed,
    calibrated rig has ONE camera->lab transform per trial, so the honest loss shares a SINGLE rotation
    across the whole window: temporal errors then survive into the loss (a smooth motion made jerky IS
    penalised), while the global offset is still removed (offset-invariant, no wasted capacity).

    One Procrustes fit per window over ALL its valid (frame,joint) points, applied to every frame,
    then masked MSE. R detached (same NaN-safety as procrustes_align)."""
    B, T, J, _ = pred.shape
    dev = pred.device
    eye = torch.eye(3, device=dev).unsqueeze(0)
    m = mask.unsqueeze(-1).float()                             # (B,T,J,1)
    # per-window centroids over all valid points
    n = mask.reshape(B, -1).sum(1).clamp(min=1).view(B, 1, 1, 1)
    muP = (pred * m).reshape(B, -1, 3).sum(1).view(B, 1, 1, 3) / n
    muT = (targ * m).reshape(B, -1, 3).sum(1).view(B, 1, 1, 3) / n
    P0 = (pred - muP) * m
    T0 = (targ - muT) * m
    # H per window = sum over all (frame,joint) of P0^T T0
    Hf = P0.reshape(B, -1, 3).transpose(1, 2) @ T0.reshape(B, -1, 3)   # (B,3,3)
    with torch.no_grad():
        ok = (mask.reshape(B, -1).sum(1) >= 3).view(B, 1, 1).float()
        scale = torch.linalg.matrix_norm(Hf, keepdim=True).clamp(min=1e-6)
        Hs = ok * (Hf + eye * 1e-4 * scale) + (1 - ok) * eye
        U, _, Vt = torch.linalg.svd(Hs)
        d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)         # (B,3,3) shared across the window
    # apply the single R to every frame: (B,T,J,3) @ (B,3,3)
    aligned = torch.einsum("btjk,bmk->btjm", P0, R) + muT
    err = torch.linalg.norm(aligned - targ, dim=-1)            # (B,T,J)
    w = mask.float()
    if joint_weights is not None:
        w = w * joint_weights.to(dev).view(1, 1, J)
    return (err * w).sum() / w.sum().clamp(min=1.0)


# arm chains (shoulder->elbow->wrist) for the relational loss, both sides
_ARM_CHAINS = [("left_shoulder", "left_elbow", "left_wrist"),
               ("right_shoulder", "right_elbow", "right_wrist")]


def relational_loss(pred, targ, mask, bone_w=0.3):
    """IMPAIRMENT-AGNOSTIC 'arm as a coupled linkage' loss. pred,targ (B,T,J,3); mask (B,T,J).

    WHY (user): a model that reproduces ABSOLUTE pose bakes in a HEALTHY-pose prior -> regresses the
    affected arm toward normal (measured: harms affected within-5deg 52->30%). Instead, supervise only
    RELATIVE structure -- the bone VECTORS (elbow-shoulder, wrist-elbow) -- which is body-UNIVERSAL
    (any arm is a rigid linkage, affected or not). No absolute-position term, no global frame: the loss
    literally cannot encode 'where a healthy pose goes', only 'is this arm a coherent linkage matching
    the true configuration'. Two terms per arm segment:
      (1) VECTOR error: |(child-parent)_pred - (child-parent)_targ|  -- the segment's direction+length
          in the world frame BUT since it's a difference of two joints, a global translation cancels
          (offset-invariant). This is 'the segment points the right way and is the right length'.
      (2) BONE-LENGTH constancy: penalise |len_pred - len_targ| (rigidity toward the true bone length).
    Measured signal: raw upper-arm bone-len within-trial STD 13.1mm vs OMC 6.0mm -> the shoulder<->elbow
    linkage IS violated by triangulation, so this has something to fix.
    NOTE: rotation-alignment is NOT applied -- vector differences are already translation-invariant, and
    we WANT the loss to see orientation error (that's the elbow-angle signal). A per-window rotation
    would remove exactly the orientation info we care about."""
    B, T, J, _ = pred.shape
    dev = pred.device
    # FRAME-INVARIANT ONLY. ⚠ Bone VECTORS are NOT frame-invariant (MMC=camera-frame, OMC=lab-frame
    # differ by a rotation, so even a perfect pred scores ~a full bone length -- measured 389mm, the
    # "Way 0" trap). The impairment-agnostic 'coupled linkage' idea must use quantities that don't
    # depend on the coordinate frame: BONE LENGTHS (scalar) and JOINT ANGLES (angle between bones).
    # Both are exactly 'the arm is a rigid linkage in a particular configuration', body-universal, and
    # they're the Murphy clinical quantities. No absolute pose, no frame -> can't learn a healthy prior.
    tot = torch.zeros((), device=dev); wsum = torch.zeros((), device=dev)
    for sh, el, wr in _ARM_CHAINS:
        si, ei, wi = _JI[sh], _JI[el], _JI[wr]
        vp_u = pred[:, :, ei] - pred[:, :, si]; vt_u = targ[:, :, ei] - targ[:, :, si]   # upper-arm
        vp_f = pred[:, :, wi] - pred[:, :, ei]; vt_f = targ[:, :, wi] - targ[:, :, ei]   # forearm
        mu = (mask[:, :, ei] & mask[:, :, si]).float()
        mf = (mask[:, :, wi] & mask[:, :, ei]).float()
        ma = mu * mf                                   # elbow angle needs all three
        # (1) BONE LENGTH error (mm) -- rigidity toward true length
        for vp, vt, m in [(vp_u, vt_u, mu), (vp_f, vt_f, mf)]:
            len_err = (torch.linalg.norm(vp, dim=-1) - torch.linalg.norm(vt, dim=-1)).abs()
            tot = tot + bone_w * (len_err * m).sum(); wsum = wsum + bone_w * m.sum()
        # (2) ELBOW ANGLE error (deg->mm-ish via *4 so it's comparable scale) -- the linkage config
        def ang(a, b):
            c = (a * b).sum(-1) / (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1) + 1e-6)
            return torch.rad2deg(torch.arccos(c.clamp(-1 + 1e-6, 1 - 1e-6)))
        # angle at elbow = between (shoulder->elbow) reversed and (elbow->wrist) => between -vp_u and vp_f
        ae_p = ang(-vp_u, vp_f); ae_t = ang(-vt_u, vt_f)
        ang_err = (ae_p - ae_t).abs()
        tot = tot + (ang_err * ma).sum(); wsum = wsum + ma.sum()
    return tot / wsum.clamp(min=1.0)


def velocity_loss(pred, targ, mask, joint_weights=None, peak_w=0.0, wrist_i=None, smooth_w=0.0):
    """Velocity-domain loss on a WINDOW: pred,targ (B,T,J,3); mask (B,T,J) bool.

    Penalises the per-frame DISPLACEMENT (velocity) error |Δpred - Δtarget| over consecutive frames
    where BOTH frames' joint is valid. Offset-INVARIANT by construction: a constant positional bias
    (the ~70mm rig-geometry offset that overfit and BROKE P17/P19 under the position loss) differences
    away -- d/dt of a constant is 0 -- so this loss cannot learn a rig-specific placement correction.

    WHY plain |Δpred-Δtarget| PARKED AT IDENTITY (measured): it just matches the model's NOISY velocity
    to the target's NOISY velocity, and identity already does that -- it gives NO reason to DENOISE (the
    actual speed error is jitter). `smooth_w` fixes this: it penalises the model's own ACCELERATION
    ||pred[t+1]-2pred[t]+pred[t-1]||, a smoothness prior that rewards the model for producing a cleaner
    trajectory than the raw input -- the denoising incentive the plain loss lacked. peak_w targets the
    Murphy peak-wrist-speed scalar directly. Units mm/frame."""
    B, T, J, _ = pred.shape
    dP = pred[:, 1:] - pred[:, :-1]                        # (B,T-1,J,3)
    dT = targ[:, 1:] - targ[:, :-1]
    vmask = (mask[:, 1:] & mask[:, :-1]).float()           # both frames valid  (B,T-1,J)
    err = torch.linalg.norm(dP - dT, dim=-1)               # (B,T-1,J)
    w = vmask
    if joint_weights is not None:
        w = w * joint_weights.to(pred.device).view(1, 1, J)
    loss = (err * w).sum() / w.sum().clamp(min=1.0)

    if smooth_w > 0:
        # acceleration of the PREDICTION only (denoising prior; makes the output smoother than raw).
        acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]         # (B,T-2,J,3)
        amask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
        aw = amask
        if joint_weights is not None:
            aw = aw * joint_weights.to(pred.device).view(1, 1, J)
        acc_mag = torch.linalg.norm(acc, dim=-1)
        loss = loss + smooth_w * (acc_mag * aw).sum() / aw.sum().clamp(min=1.0)

    if peak_w > 0 and wrist_i is not None:
        # per-window peak wrist SPEED, matched. Masked frames -> speed 0 (won't be the max of a
        # real reach). Encourages the refined peak to match the OMC peak, not over/under-shoot.
        spP = torch.linalg.norm(dP[:, :, wrist_i], dim=-1) * vmask[:, :, wrist_i]   # (B,T-1)
        spT = torch.linalg.norm(dT[:, :, wrist_i], dim=-1) * vmask[:, :, wrist_i]
        peak = (spP.amax(dim=1) - spT.amax(dim=1)).abs().mean()
        loss = loss + peak_w * peak
    return loss


def wrist_idx(side):
    return _JI[f"{side}_wrist"]


# ---------------------------------------------------------------------------- reprojection loss
def project_torch(X, K, dist, R, t):
    """Differentiable Brown-Conrady projection of world points into ONE camera.

    X    (B,T,J,3) world mm ;  per-SAMPLE calib: K (B,3,3) ; dist (B,5) ; R (B,3,3) world->cam ;
    t (B,3). (K/dist/R/t also accept an un-batched (3,3)/(5,)/(3,3)/(3,) form -> broadcast to all.)
    Returns (uv (B,T,J,2), in_front (B,T,J) bool). Mirrors cv2.projectPoints so the reproj residual
    matches the cached-2D sanity check. All torch ops -> gradient flows back to X (the refined 3D)."""
    B, T, J, _ = X.shape
    # reshape per-sample calib to broadcast over (T,J): (B,1,1,3,3) etc.
    if R.ndim == 2:      # un-batched -> add a batch dim
        R = R.unsqueeze(0).expand(B, 3, 3)
        t = t.unsqueeze(0).expand(B, 3)
        K = K.unsqueeze(0).expand(B, 3, 3)
        dist = dist.unsqueeze(0).expand(B, 5)
    Rb = R.view(B, 1, 1, 3, 3)
    tb = t.view(B, 1, 1, 3)
    Xc = torch.einsum("btjk,btjmk->btjm", X, Rb.expand(B, T, J, 3, 3)) + tb   # (B,T,J,3) camera frame
    z = Xc[..., 2].clamp(min=1e-3)
    in_front = Xc[..., 2] > 1e-3
    xn = Xc[..., 0] / z
    yn = Xc[..., 1] / z                                   # normalized image coords
    d = dist.view(B, 1, 1, 5)
    k1, k2, p1, p2, k3 = d[..., 0], d[..., 1], d[..., 2], d[..., 3], d[..., 4]
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    fx = K[:, 0, 0].view(B, 1, 1); sk = K[:, 0, 1].view(B, 1, 1); cx = K[:, 0, 2].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1); cy = K[:, 1, 2].view(B, 1, 1)
    u = fx * xd + sk * yd + cx
    v = fy * yd + cy
    return torch.stack([u, v], dim=-1), in_front


def reprojection_loss(pred, uv, uv_conf, uv_valid, Ks, dists, Rs, ts,
                      smooth_w=0.0, mask=None, joint_weights=None, huber_px=20.0):
    """SELF-SUPERVISED reprojection loss. NO OMC target, NO frame alignment.

    pred      (B,T,J,3)      refined 3D (mm), the GNN output
    uv        (B,T,C,J,2)    per-cam 2D keypoints (px)
    uv_conf   (B,T,C,J)      YOLO kp confidence (per-observation weight)
    uv_valid  (B,T,C,J)      bool: this cam saw this joint this frame
    Ks,dists,Rs,ts           (B,C,3,3)/(B,C,5)/(B,C,3,3)/(B,C,3)  per-sample per-cam calib
    smooth_w                 weight on the model's own acceleration (jerk penalty = denoise prior)
    mask      (B,T,J)        optional: also require the 3D point defined (finite)
    huber_px                 robust threshold -- a wildly-off cam (glass/other-object 2D) shouldn't
                             yank the 3D; Huber caps its pull so consensus cams win (like the
                             REPROJ_PX consensus gate, but soft + differentiable).

    WHY this is the less-biased signal: it pulls pred toward where the cameras ACTUALLY saw the
    joint, so where the good cams agree the raw 3D is already on the rays -> loss ~0 -> pred left
    ALONE (self-gating, the thing OMC-supervised losses couldn't do). smooth_w then buys a smoother
    trajectory at the cost of a little reproj error -> the jitter fix, physically grounded."""
    B, T, C, J, _ = uv.shape
    dev = pred.device
    Xp = pred.unsqueeze(2).expand(B, T, C, J, 3)          # broadcast 3D across cams
    # project per cam. loop C (small, <=5) -- vectorized inside.
    tot = torch.zeros((), device=dev); wsum = torch.zeros((), device=dev)
    for c in range(C):
        uvp, infront = project_torch(Xp[:, :, c], Ks[:, c], dists[:, c], Rs[:, c], ts[:, c])  # (B,T,J,2)
        res = torch.linalg.norm(uvp - uv[:, :, c], dim=-1)         # (B,T,J) px
        # Huber: quadratic near 0, linear past huber_px -> outlier cams capped
        h = huber_px
        rob = torch.where(res <= h, 0.5 * res * res / h, res - 0.5 * h)
        w = uv_valid[:, :, c].float() * uv_conf[:, :, c] * infront.float()
        if mask is not None:
            w = w * mask.float()
        if joint_weights is not None:
            w = w * joint_weights.to(dev).view(1, 1, J)
        tot = tot + (rob * w).sum(); wsum = wsum + w.sum()
    loss = tot / wsum.clamp(min=1.0)

    if smooth_w > 0:
        acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]        # (B,T-2,J,3) mm
        if mask is not None:
            am = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
        else:
            am = torch.ones(pred[:, 2:].shape[:-1], device=dev)
        if joint_weights is not None:
            am = am * joint_weights.to(dev).view(1, 1, J)
        acc_mag = torch.linalg.norm(acc, dim=-1)
        loss = loss + smooth_w * (acc_mag * am).sum() / am.sum().clamp(min=1.0)
    return loss
