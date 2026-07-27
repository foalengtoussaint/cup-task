"""ONE shared image+geometry transform for the whole mv3d pipeline — the generalization fix.

The invariant that makes everything rig-agnostic: a loader emits `img`, `P`, and any 2D keypoints
ALL in the SAME coordinate space — the pixel space of the model-input image tensor. Then the model
never needs native resolution, aspect ratio, or resize details: it just uses P and P/hm_stride.

We letterbox (aspect-preserving resize + pad) to a square `imgsz`, exactly like YOLO's backbone was
trained — NOT a squash-to-square (which distorts non-square rigs and is off-distribution for the
CSPDarknet). The SAME affine (scale r, pad dw/dh) is applied to the image, to P (left-multiply), and
to any native-px 2D. Works for ANY camera resolution / aspect ratio.

letterbox affine (native px -> input px):  u' = r*u + dw,  v' = r*v + dh   (r=min(imgsz/W,imgsz/H))
"""
import numpy as np
import cv2
import torch


def letterbox_params(W, H, imgsz):
    """r (uniform scale), dw, dh (pad, px) mapping native (W,H) into a square imgsz canvas."""
    r = min(imgsz / W, imgsz / H)
    nw, nh = round(W * r), round(H * r)
    dw, dh = (imgsz - nw) / 2.0, (imgsz - nh) / 2.0
    return r, dw, dh, nw, nh


def letterbox_image(im, imgsz, color=(114, 114, 114)):
    """Resize `im` (H,W,3) aspect-preserving + pad to (imgsz,imgsz,3). Returns (canvas, r, dw, dh).

    VERIFIED pixel-identical to ultralytics.data.augment.LetterBox with its default TRAINING config
    (new_shape=imgsz, center=True, auto=False, scaleup=True, scale_fill=False) — including the
    round(dw/dh - 0.1) pad rounding — on 1920x1080 / 1024^2 / 1280x720. We keep our own copy (not a
    call into ultralytics) because we also need the (r, dw, dh) affine to transform P/2D identically,
    which their __call__ does not return cleanly.
    """
    H, W = im.shape[:2]
    r, dw, dh, nw, nh = letterbox_params(W, H, imgsz)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), color, dtype=im.dtype)
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, dw, dh


def letterbox_P(P_native, r, dw, dh):
    """Left-multiply P (3,4) by the letterbox affine so it maps 3D -> INPUT-image px (not native)."""
    A = np.array([[r, 0.0, dw], [0.0, r, dh], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (A @ P_native.astype(np.float64)).astype(np.float32)


def letterbox_pts(uv_native, r, dw, dh):
    """Map native-px 2D points (...,2) into input-image px."""
    uv = np.asarray(uv_native, np.float64)
    return (uv * r + np.array([dw, dh])).astype(np.float32)


def prep_view(im, P_native, imgsz, kp_native=None):
    """One call per view: returns (img_tensor CHW[0,1] RGB, P_input (3,4), kp_input or None).

    im: native BGR (cv2). P_native: (3,4) native. kp_native: optional (...,2) native px 2D.
    All outputs share the SAME input-image coordinate space -> the model needs nothing else.
    """
    canvas, r, dw, dh = letterbox_image(im, imgsz)
    t = torch.from_numpy(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.
    P_in = torch.from_numpy(letterbox_P(np.asarray(P_native), r, dw, dh))
    kp_in = None if kp_native is None else torch.from_numpy(letterbox_pts(kp_native, r, dw, dh))
    return t, P_in, kp_in
