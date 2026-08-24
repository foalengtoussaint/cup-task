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


# ── PHOTOMETRIC augmentations (multi-view SAFE: don't move pixels -> P & 2D unchanged) ──
# Copied from ultralytics with its default pose gains. Geometric augs (mosaic/flip/translate/scale)
# are DELIBERATELY EXCLUDED: they'd break multi-view correspondence / need P-composition. The
# domain-gap generalization (synthetic->real) comes mainly from these photometric ops anyway.
HSV_H, HSV_S, HSV_V = 0.015, 0.7, 0.4        # ultralytics defaults
ERASE_P = 0.4                                 # ultralytics 'erasing' default


def random_hsv(img, hgain=HSV_H, sgain=HSV_S, vgain=HSV_V):
    """ultralytics RandomHSV (exact math), in-place on a BGR uint8 image. Photometric only."""
    if img.shape[-1] != 3 or not (hgain or sgain or vgain):
        return img
    dtype = img.dtype
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
    lut_sat[0] = 0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img


def random_erase(img, p=ERASE_P, sl=0.02, sh=0.1):
    """Random-erasing (occlusion robustness): blank a random rectangle. Photometric (labels unchanged).
    Applied on the LETTERBOXED canvas so erased area is real image, not pad."""
    if np.random.rand() > p:
        return img
    Hh, Ww = img.shape[:2]
    area = Hh * Ww
    for _ in range(10):
        s = np.random.uniform(sl, sh) * area
        ar = np.random.uniform(0.3, 3.3)
        w, h = int(round((s / ar) ** 0.5)), int(round((s * ar) ** 0.5))
        if w < Ww and h < Hh:
            x, y = np.random.randint(0, Ww - w), np.random.randint(0, Hh - h)
            img[y:y + h, x:x + w] = np.random.randint(0, 256, (h, w, img.shape[2]), img.dtype)
            break
    return img


def prep_view_cached(canvas_bgr, native_wh, P_native, imgsz, kp_native=None, augment=False):
    """Fast path: `canvas_bgr` is an ALREADY-letterboxed 640 BGR frame (from the cache) and native_wh
    is (W,H) of the ORIGINAL frame. Letterbox params are deterministic from native_wh, so we recompute
    (r,dw,dh) and transform P/2D identically — no native pixels needed. augment applies photometric ops
    on the canvas (erase) + (HSV also fine on the canvas since it's photometric)."""
    W, H = native_wh
    r, dw, dh, _, _ = letterbox_params(W, H, imgsz)
    canvas = canvas_bgr
    if augment:
        canvas = random_hsv(canvas.copy())
        canvas = random_erase(canvas)
    t = torch.from_numpy(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.
    P_in = torch.from_numpy(letterbox_P(np.asarray(P_native), r, dw, dh))
    kp_in = None if kp_native is None else torch.from_numpy(letterbox_pts(kp_native, r, dw, dh))
    return t, P_in, kp_in


def prep_view(im, P_native, imgsz, kp_native=None, augment=False):
    """One call per view: returns (img_tensor CHW[0,1] RGB, P_input (3,4), kp_input or None).

    im: native BGR (cv2). P_native: (3,4) native. kp_native: optional (...,2) native px 2D.
    augment=True applies multi-view-SAFE photometric aug (HSV before letterbox, erase after) — used
    ONLY in training, per view independently. All geometry outputs share the input-image space.
    """
    if augment:
        im = random_hsv(im.copy())               # HSV on native BGR (photometric, geometry unchanged)
    canvas, r, dw, dh = letterbox_image(im, imgsz)
    if augment:
        canvas = random_erase(canvas)            # erase on the letterboxed canvas
    t = torch.from_numpy(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.
    P_in = torch.from_numpy(letterbox_P(np.asarray(P_native), r, dw, dh))
    kp_in = None if kp_native is None else torch.from_numpy(letterbox_pts(kp_native, r, dw, dh))
    return t, P_in, kp_in
