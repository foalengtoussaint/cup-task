"""SmoothNet 3D pose refinement stage (v2 pipeline).

Temporal-only refinement of the triangulated 3D pose tracks. Validated 2026-07-21 on the 18 DELTA
c3d trials: wrist jitter 14417 -> 1018 mm/s^2 (-93%), OMC speed-corr +0.923 -> +0.984, peak-velocity
error +31% -> +4%, reproduction unchanged (38mm rig floor). Beats a Butterworth low-pass on peak
velocity (SmoothNet +4% vs Butter +11%) -- i.e. it denoises without attenuating the reach peak, which
is the whole point for the Murphy peak_velocity measure. See docs/PIPELINE_V2_PLAN.md.

Two rules from the validation (baked in):
  * smooth PER JOINT -- a gap in one joint must not blank the frame (root-relative-by-hip did that,
    NaN'd 95% of frames; retracted).
  * NO root subtraction -- SmoothNet's temporal filter is offset-covariant per channel.

Uses the pretrained H36M-FCN checkpoint (window 32), no retraining. The model is joint-count-agnostic
(acts on the T axis only), so our 4-target set (cup/mouth/wrists) loads clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SMOOTHNET = _ROOT / "external" / "SmoothNet"

# checkpoint search order: repo-local copy first, then the session scratchpad download
_CKPT_CANDIDATES = [
    _ROOT / "models" / "smoothnet_h36m_fcn_ckpt_32.pth.tar",
    Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
         "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/smoothnet_ckpts/"
         "checkpoints/h36m_fcn_3D/checkpoint_32.pth.tar"),
]
WINDOW = 32

_model = None   # lazy singleton


def _ckpt() -> Path:
    for c in _CKPT_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "SmoothNet checkpoint not found. Expected one of:\n  "
        + "\n  ".join(str(c) for c in _CKPT_CANDIDATES)
        + "\nDownload window-32 h36m_fcn from the SmoothNet gdrive (see project_smoothnet_pose).")


def _load_model():
    global _model
    if _model is not None:
        return _model
    import importlib.util
    import torch
    # SmoothNet and UETrack BOTH ship a top-level `lib` package -- a plain `import lib.models...`
    # resolves to whichever won sys.modules first (UETrack, if the tracker ran). Load SmoothNet's
    # model file by ABSOLUTE PATH so the `lib` name collision is irrelevant.
    smod = _SMOOTHNET / "lib" / "models" / "smoothnet.py"
    spec = importlib.util.spec_from_file_location("_smoothnet_model", smod)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SmoothNet = mod.SmoothNet
    m = SmoothNet(window_size=WINDOW, output_size=WINDOW,
                  hidden_size=512, res_hidden_size=128, num_blocks=5, dropout=0.5)
    c = torch.load(str(_ckpt()), map_location="cpu", weights_only=False)
    m.load_state_dict(c["state_dict"] if "state_dict" in c else c, strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _model = m.to(dev).eval()
    _model._device = dev
    return _model


def smooth_track(track: list[dict], min_len: int = WINDOW) -> list[dict]:
    """Refine ONE triangulated target track (fuse_3d format: [{frame, X, n_cams, reproj_px}, ...]).

    Returns a new track list with X replaced by the smoothed 3D point, frames that were None still
    None, and a `smoothed: True` flag. Short tracks (< window) pass through unchanged.
    """
    import torch
    T = len(track)
    X = np.array([t["X"] if t.get("X") is not None else [np.nan, np.nan, np.nan] for t in track],
                 dtype=float)                                   # (T,3) mm
    if T < min_len or np.isfinite(X).all(1).sum() < 2:
        return track

    # per-joint (here single 3D point) gap handling: interpolate its own gaps for the conv,
    # smooth, then restore the originally-missing frames to None.
    nan = ~np.isfinite(X).all(1)
    filled = X.copy()
    idx = np.arange(T)
    good = ~nan
    for c in range(3):
        filled[:, c] = np.interp(idx, idx[good], X[good, c])
    filled = filled / 1000.0                                    # mm -> m (SmoothNet 3D trained in metres)

    m = _load_model()
    acc = np.zeros((T, 3)); cnt = np.zeros(T)
    with torch.no_grad():
        for s in range(0, T - WINDOW + 1):
            win = filled[s:s + WINDOW]                          # (window,3)
            # CENTRE each window on its own mean before the net, and add the mean back after.
            # The pretrained h36m model was trained on ROOT-RELATIVE poses (metres, near the
            # origin); our world coordinates sit ~1.5m out, which is off-distribution and makes the
            # net's learned bias land as a CONSTANT TRANSLATION of the track (measured: 84mm on the
            # P07 wrist). Centring is exactly offset-covariant -- a temporal filter must never move
            # the track's position -- and drops that to 1.9mm. Do NOT centre on another joint (the
            # old root-subtraction bug): that propagates the root's gaps into every joint.
            off = win.mean(0)
            x = torch.from_numpy((win - off).T[None]).float().to(m._device)   # (1,3,window)
            y = m(x)[0].cpu().numpy().T + off                   # (window,3)
            acc[s:s + WINDOW] += y
            cnt[s:s + WINDOW] += 1
    out = (acc / np.maximum(cnt[:, None], 1)) * 1000.0          # back to mm

    refined = []
    for f, t in enumerate(track):
        nt = dict(t)
        if nan[f]:
            nt["X"] = None
        else:
            nt["X"] = [round(float(v), 1) for v in out[f]]
            nt["smoothed"] = True
        refined.append(nt)
    return refined


def smooth_tracks(tracks: dict, targets=("mouth", "left_wrist", "right_wrist")) -> dict:
    """Refine the POSE targets of a fuse_3d result in place-ish (returns a new dict).

    Cup is left alone by default -- the cup has its own detect-once tracker (cup_track stage); this
    stage only de-jitters the triangulated body joints that feed segmentation + Murphy.
    """
    out = dict(tracks)
    out["targets"] = dict(tracks["targets"])
    for tgt in targets:
        if out["targets"].get(tgt):
            out["targets"][tgt] = smooth_track(out["targets"][tgt])
    out["pose_smoothed"] = list(targets)
    return out
