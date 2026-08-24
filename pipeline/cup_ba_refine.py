"""Bundle-adjust the cup 3D point AFTER greedy consensus -- geometry only, gating untouched.

WHY THIS IS SEPARATE FROM THE GATE. consensus3 answers "WHICH blob is the cup": it picks the
agreeing camera subset and applies a velocity gate that rejects static distractors (the cam_10 glass).
BA answers "WHERE exactly": given that camera set, refine the point. The two are complementary and
must stay in that order -- running BA over ALL cameras instead (bypassing the gate) measurably
degrades segmentation (drink-offset p90 20 -> 47 frames), because a better solver fitted to the wrong
blob is just a better-fitted wrong answer.

So this refines ONLY over the cameras consensus kept, on the frames consensus accepted. It cannot
change which frames are claimed, and it cannot re-admit a rejected camera.

What it adds over the linear DLT inside consensus:
  * lens DISTORTION -- consensus triangulates from raw pixels; project_torch models k1,k2,k3,p1,p2
  * a HUBER kernel -- a slightly-off kept camera is down-weighted rather than trusted equally

Measured on 636 DELTA trials (scripts/cup_ba.py): reprojection 8.46 -> 8.16 px, OMC speed-profile
correlation 0.858 -> 0.869, coverage unchanged, segmentation neutral (drink-onset p90 30.0 -> 24.2).
Small -- the point of shipping it is that every 3D point in the pipeline is then reconstructed the
same way (robust triangulation -> BA), rather than pose and cup using different solvers.
"""
from __future__ import annotations

import numpy as np

HUBER_PX = 20.0
ITERS = 60


def refine_track(track: list[dict], calib: dict, huber_px: float = HUBER_PX,
                 iters: int = ITERS) -> list[dict]:
    """track = track_cup_3d_from_cache output. Returns a new list with X bundle-adjusted.

    Frames with X=None stay None. Cameras outside `kept` are never used. Falls back to the input
    track unchanged if torch is unavailable or nothing is solvable.
    """
    try:
        import torch
    except Exception:
        return track
    cams = sorted(calib)
    idx = {c: i for i, c in enumerate(cams)}
    ok = [i for i, t in enumerate(track) if t.get("X") is not None and t.get("kept")]
    if not ok:
        return track
    T, C = len(ok), len(cams)
    uv = np.zeros((T, C, 2), np.float32)
    valid = np.zeros((T, C), bool)
    X0 = np.zeros((T, 3), np.float32)
    for r, i in enumerate(ok):
        t = track[i]
        X0[r] = t["X"]
        for c in t["kept"]:
            if c in idx and t.get("uv", {}).get(c) is not None:
                uv[r, idx[c]] = t["uv"][c]
                valid[r, idx[c]] = True
    if not valid.any():
        return track                      # no per-camera 2D retained -> nothing to refine
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from gnn_refiner import project_torch  # noqa: E402  (same projection model as the pose BA)
    K = torch.from_numpy(np.stack([np.asarray(calib[c].K, np.float32) for c in cams])).to(dev)
    D = torch.from_numpy(np.stack([np.asarray(calib[c].dist, np.float32).ravel()[:5]
                                   for c in cams])).to(dev)
    R = torch.from_numpy(np.stack([np.asarray(calib[c].R, np.float32) for c in cams])).to(dev)
    tt = torch.from_numpy(np.stack([np.asarray(calib[c].t, np.float32).ravel()
                                    for c in cams])).to(dev)
    UV = torch.from_numpy(uv).to(dev)
    V = torch.from_numpy(valid).to(dev).float()
    X = torch.from_numpy(X0[None, :, None, :]).to(dev).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([X], lr=1.0, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        tot = torch.zeros((), device=dev); wsum = torch.zeros((), device=dev)
        for ci in range(C):
            uvp, infront = project_torch(X[:, :, 0:1].expand(1, T, 1, 3), K[ci], D[ci], R[ci], tt[ci])
            uvp = uvp.clamp(-1e6, 1e6)          # see ba_refine.UVMAX -- inf*0 -> NaN otherwise
            res = torch.linalg.norm(uvp[0, :, 0] - UV[:, ci], dim=-1)
            rob = torch.where(res <= huber_px, 0.5 * res * res / huber_px, res - 0.5 * huber_px)
            w = V[:, ci] * infront[0, :, 0].float()
            tot = tot + (rob * w).sum(); wsum = wsum + w.sum()
        loss = tot / wsum.clamp(min=1.0)
        loss.backward()
        return loss

    opt.step(closure)
    Xr = X.detach()[0, :, 0].cpu().numpy()
    out = [dict(t) for t in track]
    for r, i in enumerate(ok):
        if np.isfinite(Xr[r]).all():
            out[i]["X"] = [round(float(v), 1) for v in Xr[r]]
            out[i]["ba"] = True
    return out
