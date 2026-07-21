"""Greedy biggest-agreeing-subset consensus with temporal continuity (v2 pipeline).

The consensus rule for the detect-once cup tracker. Prefers the LARGEST subset of cameras whose
pairwise reprojection all agrees (<= gate px); a size-2 point is only accepted if it's within `jump`
mm of the previous accepted point (continuity), so a frozen camera + a noise camera momentarily
lining up can't teleport the track. Validated 2026-07-21: median trajectory corr 0.9995 vs OMC,
17/18 trials >= 0.998. See docs/PIPELINE_V2_PLAN.md, project_tracker_shootout_uetrack.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from cup_task.kalman_3d import project, triangulate_dlt

GATE = 30.0
JUMP = 150.0


def best_subset(obs, calib, gate, minc):
    """obs: {cam: (u,v)}  calib: {cam: CamCalib}. Returns (k, -maxerr, X, subset) or None."""
    cams = list(obs)
    best = None
    for k in range(len(cams), minc - 1, -1):
        if best and best[0] > k:
            break
        for sub in combinations(cams, k):
            X = triangulate_dlt([calib[c] for c in sub], [np.array(obs[c]) for c in sub])
            e = [float(np.hypot(*(project(calib[c], X)[0] - np.array(obs[c])))) for c in sub]
            if max(e) <= gate:
                cand = (k, -max(e), X, set(sub))
                if best is None or cand[:2] > best[:2]:
                    best = cand
        if best and best[0] == k:
            break
    return best


def consensus3(obs, calib, prev=None, gate=GATE, jump=JUMP):
    """One frame -> (X_mm | None, kept_cams:set, None). `prev` = last accepted 3D point (mm).

    size>=3 subset is trusted (real majority); size 2 requires continuity within `jump` mm of prev.
    """
    if len(obs) < 2:
        return None, set(), None
    b = best_subset(obs, calib, gate, 2)
    if b is None:
        return None, set(), None
    k, _, X, sub = b
    if k >= 3:
        return X, sub, None
    if prev is None:
        return X, sub, None
    if np.linalg.norm(np.asarray(X) - np.asarray(prev)) <= jump:
        return X, sub, None
    return None, set(), None
