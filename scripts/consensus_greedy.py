"""Consensus v3: greedy + temporal continuity. A 2-camera point is only accepted if consistent
with the recent trajectory, so spurious pairs (frozen cam + noise cam momentarily aligning) that
TELEPORT away from where the cup just was are rejected. Prefers bigger agreeing subsets always."""
import numpy as np
from itertools import combinations
import sys; sys.path.insert(0,'scripts')
from cup_task.kalman_3d import project, triangulate_dlt
GATE=30.0
def best_subset(obs,calib,gate,minc):
    cams=list(obs); best=None
    for k in range(len(cams),minc-1,-1):
        if best and best[0]>k: break
        for sub in combinations(cams,k):
            X=triangulate_dlt([calib[c] for c in sub],[np.array(obs[c]) for c in sub])
            e=[float(np.hypot(*(project(calib[c],X)[0]-np.array(obs[c])))) for c in sub]
            if max(e)<=gate:
                cand=(k,-max(e),X,set(sub))
                if best is None or cand[:2]>best[:2]: best=cand
        if best and best[0]==k: break
    return best
def consensus3(obs, calib, prev=None, gate=GATE, jump=150.0):
    """prev = previous accepted 3D point (mm). A subset of size<3 must be within `jump` mm of prev
    (continuity); size>=3 is trusted regardless (real majority)."""
    if len(obs)<2: return None,set(),None
    b=best_subset(obs,calib,gate,2)
    if b is None: return None,set(),None
    k,_,X,sub=b
    if k>=3: return X,sub,None                      # real majority -> trust
    # size 2: require continuity with prev
    if prev is None: return X,sub,None              # nothing to check against yet
    if np.linalg.norm(np.asarray(X)-np.asarray(prev))<=jump: return X,sub,None
    return None,set(),None                          # 2-cam point teleported -> reject
