"""Build the OMC-prep CACHE once, then answer velocity questions from it instantly.

WHY: H._load_omc parses a FULL C3D with ezc3d on every call (~0.5s/trial) and every probe script
today re-derived the same thing. This caches the DERIVED artefact (OMC pose on the video timebase,
lag-shifted, plus the MMC angle series, the MMC wrist track, the body basis and the phase windows)
to ONE npz per trial -- the "cache the expensive solve output" rule.

    python prep_cache.py build     # ~3 min once
    python prep_cache.py vel       # instant thereafter: velocity test, scorer-matched smoothing
"""
import sys, re
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import os
CACHE = ROOT/"cache"/(os.environ.get("OT_OMC_PREP_DIR") or "omc_prep")
CACHE.mkdir(parents=True, exist_ok=True)
PARTS = {"P12","P15","P17","P07","P13"}
JN = ["right_shoulder","left_shoulder","right_elbow","left_elbow","right_wrist","left_wrist",
      "right_hip","left_hip","nose"]

def build_one(t):
    import compare_pose_omc_delta as H, results_v3_delta as R
    from score_vs_automq import (_pose_variant_cached, _planar_body_angles, _elbow_series,
                                 load_automq, automq_part, automq_phases_to_video, _win)
    global _BA, _AMQ
    p, tr, side = t["part"], t["trial"], t["side"]
    out = CACHE/f"{p}__{tr}.npz"
    if out.exists(): return "skip"
    m = re.search(r"trial_(\d+)_([RL])_", tr)
    if not m: return None
    rec = _AMQ.get((automq_part(p), int(m.group(1)), m.group(2)))
    if rec is None or rec.get("phases") is None: return None
    n = t["mmc"].shape[0]
    pose = _pose_variant_cached(t, "BA", "smoothnet", _BA)
    if pose is None: return None
    omc = H._load_omc(p, tr, n); wr = f"{side}_wrist"
    if wr not in omc or not np.isfinite(omc[wr]).any() or not all(j in omc for j in JN): return None
    GRID = R._GRID_JOINTS
    # stacked multi-signal lag, same as the scorer (H.find_lag_best) -- the phase windows below
    # are placed with it, so a runaway wrist argmax would move every fitted angle window
    lag,_,_ = H.find_lag_best({j: t["mmc"][:, GRID.index(j)] for j in GRID}, omc, side)
    po = {j: R._shift(omc[j], lag) for j in JN}
    other = "right" if side=="left" else "left"
    ph = automq_phases_to_video(rec["phases"], lag, n)
    if not ph: return None
    reach, drink = _win(ph,"reaching"), _win(ph,"drinking")
    w = (reach[0], drink[1]) if (reach and drink) else reach
    if not w: return None
    nn = lambda x: x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
    sh, shO = po[f"{side}_shoulder"], po[f"{other}_shoulder"]
    down = nn((po["right_hip"]+po["left_hip"])/2.0 - (sh+shO)/2.0); sline = nn(sh-shO)
    fwd = nn(np.cross(sline, down)); lat = nn(np.cross(down, fwd))
    f_ = np.isfinite(down).all(1)&np.isfinite(fwd).all(1)
    if not f_.any(): return None
    u = lambda v: v/(np.linalg.norm(v)+1e-9)
    B = np.stack([u(np.nanmedian(down[f_],0)), u(np.nanmedian(fwd[f_],0)), u(np.nanmedian(lat[f_],0))])
    try:
        fm, am = _planar_body_angles(pose, side, other); em = _elbow_series(pose, side)
    except Exception: return None
    s_, e_ = w; sl = slice(s_, e_)
    np.savez_compressed(out, B=B, side=np.array(side), other=np.array(other),
        arm=np.array("affected" if rec.get("condition")=="affected" else "unaffected"),
        part=np.array(p), n_reach=np.array(max(int(reach[1]-reach[0]),2) if reach else e_-s_),
        flex_mmc=fm[sl], abd_mmc=am[sl], elb_mmc=em[sl],
        mmc_wrist=t["mmc"][:, GRID.index(wr)][sl],
        **{f"omc_{j}": po[j][sl] for j in JN})
    return "ok"

def build():
    import gnn_train as GT, results_v3_delta as R
    from score_vs_automq import load_automq
    global _BA, _AMQ
    _BA = R._ba_traj_cache(); _AMQ = load_automq()
    import compare_pose_omc_delta as H; H.use_good_cams()
    trials = [t for t in GT.load_clean(need_reproj=False) if t["part"] in PARTS]
    print(f"building cache for {len(trials)} trials -> {CACHE}", flush=True)
    n_ok = 0
    for i, t in enumerate(trials):
        r = build_one(t)
        if r in ("ok","skip"): n_ok += 1
        if (i+1) % 50 == 0: print(f"  [{i+1}/{len(trials)}] cached {n_ok}", flush=True)
    print(f"PROCESSING CHECK: {len(trials)} trials, cached {n_ok}, files {len(list(CACHE.glob('*.npz')))}")

def load_all():
    recs=[]
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["side"] = str(d["side"]); d["other"] = str(d["other"])
        d["arm"] = str(d["arm"]); d["part"] = str(d["part"])
        recs.append(d)
    return recs
