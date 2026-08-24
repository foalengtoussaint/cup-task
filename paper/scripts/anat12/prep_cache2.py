"""OMC-prep cache v2 -- ALL participants, FULL-LENGTH series + the scorer's window indices.

v1 (cache/omc_prep) covered 5 participants and stored series already sliced to reach->drink, so the
elbow measure (which the scorer reduces over the WHOLE reaching..returning window) could not be
reproduced from it. v2 stores the full series plus the three windows, so every scorer reduction is
reproducible from the cache with no re-derivation:
    reach   (reaching)                       -> interjoint correlation window
    rd      (reaching start -> drinking end)  -> max flexion / abduction
    allw    (reaching..returning)             -> max elbow angle
v1 is left in place (never delete experiment data).

    python prep_cache2.py build      # ~10 min once, all 11 participants
"""
import sys, re
from pathlib import Path
import numpy as np
ROOT = Path("/home/imove/Documents/cup-task")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
CACHE = ROOT/"cache"/"omc_prep2"; CACHE.mkdir(parents=True, exist_ok=True)
JN = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_wrist", "left_wrist",
      "right_hip", "left_hip", "nose"]


def build_one(t):
    import compare_pose_omc_delta as H, results_v3_delta as R
    from score_vs_automq import (_pose_variant_cached, _planar_body_angles, _elbow_series,
                                 automq_part, automq_phases_to_video, _win)
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
    # stacked multi-signal lag (H.find_lag_best), the same estimator the scorer and the trial
    # gate use. The wrist-only argmax (_find_lag) is superseded: it places the phase windows,
    # so a runaway lag would move every fitted angle window and read as per-participant spread.
    lag, _, _ = H.find_lag_best({j: t["mmc"][:, GRID.index(j)] for j in GRID}, omc, side)
    po = {j: R._shift(omc[j], lag) for j in JN}
    other = "right" if side == "left" else "left"
    ph = automq_phases_to_video(rec["phases"], lag, n)
    if not ph: return None
    reach, drink = _win(ph, "reaching"), _win(ph, "drinking")
    allw = _win(ph, "reaching", "forward_transport", "drinking", "back_transport", "returning")
    rd = (reach[0], drink[1]) if (reach and drink) else reach
    if not rd or not reach: return None
    nn = lambda v: v/(np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    sh, shO = po[f"{side}_shoulder"], po[f"{other}_shoulder"]
    down = nn((po["right_hip"] + po["left_hip"])/2.0 - (sh + shO)/2.0); sline = nn(sh - shO)
    fwd = nn(np.cross(sline, down)); lat = nn(np.cross(down, fwd))
    f_ = np.isfinite(down).all(1) & np.isfinite(fwd).all(1)
    if not f_.any(): return None
    u = lambda v: v/(np.linalg.norm(v) + 1e-9)
    B = np.stack([u(np.nanmedian(down[f_], 0)), u(np.nanmedian(fwd[f_], 0)), u(np.nanmedian(lat[f_], 0))])
    try:
        fm, am = _planar_body_angles(pose, side, other); em = _elbow_series(pose, side)
    except Exception:
        return None
    np.savez_compressed(
        out, B=B, side=np.array(side), other=np.array(other), part=np.array(p), n=np.array(n),
        arm=np.array("affected" if rec.get("condition") == "affected" else "unaffected"),
        reach=np.array(reach), rd=np.array(rd), allw=np.array(allw if allw else rd),
        flex_mmc=fm, abd_mmc=am, elb_mmc=em,
        mmc_wrist=t["mmc"][:, GRID.index(wr)],
        **{f"omc_{j}": po[j] for j in JN})
    return "ok"


def build():
    import gnn_train as GT, results_v3_delta as R
    from score_vs_automq import load_automq
    global _BA, _AMQ
    _BA = R._ba_traj_cache(); _AMQ = load_automq()
    import compare_pose_omc_delta as H; H.use_good_cams()
    trials = GT.load_clean(need_reproj=False)
    print(f"building v2 cache for {len(trials)} trials -> {CACHE}", flush=True)
    n_ok = 0
    for i, t in enumerate(trials):
        if build_one(t) in ("ok", "skip"): n_ok += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(trials)}] cached {n_ok}", flush=True)
    print(f"PROCESSING CHECK: {len(trials)} trials, cached {n_ok}, "
          f"files {len(list(CACHE.glob('*.npz')))}", flush=True)
    print("DONE", flush=True)


def load_all():
    recs = []
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        for k in ("side", "other", "arm", "part"): d[k] = str(d[k])
        d["trial"] = f.stem.split("__", 1)[1]
        recs.append(d)
    return recs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print(f"{len(load_all())} trials in {CACHE}")
