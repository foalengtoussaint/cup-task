"""Does optical-flow velocity beat differentiate-position velocity for the wrist speed?

The current speed = differentiate the triangulated 3D wrist position (Δx/Δt). Differentiation AMPLIFIES
any residual position jitter into velocity noise -- which is why even SmoothNet's speed-corr isn't 1.0.

Optical flow measures 2D PIXEL VELOCITY DIRECTLY (PyrLK tracks the wrist pixel frame t->t+1), with NO
differentiation. If we triangulate the per-camera flow VECTORS (not positions) we get a 3D velocity
measured independently of position noise. Literature: "keypoint velocity estimated by optical flow,
merged via Kalman filter" (Improved 2D Human Pose Tracking, ECCV-W 2020); JointFlow (Pattern Rec 2024).

This probe, per c3d trial, compares wrist SPEED three ways vs OMC:
  pos-diff   : differentiate the (raw) triangulated wrist position          -- the baseline
  smoothnet  : differentiate the SmoothNet-refined wrist position           -- our current best
  flow       : triangulate per-camera PyrLK flow at the wrist -> 3D speed    -- the candidate

Metric = low-passed wrist-speed correlation vs OMC (same as everywhere), + peak-velocity error.

    python scripts/flow_velocity_probe.py --part P07 --trial trial_10_L_unaffected --side left
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
from cup_task import pose_smooth, triangulate
from cup_task.kalman_3d import triangulate_dlt

FPS = H.VIDEO_FPS


def _shift(v, lag):
    out = np.full_like(v, np.nan)
    if lag >= 0:
        out[lag:] = v[:len(v) - lag] if lag else v
    else:
        out[:lag] = v[-lag:]
    return out


FLOW_METHOD = "pyrlk"   # pyrlk | pyrlk_tuned | dis | raft (set by main)
_DIS = None
_RAFT = None
RAFT_CROP = 256          # deep flow runs on a CROP around the wrist (full 1080p OOMs on 7.6GB)


def _raft_model():
    global _RAFT
    if _RAFT is None:
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        _RAFT = raft_small(weights=Raft_Small_Weights.DEFAULT).cuda().eval()
    return _RAFT


def _raft_flow_at(prev_bgr, cur_bgr, px):
    """Run RAFT-small on a RAFT_CROP window around px (memory-safe) -> flow vector at px, or None."""
    import torch
    import torch.nn.functional as Fn
    H_, W_ = prev_bgr.shape[:2]
    c = RAFT_CROP // 2
    x, y = int(px[0]), int(px[1])
    x0, y0 = max(0, x - c), max(0, y - c)
    x1, y1 = min(W_, x0 + RAFT_CROP), min(H_, y0 + RAFT_CROP)
    x0, y0 = x1 - RAFT_CROP if x1 - x0 < RAFT_CROP else x0, y1 - RAFT_CROP if y1 - y0 < RAFT_CROP else y0
    x0, y0 = max(0, x0), max(0, y0)
    a = prev_bgr[y0:y1, x0:x1, ::-1].copy(); b = cur_bgr[y0:y1, x0:x1, ::-1].copy()
    if a.shape[0] < 16 or a.shape[1] < 16:
        return None
    m = _raft_model()

    def to_t(im):
        t = torch.from_numpy(im).permute(2, 0, 1)[None].float().cuda() / 255.0
        t = t * 2 - 1
        h, w = t.shape[-2:]
        return Fn.pad(t, (0, (8 - w % 8) % 8, 0, (8 - h % 8) % 8))
    with torch.no_grad():
        fl = m(to_t(a), to_t(b))[-1][0].cpu().numpy().transpose(1, 2, 0)  # (h,w,2)
    lx, ly = x - x0, y - y0
    if 0 <= ly < fl.shape[0] and 0 <= lx < fl.shape[1]:
        return fl[ly, lx]
    return None


def _sample_dense(flow, px):
    """Bilinear-sample a dense flow field (H,W,2) at pixel px."""
    x, y = px
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= flow.shape[1] or y0 + 1 >= flow.shape[0]:
        return None
    ax, ay = x - x0, y - y0
    f = (flow[y0, x0] * (1 - ax) * (1 - ay) + flow[y0, x0 + 1] * ax * (1 - ay)
         + flow[y0 + 1, x0] * (1 - ax) * ay + flow[y0 + 1, x0 + 1] * ax * ay)
    return f


def wrist_flow_velocity_2d(clip: Path, wrist_px: np.ndarray):
    """Track the wrist pixel frame t-1->t. Returns (T,2) 2D pixel velocity, NaN where it failed.
    Method is FLOW_METHOD: sparse PyrLK (default / tuned) OR dense (DIS / RAFT) sampled at the wrist."""
    global _DIS, _RAFT
    # CACHE the per-clip flow velocity (video decode + flow = the expensive part; the metric is free
    # to recompute). Keyed by clip + method + joint so switching metrics never re-runs flow.
    cdir = ROOT / "cache" / "flow_vel"
    cdir.mkdir(parents=True, exist_ok=True)
    ckey = cdir / f"{clip.stem}__{FLOW_METHOD}.npy"
    if ckey.exists():
        v = np.load(ckey)
        if len(v) == len(wrist_px):
            return v
    cap = cv2.VideoCapture(str(clip))
    prev = None; prev_bgr = None
    T = len(wrist_px)
    vel = np.full((T, 2), np.nan)
    if FLOW_METHOD == "pyrlk":
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    elif FLOW_METHOD == "pyrlk_tuned":
        # smaller window (less blur-smear averaging) + more pyramid levels + min-eigenval quality gate
        lk = dict(winSize=(11, 11), maxLevel=5,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
                  flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=1e-3)
    if FLOW_METHOD == "dis" and _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if prev is not None and f < T and np.isfinite(wrist_px[f - 1]).all():
            p = wrist_px[f - 1]
            if FLOW_METHOD in ("pyrlk", "pyrlk_tuned"):
                p0 = p.astype(np.float32).reshape(1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk)
                if st[0, 0] == 1:
                    vel[f - 1] = (p1[0, 0] - p0[0, 0])
            elif FLOW_METHOD == "dis":
                fl = _DIS.calc(prev, gray, None)                 # (H,W,2)
                d = _sample_dense(fl, p)
                if d is not None:
                    vel[f - 1] = d
            elif FLOW_METHOD == "raft":
                d = _raft_flow_at(prev_bgr, im, p)               # RAFT on a crop around the wrist
                if d is not None:
                    vel[f - 1] = d
        prev = gray; prev_bgr = im
        f += 1
    cap.release()
    np.save(ckey, vel)
    return vel


def load_wrist_px(part, trial, joint):
    """Per-camera YOLO wrist pixel (T,2) from the cached pose JSONs, keyed cam_N. NaN where missing."""
    d = H.DELTA / part
    out = {}
    for pj in sorted(glob.glob(str(d / "dets" / f"*{trial}*.pose.json"))):
        cam = f"cam_{Path(pj).name.split('.')[1]}"
        frames = json.loads(Path(pj).read_text())["frames"]
        px = np.full((len(frames), 2), np.nan)
        for i, fr in enumerate(frames):
            k = fr.get("kps", {}).get(joint)
            if k and k[2] > 0.3:
                px[i] = k[:2]
        out[cam] = px
    return out


def flow_speed_3d(part, trial, joint, cams, n):
    """3D speed from triangulated per-camera FLOW vectors. For each frame, triangulate the wrist
    POINT and the wrist point + flow-displacement, take the 3D difference * fps = 3D velocity.
    Purely flow-driven: no position differencing across frames."""
    px = load_wrist_px(part, trial, joint)
    clips = {c: H.DELTA / part / "staged" / f"delta_{part}_{trial}.{c.split('_')[1]}.mp4" for c in px}
    flow = {c: wrist_flow_velocity_2d(clips[c], px[c]) for c in px if clips[c].exists()}
    common = [c for c in flow if c in cams]
    speed = np.full(n, np.nan)
    for f in range(n):
        obs_p, obs_pv = {}, {}
        for c in common:
            if f < len(px[c]) and np.isfinite(px[c][f]).all() and np.isfinite(flow[c][f]).all():
                obs_p[c] = px[c][f]
                obs_pv[c] = px[c][f] + flow[c][f]            # next-frame pixel via flow
        if len(obs_p) < 2:
            continue
        Xp = triangulate_dlt([cams[c] for c in obs_p], [np.array(obs_p[c]) for c in obs_p])
        Xpv = triangulate_dlt([cams[c] for c in obs_pv], [np.array(obs_pv[c]) for c in obs_pv])
        speed[f] = np.linalg.norm(np.array(Xpv) - np.array(Xp)) * FPS   # mm/s
    return speed


TRIALS = {
    "P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
    "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right"),
    "P13": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
}


def _eval_trial(part, trial, side):
    joint = f"{side}_wrist"
    mmc, n = H._load_mmc(part, trial)
    omc = H._load_omc(part, trial, n)
    lag, _ = H._find_lag(mmc[joint], omc[joint])
    omc_w = _shift(omc[joint], lag)
    cams = H._load_calib_mm(part)
    if H.GOOD_CAMS and part in H.GOOD_CAMS:
        cams = {c: v for c, v in cams.items() if c in H.GOOD_CAMS[part]}
    sp_pos = H._speed(mmc[joint])
    sn = pose_smooth.smooth_track([{"frame": i, "X": (None if not np.isfinite(p).all() else list(p))}
                                   for i, p in enumerate(mmc[joint])])
    sp_sn = H._speed(np.array([o["X"] if o["X"] else [np.nan] * 3 for o in sn]))
    sp_flow = flow_speed_3d(part, trial, joint, cams, n)
    sp_omc = H._speed(omc_w)
    # FUSION (speed-weighted): flow has the best SHAPE (direct velocity, no differentiation) but
    # UNDER/OVER-shoots the fast peak (PyrLK's window can't follow big jumps); SmoothNet has the best
    # PEAK. Blend by the local speed itself: at low speed trust flow, at high speed trust SmoothNet.
    # Weight w = sigmoid((speed - hi)/scale): 0 (flow) when slow, 1 (smoothnet) when fast.
    flow_or_sn = np.where(np.isfinite(sp_flow), sp_flow, sp_sn)
    ref = np.where(np.isfinite(sp_sn), sp_sn, flow_or_sn)   # SmoothNet gives the trustworthy magnitude
    hi = np.nanpercentile(ref, 75); scale = max(np.nanstd(ref), 50.0)
    w = 1.0 / (1.0 + np.exp(-(ref - hi) / scale))          # ->1 fast (trust sn peak), ->0 slow (trust flow)
    sp_fuse = (1 - w) * flow_or_sn + w * sp_sn

    o = H._lp(sp_omc)

    def err(a_):
        """DIRECT speed error (speed is frame-invariant, so absolute mm/s is the honest metric, not
        correlation). Returns (median |Δspeed| mm/s, median |Δspeed|/OMC %, |peak err| %)."""
        a_ = H._lp(a_)
        m = np.isfinite(a_) & np.isfinite(o)
        if m.sum() < 20:
            return (np.nan, np.nan, np.nan)
        abs_mm = float(np.median(np.abs(a_[m] - o[m])))
        # % error normalized by OMC speed, only where OMC is actually moving (>50mm/s) so rest-noise
        # doesn't blow up the ratio
        mv = m & (o > 50)
        pct = float(np.median(np.abs(a_[mv] - o[mv]) / o[mv]) * 100) if mv.sum() > 20 else np.nan
        peak = abs((np.nanmax(a_) - np.nanmax(o)) / np.nanmax(o) * 100)
        return (abs_mm, pct, peak)

    return {k: err(s) for k, s in
            [("pos", sp_pos), ("smoothnet", sp_sn), ("flow", sp_flow), ("fuse", sp_fuse)]}


def batch():
    H.use_good_cams()
    agg = {k: {"mm": [], "pct": [], "peak": []} for k in ("pos", "smoothnet", "flow", "fuse")}
    for part, (trials, side) in TRIALS.items():
        for trial in trials:
            try:
                r = _eval_trial(part, trial, side)
            except Exception as e:
                print(f"  {part} {trial}: ERR {e}", flush=True); continue
            for k, (mm, pct, peak) in r.items():
                if np.isfinite(mm):
                    agg[k]["mm"].append(mm); agg[k]["pct"].append(pct); agg[k]["peak"].append(peak)
            print(f"  {part} {trial.split('_')[1]}: " +
                  "  ".join(f"{k} {r[k][0]:.0f}mm/{r[k][2]:.0f}%pk" for k in r), flush=True)
    print(f"\n{'signal':12} {'|Δspeed| mm/s':>14} {'|Δspeed| %':>11} {'|peak err| %':>13} {'n':>4}",
          flush=True)
    for k in ("pos", "smoothnet", "flow", "fuse"):
        print(f"{k:12} {np.median(agg[k]['mm']):14.1f} {np.median(agg[k]['pct']):10.1f}% "
              f"{np.median(agg[k]['peak']):12.0f}% {len(agg[k]['mm']):4d}", flush=True)
    print("\n|Δspeed| = median absolute per-frame wrist-speed error vs OMC (mm/s and % of OMC speed, "
          "moving frames);\npeak err = Murphy peak-velocity magnitude error. Speed is frame-invariant "
          "so ABSOLUTE error is the metric.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="P07")
    ap.add_argument("--trial", default="trial_10_L_unaffected")
    ap.add_argument("--side", default="left")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--method", default="pyrlk",
                    choices=["pyrlk", "pyrlk_tuned", "dis", "raft"])
    a = ap.parse_args(argv)
    global FLOW_METHOD
    FLOW_METHOD = a.method
    if a.batch:
        print(f"[flow method: {a.method}]", flush=True)
        return batch()
    H.use_good_cams()
    joint = f"{a.side}_wrist"

    mmc, n = H._load_mmc(a.part, a.trial)
    omc = H._load_omc(a.part, a.trial, n)
    lag, _ = H._find_lag(mmc[joint], omc[joint])
    omc_w = _shift(omc[joint], lag)

    # calib (mm), good-cam filtered like the harness
    cams = H._load_calib_mm(a.part)
    if H.GOOD_CAMS and a.part in H.GOOD_CAMS:
        cams = {c: v for c, v in cams.items() if c in H.GOOD_CAMS[a.part]}

    # three speed signals
    sp_pos = H._speed(mmc[joint])
    sn = pose_smooth.smooth_track([{"frame": i, "X": (None if not np.isfinite(p).all() else list(p))}
                                   for i, p in enumerate(mmc[joint])])
    mmc_sn = np.array([o["X"] if o["X"] else [np.nan] * 3 for o in sn])
    sp_sn = H._speed(mmc_sn)
    sp_flow = flow_speed_3d(a.part, a.trial, joint, cams, n)
    sp_omc = H._speed(omc_w)

    def corr(a_, b_):
        a_, b_ = H._lp(a_), H._lp(b_)
        m = np.isfinite(a_) & np.isfinite(b_)
        return float(np.corrcoef(a_[m], b_[m])[0, 1]) if m.sum() > 20 else np.nan

    def pverr(a_):
        a_, o = H._lp(a_), H._lp(sp_omc)
        return (np.nanmax(a_) - np.nanmax(o)) / np.nanmax(o) * 100

    print(f"\n{a.part} {a.trial}  (n={n}, lag {lag:+d})", flush=True)
    print(f"{'signal':14} {'OMC speed-corr':>15} {'peak-vel err %':>15} {'coverage':>10}", flush=True)
    for name, s in [("pos-diff", sp_pos), ("smoothnet", sp_sn), ("flow", sp_flow)]:
        cov = np.isfinite(s).mean()
        print(f"{name:14} {corr(s, sp_omc):15.3f} {pverr(s):15.0f} {cov*100:9.0f}%", flush=True)


if __name__ == "__main__":
    main()
