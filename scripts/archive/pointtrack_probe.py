"""Point-tracking (CoTracker3 online) as a wrist-SPEED method — does tracking the point through time
(multi-frame temporal context) fix optical-flow's peak over-shoot (motion blur) while keeping its clean
off-peak accuracy?

Optical flow matches frame t->t+1 INDEPENDENTLY, so motion blur inflates the single-step match at the
fast peak (+61mm/s over-shoot). Point trackers (CoTracker/TAPNext) track a point across a WINDOW of
frames jointly, using temporal context to stay locked through blur. Hypothesis: cleaner peak.

Per camera: seed CoTracker at the YOLO wrist pixel (frame 0), track it through the trial on a crop
around the wrist -> per-frame 2D tracked position. Then two speed readouts:
  * differentiate the TRACKED position (coherent, so less noisy than raw-YOLO diff)
Feed the tracked point through the SAME triangulation + metrics as the flow probe (P07+P08, vs OMC).

Cache: cache/pointtrack/<clip>__<model>.npy  = (T,2) tracked pixel per frame.

    python scripts/pointtrack_probe.py --part P07 --trial trial_10_L_unaffected --side left
    python scripts/pointtrack_probe.py --batch
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import compare_pose_omc_delta as H
import flow_velocity_probe as F
from pipeline.kalman_3d import triangulate_dlt

FPS = 60.0
CROP = 256          # track on a crop around the wrist (full 1080p OOMs the 7.6GB GPU)
WIN = 48            # process the clip in overlapping temporal windows (OOM-safe on 6GB)
OVERLAP = 20
MODEL = "cotracker3"
_CT = None


def _cotracker():
    global _CT
    if _CT is None:
        import torch
        # ONLINE/causal predictor: streams the whole trial with CONSTANT memory, ONE query the whole
        # way = a genuine single-seed track (no re-seeding, preserves temporal identity). The offline
        # model OOMs a full trial on 6GB; re-seeded windows would defeat the point-tracker's purpose.
        _CT = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").cuda().eval()
    return _CT


def track_wrist_px(clip: Path, wrist_px: np.ndarray) -> np.ndarray:
    """Track the wrist pixel through the clip with CoTracker (online). wrist_px (T,2) = YOLO wrist
    (seed = first finite). Returns (T,2) tracked pixel, NaN where unavailable. Crop-based + cached."""
    import torch
    cdir = ROOT / "cache" / "pointtrack"; cdir.mkdir(parents=True, exist_ok=True)
    ck = cdir / f"{clip.stem}__{MODEL}.npy"
    if ck.exists():
        v = np.load(ck)
        if len(v) == len(wrist_px):
            return v
    good = np.flatnonzero(np.isfinite(wrist_px).all(1))
    T = len(wrist_px)
    out = np.full((T, 2), np.nan)
    if len(good) < 8:
        np.save(ck, out); return out
    seed_f = int(good[0]); sx, sy = wrist_px[seed_f]
    # decode frames, crop a CROP window fixed on the seed position (wrist stays within it)
    cap = cv2.VideoCapture(str(clip)); frames = []
    while True:
        ok, im = cap.read()
        if not ok: break
        frames.append(im)
    cap.release()
    Hh, Ww = frames[0].shape[:2]
    m = _cotracker()
    # FULL FRAME (online = constant memory, fits 1080p). The wrist ranges 200-300px during reaches,
    # which overran a fixed crop — full frame avoids that entirely. ONE query at frame 0, no re-seed.
    x0 = y0 = 0
    crop_frames = [frames[f][:, :, ::-1] for f in range(T)]
    q = torch.tensor([[[float(seed_f), float(sx), float(sy)]]]).float().cuda()  # ONE query, full-frame coords
    step = m.step  # =8; window_len S=16=2*step. Feed EXACTLY the last 2*step frames each call (== S).
    tr = None

    def _proc(win_frames, first):
        chunk = torch.from_numpy(np.stack(win_frames[-2 * step:])).permute(0, 3, 1, 2)[None].float().cuda()
        return m(video_chunk=chunk, is_first_step=first, queries=q)

    with torch.no_grad():
        window = []; is_first = True
        for i in range(T):
            window.append(crop_frames[i])
            if i % step == 0 and i != 0:
                tr, vis = _proc(window, is_first); is_first = False
        # tail: last (i%step) + step + 1 frames, capped at 2*step (== demo)
        tail = window[-((T - 1) % step) - step - 1:]
        tr, vis = _proc(tail, is_first)
    trk = tr[0, :, 0].cpu().numpy()   # (T,2) crop coords — the FULL single-seed trajectory
    for k in range(min(T, len(trk))):
        out[k] = [trk[k, 0] + x0, trk[k, 1] + y0]
    del tr; torch.cuda.empty_cache()
    np.save(ck, out)
    return out


def load_wrist_px(part, trial, joint):
    return F.load_wrist_px(part, trial, joint)


def track_speed_3d(part, trial, joint, cams, n):
    """3D speed from differentiating the CoTracker-tracked point (triangulated per frame)."""
    d = H.DELTA / part
    px = load_wrist_px(part, trial, joint)
    trk = {}
    for c in px:
        cam_n = c.split("_")[1]
        clip = d / "staged" / f"delta_{part}_{trial}.{cam_n}.mp4"
        if clip.exists() and c in cams:
            trk[c] = track_wrist_px(clip, px[c])
    # triangulate the tracked point each frame -> 3D position -> differentiate -> speed
    X = np.full((n, 3), np.nan)
    for fr in range(n):
        obs = {c: trk[c][fr] for c in trk if fr < len(trk[c]) and np.isfinite(trk[c][fr]).all()}
        if len(obs) < 2:
            continue
        X[fr] = triangulate_dlt([cams[c] for c in obs], [np.array(obs[c]) for c in obs])
    sp = np.r_[np.nan, np.linalg.norm(np.diff(X, axis=0), axis=1) * FPS]
    both = np.isfinite(X[:-1]).all(1) & np.isfinite(X[1:]).all(1)
    out = np.full(n, np.nan); out[1:][both] = sp[1:][both]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="P07"); ap.add_argument("--trial", default="trial_10_L_unaffected")
    ap.add_argument("--side", default="left"); ap.add_argument("--batch", action="store_true")
    a = ap.parse_args(argv)
    H.use_good_cams()
    TRIALS = {"P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
              "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right")}
    trials = ([(a.part, a.trial, a.side)] if not a.batch
              else [(p, t, s) for p, (ts, s) in TRIALS.items() for t in ts])
    print(f"{'trial':16} {'track-speed vs OMC |Δ|':>22}", flush=True)
    from pipeline import pose_smooth
    for part, trial, side in trials:
        joint = f"{side}_wrist"
        mmc, n = H._load_mmc(part, trial); omc = H._load_omc(part, trial, n)
        lag, _ = H._find_lag(mmc[joint], omc[joint]); o = H._speed(F._shift(omc[joint], lag))
        cams = H._load_calib_mm(part)
        if part in H.GOOD_CAMS: cams = {c: v for c, v in cams.items() if c in H.GOOD_CAMS[part]}
        sp = track_speed_3d(part, trial, joint, cams, n)
        a_, o_ = H._lp(sp), H._lp(o); m = np.isfinite(a_) & np.isfinite(o_)
        err = np.median(np.abs(a_[m] - o_[m])) if m.sum() > 20 else np.nan
        print(f"{part}_{trial.split('_')[1]:10} {err:20.1f} mm/s", flush=True)


if __name__ == "__main__":
    main()
