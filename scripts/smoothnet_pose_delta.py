"""SmoothNet on DELTA pose -- does temporal smoothing improve jitter / OMC speed-corr / reproduction?

Runs on the SAME n=18 trials the tracker thread used: P07/P08/P13 trial_10-15, each has a c3d
(OMC ground truth). Two experiments the user asked for:

  3D : smooth the TRIANGULATED pose (T, 11joints*3), root-relative metres, then re-triangulate error.
  2D : smooth each camera's raw keypoints (T, 11joints*2), pixel-normalised, THEN triangulate the
       smoothed 2D and measure the 3D result.

SmoothNet is a plug-and-play temporal-only refiner (ECCV'22). Its pretrained checkpoint is joint-
count-agnostic (encoder acts on the T axis only), but it expects the SAME normalisation its training
data used: 3D = root-relative metres; 2D = (px - IMG/2)/(IMG/2). We honour both.

Three metrics, INPUT (raw) vs OUTPUT (smoothed), per joint of interest (the tracked wrist):
  1. JITTER      = mean |acceleration| = mean ||x[t+1]-2x[t]+x[t-1]|| * fps^2  (SmoothNet's Accel).
  2. SPEED CORR  = low-passed wrist-speed corr vs OMC (the harness metric score.py differentiates).
  3. REPRODUCTION= median absolute 3D error mm vs OMC after one rigid Kabsch (arm+head joints).

Pretrained transfer OR train-on-ours (LOPO): --mode {pretrained, lopo}.

    python scripts/smoothnet_pose_delta.py --mode pretrained --window 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SmoothNet"))

# reuse the PROVEN loaders / metrics from the existing harness (same triangulation, sync, Kabsch)
import compare_pose_omc_delta as H
from lib.models.smoothnet import SmoothNet
from cup_task import triangulate

SP = Path("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
          "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/smoothnet_ckpts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W, IMG_H = 1920.0, 1080.0
FPS = H.VIDEO_FPS

# the 18 c3d-matched trials, and which arm is tracked (DELTA affected=L for P07/P13, R for P08)
TRIALS = {
    "P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
    "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right"),
    "P13": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
}
JOINT_ORDER = list(H.JOINTS)          # 11 joints, fixed order
ABS_JOINTS = [j for j in JOINT_ORDER if "hip" not in j]   # arm+head, hips excluded (see harness)


# --------------------------------------------------------------------------- SmoothNet plumbing
def load_smoothnet(window, ckpt):
    m = SmoothNet(window_size=window, output_size=window,
                  hidden_size=512, res_hidden_size=128, num_blocks=5, dropout=0.5)
    if ckpt is not None:
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(c["state_dict"] if "state_dict" in c else c, strict=True)
    return m.to(DEV).eval()


def smooth_sequence(model, seq, window, dim=3):
    """seq (T, C) -> smoothed (T, C). Sliding window, step 1, average overlaps (SmoothNet eval mode).

    Gaps are handled PER JOINT, not per frame: joint j is smoothed only over the frames where j is
    present (linearly interpolated across its own short gaps for the conv, then those originally-
    missing frames restored to NaN). A gap in ONE joint must not blank out the others -- the earlier
    all-joints row-mask knocked out 95% of frames because different joints drop at different times."""
    T, C = seq.shape
    njoint = C // dim
    filled = seq.copy()
    nanmask = np.zeros((T, njoint), bool)
    idx = np.arange(T)
    for k in range(njoint):
        cols = slice(dim * k, dim * k + dim)
        jm = ~np.isfinite(seq[:, cols]).all(1)
        nanmask[:, k] = jm
        good = ~jm
        if good.sum() < 2:
            continue
        for c in range(dim * k, dim * k + dim):
            filled[:, c] = np.interp(idx, idx[good], seq[good, c])
    if T < window:                       # too short to smooth -> passthrough
        return seq
    acc = np.zeros((T, C)); cnt = np.zeros(T)
    with torch.no_grad():
        for s in range(0, T - window + 1):
            win = filled[s:s + window]                       # (window, C)
            x = torch.from_numpy(win.T[None]).float().to(DEV)  # (1, C, window)
            y = model(x)[0].cpu().numpy().T                   # (window, C)
            acc[s:s + window] += y
            cnt[s:s + window] += 1
    out = acc / np.maximum(cnt[:, None], 1)
    # restore per-joint NaN
    for k in range(njoint):
        out[nanmask[:, k], dim * k:dim * k + dim] = np.nan
    return out


# ------------------------------------------------------------------------------------- metrics
def jitter(xyz):
    """mean acceleration magnitude (units/s^2 in whatever units xyz is)."""
    v = np.isfinite(xyz).all(1)
    a = xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]
    m = v[2:] & v[1:-1] & v[:-2]
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.linalg.norm(a[m], axis=1)) * FPS * FPS)


def speed_corr(mmc_wrist, omc_wrist):
    sm, so = H._lp(H._speed(mmc_wrist)), H._lp(H._speed(omc_wrist))
    m = np.isfinite(sm) & np.isfinite(so)
    return float(np.corrcoef(sm[m], so[m])[0, 1]) if m.sum() > 20 else np.nan


def reproduction(mmc, omc):
    """median absolute mm error after ONE rigid Kabsch on the arm+head constellation."""
    A = np.vstack([omc[j] for j in ABS_JOINTS])
    B = np.vstack([mmc[j] for j in ABS_JOINTS])
    R, t, resid = H._kabsch(A, B)
    return float(np.median(resid))


def peak_vel_err(mmc_wrist, omc_wrist):
    """|peak wrist speed MMC - peak wrist speed OMC| / OMC peak, low-passed (Murphy peak-velocity).
    A low-pass that over-smooths KILLS the peak -> this error grows. Returns signed % (neg = MMC
    under-shoots the true peak)."""
    sm, so = H._lp(H._speed(mmc_wrist)), H._lp(H._speed(omc_wrist))
    pm, po = np.nanmax(sm), np.nanmax(so)
    return float((pm - po) / po * 100.0)


def butter_lp_seq(seq, dim, hz=6.0):
    """Plain Butterworth low-pass baseline, per channel, NaN-gap-safe (same _lp the harness uses)."""
    out = seq.copy()
    for c in range(seq.shape[1]):
        out[:, c] = H._lp(seq[:, c])
    # restore per-joint NaN where the whole joint was missing
    T = seq.shape[0]; njoint = seq.shape[1] // dim
    for k in range(njoint):
        jm = ~np.isfinite(seq[:, dim * k:dim * k + dim]).all(1)
        out[jm, dim * k:dim * k + dim] = np.nan
    return out


# ------------------------------------------------------------------------------- data assembly
def mmc_3d_to_matrix(mmc):
    """dict[joint]->(T,3)  ->  (T, 33) in the fixed joint order, metres (harness is mm)."""
    T = len(next(iter(mmc.values())))
    M = np.full((T, len(JOINT_ORDER) * 3), np.nan)
    for k, j in enumerate(JOINT_ORDER):
        M[:, 3 * k:3 * k + 3] = mmc[j] / 1000.0    # mm -> m (SmoothNet 3D trained in metres)
    return M


def matrix_to_mmc_3d(M):
    T = M.shape[0]
    out = {}
    for k, j in enumerate(JOINT_ORDER):
        out[j] = M[:, 3 * k:3 * k + 3] * 1000.0    # back to mm
    return out


def root_relative(M, root_joint="left_hip"):
    """subtract the root joint xyz per frame (SmoothNet 3D convention). Returns centred M + root."""
    ridx = JOINT_ORDER.index(root_joint)
    root = M[:, 3 * ridx:3 * ridx + 3].copy()
    Mc = M.copy()
    for k in range(len(JOINT_ORDER)):
        Mc[:, 3 * k:3 * k + 3] -= root
    return Mc, root


def un_root(Mc, root):
    M = Mc.copy()
    for k in range(len(JOINT_ORDER)):
        M[:, 3 * k:3 * k + 3] += root
    return M


# ---------------------------------------------------------------------- 2D smoothing + retriangulate
def smooth_2d_then_triangulate(part, trial, model, window, method="smoothnet"):
    """Load per-cam pose, smooth each camera's (T, 22) keypoints in normalised px, write the smoothed
    points back into the frame dicts, then run the SAME triangulate_target the harness uses.
    method='butter' swaps SmoothNet for a plain Butterworth low-pass baseline."""
    d = H.DELTA / part
    cams = H._load_calib_mm(part)
    import glob, json
    per_cam = {}
    for pj in sorted(glob.glob(str(d / "dets" / f"*{trial}*.pose.json"))):
        cam = Path(pj).name.split(".")[1]
        per_cam[f"cam_{cam}"] = json.loads(Path(pj).read_text())["frames"]
    if H.GOOD_CAMS is not None and part in H.GOOD_CAMS:
        keep = H.GOOD_CAMS[part]
        per_cam = {c: v for c, v in per_cam.items() if c in keep}
        cams = {c: v for c, v in cams.items() if c in keep}
    n = max(len(v) for v in per_cam.values())

    # build (T, 22) per camera, smooth, write back
    for ckey, frames in per_cam.items():
        T = len(frames)
        M = np.full((T, len(JOINT_ORDER) * 2), np.nan)
        for f, fr in enumerate(frames):
            kps = fr.get("kps", {})
            for k, j in enumerate(JOINT_ORDER):
                if j in kps:
                    M[f, 2 * k] = kps[j][0]
                    M[f, 2 * k + 1] = kps[j][1]
        # normalise to [-1,1]
        Mn = M.copy()
        Mn[:, 0::2] = (M[:, 0::2] - IMG_W / 2) / (IMG_W / 2)
        Mn[:, 1::2] = (M[:, 1::2] - IMG_H / 2) / (IMG_H / 2)
        Sn = butter_lp_seq(Mn, 2) if method == "butter" else smooth_sequence(model, Mn, window, dim=2)
        # denormalise
        S = Sn.copy()
        S[:, 0::2] = Sn[:, 0::2] * (IMG_W / 2) + IMG_W / 2
        S[:, 1::2] = Sn[:, 1::2] * (IMG_H / 2) + IMG_H / 2
        # write smoothed points back into a COPY of the frame dicts
        for f, fr in enumerate(frames):
            kps = dict(fr.get("kps", {}))
            for k, j in enumerate(JOINT_ORDER):
                if j in kps and np.isfinite(S[f, 2 * k]) and np.isfinite(S[f, 2 * k + 1]):
                    kps[j] = [float(S[f, 2 * k]), float(S[f, 2 * k + 1]), kps[j][2]]
            frames[f] = {**fr, "kps": kps}

    # triangulate the smoothed 2D, same as the harness
    out = {}
    for joint in JOINT_ORDER:
        tr = triangulate.triangulate_target(per_cam, cams, H._kp_point(joint), n)
        X = np.array([t["X"] if t.get("X") else [np.nan] * 3 for t in tr])
        out[joint] = H._despike(X)
    return out, n


# ------------------------------------------------------------------------------------- one trial
def eval_trial(part, trial, side, model, window, mode2d=False, method="smoothnet"):
    """Returns (metrics_raw, metrics_smoothed, n). method in {smoothnet, butter}.
    Raw = harness triangulation; smoothed = SmoothNet or a plain Butterworth low-pass baseline."""
    mmc_raw, n = H._load_mmc(part, trial)
    omc = H._load_omc(part, trial, n)

    # sync + shift OMC exactly as the harness does (on the RAW mmc so the lag is consistent)
    wr = f"{side}_wrist"
    lag, _ = H._find_lag(mmc_raw[wr], omc[wr])
    def shift(v, lag):
        out = np.full_like(v, np.nan)
        if lag >= 0:
            out[lag:] = v[:len(v) - lag] if lag else v
        else:
            out[:lag] = v[-lag:]
        return out
    omc = {j: shift(v, lag) for j, v in omc.items()}

    if mode2d:
        if method == "butter":
            mmc_sm, _ = smooth_2d_then_triangulate(part, trial, None, window, method="butter")
        else:
            mmc_sm, _ = smooth_2d_then_triangulate(part, trial, model, window)
    else:
        # smooth absolute metres, per-joint. SmoothNet's temporal filter is offset-covariant per
        # channel, so root subtraction is unnecessary AND harmful here (root=hip has gaps that would
        # propagate to every joint). Per-joint gap handling keeps each joint's real coverage.
        M = mmc_3d_to_matrix(mmc_raw)
        S = butter_lp_seq(M, 3) if method == "butter" else smooth_sequence(model, M, window, dim=3)
        mmc_sm = matrix_to_mmc_3d(S)

    def metrics(mmc):
        return {
            "jitter": jitter(mmc[wr]),
            "speed_corr": speed_corr(mmc[wr], omc[wr]),
            "reproduction": reproduction(mmc, omc),
            "peak_vel_err": peak_vel_err(mmc[wr], omc[wr]),
        }
    return metrics(mmc_raw), metrics(mmc_sm), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pretrained"], default="pretrained")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--ckpt", default=None, help="override checkpoint path")
    ap.add_argument("--dim", choices=["3d", "2d", "both"], default="both")
    ap.add_argument("--method", choices=["smoothnet", "butter", "both"], default="both",
                    help="smoothnet, a plain butterworth low-pass baseline, or both side by side")
    a = ap.parse_args(argv)

    H.use_good_cams()   # verified-good camera whitelist (critical: bad cams poison triangulation)

    ck3d = a.ckpt or str(SP / "checkpoints" / "h36m_fcn_3D" / f"checkpoint_{a.window}.pth.tar")
    # 2D uses the h36m 2D-trained head? repo ships h36m 2D under the same folder; reuse 3D filter
    # (temporal-only, works on normalised 2D too). Both share the checkpoint architecture.
    model = load_smoothnet(a.window, ck3d)
    print(f"loaded SmoothNet window={a.window} from {Path(ck3d).parent.name}/{Path(ck3d).name}", flush=True)

    methods = ["smoothnet", "butter"] if a.method == "both" else [a.method]
    for dim in (["3d", "2d"] if a.dim == "both" else [a.dim]):
        mode2d = dim == "2d"
        for method in methods:
            print(f"\n{'='*80}\n=== {dim.upper()} / {method.upper()} ===  "
                  f"(jitter mm/s^2 | OMC speed-corr | reprod mm | peak-vel err %)\n{'='*80}", flush=True)
            rows = []
            for part, (trials, side) in TRIALS.items():
                for trial in trials:
                    try:
                        r, s, n = eval_trial(part, trial, side, model, a.window,
                                             mode2d=mode2d, method=method)
                    except Exception as e:
                        print(f"  {part} {trial}: ERR {e}", flush=True)
                        continue
                    tag = trial.split("_")[1]
                    print(f"  {part} t{tag:>2}  n={n:4d}  "
                          f"jit {r['jitter']:7.0f}->{s['jitter']:6.0f}  "
                          f"spd {r['speed_corr']:+.3f}->{s['speed_corr']:+.3f}  "
                          f"rep {r['reproduction']:5.1f}->{s['reproduction']:5.1f}  "
                          f"pvE {r['peak_vel_err']:+5.0f}->{s['peak_vel_err']:+5.0f}", flush=True)
                    rows.append((r, s))

            def agg(key, i, absol=False):
                v = [(abs(row[i][key]) if absol else row[i][key])
                     for row in rows if np.isfinite(row[i][key])]
                return np.median(v) if v else np.nan
            print(f"  {'-'*74}", flush=True)
            print(f"  MEDIAN      jit {agg('jitter',0):7.0f}->{agg('jitter',1):6.0f}  "
                  f"spd {agg('speed_corr',0):+.3f}->{agg('speed_corr',1):+.3f}  "
                  f"rep {agg('reproduction',0):5.1f}->{agg('reproduction',1):5.1f}  "
                  f"|pvE| {agg('peak_vel_err',0,True):4.0f}->{agg('peak_vel_err',1,True):4.0f}  "
                  f"(n={len(rows)})", flush=True)


if __name__ == "__main__":
    main()
