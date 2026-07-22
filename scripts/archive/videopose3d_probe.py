"""VideoPose3D lifter probe: does a learned monocular motion prior do anything USEFUL on our data,
especially at the drink-apex occlusion where multi-view triangulation loses cameras?

⚠ KNOWN LIMITATION (quantified before running): VideoPose3D's pretrained h36m_cpn model needs the FULL
17-joint H36M skeleton. We only capture 11 COCO joints (upper body + hips + head). The 6 missing joints:
Spine + Thorax (derivable = hip-mid / shoulder-mid) and RKnee/RFoot/LKnee/LFoot (NOT capturable -- our
participants are upper-body framed). We fill the legs with a STATIC NEUTRAL pose. This is off-distribution
for the model's all-joint temporal conv, so the probe is to SEE whether the wrist survives the fabrication,
not to trust it blindly. We look at the output, per the "never conclude from a scalar" rule.

Per camera: build the 17-joint 2D sequence, normalise to VideoPose3D's screen coords, run the temporal
model -> per-camera 3D (root-relative, up to scale). Compare the WRIST vs OMC (Kabsch-aligned shape) and
vs our triangulation, phase-by-phase, focusing on the apex.

    python scripts/videopose3d_probe.py --part P07 --trial trial_10_L_unaffected --cam cam_1
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "VideoPose3D"))
import compare_pose_omc_delta as H
import smoothnet_pose_delta as S
from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

CK = ("/tmp/claude-1000/-home-imove-Documents-object-tracking/"
      "25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/vp3d_ckpts/pretrained_h36m_cpn.bin")
IMG_W, IMG_H = 1920, 1080
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# H36M 17 order: Hip RHip RKnee RFoot LHip LKnee LFoot Spine Thorax Neck/Nose Head
#                LShoulder LElbow LWrist RShoulder RElbow RWrist
H36M = ["Hip","RHip","RKnee","RFoot","LHip","LKnee","LFoot","Spine","Thorax","Neck","Head",
        "LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]
# COCO joint name feeding each H36M slot (None = fabricate)
COCO_OF = {"RHip":"right_hip","LHip":"left_hip","LShoulder":"left_shoulder","RShoulder":"right_shoulder",
           "LElbow":"left_elbow","RElbow":"right_elbow","LWrist":"left_wrist","RWrist":"right_wrist",
           "Neck":None,"Head":"nose","Hip":None,"Spine":None,"Thorax":None,
           "RKnee":None,"RFoot":None,"LKnee":None,"LFoot":None}


def build_h36m_2d(frames):
    """(T,17,2) H36M-order 2D from our COCO frames. Derive Hip/Spine/Thorax/Neck; fake legs static."""
    T = len(frames)
    def kp(fr, name):
        k = fr.get("kps", {})
        return np.array(k[name][:2], float) if name in k else np.array([np.nan, np.nan])
    out = np.full((T, 17, 2), np.nan)
    for t, fr in enumerate(frames):
        g = {n: kp(fr, n) for n in ["left_hip","right_hip","left_shoulder","right_shoulder",
                                    "left_elbow","right_elbow","left_wrist","right_wrist","nose"]}
        hip_mid = np.nanmean([g["left_hip"], g["right_hip"]], axis=0)
        sh_mid = np.nanmean([g["left_shoulder"], g["right_shoulder"]], axis=0)
        derived = {"Hip": hip_mid, "Spine": (hip_mid + sh_mid) / 2, "Thorax": sh_mid,
                   "Neck": sh_mid * 0.5 + g["nose"] * 0.5}
        for j, name in enumerate(H36M):
            c = COCO_OF[name]
            if c is not None:
                out[t, j] = g[c]
            elif name in derived:
                out[t, j] = derived[name]
        # fabricate legs: knees below hips, feet below knees (static, plausible seated-ish scale)
        span = np.nanmax([np.linalg.norm(g["left_shoulder"] - g["left_hip"]),
                          np.linalg.norm(g["right_shoulder"] - g["right_hip"]), 60.0])
        for hipname, knee, foot in [("right_hip","RKnee","RFoot"), ("left_hip","LKnee","LFoot")]:
            out[t, H36M.index(knee)] = g[hipname] + [0, span]
            out[t, H36M.index(foot)] = g[hipname] + [0, 2 * span]
    return out


def lift(frames, model):
    x2d = build_h36m_2d(frames)                     # (T,17,2) pixels
    # interpolate NaNs per joint (model can't take NaN); keep a validity mask for the wrist
    T = x2d.shape[0]
    for j in range(17):
        for c in range(2):
            col = x2d[:, j, c]; good = np.isfinite(col)
            if good.sum() >= 2:
                x2d[:, j, c] = np.interp(np.arange(T), np.flatnonzero(good), col[good])
    xn = normalize_screen_coordinates(x2d, w=IMG_W, h=IMG_H).astype(np.float32)
    rf = model.receptive_field(); pad = rf // 2
    xp = np.pad(xn, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    with torch.no_grad():
        out = model(torch.from_numpy(xp[None]).to(DEV))[0].cpu().numpy()   # (T,17,3)
    return out


def load_model():
    m = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_pos"]); return m.eval().to(DEV)


def cams_for(part, trial):
    d = H.DELTA / part
    keep = H.GOOD_CAMS.get(part) if (H.GOOD_CAMS and part in H.GOOD_CAMS) else None
    per_cam = {}
    for pj in sorted(glob.glob(str(d / "dets" / f"*{trial}*.pose.json"))):
        cam = f"cam_{Path(pj).name.split('.')[1]}"
        if keep is None or cam in keep:
            per_cam[cam] = json.loads(Path(pj).read_text())["frames"]
    return per_cam


def batch(argv):
    """Head-to-head across the 18 c3d trials: VideoPose3D per-cam lift (best cam + mean-of-cams) vs
    our triangulation vs OMC. Metrics: OMC speed-corr + jitter (both scale-free-ish for the lift).
    Reproduction is NOT reported for the lift -- it is root-relative + arbitrary scale, so absolute mm
    vs OMC is meaningless; shape/speed is the fair axis."""
    m = load_model()
    print(f"{'trial':16} {'tri corr':>8} {'lift best':>9} {'lift mean':>9}   "
          f"{'tri jit':>8} {'lift jit':>9}", flush=True)
    rows = []
    for part, (trials, side) in S.TRIALS.items():
        for trial in trials:
            mmc, n = H._load_mmc(part, trial); omc = H._load_omc(part, trial, n)
            wr = f"{side}_wrist"
            tri_corr = S.speed_corr(mmc[wr], omc[wr]); tri_jit = S.jitter(mmc[wr])
            per_cam = cams_for(part, trial)
            wr_idx = H36M.index("LWrist" if side == "left" else "RWrist")
            lifts = []
            for frames in per_cam.values():
                y = lift(frames, m)[:, wr_idx, :]
                # pad/trim to n
                if len(y) < n: y = np.vstack([y, np.full((n - len(y), 3), np.nan)])
                lifts.append(y[:n] * 1000.0)   # arbitrary scale; corr is scale-free
            corrs = [S.speed_corr(w, omc[wr]) for w in lifts]
            best = np.nanmax(corrs) if corrs else np.nan
            meanw = np.nanmean(lifts, axis=0)
            mean_corr = S.speed_corr(meanw, omc[wr])
            lift_jit = np.nanmedian([S.jitter(w) for w in lifts])
            tag = f"{part}_{trial.split('_')[1]}"
            print(f"{tag:16} {tri_corr:+8.3f} {best:+9.3f} {mean_corr:+9.3f}   "
                  f"{tri_jit:8.0f} {lift_jit:9.0f}", flush=True)
            rows.append((tri_corr, best, mean_corr, tri_jit, lift_jit))
    R = np.array(rows)
    print(f"{'-'*62}", flush=True)
    print(f"{'MEDIAN':16} {np.nanmedian(R[:,0]):+8.3f} {np.nanmedian(R[:,1]):+9.3f} "
          f"{np.nanmedian(R[:,2]):+9.3f}   {np.nanmedian(R[:,3]):8.0f} {np.nanmedian(R[:,4]):9.0f}",
          flush=True)
    print("\n(tri = our triangulation; lift best = best single camera's VideoPose3D lift; "
          "lift mean = average of per-cam lifts. Speed-corr vs OMC; jitter mm/s^2.)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P07"); ap.add_argument("--trial", default="trial_10_L_unaffected")
    ap.add_argument("--side", default="left"); ap.add_argument("--render", action="store_true")
    ap.add_argument("--batch", action="store_true", help="run all 18 c3d trials head-to-head")
    a = ap.parse_args(argv)
    H.use_good_cams()
    if a.batch:
        return batch(argv)

    m = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_pos"]); m.eval().to(DEV)

    d = H.DELTA / a.part
    # match the harness: filter to good cams ONLY when this participant HAS an audit entry;
    # participants absent from the audit (P07/P08 = clean 5-cam) use all cameras.
    keep = H.GOOD_CAMS.get(a.part) if (H.GOOD_CAMS and a.part in H.GOOD_CAMS) else None
    per_cam = {}
    for pj in sorted(glob.glob(str(d / "dets" / f"*{a.trial}*.pose.json"))):
        cam = f"cam_{Path(pj).name.split('.')[1]}"
        if keep is None or cam in keep:
            per_cam[cam] = json.loads(Path(pj).read_text())["frames"]

    wr_idx = H36M.index("LWrist" if a.side == "left" else "RWrist")
    # our triangulation + OMC (Kabsch shape) for reference
    mmc, n = H._load_mmc(a.part, a.trial); omc = H._load_omc(a.part, a.trial, n)
    lag, _ = H._find_lag(mmc[f"{a.side}_wrist"], omc[f"{a.side}_wrist"])

    print(f"{a.part} {a.trial}  good cams: {list(per_cam)}  n={n}", flush=True)
    lifted = {}
    for ckey, frames in per_cam.items():
        y = lift(frames, m)                        # (T,17,3)
        lifted[ckey] = y[:, wr_idx, :]             # wrist xyz (root-relative, arbitrary scale)
        # speed profile of the lifted wrist, and its correlation to OMC wrist speed
        w = y[:, wr_idx, :]
        sc = S.speed_corr(w * 1000.0, omc[f"{a.side}_wrist"])  # scale-free corr, units cancel
        print(f"  {ckey}: lifted wrist speed-corr vs OMC = {sc:+.3f}", flush=True)

    if a.render:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        t = np.arange(n) / H.VIDEO_FPS
        # OMC + our triangulation wrist speed
        def shift(v, l):
            o = np.full_like(v, np.nan); o[l:] = v[:len(v)-l] if l else v; return o
        omcw = shift(omc[f"{a.side}_wrist"], lag) if lag >= 0 else omc[f"{a.side}_wrist"]
        axes[0].plot(t, H._lp(H._speed(omcw)), "k--", lw=1.8, label="OMC truth")
        axes[0].plot(t, H._lp(H._speed(mmc[f"{a.side}_wrist"])), lw=1.5, label="our triangulation")
        for ckey, w in lifted.items():
            sp = H._lp(H._speed(w))
            axes[0].plot(t[:len(sp)], sp / (np.nanmax(sp) + 1e-9) * np.nanmax(H._lp(H._speed(omcw))),
                         lw=1, alpha=.7, label=f"VP3D {ckey} (scaled)")
        axes[0].set_title(f"{a.part} {a.trial}: wrist SPEED — VideoPose3D per-cam lift vs triangulation vs OMC")
        axes[0].set_ylabel("mm/s (lift scaled)"); axes[0].legend(fontsize=7)
        # coverage: how many good cams have a 3D point per frame (our triangulation)
        cov = np.array([np.isfinite(mmc[f"{a.side}_wrist"][f]).all() for f in range(n)], float)
        axes[1].plot(t, cov, label="our triangulation has wrist (1/0)")
        axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("cov"); axes[1].legend(fontsize=8)
        out = ROOT / "out" / f"vp3d_probe_{a.part}_{a.trial}.png"; out.parent.mkdir(exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=110); print(f"rendered {out}", flush=True)


if __name__ == "__main__":
    main()
