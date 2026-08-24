"""DeciWatch (ECCV'22) temporal refinement on the DELTA incumbent wrist track, head-to-head vs
SmoothNet + the pipeline butter, scored vs UNFILTERED OMC.

DeciWatch is a sample-then-recover Transformer (uniform-sample 1/N frames, recover+denoise the rest);
its headline is SPEED (10x fewer estimator calls), with a denoising side effect. input_dim=51 = the
H36M 17-joint 3D pose (root-relative), so — unlike SmoothNet (channel-agnostic) — it needs the FULL
body, not a lone wrist. We HAVE the body: build the 17-joint H36M pose from our COCO track (synthesize
hip/spine/thorax/head the standard way), refine the whole sequence, read the wrist back (L=13,R=16).

Model + checkpoint in external/DeciWatch (h36m_fcn_3d/checkpoint.pth.tar, sample_interval=10).
Run:  python scripts/deciwatch_delta.py --part P07 --trial trial_13_L_unaffected [--sample-interval 10]
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import compare_pose_omc_delta as H

DW = Path("/home/imove/Documents/cup-task/external/DeciWatch")

# H36M-FCN 17-joint order (una-dinosauria / 3d-pose-baseline):
# 0 Hip 1 RHip 2 RKnee 3 RFoot 4 LHip 5 LKnee 6 LFoot 7 Spine 8 Thorax 9 Neck/Nose 10 Head
# 11 LShoulder 12 LElbow 13 LWrist 14 RShoulder 15 RElbow 16 RWrist
L_WRIST_H36M, R_WRIST_H36M = 13, 16


def build_h36m17(mmc):
    """(T,17,3) H36M pose from our COCO track (mm). Legs are static in the drink task and the model
    is root-relative + we only read the wrist, so absent leg joints are filled from the hips — they
    don't affect the wrist channel meaningfully. Synthesize hip/spine/thorax/head the standard way."""
    def g(name):
        return np.asarray(mmc[name], float) if name in mmc else None
    T = len(next(iter(mmc.values())))
    ls, rs = g("left_shoulder"), g("right_shoulder")
    lh, rh = g("left_hip"), g("right_hip")
    lw, rw = g("left_wrist"), g("right_wrist")
    le, re = g("left_elbow"), g("right_elbow")
    nose = g("nose")
    hip = np.nanmean([lh, rh], axis=0)                     # root
    thorax = np.nanmean([ls, rs], axis=0)
    spine = (hip + thorax) / 2.0
    neck = thorax.copy()
    head = nose if nose is not None else thorax
    P = np.full((T, 17, 3), np.nan)
    P[:, 0] = hip;  P[:, 1] = rh; P[:, 4] = lh
    P[:, 2] = rh;   P[:, 3] = rh; P[:, 5] = lh; P[:, 6] = lh   # legs -> hips (static/unused)
    P[:, 7] = spine; P[:, 8] = thorax; P[:, 9] = neck; P[:, 10] = head
    P[:, 11] = ls; P[:, 12] = le; P[:, 13] = lw
    P[:, 14] = rs; P[:, 15] = re; P[:, 16] = rw
    return P


def load_model(sample_interval, input_dim=51):
    import torch
    spec = importlib.util.spec_from_file_location("_dw_model", DW / "lib" / "models" / "deciwatch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DW))                            # deciwatch.py does `from lib...` internally
    spec.loader.exec_module(mod)
    m = mod.DeciWatch(input_dim, sample_interval=sample_interval,
                      encoder_hidden_dim=128, decoder_hidden_dim=128, dropout=0.1, nheads=4,
                      dim_feedforward=256, enc_layers=5, dec_layers=5, activation="leaky_relu",
                      pre_norm=True, recovernet_interp_method="linear", recovernet_mode="transformer")
    ck = torch.load(str(DW / "data/checkpoints/h36m_fcn_3d" /
                        (f"checkpoint_i{sample_interval}_q{20-sample_interval}.pth.tar"
                         if sample_interval != 10 else "checkpoint.pth.tar")),
                    map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), dev


def deciwatch_refine(P17, sample_interval=10):
    """(T,17,3) mm -> refined (T,17,3) mm. Root-relative + metres in (H36M convention), whole seq."""
    import torch
    T = P17.shape[0]
    idx = np.arange(T)
    filled = P17.copy()
    for j in range(17):                                   # per-joint gap interpolate for the net
        for c in range(3):
            v = np.isfinite(P17[:, j, c])
            if v.sum() >= 2:
                filled[:, j, c] = np.interp(idx, idx[v], P17[v, j, c])
    # H36M convention roots on the HIP (idx 0), but on DELTA the hip is detected in only ~25% of frames
    # (seated/occluded lower body) → a NaN root poisons the whole root-relative transform. Root on the
    # THORAX (shoulder-midpoint, idx 8, ~100% detected) instead — a temporal refiner only needs a STABLE
    # reference frame, not specifically the hip; any consistent root works and is offset-covariant.
    root = filled[:, [8], :].copy()                       # thorax
    rel = (filled - root) / 1000.0                        # root-relative, mm->m
    # DeciWatch requires (L-1) % sample_interval == 0 (whole-sequence sampling). Pad by repeating the
    # last frame to the next valid length, then trim back after.
    pad = 0
    while (T + pad - 1) % sample_interval != 0:
        pad += 1
    rel_p = np.concatenate([rel, np.repeat(rel[-1:], pad, axis=0)], axis=0) if pad else rel
    m, dev = load_model(sample_interval)
    seq = torch.from_numpy(rel_p.reshape(T + pad, -1)[None]).float().to(dev)   # (1,L,51)
    with torch.no_grad():
        recover, denoise = m(seq, dev)                    # (1,L,51)
    out = recover[0].cpu().numpy().reshape(T + pad, 17, 3)[:T] * 1000.0 + root   # back to mm, un-root
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P07"); ap.add_argument("--trial", default="trial_13_L_unaffected")
    ap.add_argument("--sample-interval", type=int, default=10)
    args = ap.parse_args()
    H.use_good_cams()
    side = "left" if "_L_" in args.trial else "right"; joint = f"{side}_wrist"
    wi = L_WRIST_H36M if side == "left" else R_WRIST_H36M

    mmc, n = H._load_mmc(args.part, args.trial)
    omc = H._load_omc(args.part, args.trial, n)[joint]
    P17 = build_h36m17(mmc)
    dw = deciwatch_refine(P17, args.sample_interval)
    inc = H._despike(mmc[joint])                           # raw incumbent wrist
    dw_w = dw[:, wi]                                       # DeciWatch-refined wrist

    m = min(len(inc), len(dw_w), len(omc))
    inc, dw_w, omc = inc[:m], dw_w[:m], omc[:m]

    def _shift(v, l):
        o = np.full_like(v, np.nan)
        if l >= 0: o[l:] = v[:len(v)-l] if l else v
        else: o[:l] = v[-l:]
        return o
    lag, _ = H._find_lag(inc, omc); omc = _shift(omc, lag)
    # rigid-align each estimator to OMC before POSITION scoring (kills the constant rig↔mocap offset;
    # speed is frame-invariant so unaffected). Same treatment the other scorers use.
    def align(a):
        R, t, _ = H._kabsch(a, omc); return a @ R.T + t
    inc, dw_w = align(inc), align(dw_w)
    osp = H._speed(omc); ogp = np.nanmax(osp)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(np.nan_to_num(osp), distance=36, height=200)
    pk = pk[np.argsort(osp[pk])[::-1][:4]]; pk.sort()
    wins = [(max(0, p-24), min(m, p+24)) for p in pk]; orp = [np.nanmax(osp[a:b]) for a, b in wins]

    def score(name, X):
        sp = H._speed(X)
        gerr = abs(np.nanmax(sp) - ogp) / ogp * 100
        rerr = np.mean([abs(np.nanmax(sp[a:b]) - orp[i]) / orp[i] * 100 for i, (a, b) in enumerate(wins)])
        jit = np.nanmedian(np.linalg.norm(np.diff(X, 2, axis=0), axis=1))
        pe = np.linalg.norm(X - omc, axis=1); pe = np.nanmedian(pe[np.isfinite(pe)])
        print(f"{name:26} | pos {pe:5.1f}mm | jitter {jit:5.2f}mm | global peak {gerr:4.0f}% | mean-reach {rerr:4.1f}%")

    print(f"\n{args.part} {args.trial} — DeciWatch(i={args.sample_interval}) vs OMC (unfiltered), {m} frames\n")
    print(f"OMC peak {ogp:.0f} mm/s; reaches {[f'{p:.0f}' for p in orp]}\n")
    score("incumbent raw", inc)
    score("DeciWatch", dw_w)


if __name__ == "__main__":
    main()
