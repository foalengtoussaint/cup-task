"""Run MVGFormer (CVPR'24 multi-view-geometry pose transformer) on the DELTA rig.

MVGFormer targets OCCLUSION via volumetric multi-view feature fusion (a DIFFERENT problem
than the fast-frame wrist-SPEED thread). This adapter feeds our 5-camera DELTA calib + staged
clips into MVGFormer's forward(views, meta) contract and extracts the tracked-side wrist 3D
trajectory, so it can be compared to the YOLO-pose + robust-triangulation incumbent and to OMC.

MVGFormer lives (un-committed) in the scratchpad clone; the CUDA op `Deformable` is built into
the object_tracking env. Config = the panoptic q1024 yaml with DELTA overrides (5 cams, space
volume recentred on the capture region). Camera convention conversion (verified against
lib/dataset/panoptic.py):
    MVGFormer projects xcam = R (X - T), T = camera-CENTER in world (mm), standard_T = raw t.
    DELTA CamCalib: Xc = R X + t (t in metres in the toml; _load_calib_mm scales to mm).
    => T_mvg = -R.T @ t_mm ; standard_T = t_mm ; R_mvg = R ; fx,fy,cx,cy from K.
Run with:  MVG=<scratchpad>/MVGFormer python scripts/mvgformer_delta.py --part P07 --trial trial_13_L_unaffected [--smoke]
"""
import os, sys, argparse, glob
from pathlib import Path
import numpy as np
import cv2
import torch

MVG = Path(os.environ.get("MVG", "")).expanduser()
if not MVG.exists():
    sys.exit("set MVG=/path/to/MVGFormer clone (the scratchpad clone with the built Deformable op)")
sys.path.insert(0, str(MVG))               # repo root: `import lib.utils...`
sys.path.insert(0, str(MVG / "lib"))       # bare `import core...`, `import models...`
sys.path.insert(0, str(MVG / "run"))       # _init_paths side-effects
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")

import compare_pose_omc_delta as H  # DELTA calib/omc/sync helpers (reused, no reimplementation)

# mmcv is imported by lib/models/util/misc.py ONLY for get_dist_info + distributed result-gather
# (collect_results), neither of which single-process inference touches. Stub it so we avoid the
# finicky mmcv<->torch2.7 build. If distributed gather were ever needed this would have to go.
import types
_mmcv = types.ModuleType("mmcv")
_mmcv.mkdir_or_exist = lambda *a, **k: None
_mmcv.dump = lambda *a, **k: None
_mmcv.load = lambda *a, **k: None
_runner = types.ModuleType("mmcv.runner")
_runner.get_dist_info = lambda: (0, 1)
_mmcv.runner = _runner
sys.modules["mmcv"] = _mmcv
sys.modules["mmcv.runner"] = _runner

# --- MVGFormer internals -----------------------------------------------------------------------
from core.config import config as cfg, update_config
import models
import models.dq_transformer
from utils.transforms import get_affine_transform, get_scale

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_JOINTS = 15
L_WRIST, R_WRIST = 5, 11   # MVGFormer 15-joint skeleton (panoptic JOINTS_DEF)


def build_model(cfg_path, weights, backbone, device, space_center=None, space_size=None):
    update_config(str(cfg_path))
    cfg.NETWORK.PRETRAINED = str(backbone)   # get_pose_net loads this when is_train=True
    cfg.DECODER.t_pose_dir = str(MVG / "tpose.pt")   # repo-relative default breaks from our cwd
    cfg.DATASET.CAMERA_NUM = 5
    # sample_space query init distributes root queries over SPACE_CENTER +- SPACE_SIZE/2; the Panoptic
    # default (8m cube @ origin) is both mis-centred and too coarse for DELTA's ~2m capture volume, so
    # every query misses the subject and scores ~0. Recentre+shrink onto the actual subject (mm).
    # These are read in the decoder __init__, so must be set BEFORE get_mvp.
    if space_center is not None:
        cfg.MULTI_PERSON.SPACE_CENTER = [float(v) for v in space_center]
    if space_size is not None:
        cfg.MULTI_PERSON.SPACE_SIZE = [float(v) for v in space_size]
    model = models.dq_transformer.get_mvp(cfg, is_train=True)  # is_train=True => backbone weights load
    sd = torch.load(str(weights), map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[weights] loaded; {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model.gt_match_test = False          # inference: sample_space init, no GT matching
    model.log_val_loss = False           # else forward runs the GT criterion/matcher (needs real GT)
    return model.to(device).eval()


def delta_cam_to_mvg(c):
    """CamCalib (R world->cam, t mm world->cam, K rescaled) -> MVGFormer camera dict."""
    R = np.asarray(c.R, dtype=np.float32)
    t = np.asarray(c.t, dtype=np.float32).reshape(3)          # mm
    T = (-R.T @ t).reshape(3, 1)                              # camera centre in world (mm)
    fx, fy, cx, cy = float(c.K[0, 0]), float(c.K[1, 1]), float(c.K[0, 2]), float(c.K[1, 2])
    d = np.asarray(c.dist, dtype=np.float32).reshape(-1)      # k1,k2,p1,p2,k3
    k = np.array([[d[0]], [d[1]], [d[4] if d.size > 4 else 0.0]], dtype=np.float32)
    p = np.array([[d[2]], [d[3]]], dtype=np.float32)
    return dict(R=R, T=T, standard_T=t.reshape(3, 1),
                fx=np.float32(fx), fy=np.float32(fy), cx=np.float32(cx), cy=np.float32(cy),
                k=k, p=p)


def make_meta(cam_mvg, W, H_, image_size, device):
    """One view's meta dict. Affine full-image -> image_size (960x512), like JointsDataset."""
    c = np.array([W / 2.0, H_ / 2.0], dtype=np.float32)
    s = get_scale((W, H_), image_size)
    trans = get_affine_transform(c, s, 0, image_size, inv=0)
    inv_trans = get_affine_transform(c, s, 0, image_size, inv=1)
    aff = np.eye(3, dtype=np.float32); aff[:2] = trans
    inv_aff = np.eye(3, dtype=np.float32); inv_aff[:2] = inv_trans
    cam = cam_mvg
    intr = np.eye(3, dtype=np.float32)
    intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2] = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    def T(x, dt=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dt, device=device).unsqueeze(0)  # batch dim
    # single-person placeholders (sample_space init reads joints_3d/vp_pred shapes only)
    j3d = np.zeros((1, NUM_JOINTS, 3), np.float32)
    j3d_vis = np.zeros((1, NUM_JOINTS, 3), np.float32)
    vp = np.zeros((1, NUM_JOINTS, 5), np.float32)
    roots = np.zeros((1, 3), np.float32)
    return {
        "image": "delta",
        "num_person": T(np.array([1])),
        "joints_3d": T(j3d), "joints_3d_vis": T(j3d_vis),
        "joints_3d_voxelpose_pred": T(vp), "roots_3d": T(roots),
        "center": T(c), "scale": T(s), "rotation": T(np.float32(0)),
        "camera": {kk: T(vv) for kk, vv in cam.items()},
        "camera_Intri": T(intr), "camera_R": T(cam["R"]),
        "camera_focal": T(np.stack([cam["fx"], cam["fy"], np.float32(1.0)])),
        "camera_T": T(cam["T"]), "camera_standard_T": T(cam["standard_T"]),
        "affine_trans": T(aff), "inv_affine_trans": T(inv_aff), "aug_trans": T(inv_aff),
    }


def preprocess(frame_bgr, W, H_, image_size):
    c = np.array([W / 2.0, H_ / 2.0], dtype=np.float32)
    s = get_scale((W, H_), image_size)
    trans = get_affine_transform(c, s, 0, image_size, inv=0)
    warped = cv2.warpAffine(frame_bgr, trans, (int(image_size[0]), int(image_size[1])),
                            flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))  # (3,H,W)


def run_frame(model, views, metas, device, threshold=0.0):
    with torch.no_grad():
        out = model(views=views, meta=metas, threshold=threshold)
        if isinstance(out, tuple):
            out = out[0]
    bs, num_q = out["pred_logits"].shape[:2]
    poses = out["pred_poses"]["outputs_coord"].view(bs, num_q, NUM_JOINTS, 3)
    score = out["pred_logits"][:, :, 1].sigmoid()            # (bs, num_q)
    best = score[0].argmax().item()
    return poses[0, best].cpu().numpy(), float(score[0, best])  # (15,3) mm, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P07")
    ap.add_argument("--trial", default="trial_13_L_unaffected")
    ap.add_argument("--cfg", default=str(MVG / "configs/panoptic/knn5-lr4-q1024.yaml"))
    ap.add_argument("--weights", default=str(MVG / "models/mvgformer_q1024_model.pth.tar"))
    ap.add_argument("--backbone", default=str(MVG / "models/pose_resnet50_panoptic.pth.tar"))
    ap.add_argument("--smoke", action="store_true", help="one frame only, print shapes + wrist")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cams", default="", help="comma list of cam numbers to RESTRICT to, e.g. 1,3,5 "
                    "(camera-dropout / occlusion-robustness test); default = all good staged cams")
    ap.add_argument("--tag", default="", help="suffix on the output npz (distinguishes dropout runs)")
    args = ap.parse_args()

    device = torch.device("cuda")
    image_size = np.array([960, 512])
    H.use_good_cams()

    # Space cube: centre on the subject (from the incumbent triangulated body), size to fit a person.
    # This only sets WHERE the coarse queries are seeded — not what MVGFormer predicts.
    mmc0, _ = H._load_mmc(args.part, args.trial)
    body = np.concatenate([mmc0[k] for k in ("left_shoulder", "right_shoulder", "left_hip",
                           "right_hip", "left_wrist", "right_wrist") if k in mmc0], 0)
    ctr = np.nanmean(body, 0)
    space_center = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
    space_size = [3000.0, 3000.0, 2000.0]
    print(f"[space] centre(mm)={np.round(space_center)}  size={space_size}", flush=True)
    model = build_model(args.cfg, args.weights, args.backbone, device,
                        space_center=space_center, space_size=space_size)

    # DELTA volume: set the space cube from the good cams' centres (recentre from panoptic default).
    cams = H._load_calib_mm(args.part)
    good = H.GOOD_CAMS.get(args.part)
    if good:
        cams = {c: v for c, v in cams.items() if c in good}
    # staged clips  delta_<part>_<trial>.<n>.mp4  (n = cam number); keep only good cams that have one.
    def clip_path(c):
        return H.DELTA / args.part / "staged" / f"delta_{args.part}_{args.trial}.{c.split('_')[1]}.mp4"
    cam_names = [c for c in sorted(cams) if clip_path(c).exists()]
    if args.cams:                                    # camera-dropout test: restrict to a subset
        want = {f"cam_{x.strip()}" for x in args.cams.split(",")}
        cam_names = [c for c in cam_names if c in want]
    if not cam_names:
        sys.exit(f"no staged clips for {args.part} {args.trial}")
    cams = {c: cams[c] for c in cam_names}
    centres = np.array([(-v.R.T @ v.t) for v in cams.values()])  # mm
    print(f"[{args.part}] cams={cam_names}  centroid(mm)={centres.mean(0).round(0)}", flush=True)
    clips = {c: clip_path(c) for c in cam_names}
    caps = {c: cv2.VideoCapture(str(clips[c])) for c in cam_names}
    for c in cam_names:
        if not caps[c].isOpened():
            sys.exit(f"cannot open {clips[c]}")
    Wc = {c: int(caps[c].get(cv2.CAP_PROP_FRAME_WIDTH)) for c in cam_names}
    Hc = {c: int(caps[c].get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in cam_names}
    nfr = min(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in cam_names)
    print(f"[{args.part} {args.trial}] {len(cam_names)} cams, {nfr} frames, sizes={set(Wc.values())}x{set(Hc.values())}", flush=True)

    cam_mvg = {c: delta_cam_to_mvg(cams[c]) for c in cam_names}
    metas_static = [make_meta(cam_mvg[c], Wc[c], Hc[c], image_size, device) for c in cam_names]

    side = "left" if "_L_" in args.trial else "right"
    widx = L_WRIST if side == "left" else R_WRIST
    limit = args.max_frames or (1 if args.smoke else nfr)

    wrist = np.full((nfr, 3), np.nan, np.float32)
    scores = np.full(nfr, np.nan, np.float32)
    import time; t0 = time.time()
    for f in range(limit):
        views = []
        ok = True
        for c in cam_names:
            r, fr = caps[c].read()
            if not r:
                ok = False; break
            views.append(preprocess(fr, Wc[c], Hc[c], image_size).unsqueeze(0).to(device))
        if not ok:
            break
        pose15, sc = run_frame(model, views, metas_static, device)
        wrist[f] = pose15[widx]; scores[f] = sc
        if args.smoke:
            print(f"[smoke] pred pose (15,3) mm:\n{pose15.round(0)}")
            print(f"[smoke] best-query score={sc:.3f}  {side}-wrist(mm)={pose15[widx].round(1)}", flush=True)
            return
        if f in (5, 10) or (f % 25 == 0 and f):
            fps = f / (time.time() - t0)
            print(f"  frame {f}/{limit}  {fps:.1f} fps  ETA {(limit-f)/max(fps,1e-3):.0f}s  score~{sc:.2f}", flush=True)
    for c in cam_names:
        caps[c].release()

    outdir = Path("/home/imove/Documents/cup-task/cache/mvgformer"); outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{args.part}_{args.trial}{('_' + args.tag) if args.tag else ''}.npz"
    np.savez(outp, wrist=wrist, score=scores, side=side, joint_idx=widx, cams=cam_names)
    print(f"[saved] {outp}  valid frames={np.isfinite(wrist).all(1).sum()}/{nfr}", flush=True)


if __name__ == "__main__":
    main()
