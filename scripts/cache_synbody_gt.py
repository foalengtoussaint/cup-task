"""Pre-cache SynBody 3D-joint GT + camera projection matrices (one-time, OFF the training loop).

SynBody ships SMPL-X PARAMS, not joints. Forwarding SMPL-X is expensive -> do it ONCE here per
sequence (300 frames), save joints (T,J,3) world-frame + per-view P (K[R|t]) to a compact .npz.
The training loop then just loads joints + P + reads JPEGs (no SMPL-X, no body model on GPU).

Perfect in-frame GT: SynBody is synthetic, so joints and cameras share ONE world frame -> plain
MPJPE (the CDRNet metric), no MMC/OMC alignment needed. Verified: pelvis projects dead-centre in
all 8 cams.

Output: cache/synbody_gt/<seq>.npz  with joints (T,127,3), P (V,3,4), cam_names, img_size.
"""
import sys, os, glob, json
sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, torch, smplx

ROOT = '/home/imove/Documents/mv3d_data/synbody/HumanNeRF-subset'
MODELDIR = '/home/imove/Documents/iMOVE/DEV/isr-containers/smpl_data/data/smplx'
OUT = '/home/imove/Documents/cup-task/cache/synbody_gt'
os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# COCO-17-ish subset of SMPL-X joints for our 17-kpt head (SMPL-X joint indices).
# SMPL-X body joints: 0 pelvis,15 head,16/17 shoulders,18/19 elbows,20/21 wrists,1/2 hips,
#   4/5 knees,7/8 ankles. Map to a stable 17-set (approx COCO order used elsewhere).
SMPLX_KPT = [15, 15, 15, 15, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]  # head-ish for face slots


def P_of(cam):
    K, R, T = np.array(cam['K']), np.array(cam['R']), np.array(cam['T'])
    return (K @ np.hstack([R, T.reshape(3, 1)])).astype(np.float32)


def cache_seq(seq):
    out = f'{OUT}/{seq}.npz'
    if os.path.exists(out):
        print(f'  {seq}: exists, skip', flush=True); return
    d = np.load(f'{ROOT}/smplx/{seq}/smplx.npz', allow_pickle=True)
    sm = d['smplx'].item()
    gender = str(d['meta'].item().get('gender', 'neutral'))
    gender = gender if gender in ('male', 'female') else 'neutral'
    T = sm['body_pose'].shape[0]
    model = smplx.create(MODELDIR, model_type='smplx', gender=gender, use_pca=False,
                         num_betas=10, batch_size=T, flat_hand_mean=True).to(dev)
    with torch.no_grad():
        out_m = model(
            betas=torch.tensor(np.repeat(sm['betas'], T, 0), dtype=torch.float32, device=dev),
            global_orient=torch.tensor(sm['global_orient'], dtype=torch.float32, device=dev),
            body_pose=torch.tensor(sm['body_pose'], dtype=torch.float32, device=dev),
            transl=torch.tensor(sm['transl'], dtype=torch.float32, device=dev),
            left_hand_pose=torch.tensor(sm['left_hand_pose'], dtype=torch.float32, device=dev),
            right_hand_pose=torch.tensor(sm['right_hand_pose'], dtype=torch.float32, device=dev))
    joints = out_m.joints[:, :127, :].cpu().numpy().astype(np.float32)   # (T,127,3) world frame
    cams = json.load(open(f'{ROOT}/camera_params/{seq}/camera.json'))
    cam_names = sorted(cams)
    P = np.stack([P_of(cams[c]) for c in cam_names])                     # (V,3,4)
    K0 = np.array(cams[cam_names[0]]['K'])
    img_size = (int(K0[0, 2] * 2), int(K0[1, 2] * 2))
    np.savez(out, joints=joints, P=P, cam_names=np.array(cam_names),
             img_size=np.array(img_size), kpt_idx=np.array(SMPLX_KPT))
    print(f'  {seq}: joints {joints.shape} {len(cam_names)}cams img {img_size} -> saved', flush=True)


if __name__ == '__main__':
    seqs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(f'{ROOT}/smplx/*/smplx.npz'))
    print(f'{len(seqs)} sequences', flush=True)
    for s in seqs:
        cache_seq(s)
    print('DONE synbody GT cache', flush=True)
