"""SynBody multi-view Dataset — pretraining corpus for the CDRNet lifter (in-frame GT -> MPJPE).

100 synthetic sequences x 300 frames x 8 synchronized cameras, perfect 3D joints + exact calib
(all one world frame). A sample = one (seq, frame) with a RANDOM 3-6 view subset (the plan's
view-invariance trick: forces features that work on any camera subset, so our unseen 5-cam DELTA
rig is "just another subset"). Distill target = the PROJECTED GT joints (synthetic -> exact 2D),
so this doubles as the 2D-MSE teacher AND the 3D MPJPE truth.

Mirrors dataset.py's collate (variable #cams -> list). Images: img_extracted/<seq>/img/<cam>/<f>.jpg
resized to 640. GT from cache/synbody_gt/<seq>.npz (joints, P native-res, kpt_idx -> 17 joints).
"""
import os, glob
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
from cup_task.mv3d.imgproc import prep_view

IMG_ROOT = '/home/imove/Documents/mv3d_data/synbody/HumanNeRF-subset/img_extracted'
GT_ROOT = '/home/imove/Documents/cup-task/cache/synbody_gt'
IMSZ = 640


def _list_seqs():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{GT_ROOT}/*.npz'))


class SynBodyViews(Dataset):
    def __init__(self, seqs=None, min_v=3, max_v=6, frame_stride=3, augment=False):
        self.seqs = seqs if seqs is not None else _list_seqs()
        self.min_v, self.max_v = min_v, max_v
        self.augment = augment
        self.gt = {s: dict(np.load(f'{GT_ROOT}/{s}.npz', allow_pickle=True)) for s in self.seqs}
        self.index = []                                   # (seq, frame)
        for s in self.seqs:
            T = self.gt[s]['joints'].shape[0]
            for f in range(0, T, frame_stride):
                self.index.append((s, f))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, f = self.index[i]
        g = self.gt[s]
        cam_names = g['cam_names']; nC = len(cam_names)
        kpt_idx = g['kpt_idx']                            # (17,) SMPL-X joint indices
        J = g['joints'][f][kpt_idx].astype(np.float32)    # (17,3) world-frame GT 3D
        # random 3..6 view subset
        k = np.random.randint(self.min_v, min(self.max_v, nC) + 1)
        vsel = sorted(np.random.choice(nC, k, replace=False).tolist())
        imgs, Ps, kp2d = [], [], []
        for v in vsel:
            camdir = str(cam_names[v]).replace('camera', '')   # 'camera02' -> '02'
            fp = f'{IMG_ROOT}/{s}/img/{camdir}/{f:04d}.jpg'
            im = cv2.imread(fp)                            # native BGR
            if im is None:
                continue
            P = g['P'][v].astype(np.float32)              # (3,4) native res
            # projected GT joints = exact 2D teacher (native px)
            Xh = np.concatenate([J, np.ones((17, 1), np.float32)], 1)
            uvw = (P @ Xh.T).T
            uv_nat = (uvw[:, :2] / uvw[:, 2:3].clip(1e-6)).astype(np.float32)   # (17,2) native
            # SHARED letterbox: img + P + 2D all -> input-image space (rig-agnostic, like YOLO)
            t, P_in, kp_in = prep_view(im, P, IMSZ, kp_native=uv_nat, augment=self.augment)
            imgs.append(t); Ps.append(P_in); kp2d.append(kp_in)
        if len(imgs) < self.min_v:
            return None
        return {
            'imgs': torch.stack(imgs),
            'P_native': torch.stack(Ps),                  # input-image-space P (name kept for callers)
            'kp2d_tgt': torch.stack(kp2d),                # (V,17,2) exact projected GT, input-img px
            'kp_conf': torch.ones(len(imgs), 17),         # synthetic -> all visible, conf 1
            'X_gt': torch.from_numpy(J),                  # (17,3) in-frame 3D truth for MPJPE
            '_key': (s, f, tuple(vsel)),
        }


def collate_group(batch):
    return [b for b in batch if b is not None]


def make_synbody_loader(seqs=None, batch=8, workers=6, shuffle=True, min_v=3, max_v=6, frame_stride=3, augment=False):
    ds = SynBodyViews(seqs, min_v, max_v, frame_stride, augment=augment)
    dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True,
                    collate_fn=collate_group, persistent_workers=workers > 0, drop_last=False)
    return ds, dl


def split_seqs(val_frac=0.1):
    seqs = _list_seqs()
    nval = max(1, int(len(seqs) * val_frac))
    return seqs[nval:], seqs[:nval]                       # train, val (held-out sequences)
