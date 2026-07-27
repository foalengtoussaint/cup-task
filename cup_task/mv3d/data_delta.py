"""Single-person multi-view DELTA loader for the CDRNet lifter.

Per (trial, frame) sample provides:
  imgs      (V,3,640,640) RGB[0,1]   -- live-decoded staged videos, letterbox-free resize to 640
  P_native  (V,3,4)                  -- calib K[R|t] at NATIVE res (results_v3_delta._calib)
  kp2d_tgt  (V,K,2) + kp_conf (V,K)  -- YOLO Pose26 2D (the DISTILL TARGET), native px
  omc_wrist (3,) or None             -- AutoMQ filtered OMC wrist, EVAL ONLY (frame-aligned)
Only cameras with a cached pose.json are used (cam_7/cam_9 have none). V varies 3..5.

Reuses the proven loaders: compare_pose_omc_delta (H) for dets+DELTA paths, results_v3_delta (R)
for calib, AutoMQ pkl for filtered OMC truth. COCO-17 keypoint order (YOLO).
"""
import sys, json, os
sys.path.insert(0, '/home/imove/Documents/cup-task/scripts')
sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, cv2, torch
import pandas as pd
import compare_pose_omc_delta as H
import results_v3_delta as R
from cup_task.mv3d.imgproc import prep_view
H.use_good_cams()

COCO17 = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
          'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
          'left_knee','right_knee','left_ankle','right_ankle']
WRIST_IDX = COCO17.index('left_wrist')
IMSZ = 640
AMQ_PKL = '/home/imove/Documents/AutoMQ/P07/combined_data_with_kinematics.pkl'


def _pmat(cam_calib):
    c = cam_calib
    return c.K @ np.hstack([c.R, c.t.reshape(3, 1)])


class DeltaTrial:
    """Holds one trial's per-frame cache: dets(2D+conf), P, video handles, OMC wrist (eval)."""

    def __init__(self, part, trial, trial_num, amq=None):
        self.part, self.trial, self.tn = part, trial, trial_num
        self.calib = R._calib(part)
        # cams with a cached pose.json
        self.cams, self.KP = [], {}
        for c in self.calib:
            pj = H.DELTA/part/'dets'/f'delta_{part}_{trial}.{c.split("_")[1]}.pose.json'
            if not pj.exists():
                continue
            fr = json.loads(pj.read_text())['frames']
            arr = np.full((len(fr), 17, 3), np.nan, np.float32)     # x,y,conf per kpt
            for fi, f in enumerate(fr):
                kps = (f or {}).get('kps', {})
                for ki, name in enumerate(COCO17):
                    k = kps.get(name)
                    if k:
                        arr[fi, ki] = [k[0], k[1], k[2]]
            self.cams.append(c); self.KP[c] = arr
        self.P = {c: _pmat(self.calib[c]) for c in self.cams}
        self.size = {c: self.calib[c].size for c in self.cams}      # (W,H) native
        self.n = min(len(v) for v in self.KP.values())
        self._caps = {}
        # NATIVE-resolution flat frame cache (fast path); memmap per cam if present. Native (not 640)
        # so the shared letterbox in prep_view is applied consistently at load time. (The old squashed
        # _frames640 cache is incompatible with letterboxing and no longer used.)
        self._frames = {}
        fdir = f'{H.DELTA}/_frames_native'
        for c in self.cams:
            fp = f'{fdir}/{part}_{trial}_{c}.npy'
            if os.path.exists(fp):
                self._frames[c] = np.load(fp, mmap_mode='r')     # (n,H,W,3) uint8 BGR native, lazy
        # OMC wrist truth (eval only), frame-aligned via speed lag
        self.omc = None
        if amq is not None:
            self.omc = self._load_omc(amq)

    def _load_omc(self, amq):
        try:
            mmc, nn_ = H._load_mmc(self.part, self.trial)
            raw = H._load_omc(self.part, self.trial, nn_)['left_wrist']
            lag = H._find_lag(mmc['left_wrist'], raw)[0]
            return R._shift(raw, lag)                                # (n,3) OMC wrist, aligned
        except Exception:
            return None

    def _cap(self, cam):
        if cam not in self._caps:
            p = H.DELTA/self.part/'staged'/f'delta_{self.part}_{self.trial}.{cam.split("_")[1]}.mp4'
            self._caps[cam] = cv2.VideoCapture(str(p))
        return self._caps[cam]

    def _native_frame(self, cam, f):
        """Native-resolution BGR frame (H,W,3): from the native memmap cache, else live decode."""
        if cam in self._frames and f < len(self._frames[cam]):
            return np.asarray(self._frames[cam][f])
        cap = self._cap(cam); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
        return im if ok else None

    def valid_frames(self, min_cams=3, conf=0.3):
        out = []
        for f in range(self.n):
            v = sum(np.isfinite(self.KP[c][f, WRIST_IDX, 0]) and self.KP[c][f, WRIST_IDX, 2] > conf
                    for c in self.cams)
            if v >= min_cams:
                out.append(f)
        return out

    def sample(self, f, device="cpu", conf=0.3, augment=False):
        """Return dict for frame f. imgs/P_in/kp2d_tgt ALL in LETTERBOXED input-image space (shared
        imgproc.prep_view) -> rig-agnostic, matches how YOLO's backbone sees input."""
        cams_used, imgs, Ps, kp2d, kpc = [], [], [], [], []
        for c in self.cams:
            w = self.KP[c][f, WRIST_IDX]
            if not (np.isfinite(w[0]) and w[2] > conf):
                continue
            im = self._native_frame(c, f)                            # native BGR (H,W,3)
            if im is None:
                continue
            kp_nat = self.KP[c][f, :, :2]                            # (17,2) native px (nan where undet)
            t, P_in, kp_in = prep_view(im, self.P[c], IMSZ, kp_native=np.nan_to_num(kp_nat), augment=augment)
            # restore nan on originally-undetected kpts (prep_view nan_to_num'd them for the affine)
            bad = ~np.isfinite(kp_nat).all(-1)
            kp_in[bad] = float('nan')
            cams_used.append(c); imgs.append(t); Ps.append(P_in)
            kp2d.append(kp_in)
            kpc.append(torch.tensor(self.KP[c][f, :, 2], dtype=torch.float32))     # (17,)
        if len(cams_used) < 3:
            return None
        return {
            'imgs': torch.stack(imgs).to(device),
            'P_native': torch.stack(Ps).to(device),                  # NOW input-image-space P (name kept)
            'kp2d_tgt': torch.stack(kp2d).to(device),
            'kp_conf': torch.stack(kpc).to(device),
            'cams': cams_used,
            'omc_wrist': (torch.tensor(self.omc[f], dtype=torch.float32) if self.omc is not None
                          and np.isfinite(self.omc[f]).all() else None),
        }

    def release(self):
        for c in self._caps.values():
            c.release()
        self._caps = {}


def load_amq():
    return pd.read_pickle(AMQ_PKL)
