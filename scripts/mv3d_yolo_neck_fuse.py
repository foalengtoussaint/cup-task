"""REAL architecture: YOLO CSPDarknet neck features -> feature-sampled DLT -> 3D, backprop into neck.
Panoptic, 6 synced HD cams (already downloaded), real 3D GT + calib.

Per frame:
  - decode 6 cam images, run each through YOLO neck -> stride-8 feature map (128ch) per cam
  - for each 3D joint: project a COARSE estimate into each cam -> bilinear-sample neck feature there
  - a small head regresses (2D refine offset dx,dy + log-confidence) per (cam,joint) from the feature
  - weighted-DLT the refined 2D -> 3D  -> loss vs GT 3D
  - gradient flows through the sampling into the NECK (when unfrozen).

Stage 1: neck FROZEN (verify the head learns). Stage 2: unfreeze neck (the actual fine-tune).
Uses the coarse init = projected GT + noise (stand-in for a real detector prior; the head must refine).
"""
import os, glob, json, time, cv2
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ultralytics import YOLO
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
SEQ = '160422_ultimatum1'; ROOT = f'/home/imove/Documents/mv3d_data/panoptic/{SEQ}'
CAM_IDS = ['00_00', '00_03', '00_06', '00_12', '00_18', '00_23']   # the 6 downloaded HD cams
IMSZ = 640

# ---- calib: projection per downloaded cam (Panoptic cm; scale px to IMSZ later) ----
calib = json.load(open(f'{ROOT}/calibration_{SEQ}.json'))
byname = {c['name']: c for c in calib['cameras']}
CAMS = []
for cid in CAM_IDS:
    c = byname[f'00_{cid.split("_")[1]}']
    K = np.array(c['K'], np.float64); R = np.array(c['R'], np.float64); t = np.array(c['t'], np.float64).reshape(3, 1)
    res = c['resolution']            # [W,H] native
    CAMS.append({'K': K, 'R': R, 't': t, 'W': res[0], 'H': res[1]})
print(f'{len(CAMS)} cams')

# ---- YOLO neck feature extractor (tap layer 16: 128ch stride-8) ----
yolo = YOLO('models/yolo26s-pose.pt' if os.path.exists('models/yolo26s-pose.pt')
            else '/home/imove/Documents/cup-task/models/yolo26s-pose.pt')
net = yolo.model.to(dev).eval()
_feat = {}
net.model[16].register_forward_hook(lambda m, i, o: _feat.__setitem__('f', o))
def neck_feat(imgs):   # imgs (N,3,640,640) -> (N,128,80,80)
    _feat.clear(); net(imgs); return _feat['f']

# ---- data: frames with a single person, cache 3D + which cams see it ----
def load_frames(maxf=60):
    out = []
    for f in sorted(glob.glob(f'{ROOT}/hdPose3d_stage1_coco19/*.json')):
        d = json.load(open(f))
        if len(d['bodies']) == 1:
            j = np.array(d['bodies'][0]['joints19']).reshape(-1, 4)
            if j.shape[0] == 19 and (j[:, 3] > 0.1).all():
                fi = int(os.path.basename(f).split('_')[-1].split('.')[0])
                out.append((fi, j[:, :3].astype(np.float32)))
        if len(out) >= maxf: break
    return out
FR = load_frames(); print(f'{len(FR)} single-person frames')

caps = {cid: cv2.VideoCapture(f'{ROOT}/hdVideos/hd_{cid}.mp4') for cid in CAM_IDS}
def read_views(frame_idx):
    ims = []
    for cid in CAM_IDS:
        caps[cid].set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ok, im = caps[cid].read()
        if not ok: return None
        ims.append(im)
    return ims   # list of native-res BGR

def project(cam, X):    # X (...,3) cm -> px (native res)
    Xc = (cam['R'] @ X.T + cam['t'])                  # 3xN
    uv = cam['K'] @ Xc; return (uv[:2] / uv[2]).T     # Nx2

def prep(im, cam):      # native BGR -> 640x640 tensor + scale factors
    h, w = im.shape[:2]; sx, sy = IMSZ / w, IMSZ / h
    r = cv2.resize(im, (IMSZ, IMSZ)); r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(r).permute(2, 0, 1).float().to(dev) / 255., (sx, sy)

def weighted_dlt(Ps, uv, w):   # Ps (C,3,4) tensor, uv (C,J,2), w (C,J) -> (J,3)
    C, J = uv.shape[:2]
    P = Ps[:, None].expand(C, J, 3, 4).double(); uvd = uv.double()
    p0, p1, p2 = P[..., 0, :], P[..., 1, :], P[..., 2, :]
    A = torch.cat([uvd[..., 0:1] * p2 - p0, uvd[..., 1:2] * p2 - p1], 0)   # (2C,J,4)
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-9)
    wr = torch.cat([w, w], 0).double().clamp(min=1e-4).sqrt()[..., None]
    A = (A * wr).permute(1, 0, 2)                                          # (J,2C,4)
    _, _, V = torch.linalg.svd(A, full_matrices=False); Xh = V[..., -1, :]
    den = Xh[..., 3:4]; den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
    return (Xh[..., :3] / den).float()

class SampleHead(nn.Module):
    def __init__(self, cin=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(cin, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, feat):    # feat (C,J,128) -> refine (dx,dy) px + logconf
        o = self.net(feat); return o[..., :2] * 5.0, o[..., 2]   # offset scaled to +-few px

head = SampleHead().to(dev)

# projection matrices at NATIVE res, tensor
def Pmat(cam):
    return torch.tensor(cam['K'] @ np.hstack([cam['R'], cam['t']]), dtype=torch.float32, device=dev)
Ps = torch.stack([Pmat(c) for c in CAMS])    # (6,3,4)

def sample_feats(feats640, uv_native, cam):
    """feats640 (128,80,80) at 640-input; uv_native px -> grid_sample. returns (J,128)."""
    sx, sy = IMSZ / cam['W'], IMSZ / cam['H']
    uv640 = uv_native.clone(); uv640[..., 0] *= sx; uv640[..., 1] *= sy
    g = uv640 / (IMSZ - 1) * 2 - 1                          # normalize to [-1,1]
    g = g.view(1, -1, 1, 2)
    s = F.grid_sample(feats640[None], g, align_corners=True)   # (1,128,J,1)
    return s[0, :, :, 0].T                                   # (J,128)

def run(train_neck, epochs=40, lr=3e-4):
    for p in net.parameters(): p.requires_grad_(train_neck)
    params = list(head.parameters()) + ([p for p in net.parameters() if p.requires_grad] if train_neck else [])
    opt = torch.optim.Adam(params, lr)
    tr, va = FR[:int(len(FR) * .8)], FR[int(len(FR) * .8):]
    tag = 'UNFROZEN neck' if train_neck else 'frozen neck'
    print(f'--- {tag} ---')
    for ep in range(epochs):
        net.train(train_neck); head.train()
        fi, J3 = tr[np.random.randint(len(tr))]
        ims = read_views(fi)
        if ims is None: continue
        X = torch.tensor(J3, device=dev)                    # (19,3) GT
        # coarse init = projected GT + noise; corrupt 2 views
        uv0 = torch.stack([torch.tensor(project(c, J3), dtype=torch.float32, device=dev) for c in CAMS])  # (6,19,2)
        uv0 = uv0 + torch.randn_like(uv0) * 4
        bad = np.random.choice(6, 2, replace=False)
        for b in bad: uv0[b] += torch.randn(19, 2, device=dev) * 120
        # neck features per cam
        batch = torch.stack([prep(im, c)[0] for im, c in zip(ims, CAMS)])   # (6,3,640,640)
        feats = neck_feat(batch) if train_neck else neck_feat(batch).detach()  # (6,128,80,80)
        refined, logc = [], []
        for ci in range(6):
            fj = sample_feats(feats[ci], uv0[ci], CAMS[ci])   # (19,128)
            off, lc = head(fj); refined.append(uv0[ci] + off); logc.append(lc)
        uv = torch.stack(refined); w = torch.sigmoid(torch.stack(logc))
        pred = weighted_dlt(Ps, uv, w)
        loss = (pred - X).norm(dim=-1).clamp(max=50).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if ep % 8 == 0 or ep == epochs - 1:
            net.eval(); head.eval()
            with torch.no_grad():
                errs_l, errs_p = [], []
                for fi, J3 in va[:6]:
                    ims = read_views(fi)
                    if ims is None: continue
                    X = torch.tensor(J3, device=dev)
                    uv0 = torch.stack([torch.tensor(project(c, J3), dtype=torch.float32, device=dev) for c in CAMS]) + torch.randn(6, 19, 2, device=dev) * 4
                    for b in np.random.choice(6, 2, replace=False): uv0[b] += torch.randn(19, 2, device=dev) * 120
                    batch = torch.stack([prep(im, c)[0] for im, c in zip(ims, CAMS)])
                    feats = neck_feat(batch)
                    refined, logc = [], []
                    for ci in range(6):
                        off, lc = head(sample_feats(feats[ci], uv0[ci], CAMS[ci])); refined.append(uv0[ci] + off); logc.append(lc)
                    uv = torch.stack(refined); w = torch.sigmoid(torch.stack(logc))
                    errs_l.append((weighted_dlt(Ps, uv, w) - X).norm(dim=-1).clamp(max=100).mean().item())
                    errs_p.append((weighted_dlt(Ps, uv0, torch.ones(6, 19, device=dev)) - X).norm(dim=-1).clamp(max=100).mean().item())
            print(f'  ep{ep:3d} loss {loss.item()*10:6.0f}mm | val plain {np.mean(errs_p)*10:6.0f} | learned {np.mean(errs_l)*10:6.0f} mm', flush=True)

t0 = time.time()
run(train_neck=False, epochs=32)      # stage 1: frozen neck
print(f'(stage1 {time.time()-t0:.0f}s)')
run(train_neck=True, epochs=24, lr=1e-4)   # stage 2: fine-tune the neck
print('DONE — if learned<plain AND unfreezing helps further, the YOLO-neck fine-tune works.')
