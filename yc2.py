"""Does the YOLO-neck fusion CONVERGE? Fixes from the smoke run:
 - BATCH multiple frames per step (stable gradient, not 1-frame noise)
 - FIXED held-out eval set (same frames every eval -> comparable curve)
 - PRE-CACHE neck features for the frozen-neck stage (fast, many steps) to see the convergence curve
 - report train + val every N steps; success = val trend DOWN then FLAT, learned<plain stable.
Panoptic 6 synced cams, real 3D GT. Frozen-neck stage first (this is what 'converges' means for the
head); a short unfrozen stage after.
"""
import os, glob, json, time, cv2
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ultralytics import YOLO
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
SEQ = '160422_ultimatum1'; ROOT = f'/home/imove/Documents/mv3d_data/panoptic/{SEQ}'
CAM_IDS = ['00_00', '00_03', '00_06', '00_12', '00_18', '00_23']; IMSZ = 640

calib = json.load(open(f'{ROOT}/calibration_{SEQ}.json')); byname = {c['name']: c for c in calib['cameras']}
CAMS = []
for cid in CAM_IDS:
    c = byname[f'00_{cid.split("_")[1]}']
    CAMS.append({'K': np.array(c['K']), 'R': np.array(c['R']), 't': np.array(c['t']).reshape(3, 1),
                 'W': c['resolution'][0], 'H': c['resolution'][1]})
yolo = YOLO('/home/imove/Documents/cup-task/models/yolo26s-pose.pt'); net = yolo.model.to(dev).eval()
_feat = {}; net.model[16].register_forward_hook(lambda m, i, o: _feat.__setitem__('f', o))
def neck_feat(imgs): _feat.clear(); net(imgs); return _feat['f']

def load_frames(maxf=120):
    out = []
    for f in sorted(glob.glob(f'{ROOT}/hdPose3d_stage1_coco19/*.json')):
        d = json.load(open(f))
        if len(d['bodies']) == 1:
            j = np.array(d['bodies'][0]['joints19']).reshape(-1, 4)
            if j.shape[0] == 19 and (j[:, 3] > 0.1).all():
                out.append((int(os.path.basename(f).split('_')[-1].split('.')[0]), j[:, :3].astype(np.float32)))
        if len(out) >= maxf: break
    return out
FR = load_frames(); print(f'{len(FR)} frames')
caps = {cid: cv2.VideoCapture(f'{ROOT}/hdVideos/hd_{cid}.mp4') for cid in CAM_IDS}
def read_views(fi):
    ims = []
    for cid in CAM_IDS:
        caps[cid].set(cv2.CAP_PROP_POS_FRAMES, fi); ok, im = caps[cid].read()
        if not ok: return None
        ims.append(cv2.cvtColor(cv2.resize(im, (IMSZ, IMSZ)), cv2.COLOR_BGR2RGB))
    return np.stack(ims)   # (6,640,640,3)
def project(cam, X): uv = cam['K'] @ (cam['R'] @ X.T + cam['t']); return (uv[:2] / uv[2]).T
Ps = torch.stack([torch.tensor(c['K'] @ np.hstack([c['R'], c['t']]), dtype=torch.float32, device=dev) for c in CAMS])

# PRE-CACHE: neck features + GT + noisy-init for every frame (frozen-neck stage -> fast many-step training)
print('caching neck features for all frames...', flush=True)
CACHE = []
with torch.no_grad():
    for k, (fi, J3) in enumerate(FR):
        ims = read_views(fi)
        if ims is None: continue
        batch = torch.from_numpy(ims).permute(0, 3, 1, 2).float().to(dev) / 255.
        feats = neck_feat(batch).detach().cpu()          # (6,128,80,80)
        uv0 = np.stack([project(c, J3) for c in CAMS]).astype(np.float32)   # (6,19,2) native px
        CACHE.append((feats, torch.tensor(J3), torch.tensor(uv0)))
        if k % 30 == 0: print(f'  {k}/{len(FR)}', flush=True)
print(f'cached {len(CACHE)} frames')
NTR = int(len(CACHE) * .8); TR = CACHE[:NTR]; VA = CACHE[NTR:]

def sample_feats(feat640, uv_native, cam):
    sx, sy = IMSZ / cam['W'], IMSZ / cam['H']
    uv = uv_native.clone(); uv[..., 0] *= sx; uv[..., 1] *= sy
    g = (uv / (IMSZ - 1) * 2 - 1).view(1, -1, 1, 2)
    return F.grid_sample(feat640[None], g, align_corners=True)[0, :, :, 0].T   # (19,128)

def weighted_dlt(uv, w):
    C, J = uv.shape[:2]; P = Ps[:, None].expand(C, J, 3, 4).double(); uvd = uv.double()
    p0, p1, p2 = P[..., 0, :], P[..., 1, :], P[..., 2, :]
    A = torch.cat([uvd[..., 0:1] * p2 - p0, uvd[..., 1:2] * p2 - p1], 0)
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-9)
    wr = torch.cat([w, w], 0).double().clamp(min=1e-4).sqrt()[..., None]
    A = (A * wr).permute(1, 0, 2); _, _, V = torch.linalg.svd(A, full_matrices=False); Xh = V[..., -1, :]
    den = Xh[..., 3:4]; den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
    return (Xh[..., :3] / den).float()

class Head(nn.Module):
    def __init__(s): super().__init__(); s.n = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(s, f): o = s.n(f); return o[..., :2] * 5.0, o[..., 2]
head = Head().to(dev); opt = torch.optim.Adam(head.parameters(), 5e-4)

def make_noisy(uv0):
    uv = uv0.clone().to(dev) + torch.randn_like(uv0.to(dev)) * 4
    for b in np.random.choice(6, 3, replace=False): uv[b] += torch.randn(19, 2, device=dev) * 400
    return uv

def fwd(feats, uv0):
    uvn = make_noisy(uv0); refined, logc = [], []
    for ci in range(6):
        off, lc = head(sample_feats(feats[ci].to(dev), uvn[ci], CAMS[ci])); refined.append(uvn[ci] + off); logc.append(lc)
    uv = torch.stack(refined); w = torch.sigmoid(torch.stack(logc))
    return weighted_dlt(uv, w), weighted_dlt(uvn, torch.ones(6, 19, device=dev))

def evaluate():
    head.eval(); el, ep = [], []
    with torch.no_grad():
        torch.manual_seed(123)   # SAME noise every eval -> comparable
        for feats, J3, uv0 in VA:
            X = J3.to(dev); pl, pp = fwd(feats, uv0)
            el.append((pl - X).norm(-1).clamp(max=300).mean().item()); ep.append((pp - X).norm(-1).clamp(max=300).mean().item())
    head.train(); return np.mean(el) * 10, np.mean(ep) * 10

print('\nstep | train(mm) | val learned | val plain   [FROZEN neck, batched]')
BATCH = 8
for step in range(300):
    idx = np.random.choice(len(TR), BATCH, replace=False)
    losses = []
    for i in idx:
        feats, J3, uv0 = TR[i]; X = J3.to(dev)
        pred, _ = fwd(feats, uv0); losses.append((pred - X).norm(-1).clamp(max=150).mean())
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    if step % 25 == 0 or step == 299:
        vl, vp = evaluate()
        print(f'{step:4d} | {loss.item()*10:8.1f} | {vl:10.1f} | {vp:8.1f}', flush=True)
print('\nCONVERGES if train + val-learned trend DOWN then FLATTEN, and val-learned stays < val-plain.')
