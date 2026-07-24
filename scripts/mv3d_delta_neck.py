"""HONEST YOLO-neck refinement test on DELTA (our real data, no leaks).
Input = REAL cached YOLO 2D keypoints (genuine detector output, imperfect at apex/occlusion).
Refine each kp using the YOLO NECK feature sampled at that pixel -> weighted DLT -> 3D.
Compare vs plain DLT of the raw YOLO 2D. Truth = OMC. Focus joint = wrist (Murphy-relevant).

Everything real: real detections, real neck features, real 8..5-cam calib, real mocap 3D.
Trains the refine head (neck frozen first), reports wrist 3D error vs OMC, split fast/slow.
"""
import sys, os, glob, json, time, cv2
sys.path.insert(0, '/home/imove/Documents/cup-task/scripts'); sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ultralytics import YOLO
import compare_pose_omc_delta as H
import results_v3_delta as R
H.use_good_cams()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
PART, TRIAL, SIDE = 'P07', 'trial_11_L_unaffected', 'left'
JOINT = f'{SIDE}_wrist'; IMSZ = 640
# YOLO kp name for the joint (cached dets use these names)
KPNAME = {'left_wrist': 'left_wrist', 'left_elbow': 'left_elbow', 'left_shoulder': 'left_shoulder'}[JOINT]

calib = R._calib(PART)                      # {cam: CamCalib(K,R,t,size,dist)}
cams = list(calib)
print(f'{len(cams)} cams: {cams}')
# projection matrices (calib.t already in mm per results_v3_delta)
def Pmat(c): return c.K @ np.hstack([c.R, c.t.reshape(3, 1)])
Ps = torch.stack([torch.tensor(Pmat(calib[c]), dtype=torch.float32, device=dev) for c in cams])

# cached YOLO 2D for the joint, per cam (native px)
def cached_kp(cam):
    pj = H.DELTA/PART/'dets'/f'delta_{PART}_{TRIAL}.{cam.split("_")[1]}.pose.json'
    if not pj.exists(): return None
    fr = json.loads(pj.read_text())['frames']
    out = []
    for f in fr:
        k = f.get('kps', {}).get(KPNAME) if f else None
        out.append([k[0], k[1], k[2]] if k else [np.nan, np.nan, 0.0])
    return np.array(out, np.float32)          # (n,3) x,y,conf
KP = {c: cached_kp(c) for c in cams}
KP = {c: v for c, v in KP.items() if v is not None}
cams = list(KP)
Ps = torch.stack([torch.tensor(Pmat(calib[c]), dtype=torch.float32, device=dev) for c in cams])
n = min(len(v) for v in KP.values())
print(f'{len(cams)} cams with cached {KPNAME}, {n} frames')

# OMC 3D truth for the joint
mmc, nn_ = H._load_mmc(PART, TRIAL)
omc = R._shift(H._load_omc(PART, TRIAL, nn_)[JOINT], H._find_lag(mmc[JOINT], H._load_omc(PART, TRIAL, nn_)[JOINT])[0])
cup = R._cup_v3(PART, TRIAL, calib, nn_)
spd = np.r_[0, np.linalg.norm(np.diff(np.nan_to_num(mmc[JOINT]), axis=0), axis=1)]*R.FPS

# YOLO neck features (compute per frame per cam from video) -- CACHE to disk (heavy)
yolo = YOLO('/home/imove/Documents/cup-task/models/yolo26s-pose.pt'); net = yolo.model.to(dev).eval()
_feat = {}; net.model[16].register_forward_hook(lambda m, i, o: _feat.__setitem__('f', o))
def native_size(cam): return calib[cam].size   # (W,H)

# sample frames where OMC valid + >=3 cams have a detection
valid = [f for f in range(min(n, nn_)) if np.isfinite(omc[f]).all()
         and sum(np.isfinite(KP[c][f, 0]) and KP[c][f, 2] > 0.3 for c in cams) >= 3]
print(f'{len(valid)} usable frames (OMC valid + >=3 cam dets)')
samp = valid  # use all

# --- precompute neck features at the sampled frames (per cam) ---
CACHE_F = f'/tmp/claude-1000/-home-imove-Documents-object-tracking/25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad/delta_neckfeat_{PART}_{TRIAL}.pt'
def build_feats():
    caps = {c: cv2.VideoCapture(str(H.DELTA/PART/'staged'/f'delta_{PART}_{TRIAL}.{c.split("_")[1]}.mp4')) for c in cams}
    store = {}
    print('computing neck features (video decode + YOLO)...', flush=True)
    for k, f in enumerate(samp):
        per = {}
        for c in cams:
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = caps[c].read()
            if not ok: continue
            t = torch.from_numpy(cv2.cvtColor(cv2.resize(im, (IMSZ, IMSZ)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(dev)[None]/255.
            with torch.no_grad(): _feat.clear(); net(t); per[c] = _feat['f'][0].half().cpu()
        store[f] = per
        if k % 40 == 0: print(f'  {k}/{len(samp)}', flush=True)
    for c in caps.values(): c.release()
    torch.save(store, CACHE_F); return store
FEAT = torch.load(CACHE_F) if os.path.exists(CACHE_F) else build_feats()
print(f'neck features cached for {len(FEAT)} frames')

def sample_feat(feat, uv_native, cam):
    W, Hh = native_size(cam); sx, sy = IMSZ/W, IMSZ/Hh
    u = torch.tensor([uv_native[0]*sx, uv_native[1]*sy], device=dev)
    g = (u/(IMSZ-1)*2-1).view(1, 1, 1, 2)
    return F.grid_sample(feat.float().to(dev)[None], g, align_corners=True)[0, :, 0, 0]  # (128,)

class Head(nn.Module):
    def __init__(s): super().__init__(); s.n = nn.Sequential(nn.Linear(129, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(s, feat, conf):
        x = torch.cat([feat, torch.tensor([conf], device=feat.device)])   # 128 feat + 1 conf
        o = s.n(x); return o[:2]*8.0, o[2]
head = Head().to(dev); opt = torch.optim.Adam(head.parameters(), 5e-4)

def predict(f, use_head):
    uv_raw, w_raw, refined, logc = [], [], [], []
    for ci, c in enumerate(cams):
        kp = KP[c][f]
        if not (np.isfinite(kp[0]) and kp[2] > 0.3):   # missing det -> drop this cam (weight ~0)
            uv_raw.append([np.nan, np.nan]); w_raw.append(0.0); refined.append(None); logc.append(None); continue
        uv_raw.append([kp[0], kp[1]]); w_raw.append(float(kp[2]))
        if use_head and c in FEAT.get(f, {}):
            fe = sample_feat(FEAT[f][c], kp[:2], c); off, lc = head(fe, float(kp[2]))
            refined.append(torch.tensor([kp[0], kp[1]], device=dev)+off); logc.append(lc)
        else:
            refined.append(torch.tensor([kp[0], kp[1]], dtype=torch.float32, device=dev)); logc.append(torch.tensor(0., device=dev))
    keep = [i for i in range(len(cams)) if w_raw[i] > 0]
    if len(keep) < 3: return None, None
    uvP = torch.tensor([uv_raw[i] for i in keep], dtype=torch.float32, device=dev)
    wP = torch.tensor([w_raw[i] for i in keep], device=dev)
    global Ps_k; Ps_k = Ps[keep]
    plain = wdlt_k(uvP, wP)
    if use_head:
        uvL = torch.stack([refined[i] for i in keep]); wL = torch.sigmoid(torch.stack([logc[i] for i in keep]))
        learned = wdlt_k(uvL, wL)
    else: learned = plain
    return plain, learned
def wdlt_k(uv, w):
    C = uv.shape[0]; P = Ps_k.double(); uvd = uv.double()
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
    A = torch.cat([uvd[:, 0:1]*p2-p0, uvd[:, 1:2]*p2-p1], 0); A = A/(A.norm(dim=-1, keepdim=True)+1e-9)
    wr = torch.cat([w, w], 0).double().clamp(min=1e-4).sqrt()[:, None]; A = A*wr
    _, _, V = torch.linalg.svd(A, full_matrices=False); Xh = V[-1]
    den = Xh[3] if abs(Xh[3]) > 1e-6 else torch.tensor(1e-6, device=uv.device)
    return (Xh[:3]/den).float()

tr = [f for f in FEAT if f < samp[int(len(samp)*.7)]]; va = [f for f in FEAT if f >= samp[int(len(samp)*.7)]]
def evl():
    """frame-invariant: displacement-from-start magnitude vs OMC (MMC/OMC live in different worlds)."""
    head.eval()
    with torch.no_grad():
        Xp={}; Xl={}
        for f in va:
            p,l=predict(f,True)
            if p is None: continue
            Xp[f]=p.cpu().numpy(); Xl[f]=l.cpu().numpy()
        fs=sorted(Xp); 
        if len(fs)<5: head.train(); return (np.nan,)*4
        f0=fs[0]; dO=lambda f:np.linalg.norm(omc[f]-omc[f0]); 
        dp=[abs(np.linalg.norm(Xp[f]-Xp[f0])-dO(f)) for f in fs]
        dl=[abs(np.linalg.norm(Xl[f]-Xl[f0])-dO(f)) for f in fs]
        dpf=[abs(np.linalg.norm(Xp[f]-Xp[f0])-dO(f)) for f in fs if spd[f]>150]
        dlf=[abs(np.linalg.norm(Xl[f]-Xl[f0])-dO(f)) for f in fs if spd[f]>150]
    head.train()
    return (np.median(dp),np.median(dl),np.median(dpf) if dpf else np.nan,np.median(dlf) if dlf else np.nan)
print('\nstep | train | val plain | val learned | fast plain | fast learned  (mm, wrist vs OMC)')
for step in range(150):
    f = tr[np.random.randint(len(tr))]
    p, l = predict(f, True)
    if l is None: continue
    X = torch.tensor(omc[f], device=dev); loss = (l-X).norm()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    if step % 15 == 0 or step == 149:
        vp, vl, fp, fl = evl()
        print(f'{step:4d} | {loss.item():6.0f} | {vp:9.0f} | {vl:11.0f} | {fp:10.0f} | {fl:12.0f}', flush=True)
print('\nlearned<plain (esp fast/apex) => neck features refine real YOLO dets toward truth on OUR data.')
