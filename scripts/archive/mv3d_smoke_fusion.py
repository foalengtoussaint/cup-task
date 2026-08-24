"""SMOKE TEST of the multi-view learned-fusion pipeline on SynBody (calib + SMPL-X transl only;
no images/body-model needed yet). Proves: data load -> project real 3D into 8 real cams -> add
detector-like 2D noise (+ occasional bad views) -> LEARNED confidence-weighted triangulation head
-> recover 3D -> loss vs GT. Compares learned fusion vs plain (equal-weight) DLT triangulation.

If learned fusion beats plain DLT when some views are corrupted, the core idea works: the net learns
to down-weight bad views. Small run, CPU/GPU, a few epochs, LOPO-ish split across sequences.
"""
import os, glob, json, time
import numpy as np, torch, torch.nn as nn
ROOT = '/home/imove/Documents/mv3d_data/synbody/HumanNeRF-subset'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)

# ---- load real calib + real 3D (transl) per sequence ----
def load_seq(sid):
    cam = json.load(open(f'{ROOT}/camera_params/{sid}/camera.json'))
    cams = sorted(cam.keys())
    P = []
    for cn in cams:
        c = cam[cn]; K = np.array(c['K']); R = np.array(c['R']); T = np.array(c['T']).reshape(3, 1)
        P.append((K @ np.hstack([R, T])).astype(np.float32))     # 3x4 projection
    X = np.load(f'{ROOT}/smplx/{sid}/smplx.npz', allow_pickle=True)['smplx'].item()['transl'].astype(np.float32)
    return np.stack(P), X                                        # (8,3,4), (F,3)

seqs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{ROOT}/camera_params/*/camera.json'))
print(f'{len(seqs)} sequences; using calib+transl (images still downloading)')
data = {s: load_seq(s) for s in seqs[:40]}                       # 40 seqs is plenty for a smoke test
train_s = list(data)[:32]; val_s = list(data)[32:]

def project(P, X):                                              # P (8,3,4), X (...,3) -> (...,8,2)
    Xh = np.concatenate([X, np.ones((*X.shape[:-1], 1), np.float32)], -1)
    x = np.einsum('cij,...j->...ci', P, Xh)                     # (...,8,3)
    return x[..., :2] / x[..., 2:3]

def make_batch(seqlist, bs=64, noise=3.0, nbad=2):
    """sample frames; project to 8 cams; add gaussian px noise + make `nbad` random cams WILD."""
    Ps, uvs, Xs, badmask = [], [], [], []
    for _ in range(bs):
        s = seqlist[np.random.randint(len(seqlist))]; P, X = data[s]
        f = np.random.randint(len(X)); x3 = X[f]
        uv = project(P, x3)                                     # (8,2)
        uv = uv + np.random.randn(*uv.shape).astype(np.float32) * noise
        bad = np.zeros(8, np.float32)
        for bi in np.random.choice(8, nbad, replace=False):
            uv[bi] += np.random.randn(2).astype(np.float32) * 150   # corrupted view
            bad[bi] = 1.0
        Ps.append(P); uvs.append(uv); Xs.append(x3); badmask.append(bad)
    return (torch.tensor(np.stack(Ps)).to(dev), torch.tensor(np.stack(uvs)).to(dev),
            torch.tensor(np.stack(Xs)).to(dev), torch.tensor(np.stack(badmask)).to(dev))

# ---- differentiable weighted DLT triangulation ----
def weighted_dlt(P, uv, w):
    """P (B,8,3,4), uv (B,8,2), w (B,8) -> X (B,3). Weighted DLT with per-row normalization
    (Hartley-style: scale each row to unit norm so the SVD is well-conditioned even when some
    views are wildly off). float64 solve for stability."""
    P64, uv64, w64 = P.double(), uv.double(), w.double()
    p0, p1, p2 = P64[..., 0, :], P64[..., 1, :], P64[..., 2, :]     # (B,8,4)
    A0 = uv64[..., 0:1] * p2 - p0
    A1 = uv64[..., 1:2] * p2 - p1
    A = torch.cat([A0, A1], 1)                                     # (B,16,4)
    # normalize each row to unit norm (conditioning) THEN apply sqrt-weights
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-9)
    wr = torch.cat([w64, w64], 1).clamp(min=1e-4).sqrt()[..., None]
    A = A * wr
    _, _, V = torch.linalg.svd(A, full_matrices=False)
    Xh = V[..., -1, :]                                             # (B,4)
    denom = Xh[..., 3:4]
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    return (Xh[..., :3] / denom).float()

# ---- learned per-view confidence head (CDRNet-style camera-disentangle, tiny) ----
class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        # per-view features: reproj residual to a rough init + the uv itself -> weight
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, P, uv):
        # rough init = equal-weight DLT
        w0 = torch.ones(uv.shape[:2], device=uv.device)
        X0 = weighted_dlt(P, uv, w0)                            # (B,3)
        rp = torch.einsum('bcij,bj->bci', P, torch.cat([X0, torch.ones_like(X0[..., :1])], -1))
        rp = rp[..., :2] / rp[..., 2:3]
        resid = (rp - uv)                                      # (B,8,2) reproj error per view
        feat = torch.cat([uv * 0.001, resid * 0.001], -1)      # (B,8,4) scaled
        w = torch.sigmoid(self.net(feat).squeeze(-1))          # (B,8) learned confidence
        X = weighted_dlt(P, uv, w)
        return X, w

model = FusionHead().to(dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)
def err(pred, gt): return (pred - gt).norm(dim=-1).mean().item() * 1000  # mm (transl in metres)

print('epoch | train_loss | val: plain-DLT | learned-fusion  (mm error, 2 of 8 views corrupted)')
for ep in range(60):
    model.train()
    P, uv, X, bad = make_batch(train_s, 128)
    pred, w = model(P, uv)
    loss = (pred - X).norm(dim=-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 10 == 0 or ep == 59:
        model.eval()
        with torch.no_grad():
            Pv, uvv, Xv, badv = make_batch(val_s, 256)
            plain = weighted_dlt(Pv, uvv, torch.ones(uvv.shape[:2], device=dev))
            learned, wv = model(Pv, uvv)
            # does the learned weight correctly down-weight the bad views?
            w_bad = wv[badv > 0].mean().item(); w_good = wv[badv == 0].mean().item()
        print(f'{ep:5d} | {loss.item()*1000:9.1f} | {err(plain,Xv):12.1f} | {err(learned,Xv):8.1f}   '
              f'(w_good {w_good:.2f} vs w_bad {w_bad:.2f})', flush=True)
print('\nif learned << plain and w_bad << w_good, the fusion head LEARNS to reject bad views = pipeline works.')
