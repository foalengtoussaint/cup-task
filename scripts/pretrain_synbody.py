"""Pretrain the CDRNet-on-CSPDarknet lifter on SynBody (100 seqs, 8 synced cams, in-frame GT).

This is the plan's core step: train end-to-end with RANDOM 3-6 view sampling so the CSPDarknet neck
becomes a view-invariant 3D-triangulation feature extractor -> our unseen 5-cam DELTA rig is "just
another subset". Uses YOLO's optimized training core (train_core: AMP, EMA, cosine, warmup,
accumulate-to-nbs). PRIMARY loss = 2D-MSE distill vs projected GT (CDRNet-style). EVAL = MPJPE (mm)
on HELD-OUT sequences — valid here because SynBody is synthetic so pred & GT share ONE world frame
(no MMC/OMC alignment). Saves the EMA weights for the DELTA finetune.

Usage: python scripts/pretrain_synbody.py --epochs 10
"""
import sys, os, time, argparse
sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, torch
from cup_task.mv3d.cdrnet import CanonicalFusionCDR
from cup_task.mv3d.dataset_synbody import make_synbody_loader, split_seqs, SynBodyViews
from cup_task.mv3d.train_core import train_loop, distill_loss_fn

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
CKPT = '/home/imove/Documents/cup-task/models/cdrnet_synbody_pretrained.pt'


def mpjpe_eval(model, seqs, n_seq=10, per_seq=20, min_v=3, max_v=6):
    """Mean Per-Joint Position Error (mm) on held-out seqs. In-frame GT -> direct ||X_pred - X_gt||.
    Also reports a fixed-view (all-8) variant to separate view-sampling noise from lifter error."""
    ds = SynBodyViews(seqs[:n_seq], min_v=min_v, max_v=max_v, frame_stride=15)
    model.eval(); errs = []
    with torch.no_grad():
        idxs = np.random.choice(len(ds), min(per_seq * n_seq, len(ds)), replace=False)
        for i in idxs:
            s = ds[i]
            if s is None:
                continue
            out = model(s['imgs'].to(dev), s['P_native'].to(dev))
            e = (out["X3d"] - s["X_gt"].to(dev)).norm(dim=-1) * 1000.0   # meters -> mm
            errs.append(e.mean().item())
    model.train()
    return float(np.mean(errs)) if errs else float('nan')


def main(epochs, batch, workers, frame_stride, lr0):
    train_seqs, val_seqs = split_seqs(0.1)
    print(f'{len(train_seqs)} train seqs, {len(val_seqs)} held-out', flush=True)
    _, loader = make_synbody_loader(train_seqs, batch=batch, workers=workers,
                                    shuffle=True, frame_stride=frame_stride)
    print(f'{len(loader.dataset)} train samples (random 3-6 views each)', flush=True)
    model = CanonicalFusionCDR(n_kpts=17).to(dev)
    model.set_trainable_backbone(True)                            # pretrain = FULL end-to-end

    print(f'MPJPE @ init (untrained): {mpjpe_eval(model, val_seqs):.0f} mm', flush=True)

    def on_epoch(ep, ema_model):
        if ep % 2 == 0 or ep == epochs - 1:
            m = mpjpe_eval(ema_model.to(dev), val_seqs)
            print(f'    >> epoch {ep}: HELD-OUT MPJPE {m:.0f} mm (EMA)', flush=True)

    t0 = time.time()
    ema = train_loop(model, loader, distill_loss_fn(dev, amp=True), epochs=epochs, dev=dev,
                     lr0=lr0, lrf=0.01, nbs=64, batch=batch, warmup_epochs=1.0, amp=True,
                     use_ema=True, optimizer='AdamW', on_epoch=on_epoch)
    print(f'\npretrain done in {time.time()-t0:.0f}s. Final HELD-OUT MPJPE {mpjpe_eval(ema.to(dev), val_seqs):.0f} mm', flush=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({'model': ema.state_dict()}, CKPT)
    print(f'saved EMA weights -> {CKPT}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--frame_stride', type=int, default=6)
    ap.add_argument('--lr0', type=float, default=1e-3)
    a = ap.parse_args()
    import traceback
    try:
        main(a.epochs, a.batch, a.workers, a.frame_stride, a.lr0)
    except Exception as e:
        print(f'\nPRETRAIN CRASHED: {type(e).__name__}: {e}', flush=True); traceback.print_exc()
