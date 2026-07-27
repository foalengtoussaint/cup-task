"""Finetune the SynBody-pretrained CDRNet lifter on DELTA (real rig), YOLO-recipe training core.

Loads the pretrained EMA weights (view-invariant CSPDarknet from SynBody), finetunes on DELTA's 6
trials. Per the plan, the binding constraint is DELTA's tiny size -> the pretraining is what makes
this viable; here we finetune with a SMALL LR and the fusion/decoder + a gentle backbone touch.

2D-distill loss vs YOLO Pose26 (real detections). EVAL = frame-invariant wrist displacement + apex
good-frame-fraction vs OMC (DELTA is real -> MMC/OMC different frames, so NO raw MPJPE here; that's
what SynBody's synthetic in-frame GT was for). Compares against the FROM-SCRATCH DELTA baseline.
"""
import sys, os, time, argparse
sys.path.insert(0, '/home/imove/Documents/cup-task/scripts')
sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, torch
from cup_task.mv3d.cdrnet import CanonicalFusionCDR
from cup_task.mv3d.dataset import make_loader
from cup_task.mv3d.data_delta import load_amq
from cup_task.mv3d.train_core import train_loop, distill_loss_fn
# reuse the DELTA eval (frame-invariant displacement + apex GF) from the scratch trainer
import train_cdrnet_delta as DELTA

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
TRIALS = DELTA.TRIALS; TRAIN, VAL = DELTA.TRAIN, DELTA.VAL


def main(pretrained, epochs, batch, workers, lr0, freeze_backbone=False,
         out_path='/home/imove/Documents/cup-task/models/cdrnet_delta_finetuned.pt'):
    amq = load_amq()
    D = DELTA.build()
    model = CanonicalFusionCDR(n_kpts=17).to(dev)
    if pretrained and os.path.exists(pretrained):
        sd = torch.load(pretrained, map_location=dev)['model']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'loaded pretrained {pretrained} (missing {len(missing)}, unexpected {len(unexpected)})', flush=True)
    else:
        print('WARNING: no pretrained weights -> from scratch (control)', flush=True)
    # Plan warning: DELTA is tiny (4 train trials) -> full-backbone finetune OVERFITS and destroys
    # the pretrained view-invariant features. freeze_backbone=True trains only fusion+decoder
    # (light-touch adaptation that cannot wreck the backbone). Recommended variant.
    model.set_trainable_backbone(not freeze_backbone)
    print(f'backbone {"FROZEN (fusion+decoder only)" if freeze_backbone else "trainable"}', flush=True)

    _, loader = make_loader(TRAIN, amq=amq, batch=batch, workers=workers, shuffle=True, frame_stride=4)
    print(f'{len(loader.dataset)} DELTA train frames', flush=True)

    m0, g0, _ = DELTA.eval_trials(model, VAL, D)
    print(f'FINETUNE start: HELDOUT disp {m0:.1f}mm | apex GF {g0:.2f}', flush=True)

    def on_epoch(ep, ema_model):
        if ep % 5 == 0 or ep == epochs - 1:
            vm, vg, vn = DELTA.eval_trials(ema_model.to(dev), VAL, D)
            print(f'    >> epoch {ep}: HELDOUT disp {vm:.1f}mm apex GF {vg:.2f} (n{vn})', flush=True)

    # AMP OFF for the DELTA finetune: fp16 corrupts the fragile geometry/heatmap path -> nan loss
    # (verified: AMP nan vs AMP-off finite 495). Only 6 trials (~15min) so fp32 speed cost is moot.
    t0 = time.time()
    ema = train_loop(model, loader, distill_loss_fn(dev, amp=False), epochs=epochs, dev=dev,
                     lr0=lr0, lrf=0.01, nbs=64, batch=batch, warmup_epochs=1.0, amp=False,
                     use_ema=True, optimizer='AdamW', on_epoch=on_epoch)
    vm, vg, vn = DELTA.eval_trials(ema.to(dev), VAL, D)
    print(f'\nFINETUNE done in {time.time()-t0:.0f}s. FINAL HELDOUT disp {vm:.1f}mm | apex GF {vg:.2f}', flush=True)
    torch.save({'model': ema.state_dict()}, out_path)
    print(f'saved -> {out_path}', flush=True)
    for n, _ in TRIALS:
        D[n].release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', default='')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr0', type=float, default=1e-4)
    ap.add_argument('--freeze_backbone', action='store_true')
    ap.add_argument('--out', default='/home/imove/Documents/cup-task/models/cdrnet_delta_finetuned.pt')
    a = ap.parse_args()
    import traceback
    try:
        main(a.pretrained, a.epochs, a.batch, a.workers, a.lr0, a.freeze_backbone, a.out)
    except Exception as e:
        print(f'\nFINETUNE CRASHED: {type(e).__name__}: {e}', flush=True); traceback.print_exc()
