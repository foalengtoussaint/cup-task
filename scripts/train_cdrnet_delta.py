"""Train the CDRNet-on-CSPDarknet lifter on DELTA, held-out-TRIAL eval.

Faithful to CDRNet training: PRIMARY loss = per-view 2D MSE (distilled from YOLO Pose26); the
3D/DLT output is NOT the primary objective (CDRNet uses a dummy loss there — backprop DLT is
unstable and adds little). Optional TINY reproj term late (frame-invariant, no OMC frame issue).

Backbone: stage 1 FROZEN (the fusion+decoder learn to reproduce YOLO 2D from CSPDarknet maps),
stage 2 UNFROZEN (the actual CSPDarknet fine-tune — the whole point of the port).

EVAL vs OMC is EVAL-ONLY and frame-invariant: displacement-from-start magnitude of the wrist +
"good-frame fraction" at the drink APEX (where occlusion/jitter bite). Position is already ~5mm
by plain triangulation, so success = APEX/robustness, NOT median position (per the plan note).

Usage: python scripts/train_cdrnet_delta.py            # small smoke run (few steps)
       python scripts/train_cdrnet_delta.py --full     # longer
"""
import sys, os, time, argparse
sys.path.insert(0, '/home/imove/Documents/cup-task/scripts')
sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, torch, torch.nn.functional as F
from cup_task.mv3d.cdrnet import CanonicalFusionCDR
from cup_task.mv3d.data_delta import DeltaTrial, load_amq, WRIST_IDX, COCO17
from cup_task.mv3d.dlt import reproj_residual
from cup_task.mv3d.dataset import make_loader

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
PART = 'P07'
TRIALS = [(f'trial_{i}_L_unaffected', i) for i in [10, 11, 12, 13, 14, 15]]
TRAIN, VAL = TRIALS[:4], TRIALS[4:]
FPS = 60.0


def build():
    amq = load_amq()
    D = {}
    for name, tn in TRIALS:
        t = DeltaTrial(PART, name, tn, amq)
        t.vf = t.valid_frames()
        D[name] = t
        print(f'  {name}: {len(t.cams)}cam {len(t.vf)}valid OMC={t.omc is not None}', flush=True)
    return D


def wrist3d(model, t, f):
    """Run the model on frame f, return (X_wrist_3d, reproj_px) or None."""
    s = t.sample(f, device=dev)
    if s is None:
        return None
    out = model(s['imgs'], s['P_native'], native_size=s['native_size'])
    Xw = out['X3d'][WRIST_IDX]                       # (3,)
    rp = reproj_residual(Xw, out['kpts2d'][:, WRIST_IDX, :], s['P_native'])
    return out, s, Xw, rp


def eval_trials(model, trials, D, apex_speed=150.0, stride=5):
    """Frame-invariant: |disp-from-start| vs OMC. Returns (median_mm, apex_good_frac, n).

    stride: subsample every Nth valid frame. One model forward is ~160ms, so a full 1055-frame
    eval is ~3min — far too heavy to run every few steps. stride=5 -> ~35s and still representative.
    """
    model.eval()
    if dev == 'cuda':
        torch.cuda.empty_cache()          # release stage-2 training fragmentation before eval
    rows = []
    with torch.no_grad():
        for name, tn in trials:
            t = D[name]; fs, Xs, Os = [], [], []
            for f in t.vf[::stride]:
                r = wrist3d(model, t, f)
                if r is None or r[1]['omc_wrist'] is None:
                    continue
                X = r[2].cpu().numpy()
                if not np.isfinite(X).all():           # untrained decoder -> degenerate DLT; skip
                    continue
                fs.append(f); Xs.append(X); Os.append(r[1]['omc_wrist'].numpy())
            if len(fs) < 8:
                continue
            Xs, Os = np.array(Xs), np.array(Os)
            dX = np.linalg.norm(Xs - Xs[0], axis=1); dO = np.linalg.norm(Os - Os[0], axis=1)
            err = np.abs(dX - dO)
            # apex frames = fastest OMC motion
            ospd = np.r_[0, np.linalg.norm(np.diff(Os, axis=0), axis=1) * FPS]
            apex = ospd > apex_speed
            good = err < 30.0                          # good-frame threshold (mm)
            rows.append((np.median(err),
                         good[apex].mean() if apex.any() else np.nan,
                         len(fs)))
    model.train()
    if not rows:
        return np.nan, np.nan, 0
    a = np.array(rows, float)
    return np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), int(a[:, 2].sum())


def train(full=False, workers=4, batch=8):
    amq = load_amq()
    D = build()                                            # for eval (holds trials + OMC)
    model = CanonicalFusionCDR(n_kpts=17).to(dev)
    # YOLO-style workered dataloader: CPU workers load frames in parallel while GPU trains
    _, loader = make_loader(TRAIN, amq=amq, batch=batch, workers=workers, shuffle=True)
    print(f'{len(loader.dataset)} train frames, batch {batch}, {workers} workers', flush=True)

    def step_on_group(group, opt, params):
        """One optimizer step over a batch of grouped multi-view samples (variable cams each).

        GRAD ACCUMULATION: backward per group (frees that group's graph immediately) then step once.
        Holding all groups' graphs before one backward OOMs when the backbone is unfrozen (8 x 5-view
        graphs on an 8GB card). Accumulating keeps peak memory at ~one group's forward.
        """
        opt.zero_grad()
        n_used, tot = 0, 0.0
        for s in group:
            imgs = s['imgs'].to(dev, non_blocking=True); P = s['P_native'].to(dev, non_blocking=True)
            out = model(imgs, P, native_size=s['native_size'].to(dev))
            tgt = s['kp2d_tgt'].to(dev); conf = s['kp_conf'].to(dev)
            # MASK invisible keypoints (YOLO emits nan for undetected joints). nan_to_num BEFORE the
            # weight multiply — nan*0 = nan would poison the loss even after masking.
            vis = torch.isfinite(tgt).all(-1) & torch.isfinite(conf) & (conf > 0.3)
            if not vis.any():
                continue
            w = (torch.nan_to_num(conf).clamp(0, 1) * vis.float())[..., None]
            se = (out['kpts2d'] - torch.nan_to_num(tgt)) ** 2
            loss = (w * se).sum() / w.sum().clamp(min=1e-6)
            (loss / max(len(group), 1)).backward()                    # accumulate; graph freed here
            n_used += 1; tot += loss.item()
        if n_used == 0:
            return float('nan')
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        return tot / n_used

    def run_stage(name, unfreeze, steps, lr, eval_every=0):
        """2D-loss printed each step (cheap); heavy OMC eval only at start/end (+eval_every)."""
        model.set_trainable_backbone(unfreeze)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr)
        print(f'\n=== {name} ({sum(p.numel() for p in params)/1e6:.1f}M trainable) ===', flush=True)
        m0, g0, _ = eval_trials(model, VAL, D)
        print(f'  eval@start: HELDOUT disp {m0:.1f}mm | apex GF {g0:.2f}', flush=True)
        run_loss = None; step = 0
        while step < steps:
            for group in loader:
                if not group:
                    continue
                l = step_on_group(group, opt, params)
                run_loss = l if run_loss is None else 0.9 * run_loss + 0.1 * l
                print(f'  step{step:4d} 2Dloss {l:8.1f} (ema {run_loss:8.1f})', flush=True)
                if eval_every and step and step % eval_every == 0:
                    vm, vg, vn = eval_trials(model, VAL, D)
                    print(f'    -> eval HELDOUT disp {vm:.1f}mm apex GF {vg:.2f} (n{vn})', flush=True)
                step += 1
                if step >= steps:
                    break
        try:
            vm, vg, vn = eval_trials(model, VAL, D)
            print(f'  eval@END: HELDOUT disp {vm:.1f}mm | apex GF {vg:.2f} (n{vn})', flush=True)
        except Exception as e:
            import traceback
            print(f'  eval@END FAILED: {type(e).__name__}: {e}', flush=True); traceback.print_exc()

    t0 = time.time()
    run_stage('STAGE 1 frozen backbone', False, 200 if full else 40, 5e-4,
              eval_every=100 if full else 0)
    run_stage('STAGE 2 UNFROZEN CSPDarknet', True, 120 if full else 24, 1e-4,
              eval_every=60 if full else 0)
    print(f'\ndone in {time.time()-t0:.0f}s. Success = HELDOUT apex good-frac UP (esp. stage 2)', flush=True)
    for n, _ in TRIALS:
        D[n].release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    import traceback
    try:
        train(a.full, workers=a.workers)
    except Exception as e:
        print(f'\nTRAIN CRASHED: {type(e).__name__}: {e}', flush=True); traceback.print_exc()
