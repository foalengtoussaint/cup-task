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
    out = model(s['imgs'], s['P_native'])
    Xw = out['X3d'][WRIST_IDX]                       # (3,)
    rp = reproj_residual(Xw, out['kpts2d'][:, WRIST_IDX, :], s['P_native'])
    return out, s, Xw, rp


def eval_trials(model, trials, D, apex_speed=150.0):
    """Frame-invariant: |disp-from-start| vs OMC. Returns (median_mm, apex_good_frac, n)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for name, tn in trials:
            t = D[name]; fs, Xs, Os = [], [], []
            for f in t.vf:
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


def train(full=False):
    D = build()
    model = CanonicalFusionCDR(n_kpts=17).to(dev)
    tr_frames = [(n, f) for n, _ in TRAIN for f in D[n].vf]
    print(f'{len(tr_frames)} train frames', flush=True)

    def run_stage(name, unfreeze, steps, lr):
        model.set_trainable_backbone(unfreeze)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr)
        print(f'\n=== {name} ({sum(p.numel() for p in params)/1e6:.1f}M trainable) ===', flush=True)
        m0, g0, _ = eval_trials(model, VAL, D)
        print(f'  eval@start: HELDOUT disp-err {m0:.1f}mm | apex good-frac {g0:.2f}', flush=True)
        for step in range(steps):
            n, f = tr_frames[np.random.randint(len(tr_frames))]
            r = wrist3d(model, D[n], f)
            if r is None:
                continue
            out, s = r[0], r[1]
            # PRIMARY: per-view 2D MSE vs YOLO (distill), confidence-weighted
            w = s['kp_conf'][..., None].clamp(0, 1)
            loss = (w * (out['kpts2d'] - s['kp2d_tgt']) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % max(1, steps // 6) == 0 or step == steps - 1:
                vm, vg, vn = eval_trials(model, VAL, D)
                tm, tg, _ = eval_trials(model, TRAIN, D)
                print(f'  step{step:4d} 2Dloss {loss.item():7.1f} | HELDOUT disp {vm:6.1f}mm '
                      f'apexGF {vg:.2f} (n{vn}) || TRAIN disp {tm:6.1f}mm', flush=True)

    t0 = time.time()
    run_stage('STAGE 1 frozen backbone', False, 60 if full else 12, 5e-4)
    run_stage('STAGE 2 UNFROZEN CSPDarknet', True, 40 if full else 8, 1e-4)
    print(f'\ndone in {time.time()-t0:.0f}s. Success = HELDOUT apex good-frac UP (esp. stage 2)', flush=True)
    for n, _ in TRIALS:
        D[n].release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--full', action='store_true')
    train(ap.parse_args().full)
