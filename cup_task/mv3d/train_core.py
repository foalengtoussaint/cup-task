"""Optimized training core for the CDRNet lifter — copies YOLO's fine-tuning recipe.

Every optimization here mirrors ultralytics BaseTrainer (read from source), so pretraining
(SynBody) and finetuning (DELTA) share YOLO's exact machinery rather than a hand-rolled loop:

  1. AMP / mixed precision : autocast(forward) + GradScaler (scale->backward->unscale->step->update).
                             ~2x faster + half memory on the 8GB card.
  2. Accumulate to nominal batch (nbs): step optimizer every `accumulate=round(nbs/batch)` iters,
     warming accumulate up from 1 (YOLO's np.interp over warmup iters).
  3. Warmup: first `nw` iters ramp LR 0->target and momentum warmup->target (bias LR capped 0.01 Adam).
  4. Cosine LR schedule: one_cycle(1, lrf, epochs) via LambdaLR, scheduler.step() per epoch.
  5. EMA: ultralytics ModelEMA; update after each optimizer step; EVAL/return the EMA weights.
  6. Weight-decay scaled by batch*accumulate/nbs (YOLO's rule).

The forward+backward is passed in as `loss_fn(model, group, scaler, accumulate) -> (loss_value,
nkpts)`. loss_fn does its OWN per-view AMP forward + `scaler.scale(loss/accumulate).backward()`
per view, so only one view's graph is live at a time (bounds memory on the 8GB card). The core
handles accumulate/step/EMA/warmup/schedule. See `distill_loss_fn` below for the 2D-distill impl
used by both datasets.
"""
import math
import numpy as np
import torch
from ultralytics.utils.torch_utils import ModelEMA
try:
    from ultralytics.utils.torch_utils import one_cycle
except Exception:
    from ultralytics.utils import one_cycle


def distill_loss_fn(dev, amp=True, conf_th=0.3):
    """Return a loss_fn(model, group, scaler, accumulate) for 2D-keypoint distillation.

    PER-VIEW-GROUP backward: one group (multi-view frame) forwards once, its 2D-MSE vs the teacher
    kpts is masked (invisible joints nan-masked; nan_to_num before weighting since nan*0=nan) and
    backpropped scaled by 1/accumulate. Only one group's graph is live at a time -> memory bounded.
    Returns (running loss value, n visible kpts) or (None,0) if nothing visible.
    """
    def _fn(model, group, scaler, accumulate):
        tot, nk, used = 0.0, 0, 0
        for s in group:
            with torch.amp.autocast('cuda', enabled=amp and str(dev).startswith('cuda')):
                out = model(s['imgs'].to(dev, non_blocking=True), s['P_native'].to(dev, non_blocking=True))
                tgt = s['kp2d_tgt'].to(dev); conf = s['kp_conf'].to(dev)
                vis = torch.isfinite(tgt).all(-1) & torch.isfinite(conf) & (conf > conf_th)
                if not vis.any():
                    continue
                w = (torch.nan_to_num(conf).clamp(0, 1) * vis.float())                # (V,17)
                # per-kpt Euclidean px distance (NOT squared-sum): keeps loss O(10-100) so AMP's
                # fp16 doesn't overflow (squared native-px MSE was ~3e5 -> GradScaler inf -> nan).
                dist = (out['kpts2d'] - torch.nan_to_num(tgt)).norm(dim=-1)            # (V,17) px
                loss = (w * dist).sum() / w.sum().clamp(min=1e-6)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss / max(len(group), 1)).backward()      # per-view graph freed here
            tot += loss.item(); nk += int(vis.sum()); used += 1
        if used == 0:
            return None, 0
        return tot / used, nk
    return _fn


def build_optimizer(params, name='AdamW', lr=1e-3, momentum=0.937, decay=5e-4):
    if name in ('Adam', 'AdamW'):
        opt = (torch.optim.AdamW if name == 'AdamW' else torch.optim.Adam)(
            params, lr=lr, betas=(momentum, 0.999), weight_decay=decay)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=decay, nesterov=True)
    return opt


def train_loop(model, loader, loss_fn, *, epochs, dev, lr0=1e-3, lrf=0.01, nbs=64, batch=8,
               momentum=0.937, warmup_epochs=1.0, warmup_momentum=0.8, warmup_bias_lr=0.0,
               decay=5e-4, amp=True, use_ema=True, optimizer='AdamW', clip=10.0,
               on_epoch=None, log=print):
    """Run YOLO-style optimized training. loss_fn(model, group) -> (scalar loss, n_kpts_used).

    on_epoch(ep, ema_model) optional callback for eval each epoch (gets EMA weights).
    Returns the EMA model (YOLO evaluates/saves EMA), or the raw model if use_ema=False.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    accumulate = max(round(nbs / batch), 1)
    decay_scaled = decay * batch * accumulate / nbs                      # YOLO wd scaling
    opt = build_optimizer(params, optimizer, lr0, momentum, decay_scaled)
    lf = one_cycle(1, lrf, epochs)                                       # cosine 1 -> lrf
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dev == 'cuda')
    ema = ModelEMA(model) if use_ema else None

    nb = max(len(loader), 1)                                             # batches/epoch
    nw = max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else -1
    last_opt = -1
    ni = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        run = None
        for group in loader:
            if not group:
                continue
            # ---- warmup: ramp accumulate, LR, momentum (YOLO) ----
            xi = [0, nw]
            if ni <= nw and nw > 0:
                accumulate = max(1, int(np.interp(ni, xi, [1, nbs / batch]).round()))
                for j, pg in enumerate(opt.param_groups):
                    pg['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 0 else 0.0,
                                                  pg.get('initial_lr', lr0) * lf(ep)])
                    if 'momentum' in pg:
                        pg['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])
            # ---- forward+backward delegated to loss_fn (it backprops PER-VIEW to bound memory on
            # the 8GB card, scaling by 1/accumulate); returns (running_loss_value, n_kpts). AMP is
            # applied INSIDE loss_fn per-view so the graph of one view is freed before the next. ----
            lv, nk = loss_fn(model, group, scaler, accumulate)
            if lv is None:
                ni += 1
                continue
            run = lv if run is None else 0.9 * run + 0.1 * lv
            # ---- optimizer step every `accumulate` iters ----
            if ni - last_opt >= accumulate:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, clip)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                if ema:
                    ema.update(model)
                last_opt = ni
            ni += 1
        sched.step()
        log(f'  epoch {ep:3d}/{epochs}  loss {run if run is not None else float("nan"):.1f}  '
            f'lr {opt.param_groups[-1]["lr"]:.2e}  accum {accumulate}', flush=True)
        if on_epoch is not None:
            on_epoch(ep, ema.ema if ema else model)
    return ema.ema if ema else model
