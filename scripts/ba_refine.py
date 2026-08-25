"""Multi-view bundle adjustment refiner (classic per-frame optimisation, NO deep learning).

For every frame t we optimise the 9 joint positions X_t (27 numbers) to minimise

    E(X_t) = SUM_c  w_{c,t} * Huber( || pi_c(X_t) - x_{c,t} || )        # weighted reprojection
           + lam_bone * SUM_(u,v) in LIMBS ( ||X_u,t - X_v,t|| - L_uv )^2   # bone-length prior

This is the "simple algorithm" realisation of exactly the two terms validated in the parked
DL refiner (project_gnn_3d_refiner.md): a confidence-weighted reprojection data term + a
bone-length constancy prior -- but solved per-frame with L-BFGS instead of learned.

Lessons baked in (from the DL-refiner session, docs/context/project_gnn_3d_refiner.md):
  * w_{c,t} is FIXED = the observed YOLO kp confidence. It is NOT a prediction-dependent
    inlier gate -- that gate was self-defeating (it mutes the residual as X drifts off a camera,
    letting the solver escape that camera; reported 1.95px when the real residual was 25px).
    Camera robustness instead comes from a Huber kernel on the residual (fixed-weight + Huber
    = robust AND can't be gamed).
  * L_uv is the PER-TRIAL MEDIAN of the RAW triangulated bone length (self-supervised, no OMC
    leak). It fixes bone VARIANCE (the rubber-band artifact), not absolute bias.
  * The rigidity<->depth tension is real: too-high lam_bone slides X along the unobservable
    camera ray (reproj stays low, true depth drifts, OMC creeps up). So we SWEEP lam_bone and
    judge it against OMC, not just against the energy. There is a sweet spot.

Init from the raw DLT triangulation (mmc). Where the cameras agree the data term is already ~0
(DLT is reproj-optimal), so BA leaves those frames alone; its value is in gappy/occluded frames
where the bone prior fills the missing constraint.

Runs cache-only (needs the .reproj.npz sidecars: per-cam uv/conf/K/dist/R/t). No GPU decode, no
triangulation -- reuses gnn_refiner.project_torch for pi_c and gnn_train.score_trial for OMC truth.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gnn_train as T
import gnn_refiner as G

DEV = "cuda" if torch.cuda.is_available() else "cpu"
JOINTS = G.JOINTS
NJ = len(JOINTS)
_JI = G._JI
# limbs = the skeleton edges whose length is (near-)rigid. Reuse the model's own edge list but DROP
# the hip<->hip / shoulder<->hip / shoulder<->nose girdle+trunk edges that flex a lot; keep the arms
# and the (fairly rigid) shoulder width. These are the lengths the bone prior enforces.
LIMBS = [(_JI[a], _JI[b]) for a, b in [
    ("right_wrist", "right_elbow"), ("right_elbow", "right_shoulder"),
    ("left_wrist", "left_elbow"), ("left_elbow", "left_shoulder"),
    ("right_shoulder", "left_shoulder"),
]]


def trial_bone_lengths(mmc, valid):
    """Per-trial reference length L_uv for each limb = MEDIAN raw triangulated length over frames
    where BOTH endpoints are valid. Self-supervised (no OMC). Returns dict{(u,v): L_mm} + count."""
    L = {}
    for u, v in LIMBS:
        m = valid[:, u] & valid[:, v] & np.isfinite(mmc[:, u]).all(-1) & np.isfinite(mmc[:, v]).all(-1)
        if m.sum() >= 8:
            d = np.linalg.norm(mmc[m, u] - mmc[m, v], axis=1)
            L[(u, v)] = float(np.median(d))
    return L


def _to_dev(t):
    """Move a trial's per-cam sidecar arrays + calib to torch on DEV. Returns a dict of tensors."""
    return dict(
        uv=torch.from_numpy(np.nan_to_num(t["uv"]).astype(np.float32)).to(DEV),        # (T,C,J,2)
        uvc=torch.from_numpy(t["uv_conf"].astype(np.float32)).to(DEV),                 # (T,C,J)
        uvv=torch.from_numpy(t["uv_valid"].astype(bool)).to(DEV),                      # (T,C,J)
        K=torch.from_numpy(t["K"].astype(np.float32)).to(DEV),                         # (C,3,3)
        dist=torch.from_numpy(t["dist"].astype(np.float32)).to(DEV),                   # (C,5)
        R=torch.from_numpy(t["R"].astype(np.float32)).to(DEV),                         # (C,3,3)
        tt=torch.from_numpy(t["t"].astype(np.float32)).to(DEV),                        # (C,3)
    )


# Projected pixel coords are clamped to +-UVMAX before the residual norm. A point behind a camera has
# its z clamped to 1e-3 mm inside project_torch, so the normalised coords are ~1e3 and the distortion
# polynomial reaches ~1e22 px; squaring that in linalg.norm overflows float32 to `inf`. The in-front
# weight for such a point is 0, and `inf * 0 = NaN` (IEEE-754), so ONE behind-camera joint-frame
# poisons the whole energy. NaN compares false against every strong-Wolfe condition, so the line
# search cannot backtrack and the trial blows up. Clamping keeps the residual large-but-finite (it is
# masked to zero weight anyway) and the solve stays well-posed. Removes all 60 catastrophic trials.
UVMAX = 1e6


def _reproj_residual(X, S, huber_px):
    """Weighted Huber reprojection energy, summed over cams. X (B,T,J,3). Returns (energy, wsum).

    FIXED weight = uv_valid * uv_conf * in_front. NOT prediction-gated (the self-defeating bug)."""
    B, T, J, _ = X.shape
    C = S["uv"].shape[1]
    Xp = X.unsqueeze(2).expand(B, T, C, J, 3)
    tot = torch.zeros((), device=X.device); wsum = torch.zeros((), device=X.device)
    h = huber_px
    for c in range(C):
        uvp, infront = G.project_torch(Xp[:, :, c], S["K"][c], S["dist"][c], S["R"][c], S["tt"][c])
        uvp = uvp.clamp(-UVMAX, UVMAX)                                         # see UVMAX note above
        res = torch.linalg.norm(uvp - S["uv"][None, :, c], dim=-1)             # (B,T,J) px
        rob = torch.where(res <= h, 0.5 * res * res / h, res - 0.5 * h)        # Huber
        w = S["uvv"][None, :, c].float() * S["uvc"][None, :, c] * infront.float()
        tot = tot + (rob * w).sum(); wsum = wsum + w.sum()
    return tot, wsum


def _bone_energy(X, L):
    """Bone-length VARIANCE within the trial. X (B,T,J,3). L supplies the limb keys only.

    This penalises a bone whose length CHANGES frame to frame, without asserting what that length
    should be. The earlier form pinned each bone to the per-trial median of the raw triangulation
    (`_bone_energy_pinned` below) -- that pins the solve to a target derived from the very estimate
    being refined, and a wrong median then becomes a constraint.

    Judged against OMC on the full cohort (scripts/ba_free_bone.py, scripts/score_variant_measures.py)
    the variance form at lam=0.05 improves peak elbow angular velocity 0.896 -> 0.905, peak velocity
    0.877 -> 0.888 and max elbow angle 0.894 -> 0.912, with the gain concentrated on the participant
    whose calibration is worst (P19 elbow 0.470 -> 0.674). It costs absolute error on the peaks
    (median |err| 11.0 -> 13.2 deg/s, 46.8 -> 50.4 mm/s), so ranks improve and magnitudes do not.
    """
    e = torch.zeros((), device=X.device)
    for (u, v) in L:
        d = torch.linalg.norm(X[:, :, u] - X[:, :, v], dim=-1)                 # (B,T)
        e = e + ((d - d.mean(dim=1, keepdim=True)) ** 2).sum()
    return e


def _bone_energy_pinned(X, L):
    """Squared deviation from the per-trial reference length L_uv -- the superseded form, kept so the
    comparison is reproducible. Not used by the shipped solve."""
    e = torch.zeros((), device=X.device)
    for (u, v), L_uv in L.items():
        d = torch.linalg.norm(X[:, :, u] - X[:, :, v], dim=-1)
        e = e + ((d - L_uv) ** 2).sum()
    return e


def refine_trial_ba(t, lam_bone, huber_px=20.0, iters=60, lr=None, smooth_w=0.0, fallback_mm=None,
                    anchor_mm_w=0.0, trial_guard_mm=None):
    """Bundle-adjust ONE trial. Optimise X (init=raw mmc) with L-BFGS. Returns refined (T,J,3) np.

    Optimises ALL frames jointly (one tensor) but the energy is separable per-frame EXCEPT the
    optional smooth term -- so this is per-frame BA (+ optional accel prior coupling neighbours).

    fallback_mm: PER-FRAME guard -- any joint-frame BA moved more than this (mm) from the pipeline
    point, or made non-finite, reverts to that pipeline point.
    trial_guard_mm: TRIAL-LEVEL guard (preferred) -- if a joint moves > this mm from the pipeline
    ANYWHERE in the trial, its WHOLE trajectory reverts to the pipeline. A blown-up trial is a broken
    solve throughout (e.g. P19 trial_63: 2/3 cams miscalibrated -> corrupt triangulation, BA amplifies
    it), so reverting the whole joint beats the per-frame fallback (which leaves the between-blowup
    frames wrong too). Only trial_63/328 trips this at 150mm -> zero collateral on good trials."""
    mmc = t["mmc"]; valid = t["valid"]
    Tn = mmc.shape[0]
    L = trial_bone_lengths(mmc, valid)
    if not L:
        return mmc.copy(), {"bone_ref": {}, "note": "no bone ref"}
    S = _to_dev(t)
    # init from raw DLT; NaN raw points -> filled with 0 but masked out of the data term (uvv gates)
    X0 = np.nan_to_num(mmc).astype(np.float32)
    Xinit = torch.from_numpy(X0[None]).to(DEV)                                # (1,T,J,3) pipeline anchor
    finite_init = torch.from_numpy(np.isfinite(mmc).all(-1)[None]).to(DEV)    # only anchor real points
    X = Xinit.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([X], lr=(lr or 1.0), max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        reproj, wsum = _reproj_residual(X, S, huber_px)
        data = reproj / wsum.clamp(min=1.0)
        bone = _bone_energy(X, L) / max(len(L) * Tn, 1)
        loss = data + lam_bone * bone
        if smooth_w > 0 and Tn >= 3:
            acc = X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
            loss = loss + smooth_w * (acc ** 2).sum() / max((Tn - 2) * NJ, 1)
        if anchor_mm_w > 0:
            # BOUND the solve: soft quadratic pull toward the pipeline init (LM-style damping / trust
            # region). Moving far off the pipeline point costs quadratically -> the solver can't run
            # off into the distortion far-field. Scaled per-mm^2 so it only bites on gross departures.
            dev2 = ((X - Xinit) ** 2).sum(-1) * finite_init.float()
            loss = loss + anchor_mm_w * dev2.sum() / finite_init.float().sum().clamp(min=1.0)
        loss.backward()
        return loss

    opt.step(closure)
    Xr = X.detach()[0].cpu().numpy()
    # restore NaN where raw was NaN (BA has no data there beyond the bone prior; keep it honest for
    # scoring -- score_trial masks on `valid` anyway, but don't invent fully-unconstrained points)
    Xr[~np.isfinite(mmc).all(-1)] = np.nan
    n_fallback = 0
    if fallback_mm is not None:
        # DLT-as-BACKUP: the distortion-aware solve occasionally runs off into the far-field
        # (blow-ups up to 650mm on ~17% of trials). Any joint-frame BA moved MORE than fallback_mm
        # from the pipeline point -- or made non-finite -- reverts to the pipeline point. Guarantees
        # BA can never do WORSE than the incumbent; we keep BA's gain only where it stayed sane.
        # float64: at float32 a 1e20 blow-up squares to 1e40 and `moved` overflows to inf, which the
        # old `np.isfinite(moved) &` term then read as "not a revert" -- the guard silently skipped
        # exactly the worst cases. Non-finite `moved` over a finite pipeline point IS a revert.
        moved = np.linalg.norm(Xr.astype(np.float64) - mmc.astype(np.float64), axis=-1)   # (T,J) mm
        bad = ~np.isfinite(Xr).all(-1) | ~np.isfinite(moved) | (moved > fallback_mm)
        # only revert where the pipeline HAS a point (else leave NaN)
        bad &= np.isfinite(mmc).all(-1)
        Xr[bad] = mmc[bad]
        n_fallback = int(bad.sum())
    n_guarded = 0
    if trial_guard_mm is not None:
        # TRIAL-LEVEL guard: if a joint EVER moves > trial_guard_mm from the pipeline, its whole
        # trajectory is a broken solve -> revert that joint entirely to the pipeline. (P19 trial_63 =
        # miscalibrated cams; the only trial that trips 150mm in 328.) Per joint, so a single bad joint
        # doesn't discard the good ones.
        moved = np.linalg.norm(Xr.astype(np.float64) - mmc.astype(np.float64), axis=-1)   # (T,J)
        finite_pair = np.isfinite(Xr).all(-1) & np.isfinite(mmc).all(-1)
        for j in range(NJ):
            mj = moved[:, j]
            # same float64 / non-finite rule as the per-frame guard above
            if ((finite_pair[:, j] & ~np.isfinite(mj)) | (np.isfinite(mj) & (mj > trial_guard_mm))).any():
                col = np.isfinite(mmc[:, j]).all(-1)
                Xr[col, j] = mmc[col, j]
                n_guarded += 1
    return Xr, {"bone_ref": {f"{JOINTS[u]}-{JOINTS[v]}": round(v_, 1) for (u, v), v_ in L.items()},
                "n_fallback": n_fallback, "n_guarded_joints": n_guarded}


# --------------------------------------------------------------------------- metrics (RAW vs BA)
def bone_std_trial(P, valid):
    out = []
    for u, v in LIMBS:
        m = valid[:, u] & valid[:, v] & np.isfinite(P[:, u]).all(-1) & np.isfinite(P[:, v]).all(-1)
        if m.sum() >= 8:
            out.append(np.std(np.linalg.norm(P[m, u] - P[m, v], axis=1)))
    return float(np.mean(out)) if out else np.nan


def reproj_med(P, t):
    """median per-cam reprojection residual (px) over valid arm joints -- POSITION faithfulness."""
    S = _to_dev(t)
    X = torch.from_numpy(np.nan_to_num(P).astype(np.float32))[None].to(DEV)
    with torch.no_grad():
        rr = []
        for c in range(S["uv"].shape[1]):
            uvp, _ = G.project_torch(X, S["K"][c], S["dist"][c], S["R"][c], S["tt"][c])
            uvp = uvp[0].cpu().numpy()
            mk = t["uv_valid"][:, c] & np.isfinite(P).all(-1)
            rr.append(np.linalg.norm(uvp - t["uv"][:, c], axis=-1)[mk])
    return float(np.median(np.concatenate(rr))) if rr else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["P07", "P08", "P15", "P17", "P19"])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.2, 1.0],
                    help="lam_bone sweep. 0 = pure reprojection (should reproduce raw DLT).")
    ap.add_argument("--huber-px", type=float, default=20.0)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--smooth-w", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="cap #trials for a fast smoke run (0=all)")
    a = ap.parse_args()

    print(f"device={DEV}  loading trials (parts={a.parts}, reproj sidecars)...", flush=True)
    trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in a.parts]
    if a.limit:
        trials = trials[:a.limit]
    print(f"trials: {len(trials)}   limbs: {len(LIMBS)}   lams: {a.lams}", flush=True)
    print(f"NOTE: judging each lam_bone against OMC (wrist/PA), not just the energy -- there is a\n"
          f"      rigidity<->depth sweet spot (too rigid => depth drifts along the camera ray).\n",
          flush=True)

    # RAW baseline once
    print("=== RAW (DLT) baseline ===", flush=True)
    raw = {"bone": [], "reproj": [], "wr": [], "pa": [], "elb": [], "jit": []}
    for t in trials:
        raw["bone"].append(bone_std_trial(t["mmc"], t["valid"]))
        raw["reproj"].append(reproj_med(t["mmc"], t))
        sc = T.score_trial(t["mmc"], t["omc"], t["valid"], t["side"])
        raw["wr"].append(sc["wr"]); raw["pa"].append(sc["pa"]); raw["elb"].append(sc["elb"])
        raw["jit"].append(sc["jit"])
    med = lambda arr: float(np.nanmedian([z for z in arr if np.isfinite(z)]))
    print(f"  RAW   bone {med(raw['bone']):5.2f}mm  reproj {med(raw['reproj']):5.1f}px  "
          f"OMCwrist {med(raw['wr']):5.1f}mm  PA {med(raw['pa']):5.1f}mm  elb {med(raw['elb']):4.1f}deg  "
          f"jit {med(raw['jit']):6.0f}", flush=True)

    for lam in a.lams:
        t0 = time.time()
        agg = {"bone": [], "reproj": [], "wr": [], "pa": [], "elb": [], "jit": []}
        for i, t in enumerate(trials):
            Xr, _ = refine_trial_ba(t, lam, huber_px=a.huber_px, iters=a.iters, smooth_w=a.smooth_w)
            agg["bone"].append(bone_std_trial(Xr, t["valid"]))
            agg["reproj"].append(reproj_med(Xr, t))
            sc = T.score_trial(Xr, t["omc"], t["valid"], t["side"])
            agg["wr"].append(sc["wr"]); agg["pa"].append(sc["pa"]); agg["elb"].append(sc["elb"])
            agg["jit"].append(sc["jit"])
            if (i + 1) % 20 == 0:
                print(f"    [lam={lam}] {i+1}/{len(trials)} trials  ({time.time()-t0:.0f}s)", flush=True)
        print(f"  lam={lam:<5g} bone {med(agg['bone']):5.2f}mm  reproj {med(agg['reproj']):5.1f}px  "
              f"OMCwrist {med(agg['wr']):5.1f}mm  PA {med(agg['pa']):5.1f}mm  elb {med(agg['elb']):4.1f}deg  "
              f"jit {med(agg['jit']):6.0f}   ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== BA SWEEP DONE ===", flush=True)


if __name__ == "__main__":
    main()
