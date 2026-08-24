"""peak_elbow_angular_velocity from TWO rigid segment optical-flow clouds vs OMC (reproduction).

REPRODUCES the lost 2026-07-23 result (WORKLOG ~line 4098-4113):
    PEAK (p95): raw-kp-diff +29.2% | SmoothNet-diff +25.2% | CLOUD +3.6% (|err| 11.9%)
The driver + its scratchpad cache (elbow_cohort.pkl) were lost; only the reusable classes
survive. This rebuilds the driver from those classes and re-scores it. It is a VALIDATION run:
if it does not reproduce, the numbers are reported honestly.

THE METHOD (the articulated form of the arm-cloud idea):
  * Seed a FILLED cylinder of ~n points on the FOREARM segment (wrist<->elbow 3D joints) and
    another on the UPPER-ARM segment (elbow<->shoulder). Correspondence across cameras is exact
    by construction (one 3D seed projects to the same track identity in every view).
  * Track each seed with PyrLK within each camera frame t->t+1 (forward-backward check),
    triangulate the tracked points back to 3D each frame (consensus-gated per track), then Kabsch
    consecutive 3D clouds -> per-segment angular velocity omega (rad/s). A SINGLE whole-arm rigid
    fit FAILS because elbow flexion is a 2nd DOF -- hence TWO separate segment clouds.
  * Elbow flexion angular rate = (omega_forearm - omega_upperarm) . n_hat, where n_hat is the
    flexion axis = normalized cross(forearm_dir, upperarm_dir) from LOW-PASSED keypoint segment
    vectors (the RATE never touches a differentiated keypoint; only the axis DIRECTION does, and
    it is low-passed). peak = p95 of the low-passed |elbow flexion rate| over the trial.

TRUTH (OMC): elbow angle = shoulder-elbow-wrist from the mocap markers (H._load_omc), low-passed,
d/dt -> angular rate, peak = p95. BASELINE for contrast: the same angle-rate peak from the
triangulated MMC pose (raw) and from the SmoothNet-refined pose (keypoint-differentiated). This is
exactly the eav = |gradient(elbow_angle)|*fps used in results_v3_delta._angle_scalars.

CLEAN-calib cohort only: P07 (calib RMS 1.55) and P08 (0.85). P17/P19 are miscalibrated and
excluded. Cache-first: per-trial results in cache/elbow_flow/<part>__<trial>.npz.

    python scripts/elbow_flow_angular.py --trials 3            # 3 trials each of P07,P08
    python scripts/elbow_flow_angular.py --parts P07 --trials 6
    python scripts/elbow_flow_angular.py --force               # ignore cache, recompute
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "cache" / "elbow_flow"
FPS = 60.0
PEAK_PCTL = 95.0          # the WORKLOG's "PEAK (p95)"

# CLEAN-calibration cohort (study CSV: P07 RMS 1.55, P08 0.85). P17/P19 miscalibrated -> excluded.
COHORT = {
    "P07": ([f"trial_{i}_L_unaffected" for i in range(10, 16)], "left"),
    "P08": ([f"trial_{i}_R_unaffected" for i in range(10, 16)], "right"),
}


# ============================================================================================
# Segment-axis cloud tracker: CloudTracker machinery, but the seed cylinder is aligned to an
# ARBITRARY segment axis (between two joints) instead of world-Z, and FILLED (interior radii,
# not just the surface) so a thin near-cylindrical limb still populates enough trackable texture.
# ============================================================================================
def segment_seed(p0: np.ndarray, p1: np.ndarray, radius: float, n: int, seed: int = 0
                 ) -> np.ndarray:
    """Filled cylinder of n points spanning the segment p0<->p1, radius `radius` -> (n,3) world.

    Adapts cloud_track.surface_seed (which is WORLD-Z aligned and hollow) to (a) lie along the
    arbitrary axis (p1 - p0) between two 3D joints and (b) FILL the volume: a limb is thinner and
    less textured than a cup, so sampling interior radii (sqrt for area-uniform) as well as the
    surface gives PyrLK more real texture to latch onto. Being wrong about the exact surface costs
    nothing -- the seeds are only a STARTING PLACE for the flow, which then follows whatever real
    texture sits at that pixel; what matters is that the SAME 3D point defines the seed in every
    camera (exact cross-view correspondence by construction).
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-6:
        return np.repeat(p0[None], n, 0)
    axis = axis / L
    # two orthonormal directions perpendicular to the axis
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.15, 0.85, n)                 # fraction along the segment (avoid the joints)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n)) # area-uniform fill
    th = rng.uniform(0, 2 * np.pi, n)
    pts = (p0[None] + t[:, None] * (p1 - p0)[None]
           + (r * np.cos(th))[:, None] * e1[None]
           + (r * np.sin(th))[:, None] * e2[None])
    return pts


def _import_cloud():
    from pipeline import cloud_track as CT
    return CT


def make_segment_tracker(calib, n_seed, radius):
    """A CloudTracker whose seed()/topup() lay the cylinder along the segment axis.

    Everything else (PyrLK forward-backward, consensus-gated 3D lift, Kabsch+RANSAC, the rigidity
    gate, the keypoint anchor) is the stock class, unchanged -- only the seed GEOMETRY differs.
    The segment endpoints are supplied per-frame via `seg_ends` before update() (see run_segment).
    """
    CT = _import_cloud()

    class SegmentCloudTracker(CT.CloudTracker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._seg_ends = None   # (p0, p1) world, set each frame before update()

        def set_ends(self, p0, p1):
            self._seg_ends = (np.asarray(p0, float), np.asarray(p1, float))

        def _seed_points(self, seed_i):
            p0, p1 = self._seg_ends
            return segment_seed(p0, p1, self.radius, self.n_seed, seed=seed_i)

        # override seed() to place a segment-axis cylinder (centre arg unused for geometry, but
        # kept for the near-hemisphere visibility test -> use the segment midpoint).
        def seed(self, gray, centre):
            import numpy as _np
            from pipeline.cloud_track import _project, _visible
            P = self._seed_points(0)
            mid = 0.5 * (self._seg_ends[0] + self._seg_ends[1])
            self._px = {}
            for cam, cal in self.calib.items():
                if cam not in gray:
                    continue
                pts = _np.full((len(P), 2), _np.nan)
                h, w = gray[cam].shape
                for kk, X in enumerate(P):
                    if not _visible(cal, X, mid):
                        continue
                    u = _project(cal, X)
                    if u is not None and 5 <= u[0] < w - 5 and 5 <= u[1] < h - 5:
                        pts[kk] = u
                self._px[cam] = pts
            self._prev_gray = {c: g.copy() for c, g in gray.items()}
            self._offset_ref = {}
            self._cohort_age = 0
            self._prev_cloud = None
            self._prev_ids = None
            self._track_prev3d = {}

        def topup(self, gray, centre):
            import numpy as _np
            from pipeline.cloud_track import _project, _visible
            live = None
            for pts in self._px.values():
                m = _np.isfinite(pts).all(1)
                live = m if live is None else (live | m)
            if live is None:
                return 0
            dead = _np.flatnonzero(~live)
            if not len(dead):
                return 0
            P = self._seed_points(int(self._topup_n) + 1)
            mid = 0.5 * (self._seg_ends[0] + self._seg_ends[1])
            self._topup_n += 1
            for cam, cal in self.calib.items():
                if cam not in gray or cam not in self._px:
                    continue
                h, w = gray[cam].shape
                for kk in dead:
                    X = P[kk % len(P)]
                    if not _visible(cal, X, mid):
                        continue
                    u = _project(cal, X)
                    if u is not None and 5 <= u[0] < w - 5 and 5 <= u[1] < h - 5:
                        self._px[cam][kk] = u
                self._offset_ref.pop(cam, None)
            for kk in dead:
                self._track_prev3d.pop(int(kk), None)
            return len(dead)

    return SegmentCloudTracker(calib, radius=radius, height=radius * 2, n_seed=n_seed,
                               units_per_metre=1.0, anchor_px=None, gate_consensus=True,
                               min_inliers=6, max_scale_dev=None, warmup=3)


# ============================================================================================
# per-segment angular velocity: run one segment cloud over the trial -> omega(3,) per frame
# ============================================================================================
def run_segment(part, trial, calib, ends3d, kp_by_cam_all, caps_frames, n, radius, n_seed,
                label=""):
    """Track one segment cloud through the trial. Returns omega (n,3) rad/s (NaN where no fit).

    ends3d[f] = (p0, p1) the two 3D joint positions (mm) bounding the segment at frame f (from the
    LOW-PASSED triangulated pose -- used only to (re)seed, never differentiated).
    kp_by_cam_all[f] = {cam: 2d px} of the segment MIDPOINT-ish keypoint for the anchor (optional).
    caps_frames[f] = {cam: gray frame}.
    """
    import numpy as _np
    trk = make_segment_tracker(calib, n_seed, radius)
    omega = _np.full((n, 3), _np.nan)
    for f in range(n):
        gray = caps_frames[f]
        p = ends3d[f]
        if len(gray) < 2 or p is None or not (_np.isfinite(p[0]).all() and _np.isfinite(p[1]).all()):
            continue
        mid = 0.5 * (p[0] + p[1])
        trk.set_ends(p[0], p[1])
        kp = kp_by_cam_all[f] if kp_by_cam_all is not None else None
        r = trk.update(gray, mid, dt=1.0 / FPS, kp_by_cam=kp)
        if r is not None and r.angular_velocity is not None:
            omega[f] = r.angular_velocity
    return omega


def _lp_vec(x):
    """low-pass each column of an (n,k) array, interpolating across NaN gaps (H._lp per column)."""
    import compare_pose_omc_delta as H
    x = np.asarray(x, float)
    if x.ndim == 1:
        return H._lp(x)
    return np.stack([H._lp(x[:, k]) for k in range(x.shape[1])], 1)


# ============================================================================================
# one trial
# ============================================================================================
def process_trial(part, trial, side, radius, n_seed, force=False, t0=None):
    import cv2
    import compare_pose_omc_delta as H
    import results_v3_delta as R

    cache_f = CACHE / f"{part}__{trial}.npz"
    if cache_f.exists() and not force:
        d = dict(np.load(cache_f, allow_pickle=True))
        return {k: (v.item() if getattr(v, "ndim", 1) == 0 else v) for k, v in d.items()}, "cache"

    sh_j, el_j, wr_j = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    calib = R._calib(part)

    mmc, n = H._load_mmc(part, trial)
    omc = H._load_omc(part, trial, n)
    lag, synccorr = H._find_lag(mmc[wr_j], omc[wr_j])
    omc = {j: R._shift(v, lag) for j, v in omc.items()}

    # --- OMC truth: elbow angle rate peak (p95) ---
    def elbow_angle(P):
        u, v = P[sh_j] - P[el_j], P[wr_j] - P[el_j]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    ao = H._lp(elbow_angle(omc))
    eav_o = np.abs(np.gradient(ao)) * FPS
    peak_omc = float(np.nanpercentile(eav_o[np.isfinite(eav_o)], PEAK_PCTL))

    # --- BASELINE: keypoint-differentiated angle rate, raw pose + SmoothNet pose ---
    am = H._lp(elbow_angle(mmc))
    eav_raw = np.abs(np.gradient(am)) * FPS
    peak_raw = float(np.nanpercentile(eav_raw[np.isfinite(eav_raw)], PEAK_PCTL))

    sn = {j: R._smooth_joint(mmc[j]) for j in (sh_j, el_j, wr_j)}
    a_sn = H._lp(elbow_angle(sn))
    eav_sn = np.abs(np.gradient(a_sn)) * FPS
    peak_sn = float(np.nanpercentile(eav_sn[np.isfinite(eav_sn)], PEAK_PCTL))

    # --- CLOUD: two segment clouds ---
    # segment endpoints from the LOW-PASSED, gap-filled triangulated pose (seed only, no derivative)
    sh_f = _lp_vec(R._fill(mmc[sh_j])); el_f = _lp_vec(R._fill(mmc[el_j]))
    wr_f = _lp_vec(R._fill(mmc[wr_j]))
    fore_ends = [(wr_f[f], el_f[f]) for f in range(n)]      # forearm: wrist<->elbow
    upper_ends = [(el_f[f], sh_f[f]) for f in range(n)]     # upper-arm: elbow<->shoulder

    # per-camera 2D keypoints for the anchor (elbow px for both -- it is the shared, best-tracked
    # joint; the anchor is generous, so exact choice is not load-bearing)
    import flow_velocity_probe as F
    el_px = F.load_wrist_px(part, trial, el_j)
    wr_px = F.load_wrist_px(part, trial, wr_j)
    cams = [c for c in el_px if c in calib]

    # decode all cameras once, frame-synced
    caps = {}
    for c in cams:
        v = H.DELTA / part / "staged" / f"delta_{part}_{trial}.{c.split('_')[1]}.mp4"
        if v.exists():
            caps[c] = cv2.VideoCapture(str(v))
    if len(caps) < 2:
        raise RuntimeError(f"{part} {trial}: <2 cameras with staged video")

    caps_frames = []
    for f in range(n):
        gray = {}
        for c, cap in caps.items():
            ok, im = cap.read()
            if ok:
                gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        caps_frames.append(gray)
    for cap in caps.values():
        cap.release()

    # anchors per frame (elbow px for the segment endpoints)
    def kp_at(px, f):
        return {c: px[c][f] for c in cams
                if c in px and f < len(px[c]) and np.isfinite(px[c][f]).all()}
    fore_kp = [kp_at(wr_px, f) for f in range(n)]   # forearm anchored on wrist px
    upper_kp = [kp_at(el_px, f) for f in range(n)]  # upper-arm anchored on elbow px

    cal_use = {c: calib[c] for c in caps}
    om_fore = run_segment(part, trial, cal_use, fore_ends, fore_kp, caps_frames, n,
                          radius, n_seed, "forearm")
    om_upper = run_segment(part, trial, cal_use, upper_ends, upper_kp, caps_frames, n,
                           radius, n_seed, "upperarm")

    # flexion axis n_hat = cross(forearm_dir, upperarm_dir) from LOW-PASSED keypoint segment vectors
    fore_dir = wr_f - el_f
    upper_dir = sh_f - el_f
    nhat = np.cross(fore_dir, upper_dir)
    nn = np.linalg.norm(nhat, axis=1, keepdims=True)
    nhat = nhat / (nn + 1e-9)

    # elbow flexion rate = (omega_fore - omega_upper) . n_hat  (rad/s -> deg/s for the Murphy unit)
    om_rel = om_fore - om_upper
    rate = np.abs((om_rel * nhat).sum(1)) * (180.0 / np.pi)   # deg/s, magnitude
    rate_lp = H._lp(rate)
    fin = np.isfinite(rate_lp)
    peak_cloud = float(np.nanpercentile(rate_lp[fin], PEAK_PCTL)) if fin.sum() > 10 else float("nan")
    cov = float(np.isfinite(rate).mean())

    out = {
        "part": part, "trial": trial, "side": side, "n": n, "lag": lag, "synccorr": synccorr,
        "peak_omc": peak_omc, "peak_raw": peak_raw, "peak_sn": peak_sn, "peak_cloud": peak_cloud,
        "cloud_cov": cov, "n_cams": len(caps),
        # tidy per-frame series for later inspection
        "rate_cloud": rate_lp, "eav_omc": H._lp(eav_o), "eav_raw": H._lp(eav_raw),
        "eav_sn": H._lp(eav_sn), "om_fore": om_fore, "om_upper": om_upper,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache_f, **out)
    return out, "computed"


def _pcterr(mmc, omc):
    return (mmc - omc) / omc * 100.0 if omc else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["P07", "P08"])
    ap.add_argument("--trials", type=int, default=3, help="trials per participant")
    ap.add_argument("--radius", type=float, default=45.0, help="mm, filled-cylinder radius (limb)")
    ap.add_argument("--nseed", type=int, default=48)
    ap.add_argument("--force", action="store_true", help="ignore cache, recompute")
    a = ap.parse_args(argv)

    import compare_pose_omc_delta as H
    H.use_good_cams()

    todo = []
    for part in a.parts:
        trials, side = COHORT[part]
        for trial in trials[:a.trials]:
            todo.append((part, trial, side))

    print(f"=== elbow_flow_angular: {len(todo)} trials, radius={a.radius}mm nseed={a.nseed} "
          f"peak=p{PEAK_PCTL:.0f} ===", flush=True)
    print(f"cohort (clean calib): {a.parts}; cache dir {CACHE}", flush=True)

    rows, n_ok, n_fail = [], 0, 0
    t0 = time.time()
    for i, (part, trial, side) in enumerate(todo, 1):
        tt = time.time()
        try:
            out, sr0 = process_trial(part, trial, side, a.radius, a.nseed, force=a.force, t0=t0)
            n_ok += 1
            rows.append(out)
            print(f"[{i}/{len(todo)}] {part} {trial:28} {sr0:8}  "
                  f"OMC {out['peak_omc']:6.1f}  raw {out['peak_raw']:6.1f}  "
                  f"sn {out['peak_sn']:6.1f}  cloud {out['peak_cloud']:6.1f} deg/s  "
                  f"cov {out['cloud_cov']*100:3.0f}%  ({time.time()-tt:.0f}s, {time.time()-t0:.0f}s tot)",
                  flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[{i}/{len(todo)}] {part} {trial:28} FAILED: {type(e).__name__}: {e}",
                  flush=True)

    print(f"\n=== PROCESSING CHECK: {n_ok} processed / {n_fail} failed of {len(todo)} ===",
          flush=True)
    if not rows:
        print("no trials succeeded"); return

    # ---- per-trial table ----
    print(f"\n{'='*100}")
    print("peak_elbow_angular_velocity (deg/s) — per trial, %error vs OMC")
    print(f"{'='*100}")
    print(f"{'trial':32} {'OMC':>7} | {'raw':>7} {'err%':>7} | {'SmoothN':>8} {'err%':>7} | "
          f"{'CLOUD':>7} {'err%':>7} | {'cov':>4}")
    print("-" * 100)
    for r in rows:
        tag = f"{r['part']}_{r['trial'].split('_')[1]}"
        print(f"{tag:32} {r['peak_omc']:7.1f} | "
              f"{r['peak_raw']:7.1f} {_pcterr(r['peak_raw'], r['peak_omc']):+7.1f} | "
              f"{r['peak_sn']:8.1f} {_pcterr(r['peak_sn'], r['peak_omc']):+7.1f} | "
              f"{r['peak_cloud']:7.1f} {_pcterr(r['peak_cloud'], r['peak_omc']):+7.1f} | "
              f"{r['cloud_cov']*100:3.0f}%")

    # ---- cohort summary: signed %bias (median) + |err|% (median) ----
    def summarize(key):
        errs = np.array([_pcterr(r[key], r["peak_omc"]) for r in rows if np.isfinite(r[key])])
        errs = errs[np.isfinite(errs)]
        if not len(errs):
            return float("nan"), float("nan"), 0
        return float(np.median(errs)), float(np.median(np.abs(errs))), len(errs)

    print("\n" + "=" * 70)
    print(f"COHORT SUMMARY (n={len(rows)} trials, {a.parts})   — reproduction target: "
          f"cloud +3.6% bias / 11.9% |err|")
    print("=" * 70)
    print(f"{'method':16} {'median %bias':>14} {'median |err|%':>15} {'n':>4}")
    print("-" * 55)
    for key, nm in [("peak_raw", "raw kp-diff"), ("peak_sn", "SmoothNet-diff"),
                    ("peak_cloud", "CLOUD (2-seg)")]:
        b, ae, nn = summarize(key)
        print(f"{nm:16} {b:+13.1f}% {ae:14.1f}% {nn:4d}")

    b_c, ae_c, _ = summarize("peak_cloud")
    print("\nREPRODUCTION VERDICT:")
    print(f"  target : cloud ~+3.6% bias / ~11.9% |err|")
    print(f"  measured: cloud {b_c:+.1f}% bias / {ae_c:.1f}% |err|")
    near = (abs(b_c - 3.6) <= 8.0) and (abs(ae_c - 11.9) <= 8.0)
    print(f"  -> {'REPRODUCES (within tolerance)' if near else 'DOES NOT cleanly reproduce'} "
          f"(and cloud {'beats' if ae_c < summarize('peak_raw')[1] else 'does NOT beat'} "
          f"raw kp-diff on |err|)")


if __name__ == "__main__":
    main()
