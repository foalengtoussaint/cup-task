"""Classical inverse kinematics for the iMove fast preview pipeline.

Lifts the triangulated 3D keypoint trajectory from PersonKeypointFast
into body-model qpos via two-step IK:

  1. One-shot body-scale fit on 10-20 high-confidence sampled frames
     (legs near rest-pose via soft `(scale - 1.0)` prior since they're
     desk-occluded for the iMove seated drink task).
  2. Per-frame qpos fit with frozen body_scale, warm-started from the
     previous frame's solution, velocity-penalty regularised for
     temporal smoothness.

Engine: plain MuJoCo (`mj_loadXML` + `mj_forward`) for FK + scipy
`least_squares` Levenberg-Marquardt for the IK solve. ~10-30 ms per
frame on CPU. No GPU, no JAX.

Use-case-specific specialisation (controlled by `lock_legs_seated`):
the iMove seated drink task has the subject's legs under a desk
(occluded keypoints) and the legs don't move meaningfully during the
task. We therefore (a) lock the 14 leg DOFs to a seated rest pose,
(b) drop the noisy leg-site residuals from the IK, and (c) initialise
the free-joint translation from the observed mid-hip keypoint each
frame so the pelvis tracks the subject directly instead of relying on
the optimiser to find it. This brings the fit from "wanders around"
to "tracks the upper body" for a fraction of the search space.

Output shape matches the canonical `body_models.KinematicReconstruction`
so the rerun viewer's biomech rendering works unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Seated-mode constants
# ─────────────────────────────────────────────────────────────────────────
#
# The 14 leg DOFs in humanoid_torque_rl.xml. When `lock_legs_seated=True`
# we freeze these to a seated rest pose and drop them from the optimiser's
# variables. See the corresponding LEG_SITE_NAMES below for the
# residual-side counterpart — without dropping the leg sites the locked
# leg pose still appears as a large fixed residual that the upper body
# tries to absorb via the root joint.

LEG_JOINT_NAMES = (
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r",  "ankle_angle_r",  "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l",  "ankle_angle_l",  "subtalar_angle_l", "mtp_angle_l",
)

# Sensible seated rest pose: thighs roughly horizontal (hip flexion ~80°),
# knees bent ~80° (knee_angle is the model's "knee extension", so seated
# = negative). Values in radians.
SEATED_LEG_QPOS = {
    "hip_flexion_r":   1.4,
    "hip_adduction_r": 0.0,
    "hip_rotation_r":  0.0,
    "knee_angle_r":   -1.4,
    "ankle_angle_r":   0.0,
    "subtalar_angle_r": 0.0,
    "mtp_angle_r":     0.0,
    "hip_flexion_l":   1.4,
    "hip_adduction_l": 0.0,
    "hip_rotation_l":  0.0,
    "knee_angle_l":   -1.4,
    "ankle_angle_l":   0.0,
    "subtalar_angle_l": 0.0,
    "mtp_angle_l":     0.0,
}

# Lower-limb site names from movi_joint_names — these are the keypoints
# we drop from the IK residual when running seated. We DELIBERATELY keep
# the pelvis anchors (lasis/rasis/lpsis/rpsis/lbum/rbum/LHip/RHip/CHip/
# mhip) because those constrain the free-joint translation + orientation.
LEG_SITE_NAMES = frozenset({
    # right thigh / knee region
    "rfrontthigh", "rthigh", "rfrontinnerthigh", "rinnerknee", "rknem", "RKnee",
    # right shin / ankle
    "rshin", "rankm", "RAnkle",
    # right foot
    "RHeel", "rfifthmetatarsal", "rfirstmetatarsal", "RBigToe", "rfourthtoe", "RFoot",
    # left thigh / knee
    "lfrontthigh", "lthigh", "lfrontinnerthigh", "linnerknee", "lknem", "LKnee",
    # left shin / ankle
    "lshin", "lankm", "LAnkle",
    # left foot
    "LHeel", "lfifthmetatarsal", "lfirstmetatarsal", "LBigToe", "lfourthtoe", "LFoot",
})


def _build_lock_tables(model, lock_legs_seated: bool):
    """Compute the locked-qpos indices + their fixed values.

    Returns:
        locked_qpos_idx : (L,) int64  — indices into qpos
        locked_qpos_val : (L,) float64
        free_qpos_idx   : (n_dofs - L,) int64
    """
    import mujoco
    n_dofs = int(model.nq)
    if not lock_legs_seated:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.arange(n_dofs, dtype=np.int64))
    locked_idx, locked_val = [], []
    for jname in LEG_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue                          # joint not in this XML; skip silently
        qadr = int(model.jnt_qposadr[jid])
        locked_idx.append(qadr)
        locked_val.append(SEATED_LEG_QPOS[jname])
    locked_idx = np.asarray(locked_idx, dtype=np.int64)
    locked_val = np.asarray(locked_val, dtype=np.float64)
    free_mask = np.ones(n_dofs, dtype=bool)
    free_mask[locked_idx] = False
    free_qpos_idx = np.where(free_mask)[0].astype(np.int64)
    return locked_idx, locked_val, free_qpos_idx


# ─────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────

def select_scale_frames(
    keypoints3d_conf: np.ndarray,   # (T, J) per-joint confidence at each frame
    *,
    n_frames: int = 15,
    min_visible_joints: int = 30,
) -> np.ndarray:
    """Pick `n_frames` well-distributed high-confidence frames.

    Buckets the recording into n_frames temporal segments, picks the
    highest-mean-confidence frame from each bucket. Skips buckets where
    no frame has ≥ min_visible_joints joints with conf>0.

    Returns array of frame indices, length ≤ n_frames.
    """
    T, _J = keypoints3d_conf.shape
    if T == 0:
        return np.empty(0, dtype=np.int32)
    n_visible_per_frame = (keypoints3d_conf > 0).sum(axis=-1)
    mean_conf_per_frame = keypoints3d_conf.mean(axis=-1)

    selected = []
    n_frames = min(n_frames, T)
    bucket_edges = np.linspace(0, T, n_frames + 1, dtype=int)
    for b in range(n_frames):
        lo, hi = int(bucket_edges[b]), int(bucket_edges[b + 1])
        if hi <= lo:
            continue
        eligible = np.arange(lo, hi)
        eligible = eligible[n_visible_per_frame[eligible] >= min_visible_joints]
        if len(eligible) == 0:
            continue
        # Pick highest-mean-conf within bucket
        best = eligible[mean_conf_per_frame[eligible].argmax()]
        selected.append(int(best))
    return np.array(sorted(set(selected)), dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────
# Keypoint ↔ site mapping (BML-MoVI 87 → humanoid_torque_rl)
# ─────────────────────────────────────────────────────────────────────────
#
# The body_models humanoid_torque_rl.xml has named sites that correspond
# to a subset of the 87 BML-MoVI joints. The mapping is one-directional:
# for each model site, list which BML-MoVI index represents the same
# anatomical point. Joints in BML-MoVI that aren't in the model (or vice
# versa) are simply not used as residuals.
#
# The list below is conservative — only sites where the anatomical
# correspondence is unambiguous. Extend if needed; the body_models
# repo's existing `joint_names = None` default uses the canonical
# `get_joint_names()` mapping internally.

def bml_movi_87_to_humanoid_site_mapping() -> dict[str, int]:
    """Return {site_name: bml_movi_87_index}.

    In `humanoid_torque_rl.xml`, sites are named identically to the
    BML-MoVI joints — so the mapping is trivially the inverse of the
    canonical `movi_joint_names` list from
    `body_models.biomechanics_mjx.forward_kinematics`. Importing it
    from the maintainer module keeps the mapping authoritative (any
    future reorder in the upstream list propagates automatically).

    A site_name not present in the model XML contributes no residual;
    a bml_movi_87 index not in the model is just ignored.
    """
    from movi_names import movi_joint_names   # vendored (see scripts/biomech/README)
    return {name: i for i, name in enumerate(movi_joint_names)}


# ─────────────────────────────────────────────────────────────────────────
# MuJoCo loading + FK
# ─────────────────────────────────────────────────────────────────────────

def load_mujoco_model(xml_path: str | Path):
    """Load the body XML into a MuJoCo model + data pair.

    Returns (model, data, site_names, jnt_range).
    """
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
        for i in range(model.nsite)
    ]
    jnt_range = np.asarray(model.jnt_range, dtype=np.float64)  # (njnt, 2)
    return model, data, site_names, jnt_range


def fk_site_positions(model, data, qpos: np.ndarray) -> np.ndarray:
    """Forward kinematics. Returns (nsite, 3) site world positions in mm.

    The MuJoCo model's distance unit is metres; we multiply by 1000 so the
    output matches the calibration's mm convention (PersonKeypointFast
    keypoints3d are mm).
    """
    import mujoco
    data.qpos[:] = qpos[: model.nq]
    mujoco.mj_forward(model, data)
    return np.asarray(data.site_xpos) * 1000.0


# ─────────────────────────────────────────────────────────────────────────
# Step 1: global scale + sample-frame qpos joint optimisation
# ─────────────────────────────────────────────────────────────────────────

def _scale_step_residual(
    flat_vars: np.ndarray,
    *,
    n_scales: int,
    n_sample_frames: int,
    n_free: int,
    qpos_templates: np.ndarray,      # (n_sample_frames, n_dofs) — per-frame template with locks
    free_qpos_idx: np.ndarray,       # (n_free,) — qpos indices the optimiser controls
    model,
    data,
    site_indices: np.ndarray,        # (K,) — model site indices to match
    kp_indices: np.ndarray,          # (K,) — bml_movi_87 indices for those sites
    sampled_kp3d: np.ndarray,        # (n_sample_frames, 87, 4) [x,y,z,conf]
    scale_prior_weight: float,
) -> np.ndarray:
    """Residual for the (body_scale, qpos[fi]_per_sample) joint fit.

    Operates on the locked-qpos representation: the optimiser sees only
    `n_free` qpos per sample frame, which we splice into the per-frame
    template (carrying locked leg DOFs + mhip-derived translation init).
    """
    scales = flat_vars[:n_scales]
    free_block = flat_vars[n_scales:].reshape(n_sample_frames, n_free)
    apply_uniform_scale(model, scales[0])

    residuals = []
    for fi in range(n_sample_frames):
        qpos = qpos_templates[fi].copy()
        qpos[free_qpos_idx] = free_block[fi]
        sites_mm = fk_site_positions(model, data, qpos)         # (nsite, 3)
        pred = sites_mm[site_indices]                            # (K, 3)
        obs = sampled_kp3d[fi, kp_indices, :3]                   # (K, 3)
        conf = sampled_kp3d[fi, kp_indices, 3]                   # (K,)
        w = np.sqrt(np.clip(conf, 0.0, 1.0))
        diff = (pred - obs) * w[:, None]
        residuals.append(diff.ravel())

    residuals.append(scale_prior_weight * (scales - 1.0))
    return np.concatenate(residuals)


def apply_uniform_scale(model, scale: float) -> None:
    """v1 placeholder: scale the model by a single uniform factor.

    Per-bone scaling requires walking the body tree and modifying
    body_pos / geom_pos / etc. For Phase 3 v1 we use a single scalar
    that approximates "the subject is taller / shorter than the
    rest-pose adult." More fine-grained scaling deferred to v2.

    In current MuJoCo, uniform model scaling at runtime is awkward —
    the cleanest path is to skip and use scale=1.0 for v1, with the
    `(scale - 1.0)` prior making this effectively a no-op. The two-step
    structure stays in place so v2 can add per-bone scaling without
    re-architecting.
    """
    # No-op for v1 — see docstring.
    return


# ─────────────────────────────────────────────────────────────────────────
# Step 2: per-frame IK with frozen scale
# ─────────────────────────────────────────────────────────────────────────

def _per_frame_residual(
    free_vars: np.ndarray,
    *,
    qpos_template: np.ndarray,       # (n_dofs,) with locks + translation init
    free_qpos_idx: np.ndarray,       # (n_free,)
    model,
    data,
    site_indices: np.ndarray,
    kp_indices: np.ndarray,
    kp3d_frame: np.ndarray,          # (87, 4)
    free_vars_prev: np.ndarray | None,
    velocity_weight: float,
) -> np.ndarray:
    """Per-frame IK residual on the free-DOF representation.

    The locked leg DOFs (when in seated mode) stay at their template
    values; the optimiser only sees `n_free` variables. Velocity penalty
    runs on the free vars too, so locked DOFs don't appear in the residual.
    """
    qpos = qpos_template.copy()
    qpos[free_qpos_idx] = free_vars
    sites_mm = fk_site_positions(model, data, qpos)
    pred = sites_mm[site_indices]
    obs = kp3d_frame[kp_indices, :3]
    conf = kp3d_frame[kp_indices, 3]
    w = np.sqrt(np.clip(conf, 0.0, 1.0))
    pos_residual = ((pred - obs) * w[:, None]).ravel()
    if free_vars_prev is not None and velocity_weight > 0:
        vel_residual = velocity_weight * (free_vars - free_vars_prev)
        return np.concatenate([pos_residual, vel_residual])
    return pos_residual


def fit_kinematic_classical(
    keypoints3d: np.ndarray,        # (T, 87, 4) [x,y,z,conf] in mm, W0 frame
    *,
    xml_path: str,
    n_scale_frames: int = 15,
    scale_prior_weight: float = 100.0,
    velocity_weight: float = 0.1,
    min_visible_joints: int = 30,
    smoothing_window: int = 7,
    lock_legs_seated: bool = True,
    pelvis_init_kp: str = "mhip",
    progress_cb=None,
) -> dict[str, Any]:
    """Two-step classical IK fit. See module docstring for the architecture.

    When `lock_legs_seated=True`:
      - The 14 leg DOFs are pinned to a seated rest pose (see
        SEATED_LEG_QPOS) and excluded from the optimisation variables.
      - The leg sites (see LEG_SITE_NAMES) are dropped from the residual.
      - Pelvis sites (ASIS/PSIS/Hip/bum/mhip/CHip) stay — they anchor
        the root pose.
      - Each frame's free-joint translation is initialised from the
        observed mhip (or `pelvis_init_kp`) keypoint, then fine-tuned
        by the IK. This kills the "static pelvis" failure mode.

    Returns dict with:
        body_scale              : (n_bodies, 1) — scales applied (v1: ones)
        qpos                    : (T, n_dof)  — full nq, locked DOFs included
        qvel                    : (T, n_dof) — finite-diff
        joints                  : (T, n_joints, 3) world positions, mm
        sites                   : (T, n_sites, 3) world positions, mm
        mean_reprojection_error : float — placeholder (we don't reproject
                                  here; reprojection is a PKR-Fast field)
        elapsed_s               : float
    """
    from scipy.optimize import least_squares
    import mujoco

    model, data, site_names, jnt_range = load_mujoco_model(xml_path)
    n_dofs = int(model.nq)
    T = keypoints3d.shape[0]

    # ── Locked-DOF tables ──────────────────────────────────────────────
    locked_idx, locked_val, free_qpos_idx = _build_lock_tables(model, lock_legs_seated)
    n_free = int(free_qpos_idx.size)

    # ── Per-DOF bounds for trf solver ──────────────────────────────────
    # Build (lo, hi) arrays of length n_free. Free-joint components and
    # unlimited joints stay ±inf; limited hinge joints come from
    # model.jnt_range. Without these, lm wanders into crazy poses on
    # large-motion frames and the warm-start drags every subsequent
    # frame with it.
    qpos_lo = np.full(n_dofs, -np.inf, dtype=np.float64)
    qpos_hi = np.full(n_dofs, np.inf, dtype=np.float64)
    for jid in range(int(model.njnt)):
        if not bool(model.jnt_limited[jid]):
            continue
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        # The free joint qpos has 7 entries (3 trans + 4 quat); jnt_range
        # for a freejoint is meaningless. Hinge joints are 1 qpos entry.
        qpos_lo[qadr] = lo
        qpos_hi[qadr] = hi
    free_lo = qpos_lo[free_qpos_idx]
    free_hi = qpos_hi[free_qpos_idx]

    # ── Site↔keypoint mapping, with leg-site filter in seated mode ────
    raw_map = bml_movi_87_to_humanoid_site_mapping()
    site_indices, kp_indices = [], []
    for site_name, kp_idx in raw_map.items():
        if site_name not in site_names:
            continue
        if lock_legs_seated and site_name in LEG_SITE_NAMES:
            continue
        site_indices.append(site_names.index(site_name))
        kp_indices.append(kp_idx)
    site_indices = np.asarray(site_indices, dtype=np.int32)
    kp_indices = np.asarray(kp_indices, dtype=np.int32)
    n_residual_sites = len(site_indices)
    if n_residual_sites < 6:
        raise RuntimeError(
            f"only {n_residual_sites} site↔keypoint pairs available — "
            f"check bml_movi_87_to_humanoid_site_mapping and that {xml_path} "
            f"has the expected named sites"
        )

    # ── Pelvis anchor: mhip → free-joint translation init (mm → m) ───
    # raw_map maps site_name → bml_movi_87 keypoint index; pelvis_init_kp
    # is a site name (default "mhip"), and we look up its kp index.
    pelvis_kp_idx: int | None = raw_map.get(pelvis_init_kp)
    if pelvis_kp_idx is None:
        raise RuntimeError(
            f"pelvis_init_kp={pelvis_init_kp!r} not found in BML-MoVI 87 names"
        )

    def make_template_for_frame(fi: int, fallback_xyz_m: np.ndarray | None) -> np.ndarray:
        """Build qpos_template (n_dofs,): locked legs + mhip-derived translation."""
        tpl = np.zeros(n_dofs, dtype=np.float64)
        # Free joint quat starts as the identity rotation
        if n_dofs >= 7:
            tpl[3] = 1.0
        # Locked leg DOFs
        if locked_idx.size:
            tpl[locked_idx] = locked_val
        # Pelvis translation from observed mhip (mm → m)
        obs = keypoints3d[fi, pelvis_kp_idx]
        if obs[3] > 0:
            tpl[0:3] = obs[:3] / 1000.0
        elif fallback_xyz_m is not None:
            tpl[0:3] = fallback_xyz_m
        return tpl

    # ── Step 1: scale + sample-frame qpos joint fit ────────────────────
    t0 = time.time()
    sample_idx = select_scale_frames(
        keypoints3d[..., 3], n_frames=n_scale_frames,
        min_visible_joints=min_visible_joints,
    )
    n_sample_frames = len(sample_idx)
    if n_sample_frames < 3:
        raise RuntimeError(
            f"only {n_sample_frames} frames meet min_visible_joints={min_visible_joints} — "
            f"recording quality too low for the scale pre-pass"
        )
    sampled_kp3d = keypoints3d[sample_idx]
    qpos_templates_sample = np.stack(
        [make_template_for_frame(int(s), None) for s in sample_idx],
        axis=0,
    )

    n_scales = 1
    # Sample-frame free vars warm-start: zeros (templates already carry
    # the translation init in qpos[0:3]).
    x0 = np.concatenate([
        np.ones(n_scales),
        np.zeros(n_sample_frames * n_free),
    ])

    # diff_step: scipy's default (~1.5e-8 relative) is below MuJoCo's
    # numerical noise floor for joint angles — a 1.5e-8 rad perturbation
    # moves a 1 m arm by ~1.5e-8 m, indistinguishable from FK noise, so
    # the Jacobian column reads as zero and LM thinks it's converged.
    # 1e-3 (~0.057°) is large enough to produce real signal at joints
    # and small enough to stay in the linear regime.
    IK_DIFF_STEP = 1e-3

    res1 = least_squares(
        _scale_step_residual, x0,
        method="lm", max_nfev=200, diff_step=IK_DIFF_STEP,
        kwargs=dict(
            n_scales=n_scales,
            n_sample_frames=n_sample_frames,
            n_free=n_free,
            qpos_templates=qpos_templates_sample,
            free_qpos_idx=free_qpos_idx,
            model=model, data=data,
            site_indices=site_indices, kp_indices=kp_indices,
            sampled_kp3d=sampled_kp3d,
            scale_prior_weight=scale_prior_weight,
        ),
    )
    scales = res1.x[:n_scales]
    sampled_free = res1.x[n_scales:].reshape(n_sample_frames, n_free)
    apply_uniform_scale(model, scales[0])
    t_scale = time.time() - t0
    if progress_cb:
        progress_cb("scale_step", {"elapsed_s": t_scale, "scales": scales.tolist(),
                                    "n_sample_frames": n_sample_frames,
                                    "cost": float(res1.cost)})

    # ── Step 2: per-frame IK with frozen scale ─────────────────────────
    t_step2 = time.time()
    qpos_traj = np.zeros((T, n_dofs), dtype=np.float64)
    free_traj = np.zeros((T, n_free), dtype=np.float64)

    def warm_start_free_for(fi: int) -> np.ndarray:
        i = int(np.argmin(np.abs(sample_idx - fi)))
        return sampled_free[i].copy()

    per_frame_nfev_first = 1000
    per_frame_nfev = 300

    free_prev: np.ndarray | None = None
    last_pelvis_m = qpos_templates_sample[0, 0:3].copy()
    n_failed = 0
    last_error: Exception | None = None
    first_initial_cost: float | None = None
    first_final_cost: float | None = None
    for fi in range(T):
        # Per-frame template carries the mhip-derived translation init
        # (with previous-frame fallback when mhip is not visible).
        tpl = make_template_for_frame(fi, last_pelvis_m)
        last_pelvis_m = tpl[0:3].copy()

        # Skip frames with too few visible joints — propagate previous.
        conf_f = keypoints3d[fi, kp_indices, 3]
        if (conf_f > 0).sum() < 6:
            x0_f = free_prev if free_prev is not None else warm_start_free_for(fi)
            qpos = tpl.copy(); qpos[free_qpos_idx] = x0_f
            qpos_traj[fi] = qpos
            free_traj[fi] = x0_f
            free_prev = x0_f.copy()
            continue
        x0_f = free_prev if free_prev is not None else warm_start_free_for(fi)
        nfev = per_frame_nfev_first if free_prev is None else per_frame_nfev
        try:
            # Clip the warm-start into the bounded region so trf accepts it.
            x0_clipped = np.clip(x0_f, free_lo, free_hi)
            res_f = least_squares(
                _per_frame_residual, x0_clipped,
                method="trf",
                bounds=(free_lo, free_hi),
                max_nfev=nfev,
                diff_step=IK_DIFF_STEP,
                kwargs=dict(
                    qpos_template=tpl,
                    free_qpos_idx=free_qpos_idx,
                    model=model, data=data,
                    site_indices=site_indices, kp_indices=kp_indices,
                    kp3d_frame=keypoints3d[fi],
                    free_vars_prev=free_prev, velocity_weight=velocity_weight,
                ),
            )
            if fi == 0:
                r0 = _per_frame_residual(
                    x0_f, qpos_template=tpl, free_qpos_idx=free_qpos_idx,
                    model=model, data=data,
                    site_indices=site_indices, kp_indices=kp_indices,
                    kp3d_frame=keypoints3d[0],
                    free_vars_prev=None, velocity_weight=0.0,
                )
                first_initial_cost = float(0.5 * (r0 ** 2).sum())
                first_final_cost = float(res_f.cost)
            free_traj[fi] = res_f.x
            qpos = tpl.copy(); qpos[free_qpos_idx] = res_f.x
            qpos_traj[fi] = qpos
        except Exception as e:
            n_failed += 1
            last_error = e
            free_traj[fi] = x0_f
            qpos = tpl.copy(); qpos[free_qpos_idx] = x0_f
            qpos_traj[fi] = qpos
        free_prev = free_traj[fi].copy()

        if progress_cb and fi % 100 == 0:
            progress_cb("per_frame", {"i": fi, "n": T,
                                       "elapsed_s": time.time() - t_step2})

    if first_initial_cost is not None:
        import sys
        print(f"[fit_kinematic_classical] frame 0 cost: "
              f"initial={first_initial_cost:.1f} → final={first_final_cost:.1f} "
              f"(n_residual_sites={n_residual_sites}, n_free={n_free}, "
              f"n_locked={int(locked_idx.size)})",
              file=sys.stderr)
    if n_failed > 0:
        import sys
        print(f"[fit_kinematic_classical] WARNING: {n_failed}/{T} frames "
              f"failed IK; last error: {type(last_error).__name__}: {last_error}",
              file=sys.stderr)

    # ── Step 3a: detect-and-interpolate twitch frames in JOINT space ───
    # A "twitch" is a single frame where the model briefly snaps to a
    # different pose; this shows up in diagnostics as two consecutive
    # high-velocity frames (one going in, one coming out). Per-DOF
    # median filtering can't see this — the outlier is in the *combined*
    # pose (each individual joint moves only ~30°, but the end effector
    # like the head shifts a metre). We detect in joint-position mm
    # space and interpolate qpos linearly through the bad frame; the
    # downstream FK normalises the quaternion so linear-interp-of-quat
    # is fine for preview quality.
    if T >= 3:
        prelim_joints = np.zeros((T, int(model.nbody), 3), dtype=np.float64)
        for fi in range(T):
            data.qpos[:] = qpos_traj[fi]
            mujoco.mj_forward(model, data)
            prelim_joints[fi] = np.asarray(data.xpos) * 1000.0
        body_vel = np.linalg.norm(np.diff(prelim_joints, axis=0), axis=-1)  # (T-1, nb) mm
        max_vel = body_vel.max(axis=-1)                                     # (T-1,) mm
        # Threshold: 10× median or 100mm/frame, whichever is larger. The
        # 100mm floor prevents over-triggering on a sequence with no
        # outliers (where median ≈ 0).
        thresh = max(100.0, float(np.median(max_vel) * 10.0))
        # Bad frame f has both incoming velocity (max_vel[f-1]) AND
        # outgoing velocity (max_vel[f]) above threshold.
        bad = np.zeros(T, dtype=bool)
        for f in range(1, T - 1):
            if max_vel[f - 1] > thresh and max_vel[f] > thresh:
                bad[f] = True
        n_bad = int(bad.sum())
        # Linear-interpolate qpos through bad frames using the nearest
        # non-bad neighbours on each side.
        for f in np.where(bad)[0]:
            lo = f - 1
            while lo >= 0 and bad[lo]:
                lo -= 1
            hi = f + 1
            while hi < T and bad[hi]:
                hi += 1
            if lo < 0 and hi < T:
                qpos_traj[f] = qpos_traj[hi]
            elif hi >= T and lo >= 0:
                qpos_traj[f] = qpos_traj[lo]
            elif lo >= 0 and hi < T:
                alpha = (f - lo) / (hi - lo)
                qpos_traj[f] = (1.0 - alpha) * qpos_traj[lo] + alpha * qpos_traj[hi]
        import sys
        print(f"[fit_kinematic_classical] joint-vel diag: median={np.median(max_vel):.1f}mm "
              f"p90={np.percentile(max_vel, 90):.1f}mm max={max_vel.max():.1f}mm "
              f"thresh={thresh:.0f}mm  bad={n_bad}",
              file=sys.stderr)

    # ── Step 3b: post-hoc Savitzky-Golay smoothing ─────────────────────
    # Applied ONLY to non-quaternion qpos components. The free-joint
    # quaternion at qpos[3:7] lives on the unit 3-sphere; per-component
    # linear averaging across non-trivial rotations produces non-unit
    # quaternions that mj_forward normalises to a wrong axis, causing
    # the 1m-pelvis-tilt frame artefacts. The quaternion is already
    # smooth from the IK's velocity penalty + warm-start chain.
    if smoothing_window >= 3 and smoothing_window <= T:
        from scipy.signal import savgol_filter
        wlen = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        poly = min(2, wlen - 1)
        # Build the mask of qpos indices that are safe to smooth:
        # everything except the free-joint quaternion qpos[3:7].
        smoothable = np.ones(n_dofs, dtype=bool)
        if n_dofs >= 7:
            smoothable[3:7] = False
        smooth_idx = np.where(smoothable)[0]
        qpos_traj[:, smooth_idx] = savgol_filter(
            qpos_traj[:, smooth_idx],
            window_length=wlen,
            polyorder=poly,
            axis=0,
        )

    # ── Outputs: joints + sites trajectories ───────────────────────────
    # Replay qpos through FK to get the trajectories the canonical KR
    # also stores.
    n_bodies = int(model.nbody)
    n_sites = int(model.nsite)
    joints_traj = np.zeros((T, n_bodies, 3), dtype=np.float64)
    sites_traj = np.zeros((T, n_sites, 3), dtype=np.float64)
    for fi in range(T):
        data.qpos[:] = qpos_traj[fi]
        mujoco.mj_forward(model, data)
        joints_traj[fi] = np.asarray(data.xpos) * 1000.0     # mm
        sites_traj[fi] = np.asarray(data.site_xpos) * 1000.0

    qvel_traj = np.zeros_like(qpos_traj)
    qvel_traj[1:] = np.diff(qpos_traj, axis=0)

    elapsed = time.time() - t0
    return {
        "body_scale": np.full((n_bodies, 1), scales[0], dtype=np.float64),
        "qpos": qpos_traj.astype(np.float32),
        "qvel": qvel_traj.astype(np.float32),
        "joints": joints_traj.astype(np.float32),
        "sites": sites_traj.astype(np.float32),
        "mean_reprojection_error": float("nan"),   # see docstring
        "elapsed_s": elapsed,
        "scale_step_elapsed_s": t_scale,
        "n_sample_frames": n_sample_frames,
    }
