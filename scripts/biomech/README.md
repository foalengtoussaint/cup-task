# Biomechanical IK (vendored from iMOVE)

`imove_ik.py` is a verbatim copy of
`~/Documents/iMOVE/DEV/imove_extensions/imove_extensions/inverse_kinematics.py`,
with one edit: `movi_joint_names` is imported from the vendored `movi_names.py`
instead of `body_models.biomechanics_mjx.forward_kinematics`, so it runs without
the ISR container.

`humanoid_torque_rl_nomesh.xml` is `humanoid_torque_rl.xml` from
`isr-containers/packages/BodyModels/.../data/humanoid/` with the 79 `<mesh>` assets
and their geoms stripped. Verified: identical nq/nsite/nbody and site positions
agree to 0.000e+00 m over random poses -- FK walks the joint tree, geoms play no
part. 21 KB instead of 50 MB, and no external asset directory.

Model choice: `torque_rl` has all nine sites we need and a ball-joint shoulder.
The richer `humanoid_arms_torso.xml` has 16 shoulder-girdle DOF, but the DELTA
C3Ds carry ONE acromion marker per side and nothing on the scapula, so those DOF
are unidentifiable -- they would absorb error rather than model anatomy.
