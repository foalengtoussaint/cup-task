# Methods ↔ code verification checklist

Every falsifiable claim the Methods section now makes, and where to check it. This is
step 2 of the plan at the top of `main.tex`: **Methods is the spec, the code is what gets
checked against it.** Four mismatches were already found by writing the section, so the
default assumption is that a doc-sourced claim is stale until read in the code.

Legend: **[c]** verified against code this session · **[d]** taken from a doc, code not
read · **[?]** not verifiable locally (needs `cache/`, `out/`, or the compute box)

---

## §II-A Dataset

| | claim | check against |
|---|---|---|
| [d] | ten participants, one recorded twice → eleven units | `cache/delta/cam_quality.json` keys |
| [c] | camera whitelist applied at load *and* in the sidecars | `compare_pose_omc_delta.py:167`, `gnn_build_dataset.py:141` |
| [d] | two exclusion passes: mis-cut (NCC in uncut) then miscalibration (leave-one-out per-joint reprojection) | `cut_placement_audit.py`, the leave-one-out check — **neither is in the repo**, find or re-add |
| [d] | both fault classes land at 25–30 px | `DATASET_STATUS.md` §3 |
| [d] | five cameras dropped as mis-cut, seven as miscalibrated | recount from `cam_quality.json`, not from the worklog prose |
| [d] | three units keep five cameras, rest three or four | same |
| [?] | 837 clips pass the integrity check | `cache/delta/clip_omc_audit.json` |
| [c] | temporal gate = best of {wrist,elbow,shoulder}×{speed,disp} + cup, ≥0.7 | `_find_lag_multi`, `load_clean` |
| [c] | wrist-validity gate ≥50 % of frames | `gnn_train.py:49` — `wr_thr=0.5` ✅ |
| [?] | 826 admitted (99 %) | scorer log |
| [?] | 697 measure / 687 trajectory | **687 recomputed from `trajectory_agreement.csv` this session; 697 is from `table3_measures.csv`.** Confirm both in one run |

## §II-B Pipeline

| | claim | check against |
|---|---|---|
| [c] | stock `yolo26s-pose`, per frame per camera | `detect_pose_multi.py:71`, `cache_pose_cohort.py:153` |
| [c] | stock `yolo26x-seg` seeds the cup once | `cache_cup_seed26x.py:37` |
| [c] | seed frame = first to agree across ≥3 cameras | `cache_cup_seed26x.py` docstring |
| [c] | UETrack-B, never re-anchored | `cup_task/cup_track.py:32,51` |
| [c] | BA: DLT init, geometric reprojection, confidence-weighted, Huber | `ba_refine.py:80-107` |
| [c] | no bone-length prior | `ba_cache_traj_all.py:40` passes `lam_bone=0.0` |
| [c] | >30 px excluded, refit on survivors | `cup_task/triangulate.py:274-280` |
| [c] | too few survive → all-camera DLT, not dropped | same, `keep = list(range(len(cams)))` |
| [c] | SmoothNet ±16-frame non-causal window | `pose_smooth.py:34` — `WINDOW = 32` ✅ (±16 assumes centred; "32-frame" would be literal) |
| [?] | BA divergence guard is not described in the text at all | decide revert-vs-exclude first (plan block) |

## §II-C Phase Segmentation

| | claim | check against |
|---|---|---|
| [c] | close–plateau–open on wrist→cup and cup→mouth | `seg_sequential.py:53-110` |
| [c] | runs must traverse ≥30 % of the channel's own range | `big_wc`, `big_cm` = `0.3 * span` |
| [c] | one forward pass, first qualifying run after the previous | same |
| [c] | cup: <3 agreeing cameras discarded | `score_e2e_seq.py:59-66`, `OT_CUP_STRICT_KF` |
| [c] | interior gaps <0.75 s KF-filled, no extrapolation | `triangulate.py:113` ✅ — **note a third guard the paper omits: nothing filled below 30 % coverage** |
| [c] | wrist proxy on the cup→mouth channel only | `score_seg_boundaries.py:73-76` (`_wr2` branch) |
| [c] | measure comparison uses optical phases for both systems | `score_vs_automq.py` step 2 |

## §II-D Measures

| | claim | check against |
|---|---|---|
| [c] | body frame from four torso points; flexion sagittal, abduction frontal | `planar_body_angles.py:1-13` |
| [c] | body frame frozen per trial | same (`freeze`) |
| [c] | peak velocity differentiates without the extra position low-pass | `score_vs_automq.py:295-313` |
| [c] | movement units at 60 mm/s, 30–50 mm/s video jitter floor | `score.py:37,50,51` ✅ — all three confirmed in code, not just the doc |
| [c] | trunk = 3D excursion of the shoulder midpoint | `score_vs_automq.py:254-262` ✅ — midpoint of the two shoulders, excursion from median rest |
| [c] | COCO-17 has no sternum keypoint | keypoint list |

## §II-F Validation Protocol

| | claim | check against |
|---|---|---|
| [c] | lag from wrist-speed cross-correlation | `paper_trajectories.py:169-176` |
| [c] | static offset removed per trial for the trajectory comparison | `paper_trajectories.py:116-124` |
| [c] | `r_s` over trials, `r_av` over per-participant/arm averages, both pooled | `paper_table3.py:47-51` |
| [?] | landmark offsets fitted per landmark in body and arm frames | **script not in the repo** |
| [?] | offsets validated on held-out trials, within 0.007 | derived from `anat12w0_omc_values.csv` this session; confirm against the script |
| [?] | applied "before the angles are compared" — which measures? | unresolved; see plan block |

---

## Still open after the code check

All five locally checkable doc-sourced claims verified correct. The four that remain
are the **camera-audit** claims — participant/unit count, the two-pass exclusion, the
25–30 px overlap, and the 5-mis-cut/7-miscalibrated split. They need
`cache/delta/cam_quality.json`, and their tooling (`cut_placement_audit.py`, the
leave-one-out check) is not in the repo. That makes §II-A the weakest part of Methods:
claims with no code behind them, in a paper whose code is being released.

## Blocking order

1. **Recover the landmark-fit script.** It gates four open items: the scope question,
   cross-participant generalisation, the P25 discriminator, and the `PIPELINE.md`
   contradiction. It has to be in the repo anyway for the code release.
2. **Copy `out/automq/score_vs_automq.csv`.** Gates the four-variant ablation table.
3. **Log `n_fallback` / `n_guarded`** over 826 trials, then decide revert-vs-exclude.
4. **Then regenerate**, per the table changes in the plan block.

## Docs to fix before the repo is public

- `results/PIPELINE.md` — the anat12 entry under "did not work" (contradicted), and the
  argmin "do not re-try" entry (does not reproduce on the current cup).
- `measures_methods.md` — describes the superseded constant-trunk-down-axis flexion, and
  says "raw wrist" where it means "without the additional 4 Hz low-pass".
