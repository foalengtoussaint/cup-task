# Methods ↔ code verification checklist

Every falsifiable claim the Methods section now makes, and where to check it. This is
step 2 of the plan at the top of `main.tex`: **Methods is the spec, the code is what gets
checked against it.** Four mismatches were already found by writing the section, so the
default assumption is that a doc-sourced claim is stale until read in the code.

Legend: **[c]** verified against code this session · **[d]** taken from a doc, code not
read · **[?]** not verifiable locally (needs `cache/`, `out/`, or the compute box) ·
**[~]** partly verified · **[!]** verified BROKEN — needs work, not just a check

---

## STATE, 2026-08-24

**Paper**: 7 pages (6 of content + a partial 7th holding the reference placeholders), 2 figures,
3 tables, 0 overfull boxes, 0 undefined references or citations, 11 `\todo`s. Every section is
written except the ones needing information not on this machine: title, authors/affiliations,
repository URL, demographics table, funding, and 12 of ~15 bibliography entries.

**Figures**: Fig. 1 exemplary trajectories (P07, trials 10 and 43, four of six trajectories);
Fig. 2 the twelve measures end-to-end (`fig4b_mmcseg_anat12.png`). The optical-windows scatter
(`fig4a`) was cut — its numbers are Table II's left column pair.

**Tables**: I trajectories (bias IQR folded in as a fifth row); II measures, six numeric columns
(optical windows, markerless windows, 30 Hz, n); III boundaries. Tables for latency, capture rate
and the alternative measures were cut for length; their scripts and CSVs remain.

**The headline numbers, all current**: 750 trials scored of 790 cached of 842 admitted; ten of
twelve measures at r_s ≥ 0.84 and r_av ≥ 0.93 end-to-end, eleven with optical windows, and ten of
twelve again at 30 Hz; 16.8 s per trial with 1.3 s after the last frame; 4.9 % segmenter declines.

**If you read one thing before editing**: five claims in this document were WRONG and are marked
[!] — the "no bone-length prior" (the term is active at 0.05), the "10 % calibration scale error"
(contradicted, ratios are segment-specific), the "30–50 mm/s jitter floor" (measured 13.4), the
arm-vs-impairment analyses (arm is confounded with recording block), and the rectified coordination
index (its arm effect was a rectifier artefact). Each was asserted from a note rather than measured,
and each survived several readings because it sounded like a fact.

## §II-A Dataset

*Counts moved to Results §Study Population 2026-08-20; **written and re-verified 2026-08-21**.* Methods
states the admission RULE and the censoring RULE only. The old parked numbers were all stale and are
replaced: 704/705 → 750/790, and 24 % censored → 19.5 %. The accounting now in §V-A is
842 admitted → −45 with no usable cup track, −3 mispaired → 794 → −4 optical faults the de-stepper
refuses → **790** (Tables I–II) → −1 with no segmentation → 789 → −39 segmenter declines → **750**
(Tables III–IV). Sources: `out/scoring/seg_inputs_refill.log` line 24 (842/45/3/794),
`out/scoring/qc_gate_scan.log` (the 4), `out/scoring/score_own_phases.log` ("no-seg 1"),
`m3a.log` (39/790). Censoring: 146 of 750 trials have `settle_observed=False` (19.5 %), and TMT's
n = 602 is those 146 plus two trials with no usable value. `unger2024` reports its exclusions the same way — one
sentence in Results, nothing in Methods. Struck rows below are those counts.

| | claim | check against |
|---|---|---|
| [c] | ten participants, one recorded twice → eleven units | `cam_quality.json` has exactly 11 keys: P07 P08 P10 P12 P13 P14 P15 P17 P19 P251 P252 ✅ |
| [c] | camera whitelist applied at load *and* in the sidecars | `compare_pose_omc_delta.py:167`, `gnn_build_dataset.py:141` |
| — | ~~two exclusion passes: mis-cut (NCC in uncut) then miscalibration (leave-one-out per-joint reprojection)~~ | **MECHANISM CUT from Methods 2026-08-20** — the paragraph now states the two fault classes and their counts only. It was also wrong as written: NCC-within-own-uncut cannot by itself flag a mis-cut (it locates the clip; the verdict comes from comparing that position against the reference camera after removing the inter-recording clock offset, recovered by motion-energy xcorr), and the miscalibration pass is RANSAC consensus from the reference cams, not leave-one-out. Tooling **is** in the repo — `scripts/archive/cut_placement_audit.py` (same-camera NCC template match in the uncut) and `scripts/archive/multijoint_reproj.py` (per-joint reproj vs RANSAC consensus of the reference cams). The earlier "neither is in the repo" was wrong: they are in `scripts/archive/`, not `scripts/`. `scripts/spatial_miscalib_check.py` is the third, finer discriminator (smooth-field vs incoherent error). **Promote all three to `scripts/` before release.** |
| — | ~~both fault classes land at 25–30 px~~ | **CUT from Methods 2026-08-20.** Was the section's only [d] claim. As written it was also wrong: per `archive/docs_20260820/DATASET_STATUS.md` §3 the smooth, position-dependent error field is the *const-offset cut* signature (P10 cam4 R²=0.64, actually a +58.5 s shift), while a *shuffled* cut gives low R² (P13 cam2 R²=0.11). The defensible finding — reprojection *magnitude alone* cannot separate the two fault classes — is not currently claimed in the paper. |
| [c] | camera exclusions decided from the recordings alone, independently of any comparison against the optical reference | ✅ Both passes are OMC-free: `cut_placement_audit.py` is calibration-free video matching (its own docstring: "all calibration-free"), and `multijoint_reproj.py` reprojects against the RANSAC consensus of the *reference cameras*, never against mocap or any measure score. This is the anti-circularity claim the paragraph now rests on. |
| [c] | 12 of 55 unit–camera pairs excluded | ✅ `cache/delta/cam_quality.json`: 11 units × 5 = 55, 43 kept, 12 dropped. |
| [~] | five cameras dropped as mis-cut, seven as miscalibrated | Totals agree: 43 cameras kept of 11×5=55, so **12 dropped** = 5+7 ✅. The 5/7 *split* is not recoverable from `cam_quality.json` (it stores only `good`) — re-derive it from the two audit scripts' own output before the claim ships. |
| [c] | three units keep five cameras, rest three or four | ✅ exactly: 5 cams — P07 P08 P15 (3 units); 4 cams — P10 P14 P251 P252; 3 cams — P12 P13 P17 P19 |
| — | ~~851 clips pass the integrity check~~ | **CUT from Methods 2026-08-20** — the paragraph was condensed to match `unger2024`'s density (their exclusions are one sentence in Results §III-A, not Methods). Removes a number that was never verifiable: `clip_omc_audit.json` is dated Aug 3, covers 9 of 11 units, lists 666 clean vs 680 in the rebuilt cache. If a paired-clip total is ever restated, re-run the audit over all 11 units first. |
| [c] | temporal gate = agreement at the best lag, consensus across joint+cup signals (thresholds not stated in Methods) | **Mechanism and constants CUT 2026-08-20**, per the plan note at the top of `main.tex` ("Unger's Methods carries no constants at all"). Code unchanged and still the stacked Fisher-z consensus: `compare_pose_omc_delta.find_lag_best`, whose correlation `gnn_build_dataset.py:263` stores as `sync_corr` tagged "APPLIED". Methods now claims only that the lag is a consensus over several signals rather than the wrist alone — which is true and is the part that matters. |
| [c] | wrist-validity gate = fraction of frames with a usable wrist (threshold not stated in Methods) | `gnn_train.py` `load_clean(wr_thr=0.5)` ✅ |
| — | ~~the gates admit 99 % of paired clips~~ | ✅ recomputed 2026-08-20 over the rebuilt cache: 842 pass both gates; rejects = 5 sync, 1 wrist, 3 both. **Gate fixed in the same pass**: `load_clean` read `max(sync_corr, sync_corr_multi)`, which admitted 2 trials on the best-of-8 argmax while the *applied* stacked lag had failed (P15 trial_47_L_affected 0.339 vs 0.894; P19 trial_55_R_affected 0.389 vs 0.755) and rescued 0 in the other direction. Now gates on `sync_corr` alone → 844→842. Neither trial reached the scored cohort, so no published number moved. |
| — | ~~704 enter the measure comparison, 705 the trajectory comparison~~ | 704 ✅ from `out/scoring/score_vs_automq_matchedangles.csv` (matches Table III's n). **705 is from `paper/trajectory_agreement.csv` dated Aug 20 04:57, which PREDATES the 15:52 relaxed-gate rebuild** — re-run `paper_trajectories.py` and re-check before shipping. |

## §II-B Pipeline

| | claim | check against |
|---|---|---|
| [c] | stock `yolo26s-pose`, per frame per camera | `detect_pose_multi.py:71`, `cache_pose_cohort.py:153` |
| [c] | stock `yolo26x-seg` seeds the cup once | `cache_cup_seed26x.py:37` |
| [c] | seed frame = first to agree across ≥3 cameras | `cache_cup_seed26x.py` docstring |
| [c] | UETrack-B, never re-anchored | `pipeline/cup_track.py:32,51` |
| [c] | BA: DLT init, geometric reprojection, confidence-weighted, Huber | `ba_refine.py:80-107` |
| — | ~~DLT-vs-ML rationale; cup/body gating asymmetry argument~~ | **CONDENSED 2026-08-20.** Pipeline stage 3 was 375 of the 545 words describing all six stages (69 %); now 192. Cut was rationale, not claims: the textbook argument that reprojection error is the ML estimate under Gaussian noise (the `hartley2004` citation carries it), and the extended defence of why the body fit is ungated where the cup fit is gated. The asymmetry itself is retained in one clause. No falsifiable claim was removed — every row in this section still corresponds to a sentence in the text. |
| [!]→[c] | ~~no bone-length prior~~ **WRONG, corrected 2026-08-21**: a bone-length VARIANCE term at lam=0.05 IS active in every published number | The committed `ba_cache_traj_all.py` passes `lam_bone=0.0`; the working tree passes `LAM_BONE = 0.05`, and the cache was written one minute after that file was last edited, so timestamps could not settle it. Settled by re-solving: `refine_trial_ba` is deterministic (L-BFGS, no sampling), and on six trials spread over participants lam=0.05 reproduces `cache/ba_traj/traj_sw0_noguard.npz` to **0.0000 mm RMS** while lam=0.00 misses it by **3.99–14.88 mm** (`scratchpad/ba_lam_check.py`). §II-B now describes the term. It is `_bone_energy`, the frame-to-frame VARIANCE of the five near-rigid limbs (`LIMBS`: both upper arms, both forearms, shoulder width), asserting no target length — not `_bone_energy_pinned`, which pinned to the per-trial median and is unused. Its own docstring records the measured effect at lam=0.05: elbow angular PV 0.896→0.905, PV 0.877→0.888, max elbow angle 0.894→0.912, concentrated on the worst-calibrated participant (P19 elbow 0.470→0.674), at the cost of absolute peak error (11.0→13.2 deg/s). **Ranks improve, magnitudes do not** — which is the honest framing for the ablation |
| [c] | >30 px excluded, refit on survivors | `pipeline/triangulate.py:274-280` |
| [c] | too few survive → all-camera DLT, not dropped | same, `keep = list(range(len(cams)))` |
| [c] | 30 px gate applies to the DLT init only; BA sees every camera | ✅ `ba_refine._reproj_residual` loops all C cameras with weight `uv_valid * uv_conf * in_front` and its docstring says "NOT prediction-gated (the self-defeating bug)". `uv_valid` is "kp present AND finite" (`gnn_build_dataset.py:144`), **not** a reprojection mask, so the 30 px inlier set never reaches BA. |
| [c] | frame left empty below two cameras, 0.6 % of joint-frames | ✅ measured 2026-08-20 over the 842 admitted trials: 21,190 / 3,742,380 joint-frames non-finite = **0.57 %**. (`robust_triangulate` returns `None` when `len(cams) < 2`.) |
| [c] | cup: <3 agreeing cameras discarded | `score_e2e_seq.py:59-66`, `OT_CUP_STRICT_KF` |
| [c] | interior gaps <0.75 s KF-filled, no extrapolation | `triangulate.py:113` ✅ — **note a third guard the paper omits: nothing filled below 30 % coverage** |
| — | *(cup reconstruction rows moved here from §II-C, 2026-08-20)* | The cup gate, KF gap-fill and wrist-proxy fill are reconstruction, not segmentation, and happen at pipeline stage 3 — Methods now describes them there. §II-C keeps only the channel-scoping consequence (the proxy applies to cup→mouth alone), which is where it does its work. |
| [c] | SmoothNet ±16-frame non-causal window | `pose_smooth.py:34` — `WINDOW = 32` ✅ (±16 assumes centred; "32-frame" would be literal) |
| [c] | every joint-frame comes from the BA; no trial excluded for failing to converge | ✅ The divergence guard is **off by design** in the shipped run: `ba_cache_traj_all.py:23` sets `FALLBACK_MM = None` ("NO GUARD") and never passes `trial_guard_mm`, so nothing reverts to the DLT init. Not describing it in Methods is therefore correct, not an omission. Rationale at `results_v3_delta.py:587` — the guard existed to catch blow-ups from a `0*inf` NaN bug in `_reproj_residual`; with that fixed it is redundant **and harmful**, since reverting scattered joint-frames to DLT splices two estimators into one series and injects derivative spikes (peak elbow angular velocity: guarded r_s 0.769 vs unguarded 0.884). **The plan block's "BA guard rate (n_fallback/n_guarded)" TODO is moot while the guard is off.** |

## §II-C Phase Segmentation

| | claim | check against |
|---|---|---|
| [c] | close–plateau–open on wrist→cup and cup→mouth | `seg_sequential.py:53-110` |
| [c] | reach onset referenced to the START rest position, settle to where the hand rests at the END | **Corrected 2026-08-20** — the text still said both were referenced to the start after `settle_rule` moved to `"end"`. `_tail(rule="end")` builds `ref_end` from the median hand position over the final 0.5 s; reach onset still uses the first-0.5 s reference (`d_rest`, `LEAVE_REST`). |
| [c] | run-magnitude thresholds are relative to the channel's own range | `big_wc`, `big_cm` = `0.3 * span` ✅. **Reworded 2026-08-20**: Methods previously said "thresholds are relative" without qualification, which overclaimed — `LEAVE_REST` (30 mm), `ARRIVE_REST` (40 mm) and `GRASP_FLAT_MMPS` (40 mm/s) are ABSOLUTE. Only the run-magnitude test is relative. |
| [c] | one forward pass, first qualifying run after the previous | same |
| [c] | settle = hand within tolerance of where it ENDS UP | **RULE CHANGED 2026-08-20** — was proximity to where it STARTED. `seg_sequential._tail(rule="end")`, now the default. The old rule bisected the data: the hand comes to rest a median 14.7 mm from its start with p90 35.6 mm (OMC) / 40.3 mm (MMC) against a 40 mm threshold, so OMC and MMC flipped it independently and a flip costs ~40 frames (the alternative is the timeout). Table IV settle: 33→17 ms median, 153→83 ms p90, 7.0→1.5% beyond 0.25 s; every other boundary provably untouched. `seg_anchor_min` is pinned to `rule="pos"` so the comparison variant still measures what it did. |
| [c] | total movement time is withheld where the movement end is not recorded (RULE; count moved to Results) | ✅ `settle_observed` is ALWAYS the stillness test, whatever rule places the boundary — proximity rules claim a settle on 84–90 % of these and cannot tell "arrived" from "passing through". 150/636 = 23.6 %. Verified not a threshold artefact: on those trials the hand is travelling at the last frame (median 80 mm/s, 42 mm net, straightness 0.84–0.89) and **0 %** are censored merely for lack of frames; on 45 % AutoMQ's own settle also lies past the last recorded frame. `total_movement_time` n≈486; `number_of_movement_units` partially affected (sums over `returning`); reaching-windowed measures unaffected. |
| [c] | cup→mouth channel is driven by the WRIST, not the cup | **CHANGED 2026-08-20** — was the cup track with an offset-fitted wrist proxy filling its gaps (`_wr2`). Now `segment_sequential(cm_source="wrist")`, the default. Rationale is physical (the hand carries the cup) and the wrist is the better-conditioned signal: nine-keypoint triangulation vs a single object below the three-camera floor on **21 %** of drinking frames. Table IV, both sides on the wrist: drink onset 133→83 ms median (p90 403→267, >0.25 s 20.8→10.4 %), drink offset 133→67 ms (p90 300→217, 16.1→4.5 %); grasp/release/reach onset/settle bit-identical. Verified SYMMETRICALLY (both sides changed) — the asymmetric test was better still on MMC alone, which would have been the misleading measurement. |
| [c] | shipped pipeline no longer applies the wrist PROXY at all | **Methods stage 3 corrected 2026-08-20** — it still said "what remains is filled from the wrist… as a constant offset estimated over the frames where the cup is held", but with `cm_source="wrist"` the shipped variant (`mmc_c3kf`) applies only the 3-camera floor and `kf_fill_gaps`; `fill_cup_from_wrist` is reached solely by the `_wr`/`_wr2` comparison variants. Long gaps in the cup are now simply left empty — the only boundaries reading that channel are grasp and release, which a fabricated cup position cannot help. The coverage guard is now stated in Methods as a rule (no constant). |
| [c] | dropping the fitted offset costs one frame | ✅ `wrist + d` (d median 142 mm) gives 83 ms median vs 100 ms for the wrist alone, i.e. 17 ms at 60 Hz — traded for removing the offset estimator, its hold detection and its 20-frame minimum, which were undefined on 37 trials. Symmetric evaluation then put the wrist alone ahead at 83/67 ms. |
| [c] | Table IV reads the `mmc_c3kf` variant | **Repointed 2026-08-20** from `mmc_c3kf_wr2`: with `cm_source="wrist"` the explicit proxy in `_wr2` would override the default, so `mmc_c3kf` (≥3-camera floor + KF fill on wrist→cup, wrist on cup→mouth) is now the shipped configuration. `_wr2` is kept only as the comparison against the offset-fitted proxy. |
| [c] | measure comparison uses optical phases for both systems | `score_vs_automq.py` step 2 ✅ |
| [c] | run-magnitude fraction (0.3) shown insensitive | Swept 0.20–0.45 (`scratchpad/frac_sweep.py`, 264 trials): the reported MMC-vs-OMC median is **33 ms at every value**, and agreement with the reference protocol's phases is best at 0.40, not at 0.30 — so the constant was demonstrably not tuned. One sentence now in §III-C |
| [c] | segmentation IS now validated against a reference segmentation | The `amq_*` windows cached in `seg_inputs_ship` are the reference protocol's own phases on the optical markers, present on **631** scored trials. §IV-E reports both quantities: input sensitivity (Table III) and, against the reference, a constant offset per boundary (+200 / −150 / +167 / −100 ms on the first four) with a **33–67 ms residual** once removed. Release and settle differ more (+483 / −433, residuals 233 / 117). This supersedes the "re-check when Results is written" note below |
| [c] | Methods no longer states HOW segmentation is validated | **Corrected 2026-08-20** — it said "validated separately, as boundary error **against the optical phases**", which contradicted both Results ("the disagreement between the two inputs **rather than** the error against the reference phases") and `make_seg_table.py:42`, which computes `|seq_mmc − seq_omc|`. Methods now says only that segmentation is validated separately and points at the section, leaving the choice of quantity to Results — which is not yet settled. Re-check this row when Results is written. |
| [c] | "the deployed pipeline segments from the cup" reworded | Since `cm_source="wrist"` the cup drives only grasp and release, so "segments from the cup" was half true. Now "segments the trial itself". |
| [!] | Table IV regenerated 2026-08-20 | ⚠ The committed `seg_boundaries.csv` **predated the relaxed-gate cache rebuild**, so regenerating moved every boundary, not just settle (reach onset p90 83→67 ms with the rule held fixed). Attribution for settle: 50→33 ms is the cache refresh, 33→17 ms is the rule change. Any other artifact generated from that CSV before 2026-08-20 is similarly stale — check before citing. |

## §II-D Measures

| | claim | check against |
|---|---|---|
| [c] | body frame from four torso points; flexion sagittal, abduction frontal | `scripts/score_vs_automq.py:221` (`_planar_body_angles`) — **this is the shipped operator**. `paper/scripts/planar_body_angles.py` is a standalone analysis copy that nothing in the scoring path imports; cite the scorer, not it. |
| [c] | body frame frozen per trial | same (`freeze`), `scripts/score_vs_automq.py` |
| [c] | peak velocity differentiates without the extra position low-pass | `score_vs_automq.py:295-313` |
| [c]→[!] | movement units at 60 mm/s; ~~30–50 mm/s video jitter floor~~ | The 60 mm/s constant is real (`score.py:50`). **The jitter floor was never measured and is wrong**: the markerless wrist moves **13.4 mm/s** at rest (IQR 8.5–22.4), optical 4.0. Both §II-D and the `score.py:37` comment corrected 2026-08-22. The threshold does not matter anyway — swept 20/40/60/80 mm/s on both systems, r_s = 0.806/0.820/0.804/0.802, so the agreement is a property of the pipeline and holds at the protocol's own 20 |
| [c] | trunk = 3D excursion of the shoulder midpoint | `score_vs_automq.py:254-262` ✅ — midpoint of the two shoulders, excursion from median rest |
| [c] | COCO-17 has no sternum keypoint | keypoint list |
| — | *(§II-D condensed 2026-08-20)* | 377 → 278 words. Cut was rationale only — every row above still maps to a sentence. Removed: why smoothing precedes differentiation; why the anatomical forward projection is unavailable; why a jittering axis dominates; why smoothing the position flattens the peak. Kept the three departures from `unger2024` and the omitted twelfth measure, which are the cases `unger2024` itself explains. Frame-frozen-per-trial moved into the frame definition as a fact. |

## §II-F Validation Protocol

| | claim | check against |
|---|---|---|
| [c] | lag is the stacked multi-signal consensus, same criterion that admits a trial | **CORRECTED 2026-08-20** — this row said "wrist-speed cross-correlation", the same stale claim already fixed in §II-A. `paper_trajectories.py:171` calls `H.find_lag_best` with the comment "STACKED multi-signal lag (same path as the kinematics scorer)"; `_find_lag` (wrist speed alone) is not used here. Methods describes the Fisher-z stacking correctly. |
| — | Methods no longer describes the stacking MECHANISM | **Cut 2026-08-20** — it was explained in §II-A and again in §II-F. Both now say only that the lag is a consensus across several joint and cup signals, which is accurate and does not re-describe `find_lag_best`. The Fisher-z / peak-weighted / single-argmax construction lives in the released code and in the row above. Watch that no future edit reintroduces "best of", which would name a different and worse estimator. |
| [c] | static offset removed per trial for the trajectory comparison | `paper_trajectories.py:116-124` |
| [c] | `r_s` = PEARSON over single trials, `r_av` over per-participant/arm averages, both pooled | **STATISTIC CHANGED 2026-08-20.** We computed **Spearman** while claiming to follow `unger2024`, which states *"**Pearson** correlations were calculated for each movement quality measure, both for individual trials (r_s) and for averaged movement measure (r_av)"* — their subscript means *single trials*, not Spearman. Table III was therefore not comparable to theirs. Now `pearsonr` in `paper_table3.py:48,51`; `fig4_correlation.py` defaults to `CORR=pearson` (`CORR=spearman` restores the rank version). Effect is small except on the two measures a rank statistic mishandles: **movement units 0.50→0.73** (small integer counts, heavily tied ranks) and **interjoint r_av 0.66→0.92** (reference values near-constant, so ranks track noise — the failure mode the Discussion block already predicted). Trajectory tables were always Pearson; `spearmanr` in `paper_trajectories.py` was an unused import. |
| [c] | landmark offsets fitted per landmark in body and arm frames | **RECOVERED** to `paper/scripts/anat12/` (2026-08-19). The fitter is `anat_frame.py` — `anat9` = shoulder TRUNK + elbow/wrist SEG (9 params), `anat12` = shoulder TRUNK+SEG (12), fitted by `scipy.least_squares`, which is exactly the "body and arm frames" the text claims ✅. `fit_landmark_offset.py` is the separate discriminator (landmark sinusoid vs axis-tilt constant vs line). The scripts were never in the repo because they were written to the session scratchpad under `/tmp`, not to `paper/scripts/`; all 32 were still on disk and are now committed. |
| [?] | offsets validated on held-out trials, within 0.007 | derived from `anat12w0_omc_values.csv` this session; confirm against the script |
| [?] | applied "before the angles are compared" — which measures? | unresolved; see plan block |

---

---

## 2026-08-21 — the optical reference was replaced, and three bugs behind it

Everything the Results now report is scored with **our** measure operator and **our**
segmenter on both sides. The DELTA study's stored measures, phase windows and kinematic
series are used nowhere. The four tables and both Fig. 4 panels share one trial set.

**Why the reference changed.** Its `kinematics` column holds six per-frame series derived by
the study's own code, and four of the six are different quantities from ours: shoulder
flexion/abduction project onto world axes rather than a body frame; `trunk_displacement` is
`-(chest_y - chest_y[0])` (exact to 1e-15 mm) — one marker, one lab axis, signed, referenced
to a single frame; `hand_*_velocity` uses the hand, not the wrist. Only `elbow_angle` was
like-for-like, and it was correspondingly the trajectory that moved least. Deriving both sides
through one operator took the trajectory correlations from 0.64–0.96 to 0.83–1.00.

**Trial count 636 → 790.** `cache_seg_inputs` refused any trial without a reference phase row,
which cost 161 usable trials; the arm label was the only thing still needed from the reference
and it is in every filename. Removed, then three exclusions applied:
`_MISPAIRED` (3, another C3D matches the clip by >0.02 at the stacked lag), `_BAD_OMC` (4,
below), and 45 with no usable optical wrist/nose. The cup caches were pinned to the same 636
by a cycle — `cache_cup_seed26x` read the seg-inputs cache, which cannot exist until the cup
track does — now broken by reading `load_clean`.

| | fixed | was |
|---|---|---|
| [c] `score_own_phases` passed the frame count as `segment_sequential`'s `fps` | 4th positional dropped | low-pass cutoff set from a frame count |
| [c] `max_trunk_displacement` built the shoulder midpoint inside `angle_measures_automq` | prefers `pose["trunk"]`, the sternum | any landmark fit moved it; anat12 drove it 0.93 → 0.48 |
| [c] mislabelled leading marker segments | `_destep` on the raw C3D | `_despike` runs post-resample and NaNs one frame, so a jump-and-stay survived |

`_destep` splits a marker at steps >60 mm/frame at the C3D rate (~6 m/s; a fast wrist is
~15 mm/frame at 100 Hz) and keeps the longest run, refusing when >35% of finite frames would
go. It fires on 20 of 858 trials, 16 of them P17, `elbow_L` in 11 — P17's left elbow is
misplaced for the first 0.6–0.75 s of a trial, reading 20 mm long on the upper arm until it
locks on. Clean trials are bit-identical. Effect: `peak_elbow_angular_velocity` 0.736 → 0.880,
every other measure ≤0.001.

`_BAD_OMC` acts on the four trials `_destep` refuses. `P08/trial_1_R_unaffected` has four
markers stepping at once including `chest`, and reports a 360.6 mm optical trunk displacement
against our 23.8 — 0.07 off Table III's trunk correlation from one trial. Threshold is not
tuned: >0.35 selects four, >0.15 selects six. **This is not the same kind of exclusion as the
segmenter's 4.9% declines and must never be summed with them** — a decline is read from the
markerless cup track and a deployed system has that signal; this needs mocap.

**Table III and Fig. 4 are landmark-matched** (`M3_CSV` selects; `table3_measures.csv` is the
uncorrected version). anat12 gained peak-velocity and elbow-angular-velocity blocks in its
loss, which resolves open mismatch 1 below: the angles-only fit was displacing the wrist by
43.6 ± 272.5 mm and destroying the derivative measures (peak wrist vel 0.758 → 0.002 held
out). With both blocks the wrist offset is 2.9 ± 6.8 mm and those measures return to baseline,
while the angles keep their gain (flexion 0.975 → 0.966 held out).

| [c] | landmark offsets do not transfer across participants | LOPO buys ~0 on every measure; per-participant held-out recovers nearly all of it. Rules out a fixed definitional offset between the two systems — the discrepancy is participant-specific. Does **not** establish which system carries it: neither is positionally authoritative. |
| [c] | held-out ≈ in-sample within a participant | flexion 0.966/0.966, abduction 0.952/0.957, elbow 0.983/0.986 — the "within 0.007" claim verifies (max gap 0.005) |
| [c] | trunk displacement takes the same treatment | 3-param body-frame offset on the optical sternum, trunk-displacement difference as loss: RMSE 4.28 → 3.45 mm held out, peak-excursion gap 3.23 → 1.40 mm, \|d\| 131 mm against the 122 mm the offset actually measures. LOPO 4.32, i.e. nothing. Not yet folded into `anat_frame` |
| [!] | `settle_observed` does not always reach the censoring logic | `P14/trial_38_R_unaffected` settles by the `rel + HOLD` timeout (239 mm from its own end reference, hand at 243 mm/s) yet reports a total movement time of 5.17 s. 4 of 602 trials differ by >0.5 s |
| [~] | interjoint is reported as a correlation | 2.4% of trials carry **91.7%** of the variance on the anat12 file the paper reports (94.5% on the uncorrected one — see 2026-08-24 below), and on those the optical elbow moves 1.4–1.7° across the whole reach against 14–16° normally — the reference is undefined, not wrong. Report agreement (median \|diff\| 0.008, 94.8% within 0.05) |

### Latency — measured 2026-08-21 (§II-E and §V-F are no longer empty)

`scripts/latency_bench.py` times the shipped stages on forced cache misses; `scripts/latency_opt.py`
is the optimised single-trial path and verifies its pose output against the shipped one. RTX 3060 Ti
8 GB, i7-13700K, torch 2.7.1+cu118. Capture is offline, so recording and inference never share the
GPU.

| | baseline | optimised |
|---|---|---|
| per trial, 5 cam / 549 fr (9.2 s of video) | 30.7 s | **16.8 s** (1.82×) |
| pose + cup tracking | 12.3 + 14.9 s serial | 15.4 s concurrent, 94 % GPU |
| SmoothNet | 2.42 s | 0.03 s |
| triangulation / BA / segment / measures | 1.08 / 0.40 / 0.01 / 0.00 s | unchanged |
| decode | 10 s NVDEC, counted | 3.7 s cv2, **excluded** |

Three wastes, all measured before being removed:
1. **Two decodes, the slower one on the GPU.** `gpu_decode` (NVDEC piped as raw BGR24) does
   268 cam-frames/s against cv2's **1149** — a 5-camera trial is 17 GB through a subprocess pipe.
   Pose used NVDEC, tracking used cv2, so the pixels were decoded twice by different means.
2. **Pose and UETrack ran in sequence** on a GPU neither saturates alone (80.6 % / 92.4 %). Two
   threads on two CUDA streams: 26.4 ms per rig-frame against 30.9 serial, 94 % GPU. Only 1.17× —
   both are compute-bound, so this is near the card's ceiling.
3. **SmoothNet ran one window at a time.** `_smooth_joint` issues ~500 batch-1 forwards per joint,
   ×9. `pose_smooth.smooth_joints_batched` already existed, stacks them into one forward, and was
   never wired into the scorer: **29.7×**, difference 0.05 mm = the shipped path's own 0.1 mm
   rounding. **Not yet wired into `score_own_phases`** — doing so would speed every rescore by
   ~2.4 s × 790 trials but perturbs the 4th decimal, so the tables would need regenerating.

Rejected after measurement, so nobody retries them: YOLO-pose fp16 (1.01×, ultralytics already runs
half precision) and `UETrackBatch(bgr=True)` (1.10× but the box moves up to **686 px** — not the
same tracker).

Equality check: the optimised path's smoothed pose matches the shipped cached pose to **0.000 mm**
on all three trials tested.

Reported in §V-F: 16.8 s per trial, 1.84× real time, of which **1.3 s** falls after the last frame.
The front end needs 28 ms per five-camera frame against 60 Hz's 16.7 ms budget, so live streaming
does not fit at five cameras on this card and is within reach at three. The `53×` against
`unger2024` is 12 GPU-days / 1160 trials = 894 s per trial, over 16.8 s.

### SmoothNet rewired to the batched path (2026-08-21)

`results_v3_delta._smooth_joint` now calls `pose_smooth.smooth_joints_batched` through a new
`_smooth_pose(dict)`, so every call site batches. Measured 1.08 s → 0.036 s for nine joints (29.7×);
the win is batching WINDOWS (~520 batch-1 forwards per joint), so even a single-joint call gains.

- `cache/pose_smoothed`, which the scorer actually reads, was **already** built with
  `smooth_joints_batched` (`cache_smoothed_pose.py:52`), so no published pose number moves.
- The 0.1 mm rounding `smooth_track` applied is preserved. Verified on 11 trials, one per
  participant: **9 identical, 2 differ by exactly one rounding step (0.1 mm)** on a few frames where
  the batched sum order lands on the other side of a tie.
- **`cache/seg_inputs_ship` is NOT rebuilt.** It was built through the per-window path, and the
  segmenter is one-frame sensitive at that magnitude: perturbing every channel of every frame by
  ±0.1 mm moves a boundary by 1–2 frames on **13 of 39** sampled trials (worst 2 frames = 33 ms).
  That is inside the boundary disagreement Table IV already reports, but there is no reason to
  perturb published boundaries for a rounding tie. `OT_SMOOTH_LEGACY=1` restores the old path
  exactly.
- Worth knowing independently: that sensitivity means Table IV's medians are not stable to the last
  frame. It does not change any conclusion — 1 frame is 17 ms against medians of 17–100 ms — but a
  reviewer asking "is the grasp boundary really 100 ms?" should be told ±1 frame.

### Ablations written 2026-08-21: capture rate, and alternatives to the two weak measures

**Capture rate (`paper/scripts/fps_ablation.py`, 750 trials; `fps_cup.py`, 40 trials).** Decimating
every markerless channel before smoothing, with the smoother, segmenter and measures at 30 Hz:
median Δr_s **+0.001**, worst −0.017, none beyond 0.02. Absolute peaks fall: PV −6.6%, elbow angular
PV −8.7%.

- **1.4 of those 8.7 points were a bug**, not the sampling: `H._lp` builds its Butterworth cutoff
  from the module constant `VIDEO_FPS = 60`, so on 30 Hz data the intended 6 Hz became 3 Hz. The
  ablation patches `H.VIDEO_FPS` for the call; the shipped code still has the latent bug and should
  take fps as an argument. `peak_velocity_reduce` is already correct (it passes FPS to its own
  butter), which is why PV is −6.6% either way.
- **Decimating the cup was invalid** and `fps_cup.py` replaces it: UETrack is recursive, so a 30 Hz
  capture produces a different track, not a subsample. Re-run at 30 Hz from the same seed at the same
  batch size, the cup sits **6.8 mm** from the 60 Hz track at the median, 28.5 mm at p90, with >5% of
  frames beyond 10 mm on **40 of 40** trials. Boundaries absorb it (0–50 ms median; grasp p90 267 ms,
  >100 ms on 8/40). Measure shifts reproduce to within 0.4 pp — **except movement units, which loses
  one unit on 22 of 40 trials**, invisible in the decimated run because ranks are preserved when
  everything shifts down together.
- UETrack determinism: identical at fixed batch size (0.0000 px over 250 frames) but **not
  batch-invariant** — batch 1 vs 5 diverges up to 7.66 px (`tf32.cudnn=True`, cuDNN algo choice).
  `cache_trial` tracks only newly-seeded cameras when merging, so **`tracks_uetrack_26x` is not
  reproducible by rebuild**. Belongs in the release notes.

**Alternatives (`paper/scripts/alt_measures.py`, 748 trials).** Reuses `vcode` from
`scripts/archive/vector_coding.py` and adds velocity-form LDLJ (Balasubramanian 2015).

| | r_s | r_av | AUC (mmc) |
|---|---|---|---|
| interjoint (incumbent) | 0.41 | 0.79 | 0.45 |
| coupling angle, circular mean | 0.57 | — | 0.42 |
| proximal-phase fraction | 0.65 | 0.84 | **0.65** |
| distal-phase fraction | 0.60 | 0.78 | 0.34 |
| movement units (incumbent) | 0.84 | 0.93 | 0.55 |
| LDLJ | 0.64 | 0.71 | 0.48 |

- **anat12 is measure-specific, and this is the important finding.** The offsets are fitted to the
  angle *maxima* (plus wrist speed and elbow angular velocity); vector coding reads the *ratio of
  the angles' derivatives*, which was never in the loss. Applying anat12 **degrades** it: proximal
  0.648→0.603 r_s and 0.841→0.726 r_av, distal 0.601→0.315, in-phase 0.562→0.350 — while interjoint
  improves 0.406→0.483. Movement units and LDLJ are indifferent (no shoulder angle). Table VII
  therefore reports the **uncorrected** numbers, and §VI states the consequence: a fit tuned on one
  functional of a trajectory is not a correction of the trajectory.
- **The proximal fraction's correlation is a threshold artefact; its separation is not.** Rotating
  the Needham bin edges ±5° moves r_s between 0.47 and 0.73, and 25.6% of frames sit within 5° of an
  edge — so do NOT claim it agrees better than interjoint. But affected-vs-unaffected AUC is
  0.596/0.621/0.610 optical and 0.601/0.650/0.673 markerless across the three placements, every one
  p ≤ 3e-7. Distal is unusable: AUC 0.474 optical against 0.340 markerless, i.e. the systems disagree
  on the direction of the effect.
- **Movement units' 0.84 is carried by a few trials.** 74.3% of optical values are exactly 1 (556/748;
  99.3% share their value with another trial — the "98.1% ties" once recorded here reproduces under no
  definition tried, see 2026-08-24); Pearson 0.841 against Spearman **0.558**; exact agreement 59.2%, within one 90.9%.
- **LDLJ is worse and window-fragile**: 0.64 optical windows, **0.34** markerless, because it
  normalises by T³ so a boundary error enters cubed. `unger2024` proposes LDLJ without testing it, so
  this is a reportable negative.

### Threshold-free coordination index, and a separation claim retracted (2026-08-21)

**The measure.** `vc_continuous` in `paper/scripts/alt_measures.py`. gamma = atan2(d elbow, d flexion)
per frame, as in vector coding; then
`cos 2*gamma = (dflex^2 - delb^2)/(dflex^2 + delb^2)`, averaged over the reach — +1 pure shoulder,
−1 pure elbow, 0 equal. The doubled angle is the standard circular-statistics handling of AXIAL data
(0 deg and 180 deg are the same coordination), so the continuous form is textbook and the four
Needham bins were the ad hoc step. The weighted variant weights each frame by
`sqrt(dflex^2 + delb^2)`, demoting near-stationary frames whose direction is least determined.

| formulation | r_s | r_av | r_s (mmc win) | Spearman | var. in top 5% |
|---|---|---|---|---|---|
| interjoint (incumbent) | 0.406 | 0.787 | 0.472 | 0.651 | 96.3% |
| proximal fraction (binned) | 0.648 | 0.841 | 0.677 | 0.588 | 75.1% |
| mean cos 2g | 0.815 | 0.921 | 0.771 | 0.762 | 43.5% |
| **mean cos 2g, weighted** | **0.903** | **0.946** | **0.882** | **0.872** | **40.9%** |

- Not outlier-driven: Pearson 0.903 vs Spearman 0.872, and the reference IQR is −0.445…−0.096 rather
  than interjoint's 0.969…0.989.
- **Landmark-fit-free, and the fit hurts it**: anat12 costs the weighted index 0.114 r_s
  (0.903 → 0.789) and 0.142 r_av. Same direction as the binned version, larger magnitude. Consistent
  with anat12 being fitted on the angle MAXIMA while this reads the angles' derivatives.
- Bins are a threshold artefact: 25.6% of frames within 5 deg of an edge; ±5 deg rotation moves the
  proximal fraction's r_s over 0.469 / 0.648 / 0.733.
- `sin 2*gamma` (in-phase axis) is recovered poorly (0.543) — report only the proximal–distal axis.

**[!] Separation claim retracted.** I reported pooled Mann-Whitney AUCs (proximal 0.65, cos2g 0.66
markerless) as affected-vs-unaffected discrimination. **That pooling is confounded**: P19 contributes
47 affected trials and 0 unaffected, so its overall level shifts the affected distribution wholesale.
The correct test is paired within participant — each participant's per-arm medians, Wilcoxon over the
10 units with both arms — and **nothing separates**:

| measure (mmc) | unaff | aff | p | direction consistent |
|---|---|---|---|---|
| interjoint | 0.986 | 0.985 | 0.28 | 4/10 |
| proximal fraction | 0.010 | 0.036 | 0.09 | 5/10 |
| cos 2g | −0.434 | −0.367 | 0.56 | 6/10 |
| cos 2g weighted | −0.320 | −0.263 | 0.63 | 5/10 |
| movement units | 1.000 | 1.000 | 1.00 | 1/10 |

n = 10 pairs, so this is **not demonstrated**, not **shown absent**. §V-G and §VI now say agreement
only, and Table VII's caption states it. **Any future pooled-arm comparison in this project must be
paired or weighted per participant** — the same confound would hit any measure.

**Positive control run afterwards, and it fails too — which is the point.** The same paired test on
the twelve headline measures separates nothing either, `total_movement_time` included:

| measure (mmc, paired n=10) | unaff | aff | p | direction |
|---|---|---|---|---|
| total movement time | 5.59 s | 6.08 s | 0.32 | 8/10 slower |
| shoulder abduction | 25.6° | 33.8° | 0.064 | 7/10 higher |
| shoulder flexion | — | — | 0.105 | 8/10 |
| peak velocity | 551 | 505 mm/s | 0.28 | 3/10 |
| peak elbow angular velocity | 95.9 | 108.1 °/s | 0.32 | 4/10 |
| trunk displacement | 51.4 | 49.9 mm | 0.63 | 6/10 |

The directions are the expected compensatory pattern (affected slower, more abducted, more trunk),
which validates the arm labels and the operators. But Wilcoxon at n=10 cannot go below p=0.002 even
for a perfect split, and 8/10 in one direction is only p≈0.11. So the honest reading is **the cohort
cannot test discrimination for ANY measure**, not that these measures fail to discriminate. Moved to
§Limitations, where it applies to Tables III and VII alike, rather than sitting in §V-G where it
would read as evidence against the new formulation specifically.

### [!] ARM IS CONFOUNDED WITH RECORDING ORDER (found 2026-08-21)

The affected and unaffected repetitions are **consecutive blocks, not interleaved**, in every unit —
unaffected first, affected second, no overlap in trial number, adjacent-trial arm switches 0.01–0.03
(0.5 would be alternating):

| part | affected trials | unaffected trials | | part | affected | unaffected |
|---|---|---|---|---|---|---|
| P07 | 43–83 | 1–42 | | P15 | 46–90 | 2–45 |
| P08 | 50–89 | 2–48 | | P17 | 43–83 | 1–42 |
| P10 | 44–83 | 1–40 | | P251 | 23–41 | 1–21 |
| P12 | 41–82 | 3–38 | | P252 | 63–83 | 43–62 |
| P13 | 41–80 | 1–40 | | P14 | 48–90 | 1–47 |

**Consequences.**
1. No arm comparison in this dataset can be attributed to impairment. A re-seated marker, a moved
   cup, calibration drift or fatigue between blocks is perfectly aliased with arm — and it hits BOTH
   systems, so OMC–MMC agreement on an asymmetry does not disentangle it.
2. Per-participant, the arms DO separate, strongly: on the movement-weighted `cos 2g`, 10/10 units
   significant optically, 8/10 markerlessly, 7/10 agreeing on significance and direction (P08 reaches
   AUC 1.000, p=3e-15 markerless). Interjoint separates too (7 OMC / 9 MMC). **None of this is
   evidence about impairment.**
3. The direction is inconsistent — affected-higher in 4/10, right-arm-higher in 6/10, and the
   cross-tab of affected-side against direction is flat — so it is neither an impairment effect nor a
   left/right effect. A universal fatigue drift would push all units the same way (unaffected is
   always first), so a *per-session* artefact or genuine idiosyncratic asymmetry are the live
   explanations; this data cannot separate them.
4. Tables I and II split by arm. For AGREEMENT metrics (RMSE, r, bias between systems) the confound
   is mild — a drift moves both systems together — but the split is a stratification of the recording,
   not of the pathology, and §Limitations now says so.

**Do not run an affected-vs-unaffected analysis on this dataset** without an interleaved protocol or
recorded block order. Two earlier claims died here: the pooled AUC (confounded by P19's 47 affected /
0 unaffected trials) and then the per-participant separation (confounded by block order).

### The two bespoke constants are now shown insensitive (2026-08-22)

Neither was tuned, and both sweeps show it. **Do not turn these into selection procedures** — picking
either value by maximising agreement with the reference protocol's phases would make the segmentation
validation circular, fitting to the signal we then validate against, on the same trials. The point of
each sweep is that the shipped value is NOT the argmax.

**Segmenter range fraction (shipped 0.3)**, `scratchpad/frac_sweep.py`, 264 trials:

| fraction | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 |
|---|---|---|---|---|---|---|
| MMC-vs-OMC median (ms) | 33 | 33 | **33** | 33 | 33 | 33 |
| vs reference protocol (ms) | 217 | 208 | **208** | 217 | 200 | 208 |

Flat on the reported quantity; the best match to the reference protocol is at 0.40, not 0.30.

**BA bone-variance weight (shipped 0.05)**, `scratchpad/bone_sweep.py`, 110 trials, BA re-solved at
each weight and re-scored through the same operators:

| measure | 0.0 | 0.025 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| peak velocity | 0.876 | 0.902 | **0.904** | 0.901 | 0.896 |
| elbow angular PV | 0.923 | 0.942 | **0.942** | 0.941 | 0.937 |
| elbow extension | 0.976 | 0.983 | **0.983** | 0.984 | 0.981 |
| shoulder flexion | 0.968 | 0.976 | **0.976** | 0.976 | 0.976 |

Flat over an 8x range; the term earns its place (removing it costs 0.03 PV, 0.02 elbow angular PV)
without the value mattering. **The subset's interjoint values (0.71→0.91) are NOT comparable to
Table II's 0.48** — interjoint's variance sits in 18 trials cohort-wide, so any 1-in-7 subset moves it
wildly. Sensitivity across weights on one subset is valid; the absolute numbers are not quotable.

One sentence each is now in §III-B and §III-C.

### Moderate concerns: the "double-counted cup failures" was MY WORDING, not a double count

**§IV-A said "45 carry no usable cup track". That is wrong.** `cache_seg_inputs.py`'s `n_bad`
counts trials with no `_affected`/`_unaffected` tag or **no usable optical wrist or nose marker**.
The cup counter in the same pass, `n_nocup`, is **0** in both fill logs. So the 45 are an
OPTICAL-side exclusion, in the same class as the four de-stepper refusals, and there is no double
counting with the 39 segmenter declines. The deployable failure rate stays 4.9%. Text corrected.

**[!] The "30–50 mm/s video jitter floor" was another unverified assertion.** Measured markerless
wrist speed before reach onset: **13.4 mm/s** median (IQR 8.5–22.4); optical 4.0 mm/s. The claim
lived in `pipeline/score.py:37` as well as §III-D. Replaced with the measurement — and with the
sensitivity sweep, which is the better answer anyway:

| MU amplitude threshold | 20 | 40 | 60 | 80 mm/s |
|---|---|---|---|---|
| r_s (end-to-end) | 0.806 | 0.820 | **0.804** | 0.802 |

**Agreement does not depend on the threshold**, including at the protocol's own 20 mm/s, so the
0.80 is a property of the pipeline. `scratchpad/mu_sweep.py`.

**Also fixed:** "none by more than 0.04" contradicted "+0.088" in the same sentence → "none
degrades by more than 0.04"; interjoint improving under worse windows is now remarked on as evidence
the statistic is noise-dominated; the `unger2024` comparison carries an uncontrolled-comparison
caveat; empty joint-frames are stated to stay empty rather than be interpolated.

**Open, not addressed:** Table I is computed on the 790 (it excludes the 45 optical-marker
failures, which is correct for it after all, since those trials have no optical wrist to compare
against — the reviewer's premise that the exclusion was cup-based was wrong). Table II's n column
(748/747 vs 750) is 2–3 trials where a reaching-windowed measure is undefined; not yet explained in
the text. No stage ablations.

### M6: two of four points taken, with numbers (2026-08-22)

**Trajectory r IS partly the shared task profile** (`scratchpad/null_r.py`). Null = pair a markerless
trajectory with the optical trajectory of a DIFFERENT trial by the same participant and arm,
resampled to length. Same task, same person, same arm, different execution:

| trajectory | real r | null r | gap |
|---|---|---|---|
| elbow extension | 0.996 | **0.934** | 0.06 |
| shoulder flexion | 0.962 | 0.850 | 0.11 |
| trunk displacement | 0.971 | 0.830 | 0.14 |
| end-effector velocity | 0.983 | 0.669 | 0.31 |
| elbow angular velocity | 0.965 | 0.648 | 0.32 |
| shoulder abduction | 0.823 | 0.527 | 0.30 |

So the criticism holds for the ANGLES and not for the velocities. §IV-B now says this and points the
reader at RMSE and bias, which is what Table I was always for.

**The pooled/within split confirms the session-constant story** (independent of the anat12 fit):

| measure | pooled, no fit | within-session, no fit | within-session, fitted |
|---|---|---|---|
| shoulder flexion | 0.683 | **0.859** | 0.850 |
| flexion (drinking) | 0.730 | 0.929 | 0.893 |
| abduction | 0.863 | 0.902 | 0.883 |
| elbow extension | 0.900 | 0.873 | 0.852 |

Most of the pooled deflation is between-session offset, and the fit leaves the within-session figure
alone — as an additive constant must, Pearson being offset-invariant. The small decreases say the
fitted 3D displacement adds a little pose-dependent distortion within a session while removing the
between-session constant. In §IV-C.

**NOT taken, deliberately:** confidence intervals on r_av (n=21) and Bland–Altman limits of
agreement. `unger2024` reports neither, the comparison is the point of the tables, and a 12-row LoA
table does not fit the page budget. If a reviewer insists, LoA is one table from
`score_own_phases_anat12.csv` with no new computation.

### M4 answered: the deployable configuration is now reported (2026-08-22)

**Table II carries a 30 Hz column** (`paper/table3_fps30.csv`, from the full re-tracked run). The
headline count is unchanged: **10 of 12 clear r_s ≥ 0.84 and r_av ≥ 0.93 at BOTH rates**, and the
same two fail (movement units, interjoint). All three measures moving >0.01 in r_s are now named:
PV −0.028, movement units +0.016, interjoint −0.275.

**Camera-count scaling measured, not extrapolated** (`scratchpad/cams.py`, pose + tracker concurrent
on two streams, frames in RAM):

| cameras | pose | track | concurrent | fits 60 Hz (16.7 ms)? |
|---|---|---|---|---|
| 3 | 11.0 | 7.4 | **14.9 ms** | yes |
| 4 | 14.4 | 9.0 | 19.1 ms | no, fits 30 Hz |
| 5 | 17.9 | 11.1 | 24.7 ms | no, fits 30 Hz |

The reviewer's extrapolation (22 ms at four, 17 at three) was pessimistic. Only 3 of 11 units keep
five cameras, so five is the worst case and the modal configuration is at or inside the 60 Hz budget.
NB the 24.7 ms here is inference alone; the 28 ms in §IV-F is the same two networks inside the
running pipeline, which adds queueing and result parsing. Both are five-camera figures — do not
mix them.

### [!] The "10% calibration scale error" was never verified, and is CONTRADICTED (2026-08-22)

§Limitations carried, from an old `\todo` note, "the calibration inflates every 3D distance and
velocity the pipeline reports by roughly 10%". Nothing in this project's outputs supports it and two
independent tests contradict it:

| scale-dependent quantity | OMC | MMC | MMC/OMC |
|---|---|---|---|
| peak velocity | 534.3 | 524.0 mm/s | **0.963** |
| trunk displacement | 49.4 | 51.7 mm | 1.015 |
| peak elbow angular velocity | 102.0 | 99.7 °/s | 1.000 |
| upper arm length | 339.1 | 264.9 mm | **0.791** |
| forearm length | 249.3 | 238.6 mm | 0.968 |
| shoulder width | 317.9 | 333.1 mm | 1.032 |

A global scale error gives the SAME ratio on every segment. These differ by segment (0.79 / 0.97 /
1.03), which is the signature of landmark correspondence, not calibration — and the shortened upper
arm is precisely what the fitted 97 mm shoulder offset predicts, so the two analyses agree
independently. In the measures there is no inflation at all: peak velocity runs 3.7% LOW.

§Limitations now states the measured, measure-specific differences and attributes them to landmark
correspondence. **Do not reinstate the 10% figure without a scale artefact of known length measured
in the volume.** Where the original note came from is unknown; it predates this session's records and
may refer to an earlier calibration or to the cup rather than the body.

### 30 Hz, whole cohort, tracker re-run (2026-08-22) — and a SmoothNet artefact behind it

`paper/scripts/fps_full.py`, 3 shards × ~264 trials, ~13–16 s/trial, 750 trials scored. Only the
30 Hz arm is computed; the 60 Hz arm is the published `score_own_phases_anat12.csv`, so the
comparison is against what the paper prints. The cup tracker is genuinely re-run on every second
frame (recursive, so decimating its output is invalid — `fps_cup.py` measured 6.8 mm median
divergence).

**[!] The peak loss attributed to sampling was a smoothing artefact.** SmoothNet's checkpoint has a
fixed 32-sample window: 0.53 s of support at 60 Hz, 1.07 s at 30 Hz. Decomposed on 154 trials by
smoothing at 60 Hz and decimating (B) versus decimating and smoothing at 30 Hz (A):

| | PV median | vs 60 Hz |
|---|---|---|
| baseline, smoothed at 60 | 654.7 mm/s | — |
| B: sampling alone | 650.1 | **−0.2%** |
| A: smoothed at 30 (the naive port) | 586.1 | **−7.6%** |

So sampling costs nothing and the filter costs everything. Confirmed cohort-wide: naive −6.6% PV /
−8.9% elbow angular PV, rate-matched −0.2% / −0.9%. Only the window-32 checkpoint exists locally
(no 8/16/64), so the rate-matched variant interpolates to 60 Hz, smooths, and decimates.

**Result at 30 Hz (rate-matched), 750 trials:** 9 of 12 measures within 0.01 of the 60 Hz r_s, all
within 1.5% in value. Two exceptions: **interjoint 0.572 → 0.297**, because it is a correlation
estimated WITHIN the reaching window and halving the rate halves its samples (~48 → ~24) — the only
measure whose statistic depends on sample count; and **PV loses 0.028 of r while recovering its
magnitude**, the bias–variance trade the naive smoothing was hiding (naive keeps r at 0.883 by
attenuating every peak 6.6%).

**Approximations that remain**, stated rather than hidden: 2D pose decimation is exact (per-frame
YOLO); BA decimation is exact per frame EXCEPT the bone-length variance term, which couples frames
and would be computed over half as many samples in a true 30 Hz solve; and the segmenter's
cup/wrist/nose channels are the shipped 60 Hz-smoothed ones decimated, i.e. rate-matched in both arms.

### Sternum offset now APPLIED (2026-08-22)

The trunk fit was never a decision to omit — it was an unfinished integration, sitting in
`trunk_offset_fit.py` as a probe. Now wired in: `paper/scripts/anat12/trunk_theta.py` fits d per
(part, arm) in sample and writes `out/scoring/trunk_theta.csv`; `score_own_phases.py --trunk-theta`
applies it to the OPTICAL `trunk` landmark.

**Table III trunk row: r_s 0.934 → 0.952, r_av 0.960 → 0.986** (matching the probe's out-of-sample
0.950 / 0.985). Nothing else in the table moves. Fitted |d| spans 54–296 mm across the 21 groups;
per-frame RMSE improves in 19 of 21.

**[!] Two ordering traps, both hit and fixed:**
1. The offset MUST be applied in a **per-frame** torso triad, not the frozen one the twelve arm
   parameters use. Trunk displacement is a distance from the trial's own rest position, so an offset
   constant in a frozen frame is a constant world vector and cancels EXACTLY — a frozen-frame sternum
   offset does nothing at all. `_trunk_basis` in the scorer duplicates `trunk_offset_fit.basis` for
   this reason; keep them in step.
2. It must be applied **before** the twelve arm offsets. anat12 displaces the SHOULDERS, which define
   the torso triad the sternum offset lives in, so applying it afterwards evaluates d in a frame the
   fit never saw. First attempt did exactly that and drove trunk to **r_s 0.557** — a 0.38 drop that
   looked like a broken fit and was a broken call order.

Table VI (capture rate) does not include trunk, so it needed no rerun.

### Redundancy pass (2026-08-22): 540 words cut, no facts removed

Cut in three categories. **Nothing measured was deleted** — only repetition and internal
justification.

*Said three times:* the "both systems get the same optical windows, so this isolates pose error"
framing (§II-C's "two roles" paragraph deleted; kept in §V-D, §V-E now just points forward); the
right-censoring rule for total movement time (rule stays in §II-A, count in §V-A, the §V-D restatement
went); the 1.3 s residual latency, which appeared three times inside §V-F.

*Said twice:* declines being self-detected and therefore reportable (kept in §V-E, cut from §V-A);
the landmark fit not transferring (Methods now points forward to §V-C); the decode exclusion (rule in
§II-E, number in §V-F); the segmentation-cost mechanism and the latency numbers, both restated in the
Discussion and now compressed there to their conclusions.

*Justifying our own choices to nobody:* the cup re-anchoring cascade argument, "a joint outlier is the
same joint seen badly", "a cup outlier is a different object --- another vessel in the room", "a
fabricated cup position cannot help them", the wrist-lag gate defence, "three further departures are
forced rather than chosen", the two-alignments recap in §II-F, and "the two withdrawals differ in kind
and are not summed". Every underlying RULE survives; only the defence of it went.

*Also cut:* the §V-C trunk-offset paragraph. It reported a fit that was tried and then not applied,
which is a lab-notebook fact. The numbers remain above under the trunk fit entry.

### SCOPE, second pass (2026-08-22): trimmed again for length

- **Figure 2 removed** — the optical-windows correlation scatter (`fig4a_omcseg_anat12.png`). The
  file and the script survive; `main.tex` inputs only the end-to-end panel, now Fig. 2, whose caption
  was rewritten to stand alone. Pose-isolation numbers remain in Table III's left column pair.
- **Table I is single-column** (`table` not `table*`, `\scriptsize`, shortened row labels). At
  `\footnotesize` in one column it overran by 29.5 pt.
- **LDLJ cut entirely** from the paper. The result stands and is recorded above (r_s 0.64, falling to
  0.34 under markerless windows because of the T³ normalisation) — it is simply out of scope for
  length. `paper/scripts/alt_measures.py` still computes it.
- **§Ablations is now §Capture Rate**, holding only that ablation. The diagnosis of why interjoint
  and movement units read low moved into §VI Discussion, with its numbers, where it explains the
  paper's own Table III rather than proposing anything.

### SCOPE (2026-08-21): the alternative measures are FUTURE WORK, not results

Validating a new measure is a different study. `tables/table7_alt.tex` is no longer input by
`main.tex` and the Ablations subsection keeps only what characterises the measures the paper DOES
report — interjoint's variance concentration and Pearson-vs-Spearman gap, movement units' tie rate
and 59%/91% agreement, and the LDLJ test, which is a response to a proposal in `unger2024`
rather than a measure of ours. The discriminative-validity discussion is CUT from §Limitations
entirely — this is a measurement-agreement study, `unger2024` makes no such claim either, and the
paired-test null was over-explaining a question nobody asks of a validation paper. The travel-share
reformulation survives as one clause in §VI (0.48 → 0.89, no landmark matching, "a separate study"). Everything below is kept as the record behind that sentence and
because the negative results are expensive to rediscover.

### The coordination statistic: pooled ratio, not a rectifier (2026-08-21)

Four formulations of the same proximal--distal axis, all threshold-free, 748 trials, no anat12:

| form | definition | r_s | r_av | r_s mmc | Spearman | var in top 5% |
|---|---|---|---|---|---|---|
| per-frame mean | mean cos 2g | 0.815 | 0.921 | 0.771 | 0.762 | 43.5% |
| per-frame, weighted | sum(r·cos2g)/sum r | 0.903 | 0.946 | 0.882 | 0.872 | 40.9% |
| **travel share (L1)** | sum\|df\| / (sum\|df\| + sum\|de\|) | 0.889 | 0.941 | 0.870 | 0.866 | 48.9% |
| **pooled (L2)** | (sum df² − sum de²)/(sum df² + sum de²) | 0.895 | 0.940 | 0.887 | 0.866 | **37.3%** |
| rectified | mean max(cos2g, 0) | 0.817 | 0.941 | 0.804 | 0.820 | 61.0% |

`cos 2g` is itself a per-frame RATIO, so a frame where neither joint moves contributes a full ±1.
The pooled forms aggregate magnitudes and divide once — same information, no explicit weight, no
rectifier — and match the weighted mean while being better conditioned. **Paper uses the travel
share (interpretable, one-sided, [0,1]) with the energy form quoted alongside.**

**[!] The rectifier manufactured an arm effect. Do not use `max()` here.** `mean max(cos2g,0)` showed
a significant affected-vs-unaffected difference MARKERLESSLY (p=0.027 signed / 0.004 deviation, 7-9
of 10) and NOTHING optically (p=0.312, 4/10) — on a measure the two systems agree on at r_s 0.82.
Checks: the MMC−OMC bias is not arm-dependent (p=0.49) nor side-dependent (p=0.49) and is ~0.007,
far too small to produce the 0.05 arm gap. The effect **vanishes under both pooled forms**
(p 0.56–0.70, 4–5/10). Since E[max(X,0)] > max(E[X],0), a rectifier converts noise into positive
bias, so a noisier arm inflates it — and the observed bias leans that way (unaff −0.007, aff +0.001).
Treat it as a rectification artefact; it is not reported.

Standing lesson for this project: **any one-sided summary built by rectifying a noisy per-frame
quantity will show a spurious difference between conditions of unequal noise.** Aggregate first,
divide once.

**Still not measured.**
Hardware for it: i7-13700K (16c/24t), 61 GB, RTX 3060 Ti 8 GB, torch 2.7.1+cu118; capture is
offline so recording and inference do not share the GPU. Camera-dropout and miscalibration
data exist (`paper/archive_20260820/augment_camdrop.csv` 10.8k rows, `augment_miscal.csv`
49.4k, `augment_measures.csv` 38.4k) but are scored against the retired reference. Study
population needs DELTA demographics, not on this machine; 21 participants have optical
processing, 11 are used on calibration validity.

**Resolved from the list below:** mismatch 1 (anat12 scope) — the fit was not a constant
offset and the velocity blocks were the fix. Blocking item 4 (the manual
`seg_boundaries.csv` copy) is now inside the regeneration script, though not inside
`score_seg_boundaries.py` itself.

## 2026-08-21 — Results §V-B..§V-E written, and what backs each number

§II-F changed with it: the landmark paragraph no longer says the fitted offsets are unreported
(§V-C reports them) and no longer carries the "generalisation across participants is untested"
todo (it is tested, and it fails). The paragraph now states the fit is per participant and arm,
that the pooled fit is reported separately, and that a deployed version would fit against a
biomechanical model.

| claim in §V | where it comes from |
|---|---|
| trajectory numbers, §V-B | `tables/table1_trajectories.tex`, `table2_bias_iqr.tex` — per-trial static offset removed, Unger-comparable, NOT landmark-matched. The bias column *is* the offset that was removed |
| angle-measure bias is a per-session constant: group means span −9.8…+19.4° (flexion), −19.2…−1.3° (elbow ext), −3.1…+15.0° (abduction); median within-group SD 1.4 / 1.3 / 1.7°; 93% / 89% / 79% of the per-trial difference variance is between groups | computed from `out/scoring/score_own_phases.csv` (`omc_omc − mmc_omc`), declines excluded by the Fig-4 rule, 750 trials, 21 participant × arm groups with ≥4 trials. Trunk is 51% by the same measure and is not claimed |
| fitted offsets 48 / 97 / 41 / 2.9 mm with SD 140 / 120 / 40 / 6.8 mm | `out/scoring/anat12_wv1wa1.log`, per-block table across the 21 fits |
| held out vs in sample 0.966/0.968, 0.951/0.957, 0.984/0.986 | same log, `within_oos` vs `within_in` |
| pooled fit does not transfer: flexion 0.72 → 0.55, others <0.02 | `out/scoring/anat12_lopo.log`. **Angles-only loss (w=0)** — the LOPO arm of the velocity-in-loss variant was never run to completion, so this is a different loss from the one Table III uses. Stated in §V-C without naming the variant; rerun `anat12_lopo.py --wv 1 --wa 1` if a reviewer presses |
| the match acts on the angles and nothing else: flexion +0.28, elbow ext +0.08, abduction +0.08, interjoint +0.08, elbow ang PV +0.03, everything else ≤0.003 | `table3_measures.csv` (no match) vs `table3_measures_anat12.csv` (matched) — same 750 trials, same code path, optical windows |
| trunk sternum fit 0.969 → 0.976 per frame, 4.28 → 3.45 mm, peak gap 3.23 → 1.40 mm, scalar 0.936 → 0.950 | `out/scoring/trunk_anat12.log`. §V-C states explicitly that Table III's trunk row is the *unfitted* sternum |
| segmentation cost: 8 of 12 measures move <0.01, none >0.04 — movement units −0.036, time to PV −0.033, elbow angular PV −0.013, interjoint +0.088; elbow ext and trunk identical | the two column pairs of `table3_measures_anat12.csv`. **An earlier draft said "nine of eleven ≤0.01"; that was a miscount even before the twelfth measure — it was eight of eleven on rounded values, seven on exact ones** |
| the cost follows the windows, not the boundary errors | window assignment read off `angle_measures_automq` / `peak_velocity_reduce`: reaching (onset→grasp) for PV, time-to-PV ×2, movement units, elbow angular PV, interjoint; reach onset→drink offset for the two max shoulder angles; whole trial for elbow extension and trunk; onset→settle for TMT. Grasp is the weakest boundary (100 ms) and windows every measure that moves; the drink boundaries window only the two maxima, which move 0.0002 |

### The twelfth measure: drinking-phase shoulder flexion (added 2026-08-21)

`unger2024` Table III has **twelve** rows, read off the PDF at `~/Downloads/2411.14992v1.pdf`
(p. 5). The arXiv HTML fetched through a summarizer returned only eleven — it silently dropped
the `Shoulder abduction` row — so **quote the PDF, not the HTML**:

| unger2024 measure | their r_s | their r_av | ours r_s | ours r_av |
|---|---|---|---|---|
| PV | 0.90 | 0.96 | 0.89 | 0.94 |
| Elbow angular PV | 0.87 | 0.93 | 0.91 | 0.97 |
| Time to PV | 0.98 | 1.00 | 0.99 | 0.99 |
| Time to first PV | 0.94 | 0.99 | 1.00 | 1.00 |
| Number of movement units | 0.77 | 0.91 | 0.84 | 0.93 |
| Total movement time | 0.98 | 1.00 | 1.00 | 1.00 |
| Interjoint coordination | 0.71 | 0.71 | 0.48 | 0.83 |
| Trunk displacement | 0.96 | 0.97 | 0.93 | 0.96 |
| Shoulder flexion | 0.97 | 0.98 | 0.97 | 0.99 |
| Elbow extension | 0.98 | 0.99 | 0.98 | 0.99 |
| Shoulder abduction | 0.93 | 0.94 | 0.94 | 0.97 |
| Shoulder flexion D | 0.95 | 0.97 | 0.97 | 0.99 |

Their PV is in m/s and ours in mm/s; the correlation is unaffected. Our set now matches theirs
one for one — the earlier "we report eleven where they report twelve" was right about the count
and wrong about which measure was missing, and a still earlier note claiming they report eleven
came from the bad HTML fetch.

`max_shoulder_flexion_drink` = max planar shoulder flexion over the DRINKING window alone
(`_win(ph, "drinking")`), against `max_shoulder_flexion` over Reaching[0]..Drinking[1]. Added in
`angle_measures_automq`, so it propagates to `score_own_phases` and both tables without further
wiring. n = 748, not 750: two trials have no usable drinking window.

- r_s 0.972 / r_av 0.994 optical windows; 0.969 / 0.993 markerless. Their value is 0.95 / 0.97.
- anat12 gain +0.242 (raw 0.730). **θ was not refit.** The offset acts on the same flexion
  trajectory, so the new measure inherits it, and refitting would invalidate the held-out
  numbers already in §II-F and §V-C.
- It is the only measure bounded by BOTH drink boundaries and reduced over the shortest window,
  and it still costs only 0.003 between window sources — which is what makes §V-E's "the cost
  follows the window, not the boundary error" claim testable rather than merely consistent.

Their Table II (bias IQR) for comparison, same PDF: elbow angular velocity 1.83/1.57 °/s, elbow
extension 3.69/3.88°, shoulder flexion 1.34/1.14°, abduction 1.20/0.91°, trunk 2.86/3.10 mm,
end-effector velocity 0.01 m/s. Ours (Table II) is equal or lower on all six, elbow extension by
a factor of three. Not yet used in the text.

### §V-B corrections (2026-08-21)

| fix | detail |
|---|---|
| the bias sign convention is now stated | `paper_trajectories._agree` computes `bias = mean(MMC − OMC)`, so Table I's +7.4° on elbow extension means the markerless elbow reads MORE extended. §V-B now says "markerless minus optical" and spells out the direction for all three angles. Before this, the signed numbers were unusable as printed |
| §V-C used the OPPOSITE convention | its group-mean ranges were computed as optical − markerless while §V-B's were markerless − optical, in adjacent subsections. §V-C is now flipped to markerless − optical: flexion −19.4…+9.8°, flexion D −22.8…+9.8°, elbow extension +1.3…+19.2°, abduction −15.0…+3.1°. The between-group variance fractions (93/93/89/79 %) are sign-free and unchanged |
| §V-B now states its n | 790 trials, 413 affected / 377 unaffected, eleven units — and says why it exceeds Tables III–IV's 750: no phase window enters the trajectory comparison, so the segmenter's declines are included |

Housekeeping in the same pass: `fig4_correlation.py` gained the twelfth measure and its
panel-count comment no longer claims eleven (its "same set and order as fig4_pair" contract holds
again); `measures_table.py`'s docstring says twelve; `paper/README.md` says 12-panel.

**Still `\todo` in Results:** Study Population (numbers in it are pre-expansion: 842/851, 704,
705, 24%), Latency (nothing measured), Ablations, and the participant/trial names in the Fig. 3
caption.

## Still open after the code check

**Updated 2026-08-19 (paper reorg session).** §II-A is no longer the weak part: the
camera-audit claims were checked against `cam_quality.json` and the audit tooling was
found. Of the four that were open, three now verify (**11 units**; **3/4/4 units at
5/4/3 cameras**; **12 cameras dropped**, consistent with 5+7) and both audit scripts
turned out to be in `scripts/archive/`, not missing. What is left in §II-A is the
**5-mis-cut/7-miscalibrated split** — the totals agree but the split itself is not
stored in `cam_quality.json`, and it survives the 2026-08-20 rewrite. The **25–30 px
overlap** is no longer a risk: that sentence was cut from Methods on 2026-08-20, so §II-A no
longer rests on any doc-sourced claim. The **exclusion mechanism** was cut in the same pass —
the paragraph now states the two fault classes, their counts, and that both were judged from
the recordings alone, without describing the audit machinery. That removes two mis-descriptions
(NCC-in-uncut cannot by itself flag a mis-cut; the miscalibration pass is RANSAC-consensus, not
leave-one-out) and narrows §II-A to claims that verify directly against `cam_quality.json`.

§II-F is also resolved: the **landmark-offset fit scripts were recovered**, not lost.
They had been written to a `/tmp` session scratchpad instead of `paper/scripts/`, which
is why they were absent from both the repo and git history — the earlier "does not exist
anywhere" finding searched `~/Documents` but not `/tmp`. All 32 are now in
`paper/scripts/anat12/`, they compile, and none of them import anything that the
2026-08-19 reorg archived.

Recovered in the same sweep: `plot_grasp_blowup.py`, `plot_seq_outliers.py` and
`plot_seg_mmc_vs_omc.py`, now in `paper/scripts/`. These generate `paper/grasp_blowup.png`,
`paper/seq_outliers.png` and `paper/seg_mmc_vs_omc.png` — three figures that were already
committed with no generator behind them.

**Standing risk this exposes:** a scratchpad script that produces a committed paper
artifact is invisible to git and is deleted whenever `/tmp` is cleared. A sweep of every
surviving scratchpad found no further orphans, but anything generated that way in future
must be moved into `paper/scripts/` at the time it is written, not recovered afterwards.

### Two mismatches found while reorganising, not yet resolved

1. **The anat12 scope prediction is contradicted by its own log.** `main.tex`'s plan
   block argues a constant offset "cancels under differentiation … so the speed, timing,
   smoothness and trunk measures cannot move". `out/scoring/anat12_w0.log` reports
   **peak wrist vel r_s 0.798 → 0.409** and time-to-peak 0.457 → 0.400 under anat12/w0.
   Either the fit is not a constant offset, or it is being applied somewhere the
   argument does not cover. Settle this before §II-F states a scope.

2. **A manual step sits in the middle of the rerun chain.** `score_seg_boundaries.py`
   writes `out/scoring/seg_boundaries.csv`; `make_seg_table.py` reads
   `paper/results/seg_boundaries.csv`. The two are byte-identical today, so the copy was
   done by hand and is not in any script. Table IV silently goes stale on a re-run
   unless someone remembers it.

## Blocking order

1. ~~Recover the landmark-fit script.~~ **DONE 2026-08-19** — `paper/scripts/anat12/`.
   Next step on it is to re-run `anat_frame.py` and confirm it reproduces the committed
   `recon_offsets_r4.csv` / `anat12w0_omc_values.csv`, which then unblocks the scope
   question, cross-participant generalisation, the P25 discriminator, and `PIPELINE.md`.
2. **Copy `out/scoring/score_vs_automq.csv`.** Gates the four-variant ablation table.
3. **Log `n_fallback` / `n_guarded`** over the 842 admitted trials, then decide revert-vs-exclude.
4. **Fold the `seg_boundaries.csv` copy into a script** so Table IV cannot go stale.
5. **Promote the three audit scripts** out of `scripts/archive/` into `scripts/`.
6. **Then regenerate**, per the table changes in the plan block.

## Docs to fix before the repo is public

- `results/PIPELINE.md` — the anat12 entry under "did not work" (contradicted), and the
  argmin "do not re-try" entry (does not reproduce on the current cup).
- `measures_methods.md` — describes the superseded constant-trunk-down-axis flexion, and
  says "raw wrist" where it means "without the additional 4 Hz low-pass".

### Internal-consistency pass over the whole paper (2026-08-24)

Every number in `main.tex` re-derived from the CSVs, caches and logs. Build clean throughout:
7 pages, 0 overfull, 0 undefined refs or citations, 11 `\todo`s. Twelve inconsistencies found and
fixed. What did NOT move: the trial accounting (842 − 45 − 3 − 4 = 790; 413 + 377; 21 groups,
P19 affected-only), every Table I figure quoted in §V-B, the anat12 deltas in §V-C, the offset
magnitudes, the LOPO drop, the null-pairing correlations, all twelve window deltas, the 30 Hz
deltas, and the 53× ratio.

**[!] Three statistics were quoted from the UNCORRECTED file into anat12 prose.** `anat12` fits the
offsets onto the OPTICAL landmarks, so the optical interjoint distribution itself moves. Anything
describing that distribution must come from `score_own_phases_anat12.csv`, not
`score_own_phases.csv`. Verified on the exact 747-row set (Pearson reproduces to 0.48296):

| statistic | uncorrected (was printed) | anat12 (now printed) |
|---|---|---|
| top 18 trials (2.4%) carry | 94.5% | **91.7%** |
| top 5% of trials carry | 96.3% | **94.4%** |
| optical IQR span | 0.969–0.989 | **0.976–0.995** |

Both wrong figures sat next to r_s = 0.48, which is the anat12 number. §V-D and §VI corrected.

**[!] "The match acts on the angles and on nothing else" was false** — and it was false in the
direction the header comment predicted back in the plan block. Peak elbow angular velocity is
differentiated from an angle whose shape changes, and it gains **+0.025** (0.887 → 0.911);
`anat12_wv1wa1.log:46` shows the same gain independently (0.899 → 0.922). It was named in neither
of that sentence's two lists. §V-C now says "on the angles and on what is derived from them" and
names it. The wrist-velocity block in the loss is what keeps peak velocity flat at −0.001.

**[!] Only three of six boundaries agree within 33 ms, not four.** Table III medians are
17 / 100 / 67 / 50 / 33 / 17 ms; drink offset at 50 ms was inside neither the "four" nor the named
exceptions. Threshold in §V-E corrected 33 → 50 ms.

**Table pointers had gone stale when the 30 Hz and n columns were added.** Column order is
r_s/r_av (optical), r_s/r_av (markerless), 30 Hz, n. "The two right-hand columns" pointed at
30 Hz + n instead of the markerless pair, and "the last column" pointed at n instead of 30 Hz.
Both now named by what they are rather than by position — do not reintroduce positional references
to this table.

**Table II's `n` does not describe its 30 Hz column.** The 30 Hz run covers all 750 trials
(`fps_full_*of3.csv`, verified) but pairs fewer against the 60 Hz values: time-to-PV, time-to-first-PV
and movement units 743 (vs 748), total movement time 600 (vs 602), interjoint 739 (vs 747). Caption
in `make_tables_tex.py` now says so.

**The two five-camera latency figures were mixed** in adjacent paragraphs of §V-F, exactly as the
M4 note above warned not to. 24.7 ms is inference alone on frames in memory; 28 ms is the same pair
inside the running pipeline. §V-F now states the distinction inline, so the warning is enforced by
the paper rather than only by this file.

**16.8 − 15.4 = 1.4, not 1.3.** The missing 0.21 s is the cup-seed scan, front-end but never
mentioned, so the subtraction did not close. §V-F now gives 15.6 s of single-frame work and 1.3 s
after it (`latency_opt.csv`: 15.37 + 0.21, tail 1.26).

**Smaller:** the Conclusion claimed agreement "within 0.01" of a fitted-model pipeline where §VI
supports 0.02; §V-D called movement units "a count derived from zero crossings of the velocity",
which contradicts §IV-B and the code (`_count_movement_units` counts min→max speed oscillations
against a 60 mm/s amplitude and a 0.15 s gap); "98% of the optical values are ties" reproduces under
no definition tried (74.3% are exactly 1, 99.3% share a value, 57.8% of pairs are tied) and is now
stated as three quarters exactly 1; the median lag is exactly one frame on all six trajectories
(0.0167 s, one value per trial), not "0.02 s, one to two video frames"; the closing `\todo` still
said refs.bib held three entries and none was checked, but `altmurphy2018` was added and verified.

**Also corrected in this file, not the paper:** the wrist offset SD was recorded twice as 7.2 and
once as 7 mm; `anat12_wv1wa1.log:56` says **6.8**, which is what the paper always had.

### Peer review round 2 — the Unger comparison premise was inverted (2026-08-24)

**[!] `unger2024` does NOT segment from its own recording.** Verified verbatim in
`~/Downloads/2411.14992v1.pdf`, Sec. III-D2: *"Drinking task phase classification ... for both
systems was based on the endeffector velocity calculated from the OMC markers on the wrist and cup,
following the approach in [8]. This eliminated any influence of difference in phase classification."*
Their Table III is therefore the **optical-windows** condition, and §VI was comparing our
markerless-window column against it on the opposite premise. Corrected to the optical column:

| | theirs | ours (optical windows) |
|---|---|---|
| movement units | 0.77 | **0.84** |
| time to first PV | 0.94 | **1.00** |
| elbow angular PV | 0.87 | **0.91** |
| interjoint, r_s | 0.71 | **0.48** |
| interjoint, r_av | 0.71 | **0.83** |

Two things this buys, both now in the paper. Their Limitations says *"Phase classification was
conducted exclusively with OMC data, meaning that MMC phase classification still requires validation
for standalone application"* — a direct citation for contribution 3, now in §II and the contribution
bullet. And our one loss is a loss on **r_s only**: on r_av interjoint is 0.83 (optical) / 0.88
(markerless) against their 0.71, which is the statistic our own range-restriction argument prefers.

**[!] The 631-trial reference is `unger2024`'s IMPLEMENTATION, not `altmurphy2018` as published.**
`~/Documents/AutoMQ` is the DELTA study's own processing; `identify_phases` in
`MurphyMeasures_allP_optimized.ipynb` reads: hand velocity > **20 mm/s** held 30 frames (reach
onset), **glass** velocity > **50 mm/s** held 30 frames (grasp), hand-to-face distance crossing
**1.2 ×** its own steady state (drink boundaries), hand velocity < **2 % of peak** (settle). So it is
part absolute and part relative and still instruments the glass — which is why §IV-E's "2 % of peak"
and §II's "15 and 10 mm/s" looked like two descriptions of one rule. They are two different rules:
§II describes [3] as published, §V-E now describes [2]'s implementation explicitly and says so.
§III-C's contrast is now qualified "as published". **The novelty that survives cleanly is the visual
cup tracking, not the relative thresholds.**

**n = 631 of 750 explained.** 119 trials carry no reference row at all — AutoMQ's detector produced
no phase set for them (`cache_seg_inputs.py` writes the `(-1,-1)` sentinel). Not partial: 0 trials
have some phases and not others. All 631 that do have one carry `validity == 1`, so the comparison
never includes a trial the reference itself flagged; separately, AutoMQ marks 178/1694 (10.5 %) of
the rows it does produce invalid. Now stated in §V-E.

**Release/settle residuals no longer dropped.** 233 and 117 ms against 33–67 on the other four.
§V-E now says what follows: total movement time is the measure that boundary closes, so its
absolute value should not be read across the two rules. (Its r_s is unaffected — both inputs get
the same rule.)

**r_av at 30 Hz is now tabulated.** It was claimed in §V-F and absent from the table. Table II is
now seven numeric columns; `\tabcolsep` 1.5 pt and two shortened row labels (`_SHORT` in
`make_tables_tex.py`) keep it inside the column — it overran by 3.2 pt at the previous settings.

**Pooling caveat added to §Limitations.** §IV-C had the numbers and drew no conclusion: pooled
flexion 0.97 against 0.85 within a participant and arm. Pooling across 21 groups supplies
between-participant range a clinic scoring one patient does not have. Matching `unger2024`'s
convention makes it the right choice for comparability, not a reason to leave it unsaid.

**[!] PAGE BUDGET BROKEN: content is now 7 pages, target 6.** The additions above cost ~3,700
characters. Redundancy removed to claw it back (all of it genuine duplication): §V-F's
"both pipelines are dominated by 2D keypoint extraction" repeated §VI verbatim; §VI restated §V-E's
"eight of twelve ... none by more than 0.04" and the 4.9 % decline rate; §V-D re-argued interjoint's
range restriction that §VI argues with better numbers; the Discussion opening restated the abstract.
Not enough. **Measured: deleting Fig. 1 restores 6 content pages exactly, and nothing else does** —
shrinking it to 0.75\linewidth does not, nor does any float placement (`!t`, `b`, `h`, `!b`), because
the float forces the break rather than its height. Left in pending a decision; the reviewer
independently questioned whether it earns a column.

**NOT done this round, deliberately:** limits of agreement at the measure level (item 7). Still the
position recorded above — `unger2024` reports none, and a 12-row LoA table does not fit a budget
already over. It is one table from `score_own_phases_anat12.csv` with no new computation whenever
the page count allows. Front matter (title, authors, affiliations, repository URL, demographics,
funding, ~11 bib entries) needs information not on this machine. Fig. 1 still carries a baked-in
"Fig 3" title and small axis labels; regenerating it is a `paper_trajectories.py` change, not an edit.

### Bibliography built and every field checked against a publisher record (2026-08-24)

Seven entries, up from four. PDFs of what could be fetched are in `paper/refs/`; provenance for
each entry is in the VERIFICATION block at the end of `refs.bib`. **Two errors caught.**

**[!] `murphy2011` had the wrong author list.** It read five authors — Alt Murphy, Häger-Ross,
Michael A. Murphy, Persson, Sunnerhagen. Crossref (`10.1177/1545968310370748`) gives **three**:
Margit Alt Murphy, Carin Willén, Katharina S. Sunnerhagen. The bad list is the one printed as
**[1] in `unger2024`'s own bibliography**, so it was inherited rather than invented here — which is
exactly why copying a reference from a citing paper is not verification. Fixed, DOI added.

**[!] `unger2024` is no longer a preprint.** Published as **IEEE Trans. Medical Robotics and
Bionics 8(1):90–97, Feb 2026**, `10.1109/TMRB.2025.3605962`. It was cited here as an arXiv preprint
throughout. Entry now points at the journal with the arXiv ID in a `note`.

**WARNING that follows from it, and that is NOT yet resolved:** every number this paper quotes from
`unger2024` — their Table III (the twelve r_s / r_av pairs recorded above), the 12-GPU-day /
1160-trial figure, the ~30 CPU-hour IK figure, and both the Sec. III-D2 and Limitations quotations
— was read off **arXiv v1**, which is what sits in `refs/`. IEEE is unreachable from this machine.
**Re-check all of them against the published version before submission.** The header comment in
`main.tex` now carries the same warning, and said "ICORR 2025", which was wrong twice over (ICORR
2025 is Cotton's differentiable-biomechanics paper, their [17], not this one).

**Added, each verified against its own downloaded PDF title page:** `yolo26`
(arXiv 2606.03748, Jocher et al., Ultralytics), `uetrack` (CVPR 2026, Kang et al., Dalian Univ. of
Technology — the CVF open-access PDF), `smoothnet` (Zeng et al., "Accepted by ECCV 2022" per the
arXiv metadata). Cited at pipeline stages 1/2, 2 and 4 respectively.

**One TODO dissolved rather than done:** the "Alt Murphy planar angle definitions (PMC5933268)" the
old note asked to add is `altmurphy2018` itself — PMC5933268 **is** that JoVE article, and its
Table 2 carries the planar definitions. §IV-B now cites it instead of leaning on `murphy2011`.

**Network reality on this machine, so nobody repeats the attempts:** only `arxiv.org` and
`openaccess.thecvf.com` serve PDFs. Publisher hosts (Springer/BMC, JoVE, IEEE, PMC's own PDF path,
`homes.cs.washington.edu`) all return a sandbox interstitial or an anti-bot page. The read-only
APIs DO work and are what verified the rest: `api.crossref.org`, `api.unpaywall.org`,
`export.arxiv.org`, and `WebFetch` against publisher landing pages.

**All three previously-uncited entries are now IN, PDFs and all** (2026-08-24, second pass). The
earlier "cannot be fetched from this machine" was premature — the dead ends were the original URLs,
not the papers. What worked: **Europe PMC's render endpoint**
(`europepmc.org/articles/<PMCID>?pdf=render`) for `lam2023`, and **web.archive.org** for
`todorov2012` (whose host `homes.cs.washington.edu` now 301s to a dead page) and `rajagopal2016`
(paywalled at IEEE; the archived copy is the PMC author manuscript, whose header carries the
published citation). Cited at: `lam2023` on §II's opening claim, which was the last substantive
uncited sentence; `rajagopal2016` + `todorov2012` naming the model and engine `unger2024` fits,
which sharpens the model-free contrast the paper turns on.

**Ten entries, all audited by DOI resolution.** Every DOI was resolved through Crossref and its
returned title, journal, volume, issue and pages compared against the bib field by field — that
catches a transposed or mistyped DOI, which a by-eye check does not. All six DOI-bearing entries
resolve to the right paper with matching metadata. The four without DOIs (`hartley2004` book,
`yolo26`, `uetrack`, `smoothnet`) are verified against their own PDFs and, for the arXiv pair, the
arXiv API. One thing the audit settled: `murphy2011`'s Crossref `issued` date is 2010-09-09
(online-first) but `published-print` is 2011-01 — the year 2011 is right.

### Crossref validated the METADATA, not the citations — three content errors found (2026-08-24)

A DOI resolving to the right title says the entry is typed correctly. It says nothing about whether
the cited work supports the sentence it is attached to. Checking the content claims found three
errors in entries that had just passed the DOI audit clean.

**[!] `lam2023` did not support the claim it was cited for.** It was attached to "Multi-view pose
estimation followed by triangulation now recovers upper-limb motion accurately enough for clinical
measures". Grepping the downloaded PDF: **65 hits for "Kinect", 0 for "multi-view", 0 for
"multi-camera", 0 for "triangulat"**. It is a Kinect-dominated review of MMC as a clinical
measurement tool, and its own conclusion is that application is "still in its preliminary stages"
and the benefits "inconclusive". §II now cites it for that, which it does support, and the
multi-view sentence stands on its own. **Check what a review actually reviews before citing it for
a method it does not cover.**

**[!] `rajagopal2016` was attached to the wrong side of `unger2024`'s pipeline.** The text said their
markerless pipeline "fits a musculoskeletal model derived from [Rajagopal] through a differentiable
physics engine [MuJoCo]". Their own words: the **OpenSim/OMC** model is Rajagopal-based, while the
**MuJoCo/MMC** side uses "a modified version of the biomechanical model presented by Hamner et al.
[27], [28] compatible with Mujoco". §II now puts each on its correct side.

**[!] The `altmurphy2018` thresholds in §II were wrong, and they were load-bearing.** The text read
"absolute velocity thresholds on the hand and glass ($15$ and $10$ mm/s)". Verified against the
protocol (PMC5933268 landing page, quoting the paper): $15$ and $10$ mm/s are **both glass**
thresholds — glass velocity above 15 for forward-transport start, below 10 for back-transport end.
The **hand** threshold is "2% of the peak velocity" with a "20 mm/s" minimum, and the drinking phase
uses face-to-glass distance "below 15% of steady state". So the published protocol is itself a mix
of relative and absolute rules.

That last one **weakened a novelty claim**. §III-C said "every threshold is relative to its own
channel's range rather than an absolute velocity" as a distinction from the reference protocol —
but the reference protocol's hand threshold is 2%-of-peak, i.e. relative, and its drinking boundary
is a ratio to steady state. The distinction that survives is narrower and is what §III-C now says:
relative to each channel's range **within the trial, with no absolute velocity and no floor
anywhere** — theirs has a 20 mm/s floor and two absolute glass crossings. The visual cup tracking is
still a clean distinction. §V-E now also says the AutoMQ implementation is the reference protocol's
own mix at different constants, rather than implying the mix is AutoMQ's departure.

Confirmed correct in the same check, so it is not re-litigated: 9 markers with 2 on the cup, the
face-to-glass distance, and the manual inspection of mis-segmented recordings.

**Still unverifiable from this machine:** the `unger2024` numbers against the PUBLISHED version
(IEEE, paywalled; only the arXiv PDF is in `refs/`). That warning stands.

### Second content sweep: three MORE mis-attributions (2026-08-24)

The first sweep checked the three citations I had just touched. This one checked every remaining
claim made *about* a cited work. Three more were wrong.

**[!] `murphy2011` was credited with longitudinal responsiveness it never claimed.** §II said the
measures "separate stroke survivors from controls **and track change over rehabilitation**". The
abstract (Europe PMC, PMID 20829411) is a **cross-sectional** study: 19 chronic stroke vs 19 healthy
controls, reporting discrimination between groups and between mild (FM 58--64) and moderate
(FM 39--57) impairment. There is no longitudinal arm. §II now claims discrimination and severity
grading only — which is what the paper supports, and is enough for the sentence's purpose.

**[!] The twelve measures were attributed to the wrong paper.** §IV-B said "the twelve measures are
those of the drinking-task protocol~\cite{murphy2011}". The measure *definitions* are the 2018
protocol's — its Table 2 — and `unger2024` likewise attributes its measure set to "[8]", which is
`altmurphy2018`. Repointed. `murphy2011` remains cited where it belongs: identifying the variables
and showing they discriminate.

**[!] MuJoCo 2012 is not a GPU engine.** §II called `todorov2012` a "GPU physics engine". That
paper states GPU support as *future work* — "we plan to port the key pieces of code to OpenCL and
run it on GPUs" — and `unger2024` uses the much later GPU-accelerated Mjx (their [24], [25]).
"GPU" dropped.

**Verified correct in this sweep, so they are settled:** `yolo26` does cover pose and instance
segmentation and names the `s`/`x` variants used here (75 / 27 / 25 / 16 hits); `uetrack` names
UETrack-B as a variant and is transformer-based (22 / 23 hits); `smoothnet` states "The input window
size T is 32", which is exactly the +/-16 frames §III-B claims, and is applied to 3D poses (38 hits);
`murphy2011`'s measure list in §I (movement time, smoothness, velocity and its timing, interjoint
coordination, compensatory movement) matches its abstract.

**Running tally across both sweeps: six content errors in ten references**, every one of them in an
entry whose Crossref metadata was perfect. Metadata validation and citation validation are different
jobs; only the second requires reading the paper.

**Two claims remain unverifiable here and are the only known-open items:** the `unger2024` numbers
against the published IEEE version (arXiv only in `refs/`), and `hartley2004`'s Huber-kernel
citation (a book, not obtainable; the attribution to its bundle-adjustment treatment is standard but
unchecked).

### The last two open citation items, closed (2026-08-24)

**`unger2024` arXiv-vs-published: ACCEPTED AS-IS, by decision.** The numbers quoted from them were
read off arXiv v1 and the published IEEE version could not be fetched. Raised, and the call was that
this is fine. Not an open action — but the fact is recorded here and in `refs.bib` so that anyone
re-reading knows which version the figures came from, rather than rediscovering it.

**`hartley2004` for the Huber-robustified BA: checked, and the citation is sound.** The book itself
is not obtainable, but its official site (`robots.ox.ac.uk/~vgg/hzbook/`) serves the front matter,
and that was enough:

- The **bibliography** (`refs/hartley2004_contents.pdf` is the TOC; the bibliography PDF was checked
  in the scratchpad) contains **`[Huber-81] P. J. Huber. Robust Statistics. John Wiley and Sons,
  1981`**. A bibliography entry exists only because the text cites it, so the Huber cost function is
  demonstrably discussed in the book.
- The **table of contents** confirms §4.7 "Robust estimation", §18.1 "Projective reconstruction --
  bundle adjustment", and Appendix 6 "Iterative Estimation Methods".
- The three free sample chapters (introduction, epipolar, trifocal) contain no "Huber" hits, which
  is expected — the robust-estimation material is in Ch. 4 and App. 6, neither of which is free. So
  the exact section number could NOT be pinned down, only the fact of coverage.

**One wording change followed.** The citation sat immediately after "Huber kernel", which reads as
attributing the kernel itself to the book; the Huber function is Huber 1981, and `hartley2004` is
the standard reference for robust reprojection bundle adjustment. The cite now attaches to "bundle
adjustment", which is what the book unambiguously covers (§18.1), and the Huber kernel is left as a
described implementation detail needing no citation. Verified rather than inferred, and the claim
is now the one the source actually supports.

**Citation state: no known-open items.** Ten references, six content errors found and fixed across
three sweeps, every claim about a cited work now either verified or (for `unger2024`'s figures)
explicitly accepted with the version recorded.

### The 1.3 s tail was stated unconditionally, and the paper contradicted itself (2026-08-24)

**[!] "16.8 s per trial, 1.3 s of it after the recording ends" was an overclaim in the measured
configuration.** The 16.8 s is measured offline on archived clips. The 1.3 s is a counterfactual: it
is what remains *if* the per-frame stages overlap capture. They cannot, at five cameras and 60 Hz on
this card — the front end takes **15.6 s for 9.2 s of video, i.e. 1.70x real time**
(`latency_opt.csv`: 15.31/15.58/16.19 s against 8.88/9.18/9.15 s, ratios 1.72/1.70/1.77). In that
configuration the whole 16.8 s follows the last frame.

The paper asserted the 1.3 s in five places as plain fact and then, three paragraphs into §V-F,
conceded "a front end that keeps pace with capture, which this card does not quite manage at five
cameras and 60 Hz". Both statements cannot hold. All five now carry the condition, and §V-F states
the 1.70x factor explicitly rather than leaving the reader to derive it.

**The trial duration was also missing** wherever the 16.8 s appeared, which made it uninterpretable
— 16.8 s is only meaningful against the 9.2 s of recording it processes. The abstract and the
contribution bullet now give both.

**Sharpened while fixing:** the abstract had said the card keeps pace "at three cameras or at
30 Hz". Three cameras is **borderline**, not clear: 14.9 ms is inference alone, and the in-pipeline
overhead measured at five cameras (28 vs 24.7 ms, x1.13) puts three cameras at ~16.9 ms against the
16.7 ms budget of 60 Hz. The abstract now claims only 30 Hz, which is comfortably inside at every
camera count (28 ms against 33.3). §V-F keeps the three-camera figure with its original hedge, "at
the 16.7 ms budget", immediately next to the numbers it comes from.

Unchanged and still true: 16.8 s total, 1.3 s of non-overlappable work, 53x against `unger2024`.

### Study-population TODO made specific — and the clinical data is probably ON this machine (2026-08-24)

The old note said "demographics and impairment scores. Not on this machine." **That is likely
wrong.** `~/Documents/AutoMQ/clinical_scores_import.ipynb` carries **cached outputs** — the
notebook's own printed DataFrames — holding per-record age, sex, stroke date, dominant hand,
more-affected side, days between stroke and measurement, Box-and-Block counts per hand, and an
FMA-UE import. No re-running or CSV needed; the values are in the committed notebook JSON.

**The one thing that must be checked before any of it is used:** those rows are keyed by DELTA
**Record ID** (1, 2, 3, ...), and our participants are P07, P08, P10, P12, P13, P14, P15, P17, P19,
P25. `Record ID == 7 -> P07` is the obvious guess and is NOT verified. Confirm the mapping against
something independent before a single number goes in the table — a demographics table with a
silently wrong key is worse than no table.

**Already known, so the table's trial column needs no work** (from `trajectory_agreement.csv`,
affected / unaffected, 790 trials over 21 participant-arm groups):

| unit | aff | unaff | | unit | aff | unaff |
|---|---|---|---|---|---|---|
| P07 | 41 | 42 | | P15 | 43 | 43 |
| P08 | 39 | 47 | | P17 | 40 | 41 |
| P10 | 40 | 38 | | P19 | 47 | — |
| P12 | 41 | 39 | | P251 | 19 | 21 |
| P13 | 40 | 40 | | P252 | 21 | 19 |
| P14 | 42 | 47 | | **total** | **413** | **377** |

P19 contributes an affected arm only, which is why there are 21 groups and not 22. P251/P252 are the
two sessions of one person.

**Inherited from the source study, verified verbatim in its Sec. II-A** — usable for the criteria and
ethics text, but note it describes THEIR full cohort, not our ten-participant subset, so any cohort
statistic must be recomputed on ours:

- Ethics: "the ethical guidelines set forth by the local ethics committee (BASEC-No: 2022-00491)".
  Recruitment from the University Hospital Zurich Stroke Registry and the cereneo clinic, single-session.
- Inclusion: at least 18, capable of informed consent, confirmed stroke diagnosis, at least partial
  reaching ability, and able to grasp a cup unassisted with a cylindrical grip using the affected hand.
- Exclusion: pre-existing upper-limb deficits (e.g. orthopedic) and other neurological conditions.
- Their cohort: mean time since stroke 24 months; **FMA-UE mean 56.3 (SD 8.4), range 39--66**. DO NOT
  copy these as ours — 1160 trials against our 790, so the subsets differ.

**Open question only the authors can answer**, now stated in the TODO: whether BASEC-No 2022-00491
covers this re-analysis or whether a separate ethics/consent statement is required. §III-A currently
defers all three to `unger2024`, which a clinical reviewer will not accept for a paper of its own.

**Consequence for §Limitations:** it asserts "its impairment gradient is mild" with no number behind
it. With per-participant FMA-UE recovered, that becomes a range and should be stated as one.

### Offset removal was described as load-bearing for r, and the code never did it (2026-08-24)

Table I's caption said the bias "is removed per trial before $r$ and RMSE are computed", and §III-F
said "agreement is then Pearson $r$ and RMSE on the offset-removed series". Both read as though the
removal matters for $r$. It does not, and the code does not even do it: `paper_trajectories.py:112`
subtracts the bias inside the RMSE, while **line 113 computes `np.corrcoef(a, b)` on the RAW
series**. Pearson $r$ is invariant to an additive constant, so the two are identical anyway
(verified: identical to 12 decimals, differing only in the last float bit).

Both now say the removal bears on RMSE alone and that $r$ is computed on the raw series. Worth the
clause because `unger2024` performs the same offset removal, so a reader primed by that paper is
likely to assume the correlations depend on it and to object that they are therefore inflated — the
objection is real against the sentence as written, and empty against what the code does.

Also worth keeping straight: this is a different removal from the anat12 landmark fit. The per-trial
static offset here is Table I's `bias` column, Unger-comparable and NOT landmark-matched; the anat12
offsets act on the measures in Tables II and the figures. Neither affects $r$ for the reason above,
which is exactly why §V-C can argue that the anat12 gain proves the fit removed *trial-varying*
error rather than a constant.

### Sweep for the padding pattern (2026-08-24)

Two cuts requested in review — "those boundaries are not measurements and are reported as a failure
rate rather than scored" and "the failure is detectable from the cup track alone... which is what
makes reporting it possible in deployment" — turned out to be instances of one recurring pattern:
a clause that either restates its own neighbour, or explains something the reader would not have
doubted. Swept the whole paper for it; six more found and cut.

Pure restatement, no judgement involved:
- §III-B, cup stage: "no movement-quality measure is computed from it" — the sentence already says
  the cup trajectory is used **only** to segment.
- §IV-E: a standalone paragraph, "Both correlations pool across participants and arms,
  following [2]", restating the sentence immediately above it. §Limitations said it a third time.
- §V-C: "as an additive constant must" — and this one was **wrong**. The anat12 fit is a 3D
  positional offset, not additive in angle space, which is exactly why the figure it describes moves
  0.86 -> 0.85. It was explaining why a number did not change, inside a sentence reporting that it did.

Judgement calls, cut after review agreed:
- §I: "A measurement that arrives days after the session cannot inform the session that produced
  it" — tautological; the following clause carries the point alone.
- §I: "The trade is deliberate:" — defensive framing against "you cut a corner".
- §V-C: "so the end-effector needs essentially no correction" — restates the 2.9 mm it follows, and
  §VI says it again.

**KEPT, deliberately, and the distinction is the useful part:** two clauses of the same shape earn
their place because they pre-empt objections a reviewer would actually raise, not ones nobody would.
§IV-D's "the archived clips being an artefact of this dataset rather than of the method" defends
excluding the decode cost, which otherwise looks self-serving; §V-E's "the offsets are shared by both
inputs and cancel" stops the +200 / -150 ms reference offsets reading as though they invalidate
Table II. The test is not "is this clause explanatory" but "would a reader object without it".

### What altmurphy2018 actually justifies about its segmentation: nothing (2026-08-24)

Asked because if the reference protocol had a rationale we had not accounted for, changing the rules
could have broken something essential. Read against the protocol (PMC5933268 landing page):

- **Its phase table gives the events but NO justification for any threshold.** No defence of 2% of
  peak, 15 mm/s, 10 mm/s, or 15% of steady state against alternatives.
- **No validation of the segmentation itself** — no reliability or repeatability figures, no
  inter-rater agreement. The only acknowledgement of failure is qualitative: phases were sometimes
  "not detected correctly... often due to the extra movements in the beginning/end of the movement
  or when the movement speed was extremely low", handled by manual inspection.

So there is no missed rationale, and the paper's §V-E comparison is against an **unvalidated**
reference, which is worth knowing but is NOT worth claiming superiority from — a reviewer who knows
the protocol knows this already.

**[!] RETRACTED: the causal story I first gave for the six offsets was mostly wrong.** It was
reasoned from `altmurphy2018`'s PUBLISHED rules, but the 631 reference boundaries are AutoMQ's
IMPLEMENTATION, whose constants and channels differ. Read the full `identify_phases` and then
measured the OMC channels at both rules' boundaries (250 trials, `seg_inputs_ship`). Median wrist
speed as % of that trial's peak, cup speed in mm/s:

| boundary | ours - theirs | wrist spd, ours / theirs | cup spd, ours / theirs | what actually causes it |
|---|---|---|---|---|
| reach onset | +200 ms | **47% / 5.5%** | 1 / 1 | their 20 mm/s sustained crossing fires on almost no motion; our displacement run needs a range fraction, so ours lands halfway up the velocity ramp |
| grasp | −150 ms | 8.9% / 5.1% | **29 / 52** | ~~hand-arrival vs glass-motion~~ WRONG. Both fire during the same lift-off; theirs at its 50 mm/s glass threshold, ours once the wrist-cup distance hits its floor, which is earlier. wrist-cup is 216 vs 212 mm — the hand is on the cup in both |
| drink onset | +167 ms | 10.6% / 19.9% | 124 / 233 | ~~face-to-glass vs hand-to-mouth~~ WRONG — **AutoMQ also uses hand-to-face**. Same channel, different threshold form (1.2x steady-state minimum vs range fraction) |
| drink offset | −100 ms | 26.2% / 44.0% | 259 / 476 | same channel, other end |
| release | +483 ms | 10.3% / 6.7% | **16 / 51** | ~~glass back on the table at 10 mm/s~~ WRONG. AutoMQ's threshold is glass speed **< 50** mm/s searched from drinking end, which fires on the first dip; ours needs a sustained opening, by when the cup is at 16 mm/s. wrist-cup 214 vs 211, so the hand has left in neither |
| settle | −433 ms | **52% / 1.8%** | 1 / 1 | the criteria differ IN KIND: theirs is kinematic (2% of peak), ours positional (first entry into the final-rest neighbourhood) |

**The settle finding is the substantive one, and it is not a censoring artefact:** restricted to the
498 trials where the hand is actually observed coming to rest, our settle still fires at **54.6% of
peak wrist speed** (IQR 44--63%). Our total movement time therefore ends when the hand first reaches
the return neighbourhood, not when it stops. That is defensible and it is why the rule was chosen —
`rule="end"` halved the MMC-vs-OMC settle disagreement (33 -> 17 ms median) — but it is **not the
protocol's quantity**, and it explains both the −433 ms offset and the 117 ms residual.

§V-E now claims only what is measured: the two rules mark the same events by different criteria,
and the settle differs in kind. **Do not reinstate a per-boundary causal story without measuring the
channels at both boundaries** — four of six were wrong on the first attempt, every one of them
because the published protocol was substituted for the implementation actually being compared.

**[!] But it exposed a claim that was wrong.** §V-E said total movement time "is the one measure
whose absolute value should not be read across the two rules". Measured on the same 630 trials, our
windows are shorter across the board and TMT is not the worst:

| window | ours | theirs | diff |
|---|---|---|---|
| reaching (PV, time-to-PV, interjoint) | 0.80 s | 1.17 s | **−31%** |
| all but drinking (movement units) | 4.58 s | 5.05 s | −9% |
| reach → drink offset (shoulder angles) | 3.67 s | 3.92 s | −6% |
| total movement time | 5.98 s | 6.67 s | −10% |

Reaching is **three times** more affected than TMT. §V-E now states the general consequence — absolute
values of the window-dependent measures are not on the protocol's scale — instead of singling out TMT,
and keeps the point that the correlations are untouched because both inputs get the same rule.

**Worth noting for the interjoint discussion:** it is a correlation computed *within* the reaching
window, so a 31% shorter window means 31% fewer samples. That is the same sensitivity the 30 Hz
ablation isolated (0.57 -> 0.30 on halving the rate). Two independent routes to the same conclusion:
interjoint's weakness is a property of its statistic and its sample count, not of the reconstruction.

**Checked while here:** movement units are counted over reaching + forward transport + back transport
+ returning (`score.py:226`), i.e. everything except drinking — NOT the reaching window alone, as an
earlier note in this file implied. And `time_to_peak_velocity` is measured from **frame 0**, not from
reach onset (`score.py:198`), so the +200 ms onset offset does not shift it; only which peak falls
inside the reaching window can.

### Swept three rule families for reach onset and settle (2026-08-24)

Cached channels only, 88 trials over 11 participants, shipped HOLD = 0.15 s. Columns: MMC-vs-OMC
median |diff| (Table IV's quantity, reference-free), signed offset against `unger2024`'s boundaries,
and wrist speed at the boundary as % of trial peak — the defect being chased.

**Reach onset**

| rule | MMCvOMC | vs amq | spd@bd |
|---|---|---|---|
| SHIPPED (pos-abs 30 mm) | 17 ms | +183 ms | 47% |
| pos-abs 10 mm | **17 ms** | +100 ms | 23.5% |
| pos-frac 0.02 | 33 ms | +67 ms | 17.2% |
| speed 0.02 (AutoMQ's value) | **67 ms** | −17 ms | 2.8% |
| **speed 0.10** | **17 ms** | +33 ms | **11.7%** |
| proj 0.02 / 0.05 / 0.10 | 50 / 33 / 33 ms | | |

**Settle**

| rule | MMCvOMC | vs amq | spd@bd |
|---|---|---|---|
| SHIPPED (pos-abs 40 mm) | 17 ms | −350 ms | 52% |
| pos-abs 15 mm | **17 ms** | −217 ms | 26.3% |
| pos-frac 0.02 | 33 ms | −142 ms | 10.3% |
| speed 0.10 | 33 ms | −117 ms | 8.3% |
| proj 0.05 / 0.10 / 0.20 | 67 / 33 / 33 ms | | |

**Projected speed (wrist velocity along the wrist->cup direction) is the WORST family** on both
boundaries — 33–50 ms onset, 33–67 ms settle, plus undefined cases. Structural reason: projecting
onto the cup direction imports the cup track's noise into a channel that was wrist-only. Do not
retry it.

**The positional rule is not broken, it is loosely tuned.** MMC-vs-OMC is flat at 17 ms across
10–50 mm, so tightening costs nothing and buys most of the defect back: onset 30 -> 10 mm takes the
boundary from 47% to 23.5% of peak and halves the reference offset; settle 40 -> 15 mm from 52% to
26.3%. **This is the cheap fix and it needs no new rule.**

**[!] RETRACTED from the earlier smoke test: "f=5% keeps MMC-vs-OMC at 17 ms".** That run used
AutoMQ's 0.30 s hold; this one used the shipped 0.15 s, and on the speed family the result flips
(onset f=5%: 17 ms at 0.30 s, 33 ms at 0.15 s). **HOLD is a second axis, not a constant**, and the two
runs were compared across an uncontrolled factor. Any decision needs the (threshold x hold) grid.

Best single cell at the shipped hold is onset `speed 0.10` — 17 ms AND 11.7%, beating every
positional variant on both. Settle has no cell that is simultaneously 17 ms and low-speed; the
longer hold is where that looked achievable.

**[!] And the sweep confirmed a false claim in §III-C**, which this session had made *worse*. The text
said "every threshold is relative to that channel's own range within the trial, with no absolute
velocity and no floor anywhere, so the rule is invariant to the scale of the reconstruction".
`seg_sequential.py:35,37` sets `LEAVE_REST = 30.0` **mm** and `ARRIVE_REST = 40.0` **mm** — absolute
distances, so neither the range-relative claim nor scale invariance holds for reach onset or settle.
§III-C now says the grasp, release and drink boundaries are range-relative and names onset/settle as
the exception. Scale invariance is no longer claimed.

Worth noting what that leaves: the two rules are nearly **complementary**, not one relative and one
absolute. Theirs is relative at onset/settle (2% of peak) and absolute at grasp/release (50 mm/s);
ours is the reverse. **The clean distinction is the visual cup, not the threshold form** — which is
where the argument was already placed. Making onset/settle range- or speed-relative would let the
uniform claim be reclaimed honestly, and is an argument in favour of the re-score rather than against.

### Reach onset and settle: full sweep, and what it settles (2026-08-24)

Four rule families for the two weak boundaries, on cached channels. Metrics: |diff| and residual IQR
against `unger2024`'s boundaries (absolute comparability, and what survives a constant offset),
MMC-vs-OMC (Table III's quantity, reference-free), speed at the boundary as % of trial peak.

**REACH ONSET -- `speed 0.10` is adopted-worthy. Full cohort, Table III's own 750 trials:**

| | median | p90 | >0.25 s | spd@bd | \|diff\| vs amq | resid IQR |
|---|---|---|---|---|---|---|
| SHIPPED (pos-abs 30 mm) | 17 ms | 67 ms | 0.8% | **47%** | **200 ms** | 50 ms |
| **speed 0.10** | 17 ms | 67 ms | 1.3% | **11.7%** | **33 ms** | 33 ms |

Table III's printed row would not move. What improves is that the boundary marks movement onset
instead of firing mid-reach, and agreement with the protocol goes 6x. Costs: 1 trial of 750 does not
fire, and the >0.25 s tail goes 0.8 -> 1.3%. Pushing to `speed 0.03` gets |diff| to 17 ms and IQR to
17 ms but costs MMC-vs-OMC (50 ms), so 0.10 is the knee.

**SETTLE -- keep the shipped rule. The trade is real and the current choice is the right side of it.**
498 observed trials (censored excluded -- and excluding them makes the shipped rule look *worse*,
|diff| 433 -> 450 ms, so it is not being flattered by them):

| | \|diff\| | resid IQR | MMC-vs-OMC | no-fire |
|---|---|---|---|---|
| SHIPPED | 450 ms | 250 ms | **17 ms** | -- |
| rel-pos 0.03 | 283 ms | 200 ms | 33 ms | 0% |
| proj-fix 0.02 | 67 ms | 100 ms | 100 ms | 4% |
| speed 0.03 | **50 ms** | **83 ms** | 83 ms | 6% |

Nine times better agreement with the protocol costs five times worse agreement between systems. **A
constant offset from the reference cancels in every correlation; disagreement between the two
systems' boundaries does not** -- it feeds straight into Table II's measures, total movement time
worst. So the shipped rule optimises what the paper reports. This supersedes the earlier framing of
the settle as a defect: it is a deliberate trade, now with numbers under it. The disclosure that
absolute values are not on the protocol's scale is the correct one and is already in §V-E.

**Three of my own claims retracted in the course of this:**
1. ~~"HOLD is a second axis"~~ -- swept explicitly, 0.15 s and 0.30 s are near-identical on every row.
   The two smoke tests differed by trial subset (44 vs 88) and NaN handling in the sustained test.
2. ~~"projected speed is the worst family"~~ -- true for the PER-FRAME direction I first coded, which
   imports cup-track noise every frame. With the direction fixed once at rest (the intended design)
   onset improves 50 -> 33 ms, and `proj-fix 0.02` hits the settle dead-on (signed +0 ms). Still not
   competitive on MMC-vs-OMC, but for a different reason than I first gave.
3. ~~"`speed 0.1` halves MMC-vs-OMC (33 -> 17 ms)"~~ -- subset artefact. On the 631 trials with an amq
   row the shipped rule reads 33 ms; on all 750 it reads 17 ms, which is what Table III prints
   (p90 67 ms both, so it is a median on a frame boundary). `speed 0.10` MATCHES, it does not beat.

**`rel-pos` is the scale-free form of the shipped rule and performs identically** (onset `rel-pos
0.02` = `pos-abs 10 mm` at 17 ms / +100 ms), since 2% of a ~500 mm reach is 10 mm. Adopting it would
let §III-C's uniform range-relative claim be made honestly instead of carrying the exception now
written in. Free, but it does not fix the mid-motion firing -- `speed 0.10` does.

**Nothing is changed in the pipeline yet.** All of the above is measurement on cached channels;
adopting either rule is a full re-score of every table, both figures and the 30 Hz run.

### WHY no velocity threshold works at the settle (2026-08-24) — diagnosed, not asserted

Measured three candidate mechanisms on 585--604 observed trials rather than reasoning about them.
MMC wrist noise floor confirmed at 13.4 mm/s.

| | onset (f=0.10) | settle (f=0.05) |
|---|---|---|
| threshold value | 66 mm/s | 33 mm/s |
| H2 threshold / noise floor | 5.3x | 2.6x |
| H1 \|dv/dt\| at the crossing | 1453 mm/s^2 | **601 mm/s^2** |
| H3 threshold crossings in the search window | 2 | 3 |
| observed mean \|MMC−OMC\| | 36 ms | 95 ms |

**Two mechanisms compound, and together they close the arithmetic.**

**H1, conditioning, sets the floor.** A crossing converts a speed disagreement into a frame
disagreement by dividing by the slope. The slope ratio is 1453/601 = **2.42x**, so onset's 36 ms mean
should become ~87 ms at the settle; measured on trials with a single clean crossing it is **78 ms**,
within 12%. The naive `dv/slope` estimate (15 / 21 ms) badly undershoots both in absolute terms, so
use the RATIO, not the absolute prediction.

**H3, non-monotonic tails, add the spread.** **71% of trials cross the threshold more than once**
after the return peak, and disagreement scales with the count: mean 78 ms at 1 crossing, 161 ms at
5+, with the >4-frame rate going 37.9% -> 58.7%. Distribution is heavy-tailed overall — median 67 ms,
p90 200 ms, p99 469 ms, only 23% inside one frame.

**H2 contributes but is not decisive** — the settle threshold sits 2.6x above the markerless noise
floor against 5.3x at onset, so there is less headroom, but that alone would not produce the gap.

**The physical cause of both is the same and it is not a rule defect: after setting the cup down the
hand does not stop.** It drifts, adjusts and comes to rest in stages, so "end of movement" has no
well-defined instant in the velocity signal — 71% of the time there genuinely is not one crossing to
find. This is a property of the task, not of the estimator.

**Which retroactively justifies the shipped rule.** A positional criterion never reads the velocity
profile's shape; it asks only whether the hand is near where it ends up, which is monotone and
immune to the dithering. So position at the settle and velocity at the onset is not an inconsistency
to be tidied up — it is each boundary using the signal that is well-conditioned there. **Do not
"unify" the two rules on aesthetic grounds.**

### The settle's real constraint: "near rest" and "well-conditioned" are the same trade-off (2026-08-24)

Refines the entry above, which attributed the settle's noise to a rise/fall kinematic asymmetry. That
was measured at DIFFERENT thresholds (66 mm/s onset vs 33 mm/s settle), so it confounded edge with
speed level. Controlled by measuring |dv/dt| at the SAME absolute speed on both edges, 603 trials:

| speed level | rising (reach) | falling (return) | ratio |
|---|---|---|---|
| 400 mm/s | 1588 mm/s^2 | 1630 mm/s^2 | **0.97x** |
| 200 mm/s | 1905 | 1712 | 1.11x |
| 100 mm/s | 1617 | 1376 | 1.17x |
| 50 mm/s | 1264 | 872 | 1.45x |
| 25 mm/s | 1015 | 485 | **2.09x** |

**At high speed the two edges are identical.** The asymmetry is real but confined below ~100 mm/s.
The dominant effect is that **the slope collapses near zero speed on BOTH edges** — 1600--1900 mm/s^2
at 200--400 mm/s falling to 485--1015 at 25 mm/s — so any threshold near zero is ill-conditioned
whichever edge it is on.

What separates the two boundaries is the QUESTION, not the kinematics:
- onset asks "has motion started", and may answer once motion is underway — threshold at 66 mm/s,
  slope ~1450, well conditioned.
- settle asks "has motion stopped", which FORCES the threshold into the collapsed-slope region —
  33 mm/s, slope ~600.

Cross-check, and it holds: raising the settle threshold to f=0.10 (~66 mm/s, falling slope ~1100)
improved cross-system agreement to mean 73 ms from 129 ms at f=0.03, at the price of firing at 8.6%
of peak — far from rest. That is the same compromise the shipped positional rule makes, reached by
another route. Both the positional and velocity families obey it because the frame error is
(inter-system signal disagreement) / (derivative at threshold) and the denominator is what vanishes.

**So the 52%-of-peak firing speed of the shipped settle is not a defect; it is the price of a
reproducible boundary,** and any rule that fires closer to true rest pays for it in frames at a rate
set by the deceleration curve. Positional precision is NOT the limit: the two systems agree on
`d_end` to 2.2 mm (p90 5.5) and the hand's residual jitter at rest is under 1 mm.

For the record, the tightening curve (positional shell, 604 observed trials): 40 mm -> mean 41 ms,
>4fr 11.8%, spd 55%; 15 mm -> 68 ms, 27.7%, 26%; 7 mm -> 103 ms, 40.3%, 11%; 5 mm -> 133 ms, 48.8%,
8.2%. `d_end` closes at 309 mm/s at the 40 mm shell and 43 mm/s at 5 mm, a 7.2x fall against a 3.2x
rise in disagreement. **15 mm is the reasonable middle if a settle nearer rest is ever wanted.**

### Is the settle the time-reverse of the onset? Architecturally no, practically yes (2026-08-24)

Raised in review: the onset searches FORWARD for the first exit from the rest shell, and the settle
also searches forward — for the first ENTRY into the end shell. The true time-reverse would search
backward, taking the frame after the LAST exit. The code is asymmetric. Tested whether it matters.

**It does not. First-entry and last-exit pick the SAME frame on 99% of trials at the 40 mm shell
(97% at 15 mm).** Mirrored vs shipped, 604 observed trials, every shell: mean 40 vs 41 ms at 40 mm,
50 vs 52 at 25, 76 vs 68 at 15 — and identical |diff|, IQR and boundary speed throughout. The
shipped rule is already effectively the mirrored one.

**Why: position is far more monotone than speed after release.**

| channel after release | median crossings | >1 crossing |
|---|---|---|
| position, 40 mm shell | 1 | 35% |
| position, 15 mm shell | 1 | 40% |
| **speed, 5% of peak** | **3** | **72%** |

Once the hand is inside a 40 mm shell it stays; its small residual movements cross a 5%-of-peak speed
line repeatedly without leaving the shell.

**[!] This corrects the H3 attribution above.** The multiple-crossing mechanism (disagreement rising
78 -> 161 ms with crossing count) is a property of the **velocity** rule, NOT of the shipped
positional one. For the shipped settle the noise is H1 conditioning alone. H3 is the reason velocity
rules lose at the settle, not a defect of the rule in use.

**A failed first attempt worth recording**, because the failure mode is easy to repeat: mirroring by
requiring the final INSIDE run to last >= HOLD gives a **33% no-fire rate** at every shell, since any
clip ending within 150 ms of the hand arriving fails the test. The onset's hold applies to the
OUTSIDE run; the mirror must too. Fixed version has 0.2--2% no-fire.

**Net: the three settle findings compose.** Not an algorithmic asymmetry (there effectively is none);
not crossing ambiguity (that is the velocity channel's problem); it is conditioning, which is
geometric. Position wins at the settle and velocity at the onset because each is the channel that
stays well-conditioned where its own question is asked.

### Does the settle's 52%-of-peak firing bias the measures? Yes, modestly — and I had the constraint wrong (2026-08-24)

Raised in review: firing mid-motion should truncate the movement, and if the truncation scales with
speed it would bias affected vs unaffected differentially. Measured against a 2.5 mm "at rest" shell,
604 observed trials.

**The truncation is real: median 450 ms, 7.5% of total movement time.**

**But it is NOT speed-dependent** — r = +0.074 with return peak speed, r = -0.077 with TMT. The
predicted mechanism (fast movers cut off earlier) is false: the last 40 mm takes a roughly fixed time
because the hand is decelerating into rest either way.

**Differential bias exists but is small, and runs opposite to the guess:** affected 333 ms vs
unaffected 283 ms (+50 ms, +67 against the 2.5 mm reference), and affected arms have the SLOWER
return. Affected TMT is therefore under-measured slightly more, which compresses the
affected-vs-unaffected difference by ~50 ms on a difference of order a second. The larger effect is
per-participant: 183--417 ms, a 292 ms spread. Both are shared between MMC and OMC, so no reported
correlation moves; both sit in the absolute TMT.

**Tightening the shell buys most of it back:**

| shell | truncation | % TMT | arm diff | participant spread | spd@bd | MMCvOMC mean | >4fr |
|---|---|---|---|---|---|---|---|
| 40 mm (shipped) | 450 ms | 7.5% | +67 ms | 292 ms | 55% | 41 ms | 11.8% |
| 15 mm | 283 ms | 4.7% | **+33 ms** | 200 ms | 26% | 68 ms | 27.7% |
| 7 mm | 167 ms | 2.8% | **+17 ms** | 167 ms | 11% | 103 ms | 40.3% |

**[!] And here is where I had the constraint wrong.** I had been treating MMC-vs-OMC settle agreement
as the binding limit on Table II, and it is not close to binding. **TMT's own SD is 1282 ms**, so
boundary noise of 41--103 ms is 3--8% of it. The classic attenuation ceiling `1/sqrt(1+e^2/sd^2)`:

| shell | boundary noise | ceiling on r |
|---|---|---|
| 40 mm | 41 ms | 0.99949 |
| 15 mm | 68 ms | 0.99860 |
| 7 mm | 103 ms | 0.99679 |

All far above the **reported 0.994**, so the boundary is not what limits TMT's correlation and
tightening the shell would not move it. **Do not defend the 40 mm shell on the grounds that it
protects Table II — it does not.**

**The real cost of tightening is Table III, not Table II:** the settle row would go from 17 ms median
/ 83 ms p90 / 2.0% beyond 0.25 s to roughly 33 / 183 / 28%. A presentational cost for a substantive
gain in definition and bias.

**OPEN, and required before adopting anything:** the settle also bounds **number of movement units**,
which sum over `returning`. MU has almost no dynamic range (median 1, 74% exactly 1), so the
attenuation argument above does NOT transfer — proportionally, boundary noise could cost far more
there than for TMT. Measure MU under the 15 mm shell before changing the rule.

### RESULT: adopted nothing, learned two things (2026-08-24) — the boundary metrics were a bad proxy

`scripts/seg_sequential.py` now carries switchable onset/settle rules
(`OT_SEG_ONSET`, `OT_SEG_SETTLE`, `OT_SEG_PEAK_FRAC`), **defaulting to the shipped `pos`/`end`, so
the pipeline and every published number are unchanged.** `speed` puts the boundary at a fraction of
the trial's own peak hand speed. Verified the patch reproduces the standalone sweep (onset fires at
10.9% of peak, settle 8.7%, against 11.7% / 8.6% predicted).

Then ran the REAL scorer (`score_own_phases.py --anat12`) for all three variants, which is what
should have been done six turns earlier.

**[!] ONSET = speed 0.10: REJECTED. It destroys interjoint coordination.**

| measure (markerless windows) | shipped | onset=speed | delta |
|---|---|---|---|
| **interjoint** | 0.571 / 0.876 | **0.333 / 0.544** | **−0.238** |
| trunk displacement | 0.952 / 0.986 | 0.934 / 0.959 | −0.018 |
| the other ten | — | — | <= 0.004 |

And it looked **strictly better on every boundary metric**: |diff| vs the reference 200 -> 33 ms,
firing speed 47% -> 11.7%, cross-system agreement unchanged, 0% no-fire. The 167 ms it gains is the
START of the reach, where shoulder and elbow angles are both nearly flat, so it appends a
noise-dominated near-constant segment to a correlation and dilutes it. **Boundary-agreement metrics
do not predict measure quality. Do not optimise a boundary without re-scoring the measures.**

**SETTLE = speed 0.10: viable, not free.** Interjoint untouched (0.571), movement units −0.004, TMT
−0.002, eight others exactly 0 — but **trunk displacement −0.018** (r_av −0.026). Buys TMT mean
6.22 -> 6.48 s against the reference protocol's 6.67 s, i.e. about half the 450 ms truncation bias.
Not adopted: trunk is a reported correlation and TMT's absolute scale is already disclosed.

**[!] Paper correction that fell out of it.** §V-E said "elbow extension and trunk displacement are
unchanged by construction, being reduced over the whole trial rather than over a phase". Trunk MOVED
when the settle moved, so it is reduced over onset->settle — the whole **movement**, not the whole
clip. Elbow extension genuinely did not move. Corrected.

**The finding worth keeping is the robustness, not the rule.** Swapping the settle from a positional
to a velocity criterion — a different definition, shifting TMT's absolute value 260 ms — moves **ten
of twelve measures by <= 0.004**. The measure agreement is not an artefact of the particular boundary
definitions. And interjoint now has a THIRD independent demonstration of the same fragility: range
restriction (18 trials carry 92% of the variance), sample count (0.57 -> 0.30 at 30 Hz), and now
window composition (−0.238 while nothing else exceeds 0.018). Three unrelated perturbations, one
measure responding to all three — a stronger case for "limited by its summary statistic rather than
by the reconstruction" than any single one. **Candidate for one sentence in §VI if the page budget
ever allows.**

### DECISION: boundary rules NOT changed (2026-08-24)

Reviewed and settled: adopting `speed 0.10` for both boundaries is acceptable in principle — the
interjoint collapse is tolerable because interjoint is not a robust measure and the paper already says
so — but it is not worth the re-score, and the effect on everything else is small either way. **The
shipped `pos` / `end` rules stand.** `scripts/seg_sequential.py` keeps the switchable
`OT_SEG_ONSET` / `OT_SEG_SETTLE` / `OT_SEG_PEAK_FRAC`, defaulting to shipped, so the finding is
reproducible without being live.

**Preserved in the repo** (the anat12 scripts were once lost to a session `/tmp`; not repeating that):

- `paper/scripts/seg_rules/` — `README.md`, `sweep_boundary_rules.py`, `compare_measures.py`
- `paper/seg_rule_sweep.csv` — 66 rows, four rule families x thresholds x hold, both boundaries
- `paper/seg_rule_measures.csv` — 48 rows, all twelve measures x four variants, r_s and r_av under
  both window conditions

The scored inputs live under `out/scoring/` (gitignored, regenerable — see the README's loop).

**The one-line summary of the whole investigation**, from `seg_rule_measures.csv`: every measure is
within 0.004 of the shipped rule under a changed settle criterion except trunk displacement (−0.018),
and within 0.004 under a changed onset except interjoint (−0.238) and trunk (−0.018). **A boundary
rule that is strictly better on every boundary statistic can still destroy a measure.**

### §III-C's threshold claim, third and final correction (2026-08-24)

The claim about how our thresholds differ from the reference protocol's has now been wrong three
times, each time because another absolute constant turned up one level down:

1. Original: "every threshold is relative to that channel's own range within the trial, with no
   absolute velocity and no floor anywhere, so the rule is invariant to the scale of the
   reconstruction." **False** — `seg_sequential.py:35,37` set `LEAVE_REST = 30.0` mm and
   `ARRIVE_REST = 40.0` mm.
2. Second attempt: "the grasp, release and drink boundaries are thresholded on a fraction of their
   own channel's range rather than on an absolute velocity; reach onset and the settle are the
   exception." **Also false** — `GRASP_FLAT_MMPS = 40.0` **mm/s** (`pipeline/segment.py:139`) is what
   DETECTS the closing/opening runs for all four of those boundaries
   (`_runs(v_wc < -GRASP_FLAT_MMPS)`). The 30%-of-range test only QUALIFIES a run once detected, so
   the rule is a conjunction of an absolute rate and a relative magnitude, not a replacement of one
   by the other.
3. **Now: no boundary in this segmenter is purely relative.** Reach onset 30 mm, settle 40 mm, and
   the other four a 40 mm/s rate — every one carries an absolute constant somewhere. §III-C now
   claims only the distinction that survives: **the cup is followed visually rather than instrumented
   with markers**, and states plainly that both rules mix absolute and relative thresholds.

Consistent with the earlier finding that the two rules are nearly complementary rather than one
relative and one absolute. **The visual cup was always the real distinction** — as was said several
rounds before this claim was finally pinned down. The run-magnitude fraction (0.3, insensitive over
0.2--0.45) is still a genuine design property and is still stated; it is simply not a replacement for
an absolute threshold.

LESSON: a claim of the form "we use no absolute constants" needs every constant in the call path
checked, not just the ones in the function being read. Three passes here, three different constants.

### Is three cameras shown to be enough? For the pose yes, for the CUP no (2026-08-24)

§V-F's three-camera claim was a **latency** claim only — 14.9 ms per rig-frame fits the 16.7 ms
budget of 60 Hz — and said nothing about accuracy at three cameras, for either channel. Checked
against the cohort, which already spans camera counts (`cam_quality.json` via `H.use_good_cams`):

**5 cameras:** P07, P08, P15. **4:** P10, P14, P251, P252. **3:** P12, P13, P17, P19.

| cameras | units | trials | segmenter declines |
|---|---|---|---|
| 3 | 4 | 287 | **37 (12.9%)** |
| 4 | 4 | 247 | 1 (0.4%) |
| 5 | 3 | 255 | 1 (0.4%) |

**95% of all declines (37 of 39) come from the four three-camera units.** The cause is structural and
already named in §IV-E without being connected to it: the cup is gated on **three AGREEING** cameras,
so at a three-camera unit every camera must agree and there is no redundancy at all. The body fit is
unaffected — it solves from two cameras up and only 0.6% of joint-frames go empty.

So the paper's 4.9% decline rate is **not uniform**, and the configuration §V-F points at for 60 Hz
live capture is the same one where the cup fails on one trial in eight. Both facts are now stated:
§V-E gives the 12.9% / 0.4% split, and §V-F says the three-camera figure is about time alone and
that the rate and the camera count cannot both be reduced.

**NOT claimed, because the data cannot support it:** that measure agreement degrades at three
cameras. Per-camera-count correlations exist (3 cam: PV 0.901, flexion 0.917, interjoint 0.637;
4 cam: PV 0.841, flexion 0.970, interjoint 0.123; 5 cam: PV 0.850, flexion 0.990, interjoint 0.760)
but each group holds only 3--4 recording units, so between-participant range differs wildly between
them and the numbers are dominated by range restriction, not by camera count — the 4-camera group
looks *worst* on the two range-restricted measures, which is not credible as a camera effect. **Do
not quote the per-camera-count correlations.** The decline rate is the finding that survives.

### How predictable are interjoint's swings? Mostly — and mostly they are not real (2026-08-25)

Interjoint's reported r ranges over 0.12--0.89 depending on condition. Asked whether that is
predictable from the data. Two answers, and the second matters more.

**1. The ordering is ~86% predictable; the level is not.** For `mmc = omc + e` with independent e,
attenuation gives `r = 1/sqrt(1 + Var(e)/Var(omc))`. Across 400 random subsamples:

| | interjoint | shoulder flexion (control) |
|---|---|---|
| Spearman(predicted, observed) | **+0.856** | +0.999 |
| mean signed model error | **+0.164** (SD 0.092) | +0.002 (SD 0.001) |
| corr(omc, error) | **−0.795** | −0.358 |

The control shows the model is exact (+/-0.001) when its assumption holds. Interjoint breaks it: it is
a correlation bounded at 1.0 and its median optical value sits **0.18 SD below that ceiling**, so
where the reference is near 1 the error must be negative. Hence corr(omc, e) = −0.795 and a
systematic over-prediction. `Var(omc)` alone predicts nothing (Spearman +0.008) — it is the ratio.

**2. [!] The headline: interjoint's r spans 0.274 to 0.979 across random subsamples of the same 747
trials.** Every interjoint value this project has reported — 0.48, 0.57, 0.30, 0.33, and the
per-camera 0.64 / 0.12 / 0.76 and per-arm 0.63 / 0.17 — sits inside that range. **Subgroup
comparisons on different trials are therefore uninterpretable and must not be quoted** (already
flagged for the camera-count table; the same applies to the arm split).

**3. [!] Two claims in the paper were over-readings, and both are corrected.** Same-trial comparisons
need a PAIRED null, not the subsample spread, so bootstrapped paired (4000 resamples):

| claim | observed | paired 95% CI | verdict |
|---|---|---|---|
| §V-E "improving by 0.088" under markerless windows | +0.088 | **[−0.086, +0.212]** | contains zero |
| §V-F "falls from 0.57 to 0.30" at 30 Hz | −0.275 | **[−0.458, +0.075]** | contains zero |

§V-E's *conclusion* survives and is now better supported: the change being inside its own interval is
stronger evidence that the statistic is unstable than the direction of the change ever was. §V-F's
causal attribution ("halving the rate halves the samples") is REMOVED — the mechanism may be real but
the observed drop does not establish it. §VI now carries the 0.27--0.98 resampling range.

Unaffected: the "ten of twelve clear r_s >= 0.84 and r_av >= 0.93" counts, at both rates. Interjoint
fails the threshold on any of these values, so no count changes.

### Bootstrap CIs added to Table II (2026-08-25)

Prompted by interjoint's 0.27--0.98 resampling range: a bare point estimate cannot be read, so every
r now carries a $95\%$ interval. `paper/scripts/measure_cis.py` -> `paper/table3_cis.csv`, read by
`make_tables_tex.py`. **`r_s` resamples TRIALS; `r_av` resamples the 21 participant-arm GROUPS**,
which is the unit that statistic averages over — resampling trials for `r_av` would understate it.

**The intervals make the paper's own argument visible.** Median width on `r_s` is 0.03 and ten of
twelve are under 0.10; the two exceptions are the two measures the paper already calls weak:

| | r_s (markerless) | CI width |
|---|---|---|
| interjoint coordination | 0.57 [0.38, 0.80] | **0.43** |
| number of movement units | 0.80 [0.70, 0.87] | **0.17** |
| the other ten | — | 0.00--0.10 |

**Table II is now `table*` (full width)** — the CIs do not fit one IEEE column. `\footnotesize`,
tabcolsep 4pt, no overfull, still 8 pages.

**[!] And the paired comparison corrected a claim.** §V-E said "eight of the twelve measures move by
less than 0.01 ... and none degrades by more than 0.04". The optical-vs-markerless change must be
bootstrapped PAIRED (same trials, same draw); done that way, **five of twelve degrade detectably**
and movement units' interval reaches **0.066**, so "none by more than 0.04" was a point estimate the
interval does not support:

| measure | change | paired 95% CI |
|---|---|---|
| number of movement units | −0.036 | [−0.066, −0.019] |
| time to peak velocity | −0.033 | [−0.096, −0.001] |
| peak elbow angular velocity | −0.013 | [−0.027, −0.003] |
| total movement time | −0.005 | [−0.011, −0.002] |
| peak velocity | −0.005 | [−0.013, −0.000] |
| the other seven | | contain zero |

§V-E now says five degrade detectably, all small, and the remaining seven cannot be distinguished
from no change — a cleaner claim than enumerating deltas, and one that survives scrutiny.

**A bug worth not repeating:** the detectability test `(lo > 0) == (hi > 0)` returns True for a
zero-width interval at zero, so trunk displacement and elbow extension — identical between the two
conditions by construction — were flagged as detectable changes. Now `lo > 0 or hi < 0`.

**P19 correction to my own analysis, not the paper's claim.** §Limitations says the participant with
the worst calibration is the one whose measures agree least well. I doubted it and was wrong: P19 IS
the worst on measure agreement (error z 0.697, next 0.223), and its calibration is known bad. My
proxies — cup camera consensus, decline rate — showed P19 as *good* (best cup consensus of the four
three-camera units, zero declines), because **they measure whether cameras agree with each other, not
whether the rig is right.** A coherently mis-calibrated rig triangulates consistently to the wrong
place: high consensus, no declines, systematically wrong geometry. Consensus is not an accuracy
proxy; do not use it as one.

### 30 Hz gets intervals too, and only ONE measure changes detectably (2026-08-25)

`measure_cis.py` now covers the 30 Hz arm as well, paired against 60 Hz on the intersection of
trials, using `mmc30r` (the rate-matched variant the paper reports) from `fps_full_*of3.csv`.
Table II's 30 Hz column carries its interval, and all five r columns now come from one source.

| measure | 60 -> 30 change | paired 95% CI | |
|---|---|---|---|
| **peak velocity** | −0.028 | **[−0.051, −0.011]** | detectable |
| time to first PV | −0.001 | [−0.002, −0.000] | detectable, trivially |
| number of movement units | +0.016 | [−0.013, +0.048] | contains zero |
| interjoint coordination | −0.275 | [−0.467, +0.073] | contains zero |
| the other eight | <= 0.010 | contain zero | |

**[!] §V-F said "three move by more than 0.01 in r_s" and named PV, movement units and interjoint.
Only PV survives.** The claim is now the stronger one the data supports: **only peak velocity changes
detectably at 30 Hz, by 0.028 (0.011--0.051), and the other eleven cannot be distinguished from no
change at all.** That is a better result for the paper than the hedged enumeration it replaces —
halving the capture rate costs one measure a little and the rest nothing measurable.

Running tally for the day: five separate claims turned out to sit inside their own uncertainty once
an interval was computed — the 17-vs-33 ms boundary medians, interjoint's window "improvement",
interjoint's 30 Hz "collapse", "none degrades by more than 0.04", and now "three move by more than
0.01". Every interval took under a minute. **Compute the interval before interpreting a difference.**

### Why the study population cannot be completed here (2026-08-25)

Chased the earlier note that the clinical scores "appear to be recoverable locally". Partly true.
**No clinical values are copied into this file** — it is committed and pushed, and participant ages,
sexes and stroke dates are health data; whether they enter the repo is the authors' decision.

**(a) FMA-UE is NOT recoverable, and it is the one the paper needs.** `clinical_scores_import.ipynb`
cell 5 reads `C:\Users\tim.unger\Desktop\FMA_clean.csv`, sums the `betroffen` / `unbetroffen` columns
into `betroffen_sum` / `unbetroffen_sum`, and **emits only figures** — a bar chart per Record ID. No
printed table, so the totals exist as pixels. §Limitations' "its impairment gradient is mild" rests
on exactly this number.

**(b) Recoverable but unusable: no Record ID -> P-label key.** The notebook's cached outputs DO carry,
for all 38 records: measurement date, age, sex, stroke date, dominant hand, more-affected side, days
between stroke and measurement, and Box-and-Block counts per hand. But they are keyed by DELTA Record
ID and no key file exists on this machine. Tested the obvious hypothesis, Record ID == the number in
the P-label, against affected side derived independently from our own trial names:

- **8 of 9 informative units match** (P07, P10, P12, P13, P14, P15, P17, P19). p(all by chance) ~ 0.02.
- **P08 is uninformative** — its record reads "beide" (both sides affected).
- **P25 CONTRADICTS**: the record says "rechts", our trials say the affected arm is L, on both
  sessions. Its Box-and-Block is 42 right against 43 left, a one-block difference, so BBT cannot
  adjudicate. No second discriminator found.

So the mapping is probably right for most units and demonstrably wrong for at least one. **A
demographics table with one silently wrong row is worse than no table**, which is why this stays a
TODO rather than being filled in from an inference.

**(c) Ethics.** Still needs an explicit statement rather than §III-A's deferral; whether BASEC-No
2022-00491 covers this re-analysis is a question only the authors can answer.

**To unblock:** the FMA-UE totals in any tabular form, and either the Record ID -> P-label key or
confirmation of P25's affected side. Trial counts per unit and arm are already recorded above, so the
table's `n` column needs no work.

### Calibration errors located on the network share (2026-08-25)

The DELTA study's own per-unit calibration error is on the institutional SMB share, not on this
machine: `smb://nslliappl01.lli.local/research_analyzed_dataset/DELTA/DELTA/DATA/data_newStruc/
2024_10_23_1138_Calibration_errors.csv` (mounted via gvfs at
`/run/user/1000/gvfs/smb-share:server=nslliappl01.lli.local,share=research_analyzed_dataset`).
Columns `id_p;cam_used;error`. The share also confirms our unit labels are the study's own —
`P251`/`P252` appear there, as do `P241`/`P242` for another two-session participant.

**[!] This file CANNOT be used to rank the units, and I briefly "corrected" §Limitations on the basis
that it could.** The error is an aggregate over whatever camera set that unit was calibrated with,
and those sets differ: 10 cameras for P14, P15, P17, P19; 7 for P07, P08; 5 for the rest. We keep
only 3--5 of them. P17 reads worst overall at 13.32, but that spans 10 cameras of which the pipeline
uses 3 — the error is inflated by cameras that were then discarded, which is why its measures are
mid-pack. **A number computed over cameras the pipeline does not use says nothing about the
reconstruction.** §Limitations' claim is correct as the authors state it — P19 is the worst on the
cameras actually used — and now says "over the cameras actually used" to foreclose the same mistake.
The naive Spearman on the aggregate (+0.645 across 11 units) is confounded by the mixed camera sets
and is NOT quoted.

**What the file DOES support cleanly, because both sessions were calibrated over the same 5 cameras:**
P25's two sessions read **0.47 and 3.13** — a sevenfold difference in calibration error for the same
person on the same rig on different days. Now in §Limitations. It is independent evidence for §V-C's
central argument, that the landmark offsets are a property of each recording SESSION rather than of
the participant, arrived at there from the fit alone.

**Still not found on the share:** the clinical scores. `DELTA/DELTA/archive/DELTA_Patients` exists and
is unexplored; the FMA-UE totals and the Record ID -> P-label key remain the two blockers on the
study-population table.

### Searched the network share for the clinical scores: not there (2026-08-25)

Where I looked, so nobody repeats it. Share
`smb://nslliappl01.lli.local/research_analyzed_dataset` (the only one mounted; gvfs has no other, and
`smbclient` is not installed here so sibling shares could not be enumerated):

- `DELTA/` subtree, every `.csv` / `.xlsx` / `.pkl` to depth 4 -> **one file only**, the calibration
  errors already used. No clinical export.
- `DELTA/DELTA/archive/DELTA_Patients/` -> a Vicon Nexus session for a single **pilot** patient
  ("Paul"), with `06_BoxAndBlock`, `07_MotricityIndex`, `08_FugelMayer` folders. Those are trial
  CAPTURE directories for one pilot, not the cohort's score table.
- Share root -> 34 study folders, no clinical/REDCap/demographics directory.
- `1. Project_content.xlsx` at the root -> a catalogue of studies, sensors and goals. Not a data map;
  no pointer to where scores live.

**Conclusion: the FMA-UE totals and the Record ID -> P-label key are not reachable from this
machine.** The notebook read REDCap exports (`1DELTABasisdaten_...csv`, `4DELTABBT_...csv`,
`FMA_clean.csv`) from a local Desktop, so they are either on that machine, on a share not mounted
here, or need a fresh REDCap export. §V-A's TODO now says the share was searched.

**What the share DID give**, and it was worth the trip: the per-unit calibration errors, and
confirmation that `P251`/`P252` are the study's own labels rather than a local invention.
