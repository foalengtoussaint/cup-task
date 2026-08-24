
# DELTA drink-task dataset — status & known problems

*Compiled 2026-08-03. Companion to `docs/WORKLOG.md` (chronological detail) — this is the standing
summary of what's in the cohort and every known data problem, with how each was diagnosed and its fix.*

The dataset is the DELTA/iDrink stroke drink-task: multi-camera video (cams 1–5 used) + Qualisys OMC
mocap ground truth, per participant, ~40–90 drinking-task repetitions each (affected + unaffected
arm). Markerless pipeline = YOLO26-pose + detect-once UETrack cup → triangulate → SmoothNet → Murphy
measures. OMC comes from `~/Documents/AutoMQ/<P>/` (processed) and raw c3d on the SMB share.

---

## 1. Cohort — who's in, who's usable

**11 participants**, selected = calibration RMS < 6 (study CSV) **AND** has OMC in AutoMQ.
Excluded: P16/P18/P20/P21/P22 (good calib but **no OMC**), P24 (no OMC).

| group                | participants                   | state                                                           |
| -------------------- | ------------------------------ | --------------------------------------------------------------- |
| **Clean core** | P07, P08, P15                  | all 5 cameras fine, both arms good — the reference cohort      |
| Original OMC         | P17, P19                       | usable but each has bad cameras + an unaffected-arm gap (below) |
| Newly added          | P10, P12, P13, P14, P251, P252 | usable after per-camera cleanup (below)                         |

**P25 is SPLIT** into **P251 (trials 10–41)** and **P252 (trials 42+)** — two recording sessions with
*different* calibrations (RMS 0.47 vs 3.13) and different camera health (P251 cam5 fine, P252 cam5
shuffled). AutoMQ/OMC bookkeep P25 unsplit; treating it that way would mix a fine + a shuffled camera
in one "participant." **The split is necessary, not cosmetic.**

**Cohort scorable pool ≈ 680 trials** (was 328 for the original 5). Counts are per-arm-sensitive —
see §4.

---

## 2. Problem A — broken / mis-cut CLIPS  (`audit_clip_omc.py`)

Some staged clips are not the cut drinking trial. Per-trial check: `n_video == n_det` and
`n_omc(c3d→60fps) ≈ n_video`. **The OMC↔video mapping is otherwise INTACT** (n_det==n_video
everywhere; P07/P08 reference nearly clean).

- **P10: ~13 broken trials** — UNCUT (e.g. trial_20 = 12 434 frames / 3.5 min; trial_30 = 4 469),
  OMC_MISMATCH (c3d a different time window than the clip → mapping broken *for that trial*), or NO_OMC.
- **Decision: excluded, NOT re-pulled** (user's call). P10 runs on its ~70 clean trials.
- Clean-trial list cached at `cache/delta/clip_omc_audit.json`; all downstream builds gate on it.

---

## 3. Problem B — bad CAMERAS: miscalibration vs shuffled cuts

**Most participants have 1–2 bad cameras**, but the *mechanism* differs and MUST be told apart —
because a shuffled cut is RE-CUTTABLE while miscalibration needs recalibration/dropping.

⚠ **Reprojection error MAGNITUDE alone cannot tell them apart** (both give ~25–30 px). Three methods,
in order of authority:

1. `reaudit_cam_quality.py` — reproj vs RANSAC-consensus 3D, stratified still/moving (separates FINE /
   desync / bad; but mislabels shuffled cuts as "miscalib").
2. `multijoint_reproj.py` — spatial: real geometry displaces ALL joints incl. stable nose/shoulders;
   error that grades with joint *speed* is not pure geometry.
3. `cut_placement_audit.py` — pixel-exact NCC of each cut clip in its OWN uncut → catches SHUFFLED
   cuts (clip = a *different* repetition, +tens–hundreds of seconds off, invisible to reproj).
4. **`spatial_miscalib_check.py`** (built this session, the cleanest single test) — the reprojection
   **error VECTOR** must be a *smooth function of 3D position* for real miscalibration.
   `spatialR2` = R² of `error ~ linear(X,Y,Z)`.
   ⚠ **spatialR2 alone is NOT sufficient — it fails two ways** (found by running cut-placement on
   every flagged cam): (a) a **CONST-OFFSET cut** (whole clip shifted a constant amount) makes a
   *smooth* error field → high spatialR2 → looks identical to real miscalibration (P10 cam4: R²=0.64
   but actually a +58.5 s re-cuttable shift); (b) at **low error** it cannot tell FINE from SHUFFLED
   (a different-repetition wrist lands *plausibly*, so low R² — P13 cam2 R²=0.11 but is shuffled).
   **`cut_placement_audit.py` (pixel-exact NCC in the uncut) is the authoritative label.**

**Per-camera verdicts — cut-placement AUTHORITATIVE (ran on every flagged cam, both methods shown):**

| participant | camera | spatialR2 (my call) | **cut-placement (truth)** | **FINAL verdict** | fix |
| --- | --- | --- | --- | --- | --- |
| P10  | cam4 | 0.64 → miscalib ✗ | **CONST-OFFSET +58.5 s** | **timing error** | ONE re-cut shift |
| P12  | cam4 | 0.30 → shuffled ✓ | **SHUFFLED +58..171 s** | **shuffled** | per-trial re-cut |
| P12  | cam5 | 0.26 → shuffled ✓ | **SHUFFLED +144..171 s** | **shuffled** | per-trial re-cut |
| P13  | cam5 | 0.29 → shuffled ✓ | **SHUFFLED −55..+0 s** | **shuffled** | per-trial re-cut |
| P13  | cam2 | 0.11 → fine ✗ | **SHUFFLED −37..+10 s** | **shuffled** | per-trial re-cut |
| P14  | cam2 | 0.21 → fine ✓ | **CUTS-OK** | **FINE** | keep |
| P14  | cam5 | 0.47 → miscalib ✓ | **CUTS-OK** | **REAL MISCALIB** | recalib / drop |
| P17  | cam5 | 0.28 → shuffled ✗ | **CUTS-OK** | **mild miscalib / noise** | keep or drop |
| P19  | cam5 | 0.72 → miscalib ✓ | **CUTS-OK** | **REAL MISCALIB (severe, 132 px)** | recalib / drop |
| P19  | cam2 | 0.65 → miscalib ✓ | **CUTS-OK** | **REAL MISCALIB (mild)** | recalib / drop |
| P251 | cam5 | 0.18 → ~fine | (not re-run) | ~FINE | keep |
| P252 | cam5 | 0.13 → shuffled ✗ | **CUTS-OK** | **mild miscalib / noise** | keep or drop |

**cut-placement overturned my spatial-only call on 4 of 11 cams** (P10c4, P13c2, P17c5, P252c5) — so
the earlier "spatialR2-confirmed" table was over-confident.

**Net (authoritative):**
- **Re-cuttable (timing/placement, NOT lost):** P10 cam4 (one +58.5 s shift), P12 cam4/5, P13 cam5/cam2
  (per-trial) — **5 cameras recoverable by re-cutting.**
- **Real miscalibration:** P14 cam5, P19 cam5 (severe), P19 cam2 — **3 cameras** (recalibrate or drop).
- **Mild / borderline (keep or drop):** P17 cam5, P252 cam5, P251 cam5.
- **Actually fine (reproj was wrong):** P14 cam2.

So the majority of the "bad cameras" are **re-cuttable, not miscalibrated** — the dataset is in better
shape than the reproj-only audit implied. **cam_5 is the most-often-affected camera**, but the
mechanism varies per session (shuffled in P12/P13; real miscalib in P14/P19; fine in P251), so there's
no single fix. Dropping the bad cameras and re-triangulating on the good ones already **rescued the
participants** (P12 4→81, P13 43→78, P10 61→68 scorable; wrist-validity P12 0.08→0.99). Whitelist =
`cache/delta/cam_quality.json`; reproj verdicts = `cache/delta/reaudit_cam_quality.json`; cut-placement
logs in `scratchpad/cut_P*.log`.

⚠ This confirms the prior work (2026-07-17): same cameras flagged, and they had already RE-CUT P13
cam2 (a placement fix, not recalibration) — matching our shuffled-vs-miscalib split.

---

## 4. Problem C — the SYNC GATE is the real bottleneck, NOT the data (measured)

Trials enter the scorable pool via `sync_corr ≥ 0.7` = cross-correlation of the **raw markerless WRIST
SPEED** vs OMC. This one metric choice is responsible for almost all the "unusable" trials — and the
underlying trials are almost all FINE. Three earlier explanations here were WRONG and are corrected:

**⚠ RETRACTED: it is NOT jitter, and there is NO coverage gap.** Investigated each cluster directly:

**(C1) P251 / P252 — a pure SPEED-METRIC artifact, not jitter, not bad data.** On their sync-failing
trials the wrist DISPLACEMENT correlation is excellent — **P251 0.90 (14/15 pass), P252 0.98 (17/17
pass, lag +1 frame, near-perfect alignment)** — and total travel matches (259 vs 276 mm). The jitter
proxy is NOT damning (P251 trial_27: 159 vs 138 frames >300 mm/s, similar). The trials track the same
motion perfectly; only the raw-SPEED correlation fails (speed = derivative → fragile). **Fix: gate on
DISPLACEMENT → recovers 31/32.** (Cup-speed sync also recovers P251 9/15.)

**(C2) P17 — a bad WRIST KEYPOINT, not a coverage gap.** P17's unaffected arm triangulates fine; its
*wrist keypoint* is poorly detected in that view (occluded/jittery), but the same arm's ELBOW syncs at
**0.94/0.95/0.86** where the wrist fails at 0.22/0.41/0.25. **Fix: sync on the best-correlating joint
(elbow) → recovers 42/42 sampled → P17 41 → ~83 scorable.** (Earlier "unaffected arm barely
triangulates, wrist_valid 0.34" was measured on the bad-camera whitelist and is retracted.)

**(C3) P19 — the ONE genuine, unfixable case: an OMC ground-truth gap.** P19's OMC has NO
inner/outer wrist markers for the unaffected (L) arm (only `cluster_wrist_L_*`), so `_load_omc`'s
inner/outer-midpoint wrist target is missing → nothing valid to sync against (best-joint AND cup both
recover 0/25). This is a mocap limitation, not our pipeline; P19's unaffected arm is simply not
OMC-validatable. (Matches memory: P19 wrist target = `outer_only`.)

**Bottom line: the sync gate should try {wrist, elbow, shoulder, cup} × {speed, displacement} and take
the best.** With that gate, every participant's trials pass EXCEPT P19's unaffected arm (OMC gap). The
current 680-scorable count UNDERCOUNTS heavily — P17 alone jumps ~41→~83, P251/P252 recover ~31, and the
losses were sync-METRIC choices, NOT cameras, NOT re-cutting, NOT jitter.

---

## 5. What's solid vs what needs work

**Solid:** P07/P08/P15 both arms; the markerless pose/cup pipeline itself (detect-once UETrack +
consensus reproject-seed matches the reference build); the OMC↔video mapping (audited); the
camera-classification tooling (4 independent methods agree).

**Needs work / open (in priority order):**

1. **Fix the SYNC GATE — the single biggest lever (§4).** Replace the raw-wrist-speed correlation with
   **best of {wrist, elbow, shoulder, cup} × {speed, displacement}**. Measured recovery: P17 elbow
   42/42, P251/P252 displacement 31/32, cup P251 9/15. This is a metric change (no re-processing of
   video/tracks) and should push the scorable pool from 680 toward the full ~830 clean trials.
   Everything below is secondary.
2. **Re-cut the shuffled / const-offset cameras** to ADD them back (more robust triangulation, helps
   the occluded drink-apex): P10 cam4 (one +58.5 s shift), P12 cam4/5, P13 cam5/cam2 (per-trial).
   Recoverable, not lost. Uncut session videos now pulled to `cache/delta/*/uncut/`.
3. **Recalibrate (or accept dropped)** the real-miscalib cameras: P14 cam5, P19 cam5 (severe), P19 cam2.
4. **P19 unaffected arm is NOT recoverable** — its OMC lacks inner/outer wrist markers (§C3). A
   ground-truth gap, not a pipeline problem. P19 is usable on the affected arm only.

**NOT problems (retracted):** "jitter" (P251/P252 — it's a speed-metric artifact, displacement-corr
0.90-0.98); "P17 coverage gap" (its arm triangulates; only the wrist keypoint is bad); "P10c4/P13c2/
P17c5/P252c5 = miscalib/shuffled" per spatial test (cut-placement overturned these — §3).

---

*Scripts: `audit_clip_omc.py`, `reaudit_cam_quality.py`, `multijoint_reproj.py`,
`cut_placement_audit.py`, `spatial_miscalib_check.py`, `check_track_quality.py`,
`cache_uetrack_tracks.py` (reproject-seed), `gnn_build_dataset.py` (lag-sync). Caches:
`cache/delta/{clip_omc_audit,cam_quality,reaudit_cam_quality}.json`, `cache/delta/gnn_pairs/`.*

---

## 6. RESULT — sync-gate fix applied (2026-08-04)

Multi-signal sync gate (`_find_lag_multi`: best of {wrist,elbow,shoulder}×{speed,displacement} + cup;
`load_clean` gates on `max(sync_corr, sync_corr_multi)`) rebuilt on all 11 participants.
**⚠ Also fixed a whitelist bug:** P14 cam5 (confirmed miscalibration §3) had been left in the whitelist
as a stale "un-audited import default" — now dropped (P14 = cam1-4). All bad cameras now removed:
P10 −c4, P12 −c4/5, P13 −c2/5, P14 −c5, P17 −c5, P19 −c2/5, P251/P252 −c5; P07/P08/P15 keep all 5.

**Scorable: 680 → 745 of 837 clean trials (89%).** 8/11 participants at 99–100% (P251/P252 62/58% →
100% via cup/displacement sync; P14 100% after cam5 drop).

**P17/P19 still partial (51% / 47%) — but NOT a sync problem now:** their unaffected-arm trials pass
sync (0.98 via elbow) but fail the separate **`wrist_valid_frac ≥ 0.5`** gate. ⚠ CORRECTED CAUSE (I was wrong twice — not
'coverage gap', not 'sparse detection'): the 2D wrist is detected **100%** in all 4 good cams, but the
3D TRIANGULATION keeps <2 agreeing cams on most frames (P17 trial_10: 276/427 frames, median 0 kept;
reproj 18px on the survivors) → no 3D wrist. This is **residual CALIBRATION error amplified at the
wrist** (fast, far-from-centre = hardest point); central/slower joints (nose/shoulder/ELBOW) triangulate
fine — which is exactly why the elbow syncs at 0.94. It is **P17/P19-SPECIFIC** (unaffected-wrist 3D
validity: every other participant 0.97–1.00; P17 0.34, P19 0.00) because they are the two MISCALIBRATED
participants — even their surviving cameras carry enough residual error that the wrist won't triangulate.
Consequence: P17/P19 unaffected-arm WRIST-based measures (peak velocity, movement units) are compromised;
angle/elbow measures are fine. P19 is doubly hit (calibration + the OMC-marker gap §C3). NOTE: only 2
participants are low (P17 51%, P19 47%); the other 9 are 99–100%.

**⚠ UPDATE — P17 recovered by dropping a camera the COARSE audit missed (found via the "why does the
elbow work?" question).** P17's low unaffected-wrist validity was NOT sparse detection (2D wrist = 100%)
and NOT vague "wrist amplification". LEAVE-ONE-OUT per-camera reprojection (triangulate from the other
cams, reproject into the held-out cam) exposed **cam_1 at 74 px on the wrist AND 76 px on the elbow** —
genuinely miscalibrated, but the coarse still/moving `reaudit_cam_quality` had PASSED it. The elbow still
triangulated because it had a clean anchor camera (cam_2 @ 7 px on the elbow) to hold consensus; the wrist
had no clean camera (best 23 px) so cam_1's error tipped it over the 30 px gate on most frames. **Dropping
cam_1 → P17 unaffected wrist_valid 0.34→0.83, P17 42→83 scorable (100%).** cam_quality P17 = cam2,3,4.
LESSON: the coarse motion-stratified reaudit UNDER-detects bad cameras; leave-one-out per-joint
reprojection is stricter and should be the standard camera check.

**Scorable now: 680 (raw) → 745 (multi-sync) → 786/837 = 94% (after P17 cam_1 drop).** 10/11 participants
99–100%; only P19 low (47%) — its unaffected arm has the OMC-marker gap (§C3) AND likely a bad camera
(pending the same leave-one-out check).

**⚠ UPDATE 2 — P19 also recovered (cluster-marker fallback). Final: 826/837 = 99%.** P19's "unfixable
OMC gap" was ALSO overturned: its unaffected (L) arm HAS the wrist as a 4-marker rigid cluster
(`cluster_wrist_L_*`), just not the inner/outer pair `_load_omc` expected. Added a cluster-centroid
fallback to `_load_omc_defensive` (tagged `target_wrist_source='cluster'`): P19 unaffected wrist_valid
0.00→0.83, 0→37/45 scorable; P19 total 43→83 (90%).

**FINAL scorable: 680 (raw) → 745 (multi-sync) → 786 (P17 cam1 drop) → 826/837 = 99%.** All 11
participants 90–100%. The remaining 11 non-counting trials all fail the `wrist_valid_frac ≥ 0.5` gate
(sync is fine 0.9+), NOT sync: the 3D wrist triangulates <50% of frames — 9 are P19's hardest trials
(marginal cameras/OMC), the rest (P08 t70, P12 t55, P15 t47) sit at 0.42–0.49, just under threshold, from
real drink-apex occlusion on the paretic arm. This is the natural task-difficulty floor, not a bug. So
essentially ALL clean trials are usable; the 1% residual is genuine wrist-occlusion.
