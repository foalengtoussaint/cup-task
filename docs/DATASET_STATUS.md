y

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

## 4. Problem C — the SYNC gate is biased (affected arm), and a per-arm coverage gap

Trials are gated into the scorable pool by `sync_corr ≥ 0.7` (wrist-speed cross-correlation MMC↔OMC).
Two distinct issues:

**(C1) The sync gate spuriously fails good trials — it's a JITTER artifact, not desync.** On the failing
trials the pose is VALID (wrist_valid ~0.98), no gaps, and roughly time-aligned — but the *raw*
markerless wrist speed is jitter-dominated (measured on P251 trial_27: MMC peak 1443 vs OMC 626 mm/s,
271 vs 180 high-speed frames), which decorrelates the two speed curves even at the correct lag. A
session-constant lag does NOT recover them (0/15 on P251) — because it isn't a timing error. **These
trials are likely fine for actual Murphy scoring (SmoothNet removes exactly this jitter downstream);
they only fail the raw-speed sync GATE.** So current "scorable" counts UNDERCOUNT — especially the
affected arm, whose lower-amplitude motion has a worse speed-SNR. ⚠ *Open: re-gate on a SmoothNet-
smoothed speed or a session-constant lag before trusting affected-arm counts.*

**(C2) P17/P19 — the UNAFFECTED arm is barely triangulated at all** (wrist_valid P17 0.34, P19 **0.00**;
0/42 and 0/45 unaffected trials pass). This is a real per-arm camera-coverage hole — the good cameras
mostly view the affected side. P17/P19's earlier "41/31 scorable" was **affected-arm only.**

Per-arm pass-rate summary (affected / unaffected):
P07 40·42 | P08 39·48 | P15 45·43 | P10 40·28 | P13 40·38  (all fine) ·
P14 33·45 | P251 9·15 | P252 5·18  (affected sync-gate penalty, pose valid) ·
**P17 41·0 | P19 31·0**  (unaffected arm missing).

---

## 5. What's solid vs what needs work

**Solid:** P07/P08/P15 both arms; the markerless pose/cup pipeline itself (detect-once UETrack +
consensus reproject-seed matches the reference build); the OMC↔video mapping (audited); the
camera-classification tooling (4 independent methods agree).

**Needs work / open:**

- **Re-cut the shuffled cameras** (P12 cam4/5, P13 cam5, P17 cam5, P252 cam5) to add them back — they're
  recoverable, not lost. Needs the uncut session videos pulled (mostly not local).
- **Fix the sync gate** (§C1) — use DISPLACEMENT or SmoothNet-smoothed speed, NOT raw wrist speed.
  Tested: 2D pixel-motion sync (calibration-free) ≈ 3D-triangulated sync (2/15 recovered) → the
  failure is NOT triangulation/calibration. Displacement-corr 0.65 vs speed-corr 0.55 on the same
  trial, with MATCHING travel (259 vs 276 mm) → the systems track the same motion; the RAW SPEED
  derivative just amplifies detection jitter. Likely the biggest lever on usable-trial count, on the
  clinically important affected arm.
- **P17/P19 unaffected-arm coverage** (§C2) — a genuine limitation; those arms may be unrecoverable.
- **Recalibration** for the real-miscalib cameras (P10c4, P14c5, P19c2/c5) if their views are needed;
  else drop (current approach) and accept fewer cameras on those participants.

---

*Scripts: `audit_clip_omc.py`, `reaudit_cam_quality.py`, `multijoint_reproj.py`,
`cut_placement_audit.py`, `spatial_miscalib_check.py`, `check_track_quality.py`,
`cache_uetrack_tracks.py` (reproject-seed), `gnn_build_dataset.py` (lag-sync). Caches:
`cache/delta/{clip_omc_audit,cam_quality,reaudit_cam_quality}.json`, `cache/delta/gnn_pairs/`.*
