# MVGFormer on DELTA — setup + eval

MVGFormer (CVPR'24, *Multiple View Geometry Transformers for 3D Human Pose*) is a multi-view
pose transformer that targets **occlusion** via volumetric multi-view feature fusion + iterative
query refinement. This is a DIFFERENT problem than the fast-frame wrist-SPEED thread
(SmoothNet / flow / σ-RTS) — MVGFormer's claim is robustness to camera dropout / occluded views,
not peak-velocity fidelity.

Repo: github XunshanMan/MVGFormer, cloned (uncommitted, ~large) to a scratchpad. Weights:
`mvgformer_q1024_model.pth.tar` (159MB) + `pose_resnet50_panoptic.pth.tar` backbone (136MB),
both placed at `<clone>/models/`.

## Build the CUDA op (the only hard part)

Runs in the **object_tracking** env (py3.10, torch 2.7.1+cu118). The deformable-attention op
must compile against nvcc 11.8 (match torch's cu118) with gcc ≤11 (system gcc-15 is rejected):

```bash
conda install -n object_tracking -c nvidia/label/cuda-11.8.0 -c conda-forge \
    cuda-nvcc=11.8 cuda-cudart-dev=11.8 gxx_linux-64=11
# cusparse/cublas/cusolver dev headers (torch's CUDAContextLight.h needs cusparse.h):
conda install -n object_tracking -c nvidia/label/cuda-11.8.0 \
    libcusparse-dev libcublas-dev libcusolver-dev cuda-cccl
# (if conda locks up, the .tar.bz2 are in ~/miniconda3/pkgs — extract include/ lib/ into $CONDA_PREFIX)

cd <clone>/lib/models/ops && rm -rf build
CUDA_HOME=$CONDA_PREFIX CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST=8.6 \
    python setup.py build install       # -> `import Deformable` works
```

### Source patches required (deprecated ATen + a bogus import + arch)
- `lib/models/ops/src/{deform.h,cuda/deform_cuda.cu}`: `.type().is_cuda()`→`.is_cuda()`,
  `AT_DISPATCH_FLOATING_TYPES(value.type()`→`.scalar_type()`, `.data<T>()`→`.data_ptr<T>()`.
- `lib/models/ops/setup.py`: add `-gencode=arch=compute_86,code=sm_86` (RTX 3060 Ti = Ampere).
- `lib/models/dq_transformer.py`: delete `from nis import cat` (dead, breaks on py3.13; unused).

## Running on DELTA — `scripts/mvgformer_delta.py`

Standalone runner (no Panoptic dataloader, no mmcv, no distributed). Key points, all learned the
hard way:

- **mmcv is stubbed** in-process (`sys.modules['mmcv']`) — it's imported only for `get_dist_info`
  + a distributed result-gather neither single-process inference touches. Avoids the finicky
  mmcv↔torch2.7 build.
- **Camera convention** (verified vs `lib/dataset/panoptic.py`): MVGFormer projects
  `xcam = R (X − T)`, `T` = camera **centre** in world (mm), `standard_T` = raw translation.
  DELTA `CamCalib` is `Xc = R X + t` → `T = −Rᵀ t`, `standard_T = t`, `R` unchanged,
  fx/fy/cx/cy from K, `k = dist[[0,1,4]]`, `p = dist[[2,3]]`. Do NOT apply panoptic's axis-swap M.
- **Space cube** (`MULTI_PERSON.SPACE_CENTER/SIZE`) MUST be recentred+shrunk onto the DELTA capture
  volume (Panoptic's default 8m cube @ origin → every coarse query misses the subject → all queries
  score 0 / pose = zeros). We seed it from the incumbent triangulated body centroid, 3×3×2 m.
  Set BEFORE `get_mvp` (baked into decoder `__init__`). This only sets WHERE queries start.
- **Inference flags**: `model.eval()`, `gt_match_test=False`, **`log_val_loss=False`** (else forward
  runs the GT criterion/matcher → crash on our dummy GT). Full checkpoint loads with 0 missing / 0
  unexpected keys (includes the backbone).
- 15-joint panoptic skeleton: **l-wrist=5, r-wrist=11**. Affected side is LEFT (trial name `_L_`).
- Only cams with a staged clip (`delta_<part>_<trial>.<n>.mp4`) are used; good-cam whitelist applies.
- Speed: **~1 fps** @ 5 cams, 1920×1080, batch=1 — and this is GENUINE MVGFormer compute, NOT the
  cv2-decode harness: GPU sits at ~88% during the forward. Cause = the q1024 config (1024 pose-query
  hypotheses × 4 decoder layers × deformable attention over 5 views) on a mid-range 8GB Ampere card
  (RTX 3060 Ti); the paper's "fast" numbers are datacenter-GPU. EVERY shipped config is q1024 and the
  query embedding (`query_embed_type: person_joint`) is LEARNED, so you can't just lower `num_instance`
  at inference without a retrain/surgery. Not worth optimising: MVGFormer is an OFFLINE occlusion
  candidate — YOLO-pose already owns the live path (40–240 fps, and wins accuracy, see
  project_pose_model_speed). 1 fps offline is acceptable for what it's for.
- Output: `cache/mvgformer/<part>_<trial>.npz` = per-frame (wrist 3D mm, score).

Score vs OMC + incumbent: `scripts/mvgformer_score.py` (reuses all sync/align/speed math from
`compare_pose_omc_delta`; rigid-aligns to OMC; reports position, coverage, the incumbent-gap
occlusion test, and speed).

## Plan / decision (2026-07-28)

Target if it had shipped: LIVE, ≥20 fps. **OUTCOME: parked — see RESULTS below.** On a 5-camera rig
(the user's actual setup) MVGFormer loses to the incumbent on accuracy AND speed, so speed optimisation
was never reached. Kept for the record: the decision order was **prove it's WORTH it BEFORE optimizing.**
1. Baseline accuracy: MVGFormer vs OMC + incumbent on a clean-OMC trial (P07 t13). ⏳
2. **Occlusion test = the whole point.** The wrist-restricted incumbent (YOLO+robust-triangulation
   on good cams) already has **100% wrist coverage on all 12 P07/P08 trials**, so there are NO
   incumbent-gap frames on the clean cohort — MVGFormer's occlusion win can't show there. The
   decisive test is **camera dropout**: run MVGFormer AND the incumbent on 2–3 cams (`--cams 1,3`)
   on a trial with clean OMC, and see if MVGFormer's volumetric fusion survives where triangulation
   collapses (KF-accuracy-budget memory: 2 cams = catastrophic >1m for triangulation).
3. **Only if it wins step 2**, chase 20 fps: q1024→~64 queries is the big lever for ONE person (but
   the query embed is learned — must verify it truncates without a retrain), + fp16 autocast +
   smaller input. 20× is roughly what the query-count lever alone could give, so not hopeless.
   If it can't clear 20 fps AND doesn't beat YOLO on occlusion, drop it.

## RESULTS (2026-07-28) — P07 trial_13_L (left wrist, 531 frames, vs OMC, rigid-aligned)

All estimators despiked (same H._despike the incumbent triangulation uses); jitter = median |d²X|.

| cameras | MVGFormer pos (med/p90) | MVGFormer cov | incumbent (robust-tri) |
|---|---|---|---|
| **5 (clean)** | 7.0 / 13.2 mm | 100% | **5.4 / 12.7 mm, 100%** (WINS) |
| **2 (dropout, cams 1,3)** | **12.6 / 26.2 mm, 79%** | 79% | **0% — COLLAPSED, produced nothing** |

**Verdict: MVGFormer earns its keep for OCCLUSION, not clean data.**
- Clean 5-cam: robust triangulation BEATS it on position (5.4 vs 7.0 mm) AND is ~3× less jittery
  (2.0 vs 6.2 mm |d²X|). MVGFormer is a per-frame regression (softmax blend over 1024 queries) →
  broadband mm-scale fuzz + occasional single-frame query-swap spikes.
- **SPEED is NOT broken.** The earlier "715% peak / unusable" was an artefact of differentiating
  unfiltered spiky position. Recipe: **despike → low-pass the POSITION @6Hz (study default) → d/dt**.
  Two rules learned the hard way here:
  1. Filter POSITION, never SPEED — speed-domain filtering can't remove spike-induced velocity
     impulses, only smears them into fake bumps: 40-90% peak overshoot at every cutoff vs ~1% for
     pos-lp. Despike first (isolated query-swap teleports), then position low-pass.
  2. **Don't chase valley-noise with a lower cutoff, and NEVER filter OMC to measure peak error**
     (softening ground truth fakes a good score). vs UNFILTERED OMC (true peak 866 mm/s), MVGFormer's
     peak err GROWS as you filter harder: **6Hz 1% → 3Hz 7% → 2.5Hz 9% → 1.5Hz 17%**. So 6Hz is the
     right cutoff — MVGFormer's despiked position already reproduces the true peak; aggression only
     loses it. (A discarded intermediate claimed "2.5Hz = 0% peak cost" — that was measured against a
     co-filtered OMC and is WRONG.) Residual per-frame valley fuzz is ~2× the incumbent but cosmetic;
     the peak (the Murphy measure) is excellent at 6Hz. Fine speed source; position-jitterier + 40× slower.
- **2-cam dropout: MVGFormer 13 mm @ 79% vs triangulation 0%.** Triangulation needs ≥3 agreeing cams
  (KF-budget floor) and dies at 2; MVGFormer's volumetric fusion degrades gracefully (7→13 mm). THIS
  is the win — the current pipeline literally cannot produce a wrist here.
- It stayed CONFIDENT on 2 cams (score ~0.83, even 0.93 at times).

**DECISION (user, 2026-07-28): rig is ALWAYS 5 cameras → MVGFormer PARKED.** Its only edge was the
camera-dropout regime; with 5 well-spread cams the incumbent wins on position (5.4<7.0), jitter
(2.0<6.2, ~3×), and is ~40× faster (40–240 fps vs 1). Speed (despike+pos-lp@6Hz) is actually a TIE-ish
(both nail the true peak: MVGFormer 1%, incumbent 13% — MVGFormer even edges peak here), so speed is
NOT the differentiator; position-jitter + throughput are. No reason to run it and no
reason to chase the 20 fps optimisation (best case = parity with a tool that's already worse on 5 cams).
Useful negative result: a heavyweight learned multi-view transformer does NOT beat plain robust
triangulation once you have enough well-spread cameras. Scripts/weights/cache kept for a future
few-camera or heavy-occlusion scenario. Graphs: out/mvgformer/*.png.
