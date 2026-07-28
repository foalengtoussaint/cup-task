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
- Speed: ~1 fps @ 5 cams, 1920×1080, batch=1 (GPU ~57%; batching across frames would help).
- Output: `cache/mvgformer/<part>_<trial>.npz` = per-frame (wrist 3D mm, score).

Score vs OMC + incumbent: `scripts/mvgformer_score.py` (reuses all sync/align/speed math from
`compare_pose_omc_delta`; rigid-aligns to OMC; reports position, coverage, the incumbent-gap
occlusion test, and speed).
