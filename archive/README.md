# archive/ — settled outputs and data, kept not deleted

Nothing here is deleted, ever. This directory exists so the repo root shows only what the current
v3 pipeline uses; everything below belongs to a **finished** thread whose conclusions are recorded in
`docs/RESULTS.md` and `docs/WORKLOG.md`.

| path | size | what it is | why archived |
|---|---|---|---|
| `runs/rfdetr/` | 3.2 G | RF-DETR cup-detector checkpoints, per participant | Thread settled: RF-DETR is more accurate but 2.4–3× slower, so **YOLO was kept**. Its scripts are in `scripts/archive/`. |
| `out640/` | 143 M | 640-px render experiments (skeletons, phase overlays, disagreement videos) | One-off figures from the detector/resolution threads. |
| `out_cup_pred/` | 141 M | cup-prediction renders | Same. |
| `out_streams/` | 5.2 M | stream-benchmark artefacts | Superseded by `scripts/bench_v3.py`. |
| `out_media/` | — | one-off `.mp4` / `.rrd` / `.html` from `out/` | Phase-comparison videos and rerun logs from settled investigations. |

## Still live in the repo root

- **`runs/segment/`** (1.6 G) — the cup YOLO weights, **still referenced** by
  `scripts/cache_tracks.py`. Do not archive.
- **`models/`** — pose weights, SmoothNet checkpoint, UETrack weights (repo-persistent on purpose:
  they were once lost to a cleaned scratchpad).
- **`cache/`** — all cached detections, tracks and flow. The whole offline results path runs from
  here with no GPU; re-deriving it costs hours of inference.
- **`out/figures/`** — the current figures cited by the docs.
- **`data/`** — clips and datasets.

## Reviving something

The scripts are in `scripts/archive/` (see its README for how to run them — they need both script
dirs on `PYTHONPATH`). Paths inside archived scripts still point at the original locations, so move
the data back to the root before re-running rather than editing the scripts.
