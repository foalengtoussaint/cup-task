# scripts/archive — settled threads, kept not deleted

Nothing here is broken or wrong; these are **finished investigations**. They are archived so that
`scripts/` shows only what the current v3 pipeline uses, and they stay in the repo because their
results are cited throughout `docs/RESULTS.md` and `docs/WORKLOG.md` — several conclusions are only
reproducible by re-running them.

**Running an archived script** needs BOTH script dirs on the path, because they import shared
libraries (`compare_pose_omc_delta`, `flow_velocity_probe`, …) that live one level up:

```bash
PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/archive" python scripts/archive/<name>.py
```

## What is in here, by thread

| thread | scripts | outcome |
|---|---|---|
| **DELTA camera quality / sync** | `cam_quality_delta`, `reaudit_cam_quality`, `multijoint_reproj`, `lag_probe_delta`, `motion_sync_delta`, `sync_fix_delta`, `delta_recut`, `constant_offset_recut`, `recut_from_audit`, `cut_placement_audit`, `uncut_offset_probe` | Bad cameras are MIS-CUT or MISCALIBRATED, not desynced. Cohort settled on P07/P08/P13. |
| **cup detector training** | `build_cup_labels_delta`, `prep_cup_dataset`, `train_cup_seg`, `train_rfdetr_cup`, `yolo_to_coco_cup`, `bench_yolo_vs_rfdetr`, `agree_yolo_rfdetr`, `viz_rfdetr_vs_yolo_disagree`, `eval_cup_3d_rfdetr` | YOLO kept: RF-DETR is more accurate but 2.4–3× slower. |
| **Siamese / tracker shootout** | `siam_from_first_detection`, `siam_gap_bridge`, `siam_percam_drift`, `viz_tracker_during_gap` | Superseded by UETrack (`scripts/uetrack_wrap.py`, still active). |
| **pose-model alternatives** | `videopose3d_probe` | No 3D→3D refiner with weights exists; SmoothNet won. |
| **point tracking** | `pointtrack_probe`, `pointtrack_peak`, `ctreseed_batch` | NEGATIVE result — CoTracker drifts, TAPNext too slow. See `project_pointtrack_negative`. |
| **renders / visualisation** | `render_*`, `reproj_render_delta`, `rerun_mmc_delta`, `log_rerun`, `plot_3d`, `animate_3d`, `visualize_*`, `overlay_phases` | One-off figures. Useful to revive when a boundary needs eyeballing — that is how the release fix was originally validated. |
| **misc probes** | `bias_vs_noise_delta`, `noncircular_delta`, `omc_effect_size_delta`, `gap_analysis_cup`, `diag_cup_delta`, `classify_rejects`, `compare_cup_head`, `affected_vs_unaffected_delta`, `validate_cohort_delta`, `build_work_set` | Answered; conclusions folded into the docs. |

## Still active (in `scripts/`, NOT here)

`results_v3_delta.py` (the metrics harness), `bench_v3.py` (speed), `cup_flow_probe.py`,
`compare_pose_omc_delta.py` + `flow_velocity_probe.py` (shared libraries), `uetrack_wrap.py`,
`gpu_decode.py`, the `cache_*` / `detect_rep_batched` caching tools, and the v2-era
`results_v2_delta.py` / `bench_realtime_v2.py` kept for comparison.
