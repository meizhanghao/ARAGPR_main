# Entry-point map

Run these commands from the repository root. The root `train.py` is the main
training entry. Supporting programs use `python -m package.module`, so imports
resolve after reorganization. Add `--help` to inspect an available CLI.
Datasets, masks, cached features and checkpoints must be prepared separately.

## Main training and baselines

| Purpose | Command |
|---|---|
| Dependency inspection / synthetic example | `python -m aragpr doctor` / `python -m aragpr demo` |
| Main shape branch | `python train.py` |
| Base DCBM training | `python -m experiments.baselines.main_train_DCBM_cub` |
| Original linear probe | `python -m experiments.baselines.linear_probe_all` |
| Zero-shot CLIP | `python -m experiments.baselines.clip_zero_image` |
| Ordinary frozen linear-base interface | `python -m experiments.baselines.run_linear_probe_shape_transfer` |

`experiments/baselines/main_dcbm_5.py`, `dcbm_seglocal_cub.py` and
`dcbm_testing.py` retain historical variants. Inspect their configured paths
before use; they are not the primary reproduction entry points.

## Controls and fixed-model diagnostics

| Purpose | Command |
|---|---|
| Geometry/random/affine/shuffled residual fits | `python -m experiments.controls.run_residual_controls` |
| Mean/paired/shuffled decoder regression | `python -m experiments.controls.run_decoder_baselines` |
| CUB image-target pairing | `python -m experiments.controls.run_cub_pairing_controls` |
| Derm7pt pairing/mechanism controls | `python -m experiments.controls.run_derm7pt_mechanism` |
| Saved-head bias-only replay | `python -m experiments.controls.replay_bias_only` |
| Additional paired bias-only fits | `python -m experiments.controls.run_bias_only_remaining` |
| Complete constant / fixed-head Teacher | `python -m experiments.diagnostics.run_posthoc_shape_diagnostics` |
| Correction retention / direction / Top-k | `python -m experiments.diagnostics.run_teacher_alignment_audit` |
| Within-class Teacher shuffling | `python -m experiments.diagnostics.run_fixed_head_within_class` |
| Case-disjoint Derm7pt pilot | `python -m experiments.diagnostics.run_derm7pt_case_disjoint` |
| Controlled image perturbation pilot | `python -m experiments.diagnostics.run_derm7pt_perturbations` |
| Mechanism summary | `python -m experiments.diagnostics.summarize_derm7pt_mechanism` |
| Pairing summary | `python -m experiments.diagnostics.summarize_pairing_controls_20261020` |

Summary modules retain historical input defaults; review them before running.
Use a fresh output directory for each new experiment.

## Preprocessing

All modules below are under `scripts.preprocessing`. Invoke them, for example,
as `python -m scripts.preprocessing.main_segment_1 --help`.

| Module suffix | Purpose |
|---|---|
| `main_create_traintestval_3` | Train/validation/test split preparation |
| `main_create_trainval_3` | Train/validation split preparation |
| `main_segment_1` | SAM2 region extraction |
| `main_image_emb_to_clip_4` | Whole-image CLIP features |
| `main_segimage_emb_to_clip_2` | Region CLIP features |
| `image_all_emb_to_clip` | Additional image-embedding utility |
| `extract_position_features` | Region position features |
| `sam2_contour` | Single-image contour inspection |

Older preprocessing variants have some settings in source rather than a full
CLI. CLIP/SAM2 operations require the optional dependencies.

## References, naming and plotting

| Purpose | Command |
|---|---|
| Shape-asset provenance | `python -m tools.audit.audit_cub_shape_assets` |
| Saved-control verification | `python -m tools.audit.audit_residual_controls` |
| CUB reference candidates | `python -m tools.references.run_cub_figure_candidates` |
| Figure-2 candidate readouts | `python -m tools.references.run_cub_figure2_candidates` |
| Cross-dataset Top-k screening | `python -m tools.references.run_topk_reference_screen` |
| Selected-case review | `python -m tools.references.run_figure_case_review` |
| CLIP naming-candidate pilot | `python -m tools.naming.run_shape_naming_pilot` |
| Top-k bars / full contribution profile | `python -m tools.plotting.plot_contributions` |
| Documentation figure export | `python -m tools.plotting.export_paper_figures` |

Reference selection requires the recorded models/assets and, for some rendering
modes, local layout blueprints. They are not bundled. The naming pilot uses
`EXEMPLARS` and author-designed candidate phrases in its source; adapt them to
your training references. It does not call a VLM API. See
[prototype_naming/](prototype_naming/README.md) for GPT-4o prompts and records.

## Code and data in separate directories

```bash
python train.py --project-root /path/to/prepared_data --dataset derm7pt --use_shape --shape_input geo --K_shape 128 --shape_mode decoded --freeze_semantic_from semantic_checkpoints/derm7pt_semantic_baseline.pt
```

`train.py` changes the working data directory and executes `aragpr.training`,
with the same options, model and losses as before. The residual-control runner
also accepts `--project-root`; other diagnostics have their own cache/path
options. The path helper distinguishes the repository root from the data root.

Model classes under `utils/` retain their module paths for checkpoint loading.
Other imports must use qualified paths from the new layout. The full migration
map is in [FILE_MIGRATIONS.json](FILE_MIGRATIONS.json).
