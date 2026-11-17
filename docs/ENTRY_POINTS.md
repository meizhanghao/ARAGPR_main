# Entry-point map

Run `python SCRIPT --help` for CLI details. Unless stated otherwise, run from the
repository root and use a fresh output directory. Supporting scripts retain the
original filenames for saved-run compatibility.

| Purpose | Entry point |
|---|---|
| Dependency inspection / synthetic example | `python -m aragpr doctor` / `demo` |
| Main DCBM-platform branch | `train.py` (alias), `cancha_cub.py` |
| Main base-model training | `main_train_DCBM_cub.py`, `main_dcbm_5.py` |
| Original linear-probe baseline | `linear_probe_all.py` |
| SAM2 regions | `main_segment_1.py` |
| Whole-image CLIP features | `main_image_emb_to_clip_4.py` |
| Region CLIP features | `main_segimage_emb_to_clip_2.py` |
| Shape-asset provenance checks | `audit_cub_shape_assets.py` |
| Geometry/random/affine/shuffled residual fits | `run_residual_controls.py` |
| Control-run verification | `audit_residual_controls.py` |
| Mean/paired/shuffled decoder regression | `run_decoder_baselines.py` |
| CUB image-target pairing | `run_cub_pairing_controls.py` |
| Derm7pt pairing and mechanism controls | `run_derm7pt_mechanism.py` |
| Complete affine constant / fixed-head Teacher | `run_posthoc_shape_diagnostics.py` |
| Correction retention / direction / Top-k | `run_teacher_alignment_audit.py` |
| Fixed-head within-class Teacher shuffling | `run_fixed_head_within_class.py` |
| Ordinary frozen linear-base interface | `run_linear_probe_shape_transfer.py` |
| Case-disjoint Derm7pt pilot | `run_derm7pt_case_disjoint.py` |
| Controlled image perturbation pilot | `run_derm7pt_perturbations.py` |
| Saved-head bias-only replay | `replay_bias_only.py`, `run_bias_only_remaining.py` |
| CLIP naming-candidate pilot | `run_shape_naming_pilot.py` |
| CSV contribution bars / full coordinate profile | `tools/plot_contributions.py` |
| Manuscript figure export for documentation | `tools/export_paper_figures.py` |

The naming pilot's example bank paths are in `EXEMPLARS`, with explicit candidate
phrases in the same file. Replace them with your own training references; it does
not call a VLM API or require an API secret. The figure-candidate and review
scripts are supporting export utilities and require their own recorded model,
reference assets and optional review files; those data-dependent files are not
bundled in this source release.

## Portable main training

```bash
python train.py --project-root /path/to/prepared_data --dataset derm7pt \
  --use_shape --shape_input geo --K_shape 128 --shape_mode decoded \
  --freeze_semantic_from semantic_checkpoints/derm7pt_semantic_baseline.pt
```

Original direct runners retain their historical CLI options. The `train.py`
wrapper changes the working data directory, then executes the same original
training implementation; it does not introduce a different model or loss.

