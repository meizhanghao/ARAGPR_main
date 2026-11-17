# Inputs and data preparation

Keep the original providers' access conditions, licenses, labels and case/patient
metadata. Obtain CUB-200-2011, Places365, Derm7pt, BUS-UCLM and ISIC-2019 from
the sources cited in the manuscript. Raw image collections and standalone
region/mask banks are not bundled. The documentation contains selected
manuscript figure montages, rather than downloadable dataset assets.

## Prepared-data layout

```text
PREPARED_ROOT/
  data/classes/
    cub_classes.txt
    derm7pt_classes.txt
    BUS_UCLM_classes.txt
    ISIC-2019_classes.txt
    places365_part_classes.txt
  DATASET_split/
    images_DATASET_train_CLIP-ViT-L14.torch
    images_DATASET_val_CLIP-ViT-L14.torch
    images_DATASET_test_CLIP-ViT-L14.torch
  semantic_checkpoints/
    DATASET_semantic_baseline.pt
  data/Segments/Seg_embs/
    DATASET_SAM2/                     # Semantic-region assets
      crops/
      masks/
    shape_csv/
      cub_geo11_features.csv
      derm7pt_geo11_features.csv
      bus_uclm_geo11_features.csv
      isic2019_geo11_features.csv
      places365_part_geo11_features.csv
    CUB_200_2011_SAM2_binary_v3/       # Strict paired CUB shape assets
      masks/
      crops/
      shape_asset_audit.json
```

`DATASET` must be one of `CUB_200_2011`, `derm7pt`, `BUS_UCLM`, `ISIC-2019`,
or `places365_part`. Copy the supplied class vocabularies into an external
prepared-data root when using `--project-root`.

The original DCBM training path additionally uses semantic segment embeddings,
segmentation statistics and its selected subset. The exact dataset-specific
paths are returned by `aragpr.training.load_dataset_info()`. A residual-only control
with an existing frozen checkpoint does not load the complete semantic segment
embedding collection.

### Image feature files

The `.torch` files use the existing dictionary format:

```python
{
    "class_directory/image.jpg": [feature_tensor, "class_name"],
    # ... one embedding per image, in a fixed class convention
}
```

Embeddings are normally 768-dimensional CLIP ViT-L/14 image features. Preserve
keys and label names across features, targets and output manifests. Preserve
full relative category paths for Places365; leaf names such as `indoor` can be
ambiguous. Training-fitted scalers are reused for validation and test images.

### Shape descriptor CSVs

Use `aragpr.training.get_shape_csv()` to construct descriptors from masks, including
the cleaned largest outer contour. The existing CSV format starts with a segment
identifier followed by the descriptor coordinates. Strict assets append paired
mask/crop relative paths and hashes. Descriptor dimensions are FD32, geo11,
Hu7, curv20 and defect7. The CLI spellings are listed by `train.py --help`.

The paired-control runner defaults to CUB's verified binary-v3 assets. To use a
separately prepared, verified CUB CSV, pass `--cub-shape-csv /path/to/file.csv`.
For the other datasets use `--shape-csv-dir`. Do not reconstruct a binary region
by thresholding an RGB crop that has lost the original segmentation mask.

The preprocessing routines are `scripts/preprocessing/main_segment_1.py`,
`scripts/preprocessing/main_image_emb_to_clip_4.py`, and `scripts/preprocessing/main_segimage_emb_to_clip_2.py`. These require
the optional CLIP/SAM2 dependencies and the original dataset metadata. They are
not run by importing the public attribution interface. SAM2 checkpoints are
downloaded only when segmentation is explicitly executed and weights are absent.

## Frozen base checkpoints

The DCBM-platform runners consume the saved semantic prototypes, training input
scaler, classifier state, class count, dataset and encoder metadata. Prefer
checkpoints produced by the matching baseline runner rather than constructing
partially populated dictionaries by hand.

Checkpoints and scaler files use the original research serialization, including
`torch.load(..., weights_only=False)`. Load trusted experiment files and preserve
their hashes. The repository contains no external model checkpoints.

## Derived teacher caches

`experiments/controls/run_residual_controls.py` creates a directory for each dataset containing:

- `dataset.json`: class order, source hashes, membership hashes and coverage;
- `input_scaler.pt` and `teacher_bank.pt`: training-fitted preprocessing/bank;
- `teacher_targets.npz`: train/val/test targets and coverage masks;
- `train_frozen.npz`, `val_frozen.npz`, `test_frozen.npz`: logits, labels and keys;
- `seed_N/MODE/`: newly fitted models, predictions and histories.

This is the cache consumed by the decoder baselines and pairing experiments.
Unavailable targets remain unavailable; they are not valid zero-valued targets.
If only non-geometric controls are run, their output will not contain a teacher
cache usable by geometry diagnostics.

For a relocated cache, relative feature paths are resolved against the manifest
directory. Alternatively, set `ARAGPR_DATA_ROOT` to `PREPARED_ROOT`; a stale
Windows feature path can then resolve to its matching `DATASET_split/` filename.
The loader still verifies the original recorded SHA256, image order and labels.
Missing or changed features are errors rather than silently substituted data.

## Splits and medical metadata

The original retained image partitions and a fresh case-disjoint experiment are
different studies. Do not label the image partitions as patient-disjoint.
`experiments/diagnostics/run_derm7pt_case_disjoint.py` requires the original Derm7pt case metadata and
rebuilds the affected training statistics for its new grouped split.
Keep any restricted image-membership or patient metadata outside the public repo.

