# ARAGPR

**Auditing Residual Attribution with Geometric Prototype References**

Research code for contour-supervised residual coordinates, signed class-margin
attribution, and correspondence diagnostics. The branch takes frozen image
embeddings and base logits. DCBM is the main experimental platform; a separate
runner uses an ordinary frozen CLIP linear classifier.

[中文使用说明](docs/README_zh-CN.md) · [Data preparation](docs/DATA.md) ·
[Experiment protocols](docs/PROTOCOLS.md) · [Entry points](docs/ENTRY_POINTS.md) ·
[Visual guide](docs/VISUAL_GUIDE.md) ·
[GitHub repository](https://github.com/meizhanghao/ARAGPR_main)

## Method

Training-region contours define a prototype bank and image-level response
targets. An affine decoder predicts these coordinates from frozen image
embeddings. A fitted head adds `eta * (W @ u + b)` to the base logits.
For a fixed class contrast, attribution separates the base margin, every signed
coordinate term, and the head bias. Retained training regions and contours are
visual references; they do not localize evidence in a test image.

`decoded` inference uses image embeddings without test masks. `oracle` is the
legacy command-line name for **Direct teacher**, a mask-dependent diagnostic.
`decoded_rel` is the optional validation-correlation-weighted variant.
Legacy names such as `cancha_cub.py`, `GlobalCBM`, and `SDR-CBM` remain in Python
identifiers and supporting scripts for compatibility with saved checkpoints.

### Architecture: learn coordinates, then inspect the added scores

![Manuscript Figure 2: frozen classifier, shape decoder, residual head, and attribution readout](docs/assets/figures/framework.png)

During training, region contours define shape prototypes and response targets
that supervise the decoder. At inference, the frozen image embedding supplies
the decoded responses; segmentation masks are no longer needed. The residual
head supplies both additional class scores and a contribution for each
prototype coordinate in a chosen class contrast.

### Prototype clusters: inspect the training regions behind a coordinate

![Manuscript Figure 4: auxiliary naming workflow and six selected prototype clusters from CUB, Derm7pt, and Places365](docs/assets/figures/prototype_clusters.png)

The six grids show two clusters per dataset, with nine selected training regions
in each grid. The central red box marks the actual representative. Original
images, region crops, and contours reveal what a coordinate refers to: for
example, leg-foot profiles, lesion outlines, or rectangular artwork. These
selected members illustrate reference content; they do not measure purity
across the complete cluster. Names are auxiliary reading aids.

### Attribution readout: connect the full profile to Top-5 references

![Manuscript Figure 5: complete 128-coordinate profiles and Top-5 signed attributions for Bank Swallow and dorm room](docs/assets/figures/contribution_audit.png)

Each row links a complete 128-coordinate profile to the five largest absolute
contributions and their training references, in the same top-to-bottom order.
These unchanged correct predictions use explicit contrasts: Bank Swallow versus
Sayornis, and dorm room versus child's room. Positive and negative values favor
opposite sides of that contrast; their magnitudes depend on both the response
and the fitted head weights. The coordinate plots omit base margins and bias.

See the [visual guide](docs/VISUAL_GUIDE.md) for the residual-interface diagram,
corrected and harmful prediction changes, validation dynamics, and the code
entry points associated with each view. Click any figure to inspect its full
resolution.

## Installation

Use Python 3.10 or 3.11 in a separate environment. Install a PyTorch build
appropriate for your CPU/CUDA environment, then install the core dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m aragpr doctor
```

CLIP image/text encoding, SAM2 segmentation, and older baseline scripts have
additional dependencies in `requirements-optional.txt`. Existing embeddings and
masks can be reused without installing or running SAM2. SAM2 is installed from
its official repository; a bundled copy of its source and weights is not included.

## Quick check without datasets

```bash
python -m unittest discover -s tests -v
python -m unittest test_shape_mechanism test_cub_pairing_controls -v
python -m aragpr demo --output outputs/synthetic_demo
```

The demo exports an explicitly synthetic 128-coordinate CSV and verifies margin
closure. It does not report experimental accuracy or reproduce a manuscript row.

The public NumPy interface also accepts logits from other frozen classifiers:

```python
from aragpr.attribution import class_margin_attribution

audit = class_margin_attribution(base_logits, coordinates, W, b, eta=0.5)
q = audit["coordinate_contributions"]  # images x prototype coordinates
```

The coordinates must use the same training-fitted transformations as the head.
Default contrasts compare each final prediction against its base prediction;
unchanged predictions have a zero contrast. Explicit class indices are supported.

## Prepare inputs

Obtain the datasets from their original providers and prepare the files described
in [docs/DATA.md](docs/DATA.md). Raw datasets, standalone region/mask banks,
embeddings, model weights, experiment run artifacts, and source-project Git
history are not bundled. The selected manuscript illustrations under
`docs/assets/figures/` are documentation assets. Five class vocabularies are
supplied under `data/classes/`.

The default layout is relative to the repository root. To keep data elsewhere,
pass `--project-root /path/to/prepared_data` to `train.py` and
`run_residual_controls.py`. For relocated cached feature manifests, set
`ARAGPR_DATA_ROOT` to that same directory; recorded feature SHA256 checks remain
active.

## Train the DCBM-platform branch

Run from the repository root. The `geo` CLI value selects the 11-dimensional
`geo11` descriptor. The examples assume prepared inputs and a frozen base model:

```bash
python train.py --dataset CUB_200_2011 --use_shape --shape_input geo \
  --K_shape 128 --shape_mode decoded \
  --freeze_semantic_from semantic_checkpoints/CUB_200_2011_semantic_baseline.pt \
  --pinggu --explain_trace

python train.py --dataset derm7pt --use_shape --shape_input geo \
  --K_shape 128 --shape_mode decoded \
  --freeze_semantic_from semantic_checkpoints/derm7pt_semantic_baseline.pt
```

Use `python train.py --help` for the original training options, including the
explanation-output directory and sample indices. On PowerShell, put commands on
one line or use PowerShell's continuation character rather than Bash `\`.
Without `--freeze_semantic_from`, the original runner fits a semantic base and
writes its checkpoint; use a separate working data directory for a new base fit.

## Residual controls and cached diagnostics

```bash
# Preflight: input checks only; no fitting or output directory creation.
python run_residual_controls.py --datasets CUB_200_2011 derm7pt \
  --modes geometry affine --seeds 42 --output-root outputs/controls --dry-run

# Fresh paired residual fits; also create the teacher caches used below.
python run_residual_controls.py --datasets CUB_200_2011 derm7pt \
  --modes geometry random_projection affine shuffled_teacher \
  --seeds 42 43 44 --output-root outputs/controls

python run_decoder_baselines.py --cache-root outputs/controls \
  --datasets CUB_200_2011 derm7pt --output-root outputs/decoder_baselines

python run_cub_pairing_controls.py --cache-root outputs/controls \
  --output outputs/cub_pairing
python run_derm7pt_mechanism.py --cache-root outputs/controls \
  --output outputs/derm_pairing

# Use the model paths produced by those pairing runs.
python run_posthoc_shape_diagnostics.py --help
python run_teacher_alignment_audit.py --help
python run_fixed_head_within_class.py --help
python run_linear_probe_shape_transfer.py --help
```

Choose a new output directory for each diagnostic run. Historical default names
in supporting runners identify the original cache layout, not downloadable
bundled experiments. Pass explicit paths when reproducing a new experiment.
The ordinary linear-base runner uses a fitted probe and a saved shape decoder;
it does not require a semantic concept bank.

## Reporting and scope

The main manuscript reports final-15-epoch accuracy means; several diagnostics
instead use validation-selected, fixed checkpoints and independent branch seeds.
These are different aggregation protocols. The provided diagnostic summaries
must not be relabeled as the main table's epoch means. Checkpoint sequences and
server experiment artifacts are not bundled with this source release.

Test targets are used only for the explicitly mask-dependent diagnostics.
Keep the head, scale, standardization, and class contrast fixed for Teacher
replacement; the separately fitted Direct teacher head is a different control.
The retained medical partitions have the limitations discussed in the manuscript;
`run_derm7pt_case_disjoint.py` provides a separate case-group experiment.

L1 regularization does not guarantee exact zeros. Algebraic closure and visual
references do not, by themselves, establish a purely geometric causal mechanism.
Auxiliary prototype names have no role in training, prediction, or attribution
arithmetic. The naming pilot supplies CLIP candidate rankings; choosing a VLM
and retaining its prompts/outputs is a separate naming step.

## Project layout

```text
aragpr/                  Public attribution and portable-path interfaces
train.py                 Portable alias for cancha_cub.py
utils/                   Existing model, controls and visualization implementation
run_*.py                 Experiment and explanation-diagnostic runners
data/classes/            Dataset class vocabularies
data/Segments/scripts/   Region/embedding adapters, without vendored frameworks
tests/                   Dataset-free regression tests
docs/                    Input formats, protocols, entry points and source provenance
docs/assets/figures/     Selected manuscript illustrations and their source hashes
```

## Citation and acknowledgements

Use `CITATION.cff` for the current manuscript title and authors. Publication
identifiers will be added when available. The DCBM platform and data utilities
adapt the [official DCBM implementation](https://github.com/KathPra/DCBM):

```bibtex
@inproceedings{prasse2025dcbm,
  title = {{DCBM}: Data-Efficient Visual Concept Bottleneck Models},
  author = {Prasse, Katharina and Knab, Patrick and Marton, Sascha and
            Bartelt, Christian and Keuper, Margret},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {267}, pages = {49752--49782}, year = {2025},
  url = {https://proceedings.mlr.press/v267/prasse25a.html}
}
```

See `LICENSE` and `THIRD_PARTY_NOTICES.md` for attribution and dependency notices.
The repository license does not grant rights to redistribute the datasets or
third-party model weights. Contact: zqmeng@gxu.edu.cn.

This work was supported by the National Natural Science Foundation of China
(Grant Nos. 62266004 and 62666007).

