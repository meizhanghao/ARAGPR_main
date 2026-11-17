# Third-party attribution

## DCBM

The DCBM platform, related training/embedding utilities and modified descendants
originate from the official implementation by Katharina Prasse and Patrick Knab:

- Repository: https://github.com/KathPra/DCBM
- License: MIT; the upstream notice is retained in `licenses/DCBM-MIT.txt`.
- Paper: https://proceedings.mlr.press/v267/prasse25a.html

This snapshot includes local extensions for shape targets, residual fitting,
signed attribution, controls and exports. It is not an official DCBM release.
Local source hashes and the release adaptations are listed in
`docs/SOURCE_MANIFEST.json`; source-project folders and machine-specific paths
are omitted from that manifest.

## External models and libraries

- OpenAI CLIP: https://github.com/openai/CLIP
- SAM2: https://github.com/facebookresearch/sam2
- PyTorch: https://pytorch.org/
- NumPy, SciPy, scikit-learn, pandas, OpenCV, Pillow and Matplotlib are installed
  as dependencies and retain their respective licenses.

CLIP and SAM2 source trees and model weights are not vendored here. Installation
instructions refer to their official repositories. Their dependencies and model
weights remain subject to the terms supplied by the respective projects.

## Datasets

The software license does not license CUB-200-2011, Places365, Derm7pt, BUS-UCLM
or ISIC-2019 data. Obtain data from the original providers and comply with the
applicable licenses, access conditions and medical-image-use restrictions.

## Manuscript illustrations

`docs/assets/figures/` contains seven selected figures from the accompanying
manuscript, rendered as PNGs for the project documentation. The figures include
selected dataset image excerpts; the repository's software license does not
relicense those underlying images or permit redistribution of the datasets.
`docs/assets/figures/manifest.json` records source figure filenames, SHA256
hashes, and image dimensions. The export preserves the complete figure pages
without changing their panels, labels, or numerical values.

