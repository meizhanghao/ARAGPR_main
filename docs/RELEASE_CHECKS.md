# Source snapshot checks

Prepared on 2026-11-17. The local verification environment is recorded in
`TEST_ENVIRONMENT.json`; it is not a dependency lockfile.

- 56 tests under `tests/` passed.
- 9 mechanism/pairing invariant tests passed.
- The combined 65-test suite also passed with CLIP, SAM2, wandb and imbalanced-learn
  imports deliberately unavailable, checking the core-only dependency boundary.
- 12 primary training, diagnostic and plotting entry points returned CLI help.
- Every exported Python source file passed syntax parsing.
- The NumPy example exported all 128 synthetic coordinate contributions and
  verified class-margin closure to floating-point precision.
- Both the full 128-coordinate profile and Top-5 horizontal contribution bars
  were generated successfully using Matplotlib/SVG. Generated check outputs are
  removed from this source snapshot.
- The exported text was scanned for the source machine's private paths and
  common credential/private-key formats. No such findings remained.
- Original source-file SHA256 values were checked again after preparing the
  export; the source project was not modified.

## Documentation figures

- Seven standalone manuscript figure PDFs were rendered to complete-page PNGs.
  Their source hashes remained unchanged; image hashes and dimensions are in
  `assets/figures/manifest.json`.
- The exported figures were visually inspected. The README embeds the framework,
  selected prototype clusters, and complete-profile/Top-5 audit; the extended
  `VISUAL_GUIDE.md` explains all seven figures and their code entry points.
- Relative Markdown links and image targets were checked locally.
- Only the seven named figure PNGs are allowed by `.gitignore`; unrelated image
  files and raw data banks remain excluded.
- The optional figure-export CLI and Python syntax were checked. Documentation
  rendering dependencies are listed separately in `requirements-docs.txt`.

These are implementation and packaging checks, not new dataset experiments or
independent reproduction of the manuscript's server-reported accuracy rows.
Full training requires the documented datasets, embeddings, masks and checkpoints.
These checks were completed locally before the initial GitHub upload. Remote
workflow results are reported separately in the repository's Actions tab.

`SOURCE_MANIFEST.json` records the origin hashes and adaptations of copied files.
`RELEASE_FILES.sha256` lists the public source snapshot files, excluding itself
and local Git metadata.
