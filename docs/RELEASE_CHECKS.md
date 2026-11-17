# Source snapshot checks

Prepared on 2026-11-17. The local verification environment is recorded in
`TEST_ENVIRONMENT.json`; it is not a dependency lockfile.

- All 65 tests now reside under `tests/` and passed after reorganization.
  On Windows, use `python -X utf8 -m unittest discover -s tests -v` for
  the legacy Unicode log messages.
- The combined 65-test suite also passed with CLIP, SAM2, wandb and imbalanced-learn
  imports deliberately unavailable, checking the core-only dependency boundary.
- 22 moved/main training, diagnostic, export and plotting entry points
  returned CLI help. The main launcher also passed with a separate working
  data directory, leaving that directory unchanged.
- All 83 exported Python source files passed syntax parsing.
- The NumPy example exported all 128 synthetic coordinate contributions and
  verified class-margin closure to floating-point precision.
- Both the full 128-coordinate profile and Top-5 horizontal contribution bars
  were generated successfully using Matplotlib/SVG. Generated check outputs are
  removed from this source snapshot.
- Module imports, worker subprocess invocations and code-hash paths were updated
  to the new layout. Core model module paths under `utils/` were retained.
- Source provenance retains the original hashes and paths; current release
  hashes were refreshed. This reorganization only changed the public checkout.

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
The checks above were repeated locally for this reorganization where indicated.
Remote workflow results are reported separately in the Actions tab.

`SOURCE_MANIFEST.json` records the origin hashes and adaptations of copied files.
`RELEASE_FILES.sha256` lists the public source snapshot files, excluding itself
and local Git metadata.

## Auxiliary naming documentation

- Added a GPT-4o selection/refinement/abstention prompt and empty UTF-8 CSV/JSON
  record templates under `prototype_naming/`.
- Template JSON parsed successfully. The templates contain no claimed model
  execution date, backend version, CLIP score or VLM naming result.
- No VLM API was called during repository reorganization.
- `FILE_MIGRATIONS.json` maps the 40 relocated source files to their new paths.
