# Verification

The checks below cover implementation behavior, command-line interfaces, and
documentation assets. Dataset experiments require the inputs and checkpoints
described in [DATA.md](DATA.md).

## Run the checks

From the repository root:

```bash
python -m aragpr doctor
python -X utf8 -m unittest discover -s tests -v
python -m aragpr demo --output outputs/synthetic_demo
```

The synthetic demo exports all 128 coordinate contributions and verifies that
the coordinate terms, bias, and base margin sum to the final margin. The test
suite runs without downloading datasets or model weights.

## Recorded implementation results

The implementation checks performed on 2026-11-17 recorded:

- All 65 tests passed, including a run with CLIP, SAM2, wandb, and imbalanced-learn
  imports unavailable to check the core dependency boundary.
- 22 training, diagnostic, export, and plotting entry points returned CLI help.
  The main launcher also accepted a separate working data directory.
- All 83 Python source files passed syntax parsing.
- The synthetic example verified class-margin closure to floating-point precision.
- Matplotlib generated both a full 128-coordinate profile and Top-5 horizontal
  contribution bars in SVG format.

The environment for these checks is recorded in
[TEST_ENVIRONMENT.json](TEST_ENVIRONMENT.json). Continuous-integration results
are available in the repository's
[Actions tab](https://github.com/meizhanghao/ARAGPR_main/actions).

## Documentation and naming assets

The [visual guide](VISUAL_GUIDE.md) includes seven complete manuscript figures
and their code entry points. Figure source hashes, exported-image hashes, and
dimensions are recorded in [the figure manifest](assets/figures/manifest.json).
Figure exports and relative Markdown links were checked. Optional rendering
dependencies are listed in `requirements-docs.txt`.

The [naming guide](prototype_naming/README.md) provides a GPT-4o selection prompt
and UTF-8 CSV/JSON templates. Template JSON was checked for valid syntax. Naming
records are generated separately when the workflow is run on prepared references.

## File integrity and compatibility

- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records source attribution,
  origin hashes, and implementation adaptations.
- [RELEASE_FILES.sha256](RELEASE_FILES.sha256) lists SHA256 hashes for the
  distributed files, excluding itself and Git metadata.
- [FILE_MIGRATIONS.json](FILE_MIGRATIONS.json) maps earlier script filenames to
  their package paths. Model modules under `utils/` retain their paths for
  existing checkpoint loading.
