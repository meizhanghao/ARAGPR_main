"""Resolve relocated feature caches without weakening their provenance checks."""
from pathlib import Path, PureWindowsPath
import os


def resolve_feature_file(value, *, anchor, dataset):
    """Find a recorded feature file; callers must still verify its saved hash.

    Relative paths are anchored at dataset.json. For a relocated cache, set
    ARAGPR_DATA_ROOT to the directory containing DATASET_split/. This also
    supports manifests produced on Windows and subsequently used on Linux.
    """
    normalized = str(value).replace("\\", "/")
    recorded = Path(normalized).expanduser()
    anchor = Path(anchor).resolve()
    filename = PureWindowsPath(normalized).name
    roots = []
    if os.environ.get("ARAGPR_DATA_ROOT"):
        roots.append(Path(os.environ["ARAGPR_DATA_ROOT"]).expanduser())
    roots.extend([Path.cwd(), Path(__file__).resolve().parents[1]])
    candidates = [recorded] if recorded.is_absolute() else [anchor / recorded]
    for root in roots:
        if not recorded.is_absolute() and not PureWindowsPath(normalized).is_absolute():
            candidates.append(root / recorded)
        candidates.append(root / f"{dataset}_split" / filename)
    for path in dict.fromkeys(candidates):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Feature file unavailable: {value}. Set ARAGPR_DATA_ROOT to the data "
        f"directory containing {dataset}_split/{filename}."
    )

