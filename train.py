"""Main training entry; implementation lives in :mod:`aragpr.training`."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", type=Path,
                        default=Path(os.environ.get("ARAGPR_DATA_ROOT", Path(__file__).resolve().parent)))
    known, remaining = parser.parse_known_args()
    root = known.project_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Project/data root does not exist: {root}")
    os.chdir(root)
    sys.argv = ["train.py", *remaining]
    runpy.run_module("aragpr.training", run_name="__main__")


if __name__ == "__main__":
    main()

