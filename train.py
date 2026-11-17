"""Portable alias for the original DCBM-platform shape-branch training entry."""
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
    script = Path(__file__).resolve().with_name("cancha_cub.py")
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

