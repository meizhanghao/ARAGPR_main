"""Dependency inspection and a synthetic attribution example."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path


def doctor():
    required = ["numpy", "torch", "pandas", "scipy", "scikit-learn",
                "opencv-python", "Pillow", "matplotlib", "threadpoolctl", "tqdm"]
    packages = {}
    for name in required:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    print(json.dumps({"dependencies": packages,
                      "all_core_dependencies_found": all(packages.values()),
                      "optional": "CLIP/SAM2 are needed only for feature/mask extraction."}, indent=2))
    return 0 if all(packages.values()) else 1


def demo(output):
    import numpy as np
    from .attribution import class_margin_attribution
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    u = rng.normal(size=(1, 128))
    w = rng.normal(scale=.04, size=(3, 128))
    b = np.array([.02, -.01, .03])
    base = np.array([[.7, .5, -.2]])
    # A fixed contrast keeps the example nontrivial even if the prediction stays.
    result = class_margin_attribution(base, u, w, b, eta=.5,
        reference_classes=np.array([0]), target_classes=np.array([1]))
    q = result["coordinate_contributions"][0]
    with (output / "contributions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prototype", "contribution"])
        writer.writeheader()
        writer.writerows({"prototype": f"P{k}", "contribution": float(v)}
                         for k, v in enumerate(q))
    summary = {"synthetic_example_only": True, "seed": 42, "n_coordinates": 128,
               "base_margin": float(result["base_margin"][0]),
               "coordinate_sum": float(q.sum()),
               "bias_contribution": float(result["bias_contribution"][0]),
               "final_margin": float(result["final_margin"][0]),
               "closure_error": float(abs(result["reconstructed_margin"][0]
                                          - result["final_margin"][0]))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect installed core dependencies.")
    d = sub.add_parser("demo", help="Export a synthetic 128-coordinate attribution.")
    d.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(doctor() if args.command == "doctor" else demo(args.output))


if __name__ == "__main__":
    main()

