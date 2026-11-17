"""Run new paired decoded/bias-only experiments with frozen semantic checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


from aragpr.paths import repository_root

ROOT = repository_root()
DATASETS = {
    "CUB_200_2011": ("cub", "cub_classes.txt"),
    "BUS_UCLM": ("bus_uclm", "BUS_UCLM_classes.txt"),
    "places365_part": ("places365_part", "places365_part_classes.txt"),
}


def build_command(dataset, mode, device, output_root):
    checkpoint = ROOT / "semantic_checkpoints" / f"{dataset}_semantic_baseline.pt"
    command = [
        sys.executable, "-u", str(ROOT / "train.py"),
        "--dataset", dataset, "--device", device,
        "--freeze_semantic_from", str(checkpoint),
        "--use_shape", "--shape_input", "geo", "--shape_mode", mode,
        "--seed", "42", "--K_sem", "2048", "--K_shape", "128",
        "--bs", "64", "--es_pat", "30",
        "--shape-standardize", "--shape-clustering", "kmeans",
        "--shape-prototype", "median_nearest",
        "--shape_decoder_lr", "0.001", "--shape_decoder_epochs", "300",
        "--eta_vals", "0.1", "0.25", "0.5", "0.75", "1.0",
        "--bias_only", "--bias_only_out", str(output_root),
    ]
    if dataset == "CUB_200_2011":
        command.extend([
            "--shape_asset_root",
            str(ROOT / "data/Segments/Seg_embs/CUB_200_2011_SAM2_binary_v3"),
            "--strict_shape_assets",
        ])
    return command


def check_inputs(dataset):
    stem, classes = DATASETS[dataset]
    bank = ROOT / "data/Segments/Seg_embs"
    required = [
        ROOT / "train.py", ROOT / "utils/bias_only.py",
        ROOT / "semantic_checkpoints" / f"{dataset}_semantic_baseline.pt",
        ROOT / "data/classes" / classes,
        bank / f"segcrop_{dataset}_SAM2_CLIP-ViT-L14.torch",
        *[ROOT / f"{dataset}_split/images_{dataset}_{split}_CLIP-ViT-L14.torch"
          for split in ("train", "val", "test")],
    ]
    if dataset == "CUB_200_2011":
        shape_root = bank / "CUB_200_2011_SAM2_binary_v3"
        required.extend([shape_root / "masks", shape_root / "crops",
                         shape_root / "shape_asset_audit.json"])
        # The training implementation generates a missing geo11 CSV from the retained binary masks.
    else:
        required.append(bank / "shape_csv" / f"{stem}_geo11_features.csv")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


def check_training_cli(commands):
    """Reject an older training entry point before creating any run directory."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--help"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
    )
    if result.returncode:
        raise RuntimeError("Cannot check the training entry point:\n" +
                           (result.stdout + result.stderr)[-4000:])
    supported = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", result.stdout))
    required = {arg for command in commands for arg in command if arg.startswith("--")}
    missing = sorted(required - supported)
    if missing:
        raise RuntimeError(
            "train.py does not support: " + ", ".join(missing) + ".\n"
            "Sync the current train.py, utils/bias_only.py and "
            "run_bias_only_remaining.py to the same project before retrying.\n"
            "No training or run-directory creation performed."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--shape-mode", choices=["decoded", "decoded_rel"], default="decoded")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bias_only_remaining_run01"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Check paths and training CLI compatibility without training.")
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Specify each dataset once.")
    output_root = (ROOT / args.output_root).resolve()
    if output_root.exists():
        parser.error(f"Preserve existing results; choose a new --output-root: {output_root}")
    jobs = []
    for dataset in args.datasets:
        check_inputs(dataset)
        command = build_command(dataset, args.shape_mode, args.device, output_root)
        jobs.append({"dataset": dataset, "command": command, "status": "pending"})
        display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        print(f"\n[{dataset}]\n{display}", flush=True)
    try:
        check_training_cli([job["command"] for job in jobs])
    except (OSError, RuntimeError) as error:
        parser.error(str(error))
    print("\nTraining entry-point options verified.", flush=True)
    if args.dry_run:
        print("\nPreflight passed. No training or output-directory creation performed.")
        return
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "launcher.json"
    manifest = {
        "scope": "New paired runs; historical table/trace results are not replaced.",
        "shape_mode": args.shape_mode,
        "frozen_checkpoints_sha256": {
            ds: hashlib.sha256((ROOT / "semantic_checkpoints" /
                               f"{ds}_semantic_baseline.pt").read_bytes()).hexdigest()
            for ds in args.datasets
        },
        "jobs": jobs,
    }
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for job in jobs:
        job["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with (output_root / f"{job['dataset']}.log").open("x", encoding="utf-8") as log:
            with subprocess.Popen(job["command"], cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        job.update(status="complete" if code == 0 else "failed", returncode=code)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if code:
            raise SystemExit(code)
    print(f"\nCompleted. Read each DATASET/{args.shape_mode}/metrics.csv under {output_root}")


if __name__ == "__main__":
    main()
