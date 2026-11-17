"""Run new paired geometry / random-basis / ordinary affine residual controls.

All test scores are computed after train/validation-only model selection.
Existing checkpoints, raw features, logs and manuscript numbers are read-only.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import csv

import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

from utils.dcbm_1 import GlobalCBM, ResidualTrainer, set_seed
from utils.residual_controls import (
    TrainingConfig, apply_affine_map, apply_residual, fit_residual,
    fit_teacher_decoder, frozen_semantic_logits, prediction_metrics,
    random_projection, standardize_splits,
)


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "CUB_200_2011": ("cub", "cub_classes.txt"),
    "places365_part": ("places365_part", "places365_part_classes.txt"),
    "derm7pt": ("derm7pt", "derm7pt_classes.txt"),
    "BUS_UCLM": ("bus_uclm", "BUS_UCLM_classes.txt"),
    "ISIC-2019": ("isic2019", "ISIC-2019_classes.txt"),
}
MODES = ("geometry", "random_projection", "affine", "shuffled_teacher")
SPLITS = ("train", "val", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_paths(root, dataset, args):
    stem, class_file = DATASETS[dataset]
    paths = {
        "checkpoint": root / "semantic_checkpoints" / f"{dataset}_semantic_baseline.pt",
        "classes": root / "data/classes" / class_file,
        **{s: root / f"{dataset}_split/images_{dataset}_{s}_CLIP-ViT-L14.torch"
           for s in SPLITS},
    }
    if set(args.modes) & {"geometry", "shuffled_teacher"}:
        if dataset == "CUB_200_2011" and args.cub_shape_csv is None:
            paths["cub_assets"] = root / "data/Segments/Seg_embs/CUB_200_2011_SAM2_binary_v3"
            paths["cub_audit"] = paths["cub_assets"] / "shape_asset_audit.json"
            paths["cub_masks"] = paths["cub_assets"] / "masks"
            paths["cub_crops"] = paths["cub_assets"] / "crops"
        else:
            paths["shape_csv"] = (args.cub_shape_csv if dataset == "CUB_200_2011"
                                  else args.shape_csv_dir / f"{stem}_geo11_features.csv")
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    return paths


def load_inputs(dataset, paths, device):
    cbm = GlobalCBM(str(paths["train"].parent), dataset, "CLIP-ViT-L14",
                    str(paths["classes"]), device)
    checkpoint = torch.load(paths["checkpoint"], weights_only=False, map_location="cpu")
    if checkpoint.get("dataset") != dataset or checkpoint.get("model_name") != "CLIP-ViT-L14":
        raise ValueError("Frozen checkpoint dataset/encoder mismatch.")
    nc = cbm.y_train_oh.shape[1]
    if checkpoint["num_classes"] != nc:
        raise ValueError("Frozen checkpoint class count mismatch.")
    cbm.K = int(checkpoint["K"])
    # Geometry setup maps CSV rows directly to image keys. It does not need
    # semantic segment embeddings; omit their potentially huge duplicate load.
    cbm.seg_to_img = []
    raw = {s: getattr(cbm, f"X_{s}").astype(np.float32) for s in SPLITS}
    if any(not np.isfinite(x).all() for x in raw.values()):
        raise ValueError("Nonfinite CLIP features after the existing dataset filters.")
    labels = {s: getattr(cbm, f"y_{s}_oh").argmax(1).astype(np.int64) for s in SPLITS}
    keys = {s: np.asarray(list(getattr(cbm, f"data_{s}_raw"))) for s in SPLITS}
    if any(len(keys[s]) != len(raw[s]) or len(labels[s]) != len(raw[s]) for s in SPLITS):
        raise ValueError("Image / label order mismatch.")
    semantic = {s: frozen_semantic_logits(raw[s], checkpoint, device) for s in SPLITS}
    h, h_scaler = standardize_splits(raw)
    del checkpoint
    return cbm, h, h_scaler, semantic, labels, keys


def verify_retained_logits(root, dataset, semantic, labels, keys):
    """Check previously exported frozen logits when available; never reuse scores."""
    candidates = [root / "outputs/bias_only_remaining_run01" / dataset / "decoded",
                  root / "outputs/bias_only_replay_20261018" / dataset]
    checks = []
    for base in candidates:
        for split in ("val", "test"):
            path = base / f"{split}_logits.npz"
            if not path.exists() and split == "test":
                path = base / "test_replay.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as saved:
                old_keys = saved["sample_keys"].astype(str).tolist()
                index = {k: i for i, k in enumerate(old_keys)}
                if len(index) != len(old_keys) or set(index) != set(keys[split]):
                    raise ValueError(f"Retained {split} membership mismatch: {path}")
                order = [index[str(k)] for k in keys[split]]
                if not np.array_equal(saved["labels"][order], labels[split]):
                    raise ValueError(f"Retained class order mismatch: {path}")
                previous = saved["semantic_logits"][order]
                error = float(np.max(np.abs(semantic[split] - previous)))
                matches = np.array_equal(semantic[split].argmax(1), previous.argmax(1))
                if error > 2e-4 or not matches:
                    raise ValueError(f"Frozen checkpoint fails retained logit verification: {path}, {error}")
                checks.append({"source": str(path), "sha256": sha256(path),
                               "max_absolute_error": error, "all_predictions_match": bool(matches)})
    return checks


def make_teacher(cbm, paths, destination, args):
    if "shape_csv" in paths:
        csv_path = paths["shape_csv"]
    else:
        from cancha_cub import get_shape_csv
        audit = json.loads(paths["cub_audit"].read_text(encoding="utf-8"))
        if not audit.get("asset_provenance_passed"):
            raise ValueError("CUB paired binary asset audit has not passed.")
        # All generated descriptors stay inside this new experiment directory.
        csv_path, _, _ = get_shape_csv(
            str(paths["cub_masks"]), str(destination / "cub_fd32_features.csv"),
            "geo", strict_assets=True)
        csv_path = Path(csv_path)
    rt = ResidualTrainer(cbm, args.device)
    rt.setup_shape_teacher(str(csv_path), K_shape=args.shape_dim, fd_dim=11,
                           shape_mode="decoded", shape_standardize=True,
                           shape_clustering="kmeans", shape_prototype="median_nearest")
    teachers = {s: getattr(rt, f"Ash_{s}_s").copy() for s in SPLITS}
    covered = {s: getattr(rt, f"has_shape_{s}").copy() for s in SPLITS}
    state = {"shape_csv": str(csv_path), "shape_csv_sha256": sha256(csv_path),
             "shape_prototypes": rt.sp, "descriptor_scaler": rt.shape_feat_scaler,
             "teacher_scaler": rt.shape_scaler,
             "prototype_metadata": rt.shape_proto_metadata,
             "prototype_representatives": rt.shape_proto_representative}
    torch.save(state, destination / "teacher_bank.pt")
    np.savez_compressed(destination / "teacher_targets.npz",
                        **{f"{s}_targets": teachers[s] for s in SPLITS},
                        **{f"{s}_covered": covered[s] for s in SPLITS})
    return teachers, covered, state["shape_csv"], state["shape_csv_sha256"]


def prepare_branch(mode, h, teachers, covered, seed, args, folder):
    metadata = {"mode": mode, "seed": seed}
    state = {}
    if mode == "affine":
        coordinates = h
        metadata.update(geometry_supervision=False, fixed_projection_parameters=0)
    elif mode == "random_projection":
        projection = random_projection(h["train"].shape[1], args.shape_dim, seed)
        state = {"projection": projection}
        coordinates = {s: x @ projection.T for s, x in h.items()}
        metadata.update(geometry_supervision=False,
                        fixed_projection_parameters=int(projection.size + args.shape_dim))
    else:
        targets = {s: teachers[s][covered[s]] for s in ("train", "val")}
        if mode == "shuffled_teacher":
            # Separate permutations within each split; no train/val exchanges.
            permutations = {s: np.random.default_rng(seed + 3000001 + i).permutation(len(y))
                            for i, (s, y) in enumerate(targets.items())}
            targets = {s: y[permutations[s]] for s, y in targets.items()}
            np.savez_compressed(folder / "teacher_permutations.npz", **permutations)
        fitted = fit_teacher_decoder(
            h["train"][covered["train"]], targets["train"],
            h["val"][covered["val"]], targets["val"], seed=seed,
            device=args.device, epochs=args.decoder_epochs, patience=args.patience,
            batch_size=args.batch_size,
            progress=lambda message: print(f"[{mode} seed={seed}] {message}", flush=True))
        state = {"decoder_state": fitted["state"]}
        coordinates = apply_affine_map(h, fitted["state"], args.device)
        write_csv(folder / "decoder_history.csv", fitted["history"])
        metadata.update(geometry_supervision=(mode == "geometry"),
                        shuffled_geometry_supervision=(mode == "shuffled_teacher"),
                        decoder_best_epoch=fitted["best_epoch"],
                        decoder_val_loss=fitted["val_loss"],
                        fixed_projection_parameters=int((h["train"].shape[1] + 1) * args.shape_dim))
    # Same train-only coordinate scaling for all controls: L1 cannot be
    # weakened/strengthened merely by choosing arbitrary projection units.
    if args.coordinate_standardize and mode != "affine":
        coordinates, scaler = standardize_splits(coordinates)
        state["coordinate_scaler"] = scaler
    metadata["coordinate_standardize"] = args.coordinate_standardize or mode == "affine"
    return coordinates, state, metadata


def run_dataset(dataset, paths, root, out, args, config):
    started = time.monotonic()
    folder = out / dataset
    folder.mkdir()
    cbm, h, h_scaler, semantic, labels, keys = load_inputs(dataset, paths, args.device)
    checks = verify_retained_logits(root, dataset, semantic, labels, keys)
    memberships = {s: {"n": len(keys[s]), "keys_sha256": hashlib.sha256(
        json.dumps(keys[s].tolist(), ensure_ascii=False).encode("utf-8")).hexdigest()}
        for s in SPLITS}
    sources = {name: {"path": str(path), "sha256": sha256(path)}
               for name, path in paths.items() if path.is_file()}
    manifest = {"dataset": dataset, "sources": sources, "split_membership": memberships,
                "retained_frozen_verification": checks,
                "class_order": sorted(np.unique(cbm.labels_train).tolist()),
                "frozen_metrics": {s: prediction_metrics(semantic[s], labels[s], semantic[s])
                                   for s in SPLITS}, "completed": False}
    torch.save({"h_scaler": h_scaler, "sources": sources}, folder / "input_scaler.pt")
    for split in SPLITS:
        np.savez_compressed(folder / f"{split}_frozen.npz", logits=semantic[split],
                            labels=labels[split], sample_keys=keys[split])
    teachers = covered = None
    if set(args.modes) & {"geometry", "shuffled_teacher"}:
        teachers, covered, csv_path, csv_hash = make_teacher(cbm, paths, folder, args)
        manifest["shape_csv"] = {"path": csv_path, "sha256": csv_hash}
        manifest["teacher_coverage"] = {s: int(covered[s].sum()) for s in SPLITS}
    write_json(folder / "dataset.json", manifest)
    rows = []
    for seed in args.seeds:
        for mode in args.modes:
            branch = folder / f"seed_{seed}" / mode
            branch.mkdir(parents=True)
            coordinates, branch_state, metadata = prepare_branch(
                mode, h, teachers, covered, seed, args, branch)
            fitted = fit_residual(
                coordinates["train"], semantic["train"], labels["train"],
                coordinates["val"], semantic["val"], labels["val"],
                config=config, seed=seed, device=args.device,
                progress=lambda message: print(f"[{dataset} {mode} seed={seed}] {message}", flush=True))
            write_csv(branch / "head_history.csv", fitted["history"])
            write_csv(branch / "eta_selection.csv", fitted["candidates"])
            selected = {k: v for k, v in fitted["selected"].items() if k != "state"}
            report = {**metadata, "selected": selected,
                      "head_trainable_parameters": fitted["trainable_parameters"],
                      "selection": "Maximum validation accuracy, first epoch/eta on ties; "
                                   "test arrays excluded from fitting and selection.", "metrics": {}}
            for split in ("val", "test"):
                scores = apply_residual(coordinates[split], semantic[split], fitted, args.device)
                metrics = prediction_metrics(scores, labels[split], semantic[split])
                report["metrics"][split] = metrics
                np.savez_compressed(branch / f"{split}_predictions.npz", logits=scores,
                                    labels=labels[split], sample_keys=keys[split])
                rows.append({"dataset": dataset, "seed": seed, "mode": mode, "split": split,
                             "eta": selected["eta"], "best_epoch": selected["best_epoch"],
                             "head_parameters": fitted["trainable_parameters"], **metrics})
            torch.save({**branch_state, "head_state": fitted["selected"]["state"],
                        "eta": selected["eta"], "metadata": metadata}, branch / "selected_model.pt")
            write_json(branch / "result.json", report)
            print(f"[DONE {dataset} {mode} seed={seed}] "
                  f"test={report['metrics']['test']['accuracy_pct']:.4f}% "
                  f"delta={report['metrics']['test']['delta_frozen_pp']:+.4f} pp", flush=True)
            write_csv(folder / "metrics.csv", rows)
            del coordinates, fitted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    manifest.update(completed=True, elapsed_seconds=time.monotonic()-started)
    write_json(folder / "dataset.json", manifest)
    return rows


def summarize(out, all_rows, args):
    write_csv(out / "metrics.csv", all_rows)
    lines = ["# Paired residual controls", "",
             "New experiments using the retained local splits and semantic checkpoints. "
             "These values do not replace manuscript main-table or Appendix E values.", "",
             "| Dataset | Frozen | " + " | ".join(args.modes) + " |",
             "|---|---:|" + "---:|" * len(args.modes)]
    summary = []
    for dataset in args.datasets:
        path = out / dataset / "dataset.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        frozen = manifest["frozen_metrics"]["test"]["accuracy_pct"]
        cells = []
        for mode in args.modes:
            values = [r["accuracy_pct"] for r in all_rows
                      if r["dataset"] == dataset and r["mode"] == mode and r["split"] == "test"]
            if not values:
                cells.append("pending")
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            cells.append(f"{mean:.2f}" + (f" ± {sd:.2f}" if sd is not None else ""))
            summary.append({"dataset": dataset, "mode": mode, "n_seeds": len(values),
                            "frozen_accuracy_pct": frozen, "mean_accuracy_pct": mean,
                            "sd_accuracy_pp": sd, "mean_delta_frozen_pp": mean-frozen})
        lines.append(f"| {dataset} | {frozen:.2f} | " + " | ".join(cells) + " |")
    lines += ["", "With multiple seeds, ± is the sample standard deviation across independent "
              "branch fits. The semantic checkpoint, membership, and teacher bank stay fixed. "
              "With one seed, no uncertainty estimate is reported.", "",
              "Geometry and random projection use the same coordinate dimension, trained head "
              "size, initialization, minibatch order, L1 coefficient, eta grid, and stopping rule. "
              "The direct affine control has a larger head; its parameter count is reported.", "",
              "Coordinate units are standardized on training images by default for all branches. "
              "The geometry decoder uses train/validation geometry only; random projection and "
              "direct affine controls never use geometry teachers. Test masks are not inference inputs.", "",
              "Derm7pt uses the retained image split. Case overlap remains; these results "
              "do not establish unseen-case generalization.", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(out / "summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES[:3]))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shape-csv-dir", type=Path, default=None)
    parser.add_argument("--cub-shape-csv", type=Path, default=None,
                        help="Explicit CSV override; otherwise rebuild CUB from audited binary-v3 assets.")
    parser.add_argument("--shape-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--decoder-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eta-values", type=float, nargs="+", default=[.1, .25, .5, .75, 1.])
    parser.add_argument("--coordinate-standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    args.shape_csv_dir = (root / (args.shape_csv_dir or "data/Segments/Seg_embs/shape_csv")).resolve()
    if args.cub_shape_csv is not None:
        args.cub_shape_csv = (root / args.cub_shape_csv).resolve()
    if out.exists():
        parser.error(f"Preserve existing results; select a new output directory: {out}")
    if any(len(x) != len(set(x)) for x in (args.datasets, args.modes, args.seeds)):
        parser.error("Datasets, modes and seeds must not repeat.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must lie in [0, 2**32).")
    if not 1 <= args.shape_dim <= 768 or args.threads < 1 or args.decoder_epochs < 1:
        parser.error("Invalid coordinate dimension, thread count, or decoder epochs.")
    config = TrainingConfig(args.epochs, args.patience, args.batch_size,
                            eta_values=tuple(args.eta_values))
    try:
        config.validate()
        paths = {ds: input_paths(root, ds, args) for ds in args.datasets}
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available.")
    print(json.dumps({"datasets": args.datasets, "modes": args.modes, "seeds": args.seeds,
                      "output_root": str(out), "training": asdict(config),
                      "decoder_epochs": args.decoder_epochs,
                      "coordinate_standardize": args.coordinate_standardize}, indent=2), flush=True)
    if args.dry_run:
        print("Preflight passed. No training or directory creation performed.")
        return
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(args.seeds[0])
    code = [Path(__file__), ROOT / "utils/residual_controls.py", ROOT / "utils/dcbm_1.py",
            ROOT / "cancha_cub.py"]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "code_sha256": {str(p): sha256(p) for p in code},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "sklearn": sklearn.__version__,
                                "device": args.device}, "status": "running", "completed_datasets": []}
    write_json(out / "run.json", manifest)
    all_rows = []
    try:
        with threadpool_limits(limits=args.threads):
            for dataset in args.datasets:
                all_rows.extend(run_dataset(dataset, paths[dataset], root, out, args, config))
                manifest["completed_datasets"].append(dataset)
                write_json(out / "run.json", manifest)
                summarize(out, all_rows, args)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "run.json", manifest)


if __name__ == "__main__":
    main()
