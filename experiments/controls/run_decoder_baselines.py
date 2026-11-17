"""Evaluate constant, paired and shuffled-target shape decoders on cached teachers.

New decoder fits use train/validation data only. Existing teachers, features,
checkpoints and manuscript results are read-only. No classifier is retrained.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from aragpr.paths import resolve_feature_file
import platform
import sys
import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from utils.residual_controls import fit_teacher_decoder, apply_affine_map


from aragpr.paths import repository_root

ROOT = repository_root()
SPLITS = ("train", "val", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def fidelity_metrics(target, prediction):
    """Uniform coordinate R2; explicitly separate undefined correlations."""
    y, p = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or len(y) < 2:
        raise ValueError("Fidelity needs aligned matrices with at least two images.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite reconstruction arrays.")
    mse = np.mean((p - y) ** 2, axis=0)
    r2 = r2_score(y, p, multioutput="raw_values", force_finite=True)
    yc, pc = y - y.mean(0), p - p.mean(0)
    yn, pn = np.sqrt(np.sum(yc ** 2, axis=0)), np.sqrt(np.sum(pc ** 2, axis=0))
    # ptp also detects a truly constant predictor despite mean-rounding noise.
    defined = (np.ptp(y, axis=0) > 1e-12) & (np.ptp(p, axis=0) > 1e-12)
    corr = np.full(y.shape[1], np.nan)
    corr[defined] = np.sum(yc[:, defined] * pc[:, defined], axis=0) / (yn[defined] * pn[defined])
    corr[defined] = np.clip(corr[defined], -1.0, 1.0)
    report = {"n_valid": len(y), "target_dimension": y.shape[1], "mse": float(mse.mean()),
              "r2_uniform": float(np.mean(r2)),
              "pearson_defined_mean": float(corr[defined].mean()) if defined.any() else None,
              "pearson_paper_zero_undefined": float(np.nan_to_num(corr, nan=0.0).mean()),
              "pearson_defined_coordinates": int(defined.sum()),
              "pearson_undefined_coordinates": int((~defined).sum())}
    return report, {"mse": mse, "r2": r2, "pearson": corr}


def load_cached_inputs(cache, dataset):
    folder = cache / dataset
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset:
        raise ValueError("Dataset manifest mismatch.")
    sources = {}
    for name in ("dataset.json", "teacher_targets.npz", "teacher_bank.pt", "input_scaler.pt"):
        path = folder / name
        sources[name] = {"path": str(path), "sha256": sha256(path)}
    scaler_state = torch.load(folder / "input_scaler.pt", map_location="cpu", weights_only=False)
    scaler = scaler_state["h_scaler"]
    with np.load(folder / "teacher_targets.npz", allow_pickle=False) as z:
        teacher = {s: z[s + "_targets"].copy() for s in SPLITS}
        valid = {s: z[s + "_covered"].astype(bool) for s in SPLITS}
    features, keys, labels = {}, {}, {}
    for split in SPLITS:
        source = manifest["sources"][split]
        path = resolve_feature_file(source["path"], anchor=folder, dataset=dataset)
        actual_hash = sha256(path)
        if actual_hash != source["sha256"] or scaler_state["sources"][split]["sha256"] != actual_hash:
            raise ValueError(f"Cached feature provenance mismatch: {dataset}/{split}")
        sources[split + "_features"] = {"path": str(path), "sha256": actual_hash}
        frozen_path = folder / f"{split}_frozen.npz"
        sources[split + "_membership"] = {"path": str(frozen_path), "sha256": sha256(frozen_path)}
        with np.load(frozen_path, allow_pickle=False) as z:
            keys[split], labels[split] = z["sample_keys"].astype(str), z["labels"].copy()
        membership_hash = hashlib.sha256(json.dumps(keys[split].tolist(),
            ensure_ascii=False).encode("utf-8")).hexdigest()
        if membership_hash != manifest["split_membership"][split]["keys_sha256"]:
            raise ValueError("Sample order differs from the teacher cache manifest.")
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if len(set(keys[split])) != len(keys[split]) or any(k not in raw for k in keys[split]):
            raise ValueError("Missing or duplicate feature keys.")
        x = np.stack([np.asarray(raw[k][0].cpu()).reshape(-1) for k in keys[split]]).astype(np.float32)
        class_names = [raw[k][1] for k in keys[split]]
        if class_names != [manifest["class_order"][int(y)] for y in labels[split]]:
            raise ValueError("Cached class order differs from raw features.")
        if not np.isfinite(x).all() or teacher[split].shape[0] != len(x) or valid[split].shape != (len(x),):
            raise ValueError("Nonfinite or misaligned cached inputs.")
        if int(valid[split].sum()) != manifest["teacher_coverage"][split]:
            raise ValueError("Teacher coverage differs from cache manifest.")
        if not np.isfinite(teacher[split][valid[split]]).all():
            raise ValueError("Nonfinite valid teachers.")
        if split == "train":
            independent = StandardScaler().fit(x)
            np.testing.assert_allclose(scaler.mean_, independent.mean_, rtol=0, atol=1e-10)
            np.testing.assert_allclose(scaler.scale_, independent.scale_, rtol=0, atol=1e-10)
        features[split] = scaler.transform(x).astype(np.float32)
        del raw
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(keys[left]) & set(keys[right]):
            raise ValueError("Image keys overlap across splits.")
    return features, teacher, valid, keys, labels, manifest, sources


def coverage_rows(dataset, labels, valid, classes):
    result = []
    for split in SPLITS:
        for index in [None, *range(len(classes))]:
            members = np.ones(len(labels[split]), bool) if index is None else labels[split] == index
            n, nv = int(members.sum()), int((members & valid[split]).sum())
            result.append({"dataset": dataset, "split": split,
                "class": "ALL" if index is None else classes[index], "n_images": n,
                "n_valid_teacher": nv, "n_invalid_teacher": n - nv,
                "valid_teacher_pct": 100.0 * nv / n if n else None})
    return result


def export_existing_classifier_groups(cache, dataset, valid, keys, labels, seed, destination):
    """Optional replay of an existing selected classifier, separate from new decoder fits."""
    rows, provenance = [], {}
    model = cache / dataset / f"seed_{seed}/geometry/test_predictions.npz"
    if not model.exists():
        return rows, {"status": "unavailable", "path": str(model)}
    frozen_path = cache / dataset / "test_frozen.npz"
    with np.load(model, allow_pickle=False) as pred, np.load(frozen_path, allow_pickle=False) as base:
        if not np.array_equal(pred["sample_keys"].astype(str), keys["test"]):
            raise ValueError("Selected classifier membership mismatch.")
        if not np.array_equal(pred["labels"], labels["test"]):
            raise ValueError("Selected classifier label mismatch.")
        yp, yb = pred["logits"].argmax(1), base["logits"].argmax(1)
    for name, mask in [("all", np.ones(len(yp), bool)), ("valid_teacher", valid["test"]),
                       ("invalid_teacher", ~valid["test"])]:
        n = int(mask.sum())
        nbase = int((yb[mask] == labels["test"][mask]).sum())
        ndecoded = int((yp[mask] == labels["test"][mask]).sum())
        rows.append({"dataset": dataset, "subset": name, "n": n,
            "frozen_correct": nbase, "decoded_correct": ndecoded,
            "frozen_accuracy_pct": 100.0 * nbase / n if n else None,
            "decoded_accuracy_pct": 100.0 * ndecoded / n if n else None,
            "delta_frozen_pp": 100.0 * (ndecoded - nbase) / n if n else None,
            "protocol": "existing_local_validation_selected_single_checkpoint"})
    provenance = {"status": "replayed", "prediction_path": str(model), "sha256": sha256(model),
                  "frozen_path": str(frozen_path), "frozen_sha256": sha256(frozen_path),
                  "scope": "Existing local classifier, not a classifier trained with the new decoder fits; not manuscript epoch means."}
    write_json(destination / "existing_classifier_group_provenance.json", provenance)
    return rows, provenance


def run_dataset(args, dataset, out):
    folder = out / dataset
    folder.mkdir()
    h, teacher, valid, keys, labels, original, sources = load_cached_inputs(args.cache_root, dataset)
    coverage = coverage_rows(dataset, labels, valid, original["class_order"])
    write_csv(folder / "teacher_coverage.csv", coverage)
    cls_rows, cls_provenance = export_existing_classifier_groups(
        args.cache_root, dataset, valid, keys, labels, args.seed, folder)
    write_csv(folder / "existing_classifier_groups.csv", cls_rows)
    mean = teacher["train"][valid["train"]].astype(np.float64).mean(0).astype(np.float32)
    np.savez_compressed(folder / "mean_predictor.npz", train_target_mean=mean)
    np.savez_compressed(folder / "memberships.npz", **{s + "_keys": keys[s] for s in SPLITS},
                        **{s + "_valid": valid[s] for s in SPLITS})
    results = []

    def evaluate(mode, predictions, best_epoch=None):
        for split in ("val", "test"):
            targets = teacher[split][valid[split]]
            predicted = predictions[split]
            metrics, coordinate = fidelity_metrics(targets, predicted)
            row = {"dataset": dataset, "model": mode, "split": split, "seed": args.seed,
                   "selected_epoch": best_epoch, **metrics}
            results.append(row)
            np.savez_compressed(folder / f"{mode}_{split}.npz", prediction=predicted,
                target=targets, sample_keys=keys[split][valid[split]], **coordinate)
            print(f"[{dataset} {mode} {split}] n={metrics['n_valid']} "
                  f"MSE={metrics['mse']:.6f} R2={metrics['r2_uniform']:.6f} "
                  f"Pearson={metrics['pearson_defined_mean']}", flush=True)

    evaluate("train_mean", {s: np.broadcast_to(mean, teacher[s][valid[s]].shape).copy()
                             for s in ("val", "test")})
    fitted_metadata = {}
    for mode in ("paired_decoder", "shuffled_target_decoder"):
        targets = {s: teacher[s][valid[s]].copy() for s in ("train", "val")}
        if mode == "shuffled_target_decoder":
            permutations = {s: np.random.default_rng(args.seed + 3000001 + i).permutation(len(y))
                            for i, (s, y) in enumerate(targets.items())}
            targets = {s: y[permutations[s]] for s, y in targets.items()}
            np.savez_compressed(folder / "shuffled_permutations.npz", **permutations)
        fitted = fit_teacher_decoder(h["train"][valid["train"]], targets["train"],
            h["val"][valid["val"]], targets["val"], seed=args.seed, device=args.device,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            progress=lambda msg: print(f"[{dataset} {mode}] {msg}", flush=True))
        torch.save({"state": fitted["state"], "model": mode, "seed": args.seed,
                    "best_epoch": fitted["best_epoch"], "selection_loss": fitted["val_loss"]},
                   folder / f"{mode}.pt")
        write_csv(folder / f"{mode}_history.csv", fitted["history"])
        predictions = apply_affine_map({s: h[s][valid[s]] for s in ("val", "test")},
                                       fitted["state"], args.device)
        evaluate(mode, predictions, fitted["best_epoch"])
        fitted_metadata[mode] = {k: fitted[k] for k in ("best_epoch", "epochs_ran", "val_loss")}
    write_csv(folder / "fidelity_metrics.csv", results)
    write_json(folder / "manifest.json", {"dataset": dataset, "sources": sources,
        "teacher_cache": str(args.cache_root / dataset), "teacher_dimension": len(mean),
        "class_order": original["class_order"], "split_membership": original["split_membership"],
        "decoder_fits": fitted_metadata, "existing_classifier_groups": cls_provenance,
        "status": "complete", "medical_scope": "Retained local image splits; case independence not established."})
    return results, coverage, cls_rows


def write_summary(out, results, coverage, groups):
    write_csv(out / "fidelity_metrics.csv", results)
    write_csv(out / "teacher_coverage.csv", coverage)
    write_csv(out / "existing_classifier_groups.csv", groups)
    lines = ["# Local decoder baseline pilot", "",
        "This is a new local experiment on cached geo11 / Ks=128 teachers. It does not replace manuscript results.",
        "The CLIP encoder and cached teacher bank are unchanged; no classification head is fitted.", "",
        "Training target means use valid training teachers only. Both linear decoders share initialization, minibatch order,",
        "architecture and optimization budget. The shuffled model independently permutes complete target vectors within",
        "train and validation, selects on its shuffled validation loss, and is evaluated against true unshuffled test targets.",
        "Test teachers are excluded from fitting, stopping and selection.", "",
        "| Dataset | Model | Valid test N | MSE | Uniform R2 | Pearson (defined coordinates) |", 
        "|---|---|---:|---:|---:|---:|"]
    for row in results:
        if row["split"] == "test":
            pearson = "N/A (constant predictor)" if row["pearson_defined_mean"] is None else f"{row['pearson_defined_mean']:.6f}"
            lines.append(f"| {row['dataset']} | {row['model']} | {row['n_valid']} | {row['mse']:.6f} | {row['r2_uniform']:.6f} | {pearson} |")
    lines += ["", "R2 averages coordinates uniformly with finite constant-target handling, matching the manuscript definition.",
        "Pearson is undefined for constant predictions; metrics CSV also provides the manuscript convention of zero for undefined coordinates.",
        "Per-coordinate scores, predictions, true targets, memberships, permutations, fitted weights and histories are saved.",
        "This single-seed pilot estimates neither seed uncertainty nor medical case-disjoint generalization.", "",
        "## Cached teacher coverage", "",
        "| Dataset | Split | Images | Valid | Invalid | Valid (%) |", "|---|---|---:|---:|---:|---:|"]
    for row in coverage:
        if row["class"] == "ALL":
            lines.append(f"| {row['dataset']} | {row['split']} | {row['n_images']} | {row['n_valid_teacher']} | {row['n_invalid_teacher']} | {row['valid_teacher_pct']:.2f} |")
    lines += ["", "Invalid flags identify missing cached teachers, not their cause. They do not distinguish SAM failures,",
        "cleaning removals, missing files or unmatched keys. Such causes require an upstream asset audit.", "",
        "## Existing classifier subset replay", "",
        "These are existing local validation-selected classifiers, separate from the newly fitted decoders above.",
        "They are single-checkpoint diagnostics, not the final-15-epoch averages reported in the manuscript.", "",
        "| Dataset | Test subset | N | Frozen (%) | Decoded (%) | Difference (pp) |",
        "|---|---|---:|---:|---:|---:|"]
    for row in groups:
        if row["n"]:
            lines.append(f"| {row['dataset']} | {row['subset']} | {row['n']} | {row['frozen_accuracy_pct']:.4f} | {row['decoded_accuracy_pct']:.4f} | {row['delta_frozen_pp']:+.4f} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/residual_controls_20261018_paired01")
    parser.add_argument("--datasets", nargs="+", default=["CUB_200_2011", "derm7pt"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Duplicate datasets.")
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.patience < 0 or not 0 <= args.seed < 2**32:
        parser.error("Invalid training settings.")
    out = args.output_root.resolve()
    if out.exists():
        parser.error("Preserve existing results: output directory already exists.")
    args.cache_root = args.cache_root.resolve()
    out.mkdir(parents=True)
    started = time.monotonic()
    run = {"started_utc": datetime.now(timezone.utc).isoformat(),
           "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "torch": torch.__version__, "device": args.device},
           "code_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                str(ROOT / "utils/residual_controls.py"): sha256(ROOT / "utils/residual_controls.py")},
           "status": "running", "completed_datasets": []}
    write_json(out / "run.json", run)
    torch.set_num_threads(args.threads)
    results, coverage, groups = [], [], []
    try:
        with threadpool_limits(limits=args.threads):
            for dataset in args.datasets:
                print(f"[START {dataset}]", flush=True)
                r, c, g = run_dataset(args, dataset, out)
                results += r
                coverage += c
                groups += g
                write_summary(out, results, coverage, groups)
                run["completed_datasets"].append(dataset)
                write_json(out / "run.json", run)
        run["status"] = "complete"
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "run.json", run)
    print(f"[COMPLETE] {out}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
