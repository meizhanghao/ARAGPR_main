"""Independently replay saved residual controls and check validation selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(directory, completed_only=False):
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "complete" and not completed_only:
        raise ValueError("Only a fully completed experiment can be certified.")
    for path, checksum in run["code_sha256"].items():
        if sha256(path) != checksum:
            raise ValueError(f"Training code changed since this run: {path}")
    cfg = run["config"]
    datasets = run["completed_datasets"] if completed_only else cfg["datasets"]
    if not datasets:
        raise ValueError("No completed dataset to replay.")
    rows, pairs, details = [], [], []
    for dataset in datasets:
        folder = directory / dataset
        manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        if not manifest["completed"]:
            raise ValueError(f"Incomplete dataset: {dataset}")
        for source in manifest["sources"].values():
            if sha256(source["path"]) != source["sha256"]:
                raise ValueError(f"Input changed since training: {source['path']}")
        if "shape_csv" in manifest:
            source = manifest["shape_csv"]
            if sha256(source["path"]) != source["sha256"]:
                raise ValueError("Shape CSV changed since training.")
        arrays, h = {}, {}
        scaler = torch.load(folder / "input_scaler.pt", weights_only=False, map_location="cpu")["h_scaler"]
        for split in ("val", "test"):
            with np.load(folder / f"{split}_frozen.npz", allow_pickle=False) as frozen:
                arrays[split] = {k: frozen[k].copy() for k in frozen.files}
            raw = torch.load(manifest["sources"][split]["path"], weights_only=False, map_location="cpu")
            features = np.stack([np.asarray(raw[str(key)][0]).reshape(-1)
                                 for key in arrays[split]["sample_keys"]]).astype(np.float32)
            h[split] = scaler.transform(features).astype(np.float32)
            expected_keys = manifest["split_membership"][split]["keys_sha256"]
            actual_keys = hashlib.sha256(json.dumps(arrays[split]["sample_keys"].tolist(),
                                                       ensure_ascii=False).encode("utf-8")).hexdigest()
            if expected_keys != actual_keys:
                raise ValueError("Frozen split membership changed.")
        predictions = {}
        for seed in cfg["seeds"]:
            for mode in cfg["modes"]:
                branch = folder / f"seed_{seed}" / mode
                report = json.loads((branch / "result.json").read_text(encoding="utf-8"))
                selected = torch.load(branch / "selected_model.pt", weights_only=False, map_location="cpu")
                candidates = read_csv(branch / "eta_selection.csv")
                history = read_csv(branch / "head_history.csv")
                if any("test" in field.lower() for field in history[0]):
                    raise ValueError("Unexpected test field in training selection history.")
                best = max(candidates, key=lambda r: int(r["val_correct"]))
                if (float(best["eta"]) != report["selected"]["eta"]
                        or int(best["best_epoch"]) != report["selected"]["best_epoch"]):
                    raise ValueError("Selected head is not the first validation maximum.")
                for candidate in candidates:
                    relevant = [r for r in history if float(r["eta"]) == float(candidate["eta"])]
                    peak = max(relevant, key=lambda r: int(r["val_correct"]))
                    if int(peak["epoch"]) != int(candidate["best_epoch"]):
                        raise ValueError("Head checkpoint is not the first best-validation epoch.")
                nc = arrays["val"]["logits"].shape[1]
                expected_width = h["val"].shape[1] if mode == "affine" else cfg["shape_dim"]
                if report["head_trainable_parameters"] != (expected_width+1)*nc:
                    raise ValueError("Reported control parameter count is incorrect.")
                max_error = 0.
                for split in ("val", "test"):
                    if mode == "affine":
                        coords = h[split]
                    elif mode == "random_projection":
                        coords = h[split] @ selected["projection"].T
                    else:
                        decoder = selected["decoder_state"]
                        coords = h[split] @ decoder["weight"].numpy().T + decoder["bias"].numpy()
                    if "coordinate_scaler" in selected:
                        coords = selected["coordinate_scaler"].transform(coords).astype(np.float32)
                    head = selected["head_state"]
                    # NumPy replay is independent of the training/inference helper.
                    scores = arrays[split]["logits"] + selected["eta"] * (
                        coords @ head["weight"].numpy().T + head["bias"].numpy())
                    with np.load(branch / f"{split}_predictions.npz", allow_pickle=False) as saved:
                        if (not np.array_equal(saved["sample_keys"], arrays[split]["sample_keys"])
                                or not np.array_equal(saved["labels"], arrays[split]["labels"])):
                            raise ValueError("Residual and Frozen predictions are not paired.")
                        error = float(np.max(np.abs(scores - saved["logits"])))
                        max_error = max(max_error, error)
                        if not np.allclose(scores, saved["logits"], rtol=3e-5, atol=3e-4):
                            raise ValueError(f"Saved head replay failed: {dataset}/{mode}, error={error}")
                        if not np.array_equal(scores.argmax(1), saved["logits"].argmax(1)):
                            raise ValueError("Replayed classification decisions differ.")
                    labels = arrays[split]["labels"]
                    base = arrays[split]["logits"].argmax(1) == labels
                    correct = scores.argmax(1) == labels
                    metrics = report["metrics"][split]
                    if (int(correct.sum()) != metrics["n_correct"]
                            or int((~base & correct).sum()) != metrics["n_corrected"]
                            or int((base & ~correct).sum()) != metrics["n_harmed"]):
                        raise ValueError("Reported accuracy/correction/harm does not match predictions.")
                    if split == "val" and int(correct.sum()) != int(best["val_correct"]):
                        raise ValueError("Restored head does not reproduce validation selection.")
                    rows.append({"dataset": dataset, "seed": seed, "mode": mode,
                                 "split": split, "n": len(labels),
                                 "n_correct": int(correct.sum()),
                                 "accuracy_pct": 100*float(correct.mean()),
                                 "replay_max_absolute_error": error})
                    if split == "test":
                        predictions[(seed, mode)] = correct
                details.append({"dataset": dataset, "seed": seed, "mode": mode,
                                "replay_max_absolute_error": max_error,
                                "head_parameters": report["head_trainable_parameters"]})
            if "geometry" in cfg["modes"]:
                geo = predictions[(seed, "geometry")]
                for mode in cfg["modes"]:
                    if mode == "geometry":
                        continue
                    control = predictions[(seed, mode)]
                    pairs.append({"dataset": dataset, "seed": seed, "control": mode,
                                  "n": len(geo), "geometry_only_correct": int((geo & ~control).sum()),
                                  "control_only_correct": int((~geo & control).sum()),
                                  "geometry_minus_control_pp": 100*float(geo.mean()-control.mean())})
    expected = len(datasets) * len(cfg["modes"]) * len(cfg["seeds"]) * 2
    if len(rows) != expected:
        raise ValueError("Missing model/dataset/split result.")
    complete = run["status"] == "complete" and set(datasets) == set(cfg["datasets"])
    result = {"passed": True, "complete": complete, "n_models": len(details), "n_split_replays": len(rows),
              "n_seeds": len(cfg["seeds"]), "details": details,
              "paired_comparisons": pairs}
    suffix = "" if complete else "_partial"
    (directory / f"verification{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (directory / f"paired_comparisons{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0])) if pairs else None
        if writer:
            writer.writeheader()
            writer.writerows(pairs)
    print(json.dumps({k: result[k] for k in ("passed", "complete", "n_models", "n_split_replays", "n_seeds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--completed-only", action="store_true",
                        help="Audit finished datasets during training; writes an explicitly partial report.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        audit(args.directory.resolve(), args.completed_only)


if __name__ == "__main__":
    main()
