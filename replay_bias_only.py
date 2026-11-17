"""Replay bias-only decisions from retained, fully identified trace exports.

No classifier or teacher is trained. Complete class-bias differences and a
verified frozen checkpoint are required; unidentified class offsets are never
filled with zero. Full decoded predictions are retained exported predictions,
not a rerun of an unavailable decoder/head checkpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.bias_only import evaluate_bias_only, recover_scaled_bias


ROOT = Path(__file__).resolve().parent
TRACES = {
    "derm7pt": "explanation_trace",
    "BUS_UCLM": "explanation_trace_medical",
    "ISIC-2019": "explanation_trace_medical",
    "CUB_200_2011": "explanation_trace_cub_binary_v3",
    "places365_part": "explanation_trace",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_bias(dataset):
    trace = ROOT / "outputs" / TRACES[dataset] / dataset / "decoded"
    manifest = json.loads((trace / "trace_run.json").read_text(encoding="utf-8"))
    rows = read_csv(trace / "all_cases.csv")
    if not manifest["export_complete"] or manifest["n_samples"] != len(rows):
        raise ValueError("Incomplete full-split trace.")
    if manifest["shape_mode"] != "decoded" or manifest["config"].get("use_cooccur", False):
        raise ValueError("Expected a decoded shape-only trace.")
    sources = [trace / "all_cases.csv", trace / "trace_run.json"]
    bundles, constraints, class_names = [], [], None
    for row in rows:
        if not row["bundle_relative_path"]:
            continue
        bundle = trace / row["bundle_relative_path"]
        summary_path = bundle / "case_summary.json"
        logits_path = bundle / "class_logits.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        logits = sorted(read_csv(logits_path), key=lambda item: int(item["class_id"]))
        names = [item["class_name"] for item in logits]
        if [int(item["class_id"]) for item in logits] != list(range(len(logits))):
            raise ValueError("An exported class-logit vector is incomplete.")
        if class_names is None:
            class_names = names
        if names != class_names or summary["sample_key"] != row["sample_key"]:
            raise ValueError("Class order or bundle membership changed.")
        if float(summary["eta_s"]) != float(manifest["eta_s"]):
            raise ValueError("Trace bundles disagree about the selected eta.")
        constraints.append((int(summary["dcbm_class"]), int(summary["sdrcbm_class"]),
                            float(summary["shape_bias_margin_f_minus_b"])))
        bundles.append((row, logits))
        sources.extend([summary_path, logits_path])
    if class_names is None:
        raise ValueError("No complete bias evidence is available.")
    offsets, recovery = recover_scaled_bias(len(class_names), constraints)
    return trace, manifest, rows, bundles, class_names, offsets, recovery, sources


def replay(dataset, out, device="cpu"):
    trace, manifest, rows, bundles, names, offsets, recovery, sources = collect_bias(dataset)
    checkpoint = ROOT / "semantic_checkpoints" / f"{dataset}_semantic_baseline.pt"
    split_root = ROOT / f"{dataset}_split"
    train_path = split_root / f"images_{dataset}_train_CLIP-ViT-L14.torch"
    test_path = split_root / f"images_{dataset}_test_CLIP-ViT-L14.torch"
    sources.extend([checkpoint, train_path, test_path])
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    train = torch.load(train_path, weights_only=False, map_location="cpu")
    test = torch.load(test_path, weights_only=False, map_location="cpu")
    classes = sorted({value[1] for value in train.values()
                      if not torch.isnan(torch.as_tensor(value[0])).any()})
    class_index = {name: i for i, name in enumerate(classes)}
    kept = {key: value for key, value in test.items()
            if not torch.isnan(torch.as_tensor(value[0])).any() and value[1] in class_index}
    by_key = {row["sample_key"]: row for row in rows}
    if len(by_key) != len(rows) or set(kept) != set(by_key):
        raise ValueError("Frozen-input and trace test memberships differ.")
    if int(ckpt["num_classes"]) != len(names) or len(classes) != len(names):
        raise ValueError("Checkpoint class dimensions differ from trace.")
    keys = list(kept)
    labels = np.array([class_index[kept[key][1]] for key in keys], dtype=np.int64)
    if any(int(by_key[key]["true_class"]) != label for key, label in zip(keys, labels)):
        raise ValueError("Checkpoint input labels and trace class order differ.")
    for key, label in zip(keys, labels):
        if by_key[key]["true_name"] != names[label]:
            raise ValueError("Exported names do not match class indices.")
    concepts = torch.as_tensor(ckpt["concepts"], dtype=torch.float32, device=device)
    norm2 = torch.linalg.vector_norm(concepts, dim=1).square()
    weight = ckpt["model_state"]["weight"].to(device)
    bias = ckpt["model_state"]["bias"].to(device)
    outputs = []
    with torch.no_grad():
        for first in range(0, len(keys), 256):
            batch = torch.stack([torch.as_tensor(kept[key][0]).reshape(-1).float()
                                 for key in keys[first:first+256]]).to(device)
            raw_logits = batch @ concepts.T / norm2[None, :]
            # Match GlobalCBM.preprocess for zero-norm semantic prototypes.
            if torch.isnan(raw_logits).any():
                raw_logits = torch.nan_to_num(raw_logits, nan=float("-inf"))
            activations = torch.softmax(raw_logits, dim=1).cpu().numpy()
            activations = ckpt["scaler"].transform(activations).astype(np.float32)
            scores = torch.nn.functional.linear(torch.from_numpy(activations).to(device), weight, bias)
            outputs.append(scores.cpu().numpy())
    semantic = np.concatenate(outputs)
    base_export = np.array([int(by_key[key]["dcbm_class"]) for key in keys])
    full_export = np.array([int(by_key[key]["sdrcbm_class"]) for key in keys])
    if not np.array_equal(semantic.argmax(1), base_export):
        raise ValueError("Frozen checkpoint does not reproduce all exported baseline predictions.")
    index = {key: i for i, key in enumerate(keys)}
    max_error = 0.0
    for row, class_logits in bundles:
        expected = np.array([float(item["semantic_logit"]) for item in class_logits])
        max_error = max(max_error, float(np.max(np.abs(semantic[index[row["sample_key"]]] - expected))))
    if max_error > 2e-4:
        raise ValueError(f"Frozen checkpoint semantic logits do not match exports: {max_error}")
    metrics, bias_logits = evaluate_bias_only(semantic, labels, offsets, full_export)
    if abs(metrics["frozen"]["accuracy_pct"] / 100 - manifest["baseline_accuracy"]) > 1e-10:
        raise ValueError("Baseline counts disagree with trace manifest.")
    if abs(metrics["decoded"]["accuracy_pct"] / 100 - manifest["final_accuracy"]) > 1e-10:
        raise ValueError("Decoded counts disagree with trace manifest.")
    report = {
        "dataset": dataset, "eligible": True, "n_classes": len(names),
        "eta_s": manifest["eta_s"], "class_names": names,
        "metrics": metrics, "bias_recovery": recovery,
        "max_semantic_logit_error": max_error,
        "baseline_prediction_matches": len(rows),
        "n_complete_logit_bundles_checked": len(bundles),
        "source_trace": str(trace),
        "bias_identification": "eta*b differences; class-0 offset fixed to zero. A common shift leaves argmax unchanged.",
        "selection": "Original validation-selected full-model eta, fixed for this audit; no training or test-set selection.",
        "full_model_scope": "Decoded predictions replayed from the complete original all_cases.csv; no decoder/head checkpoint rerun.",
        "sources": {str(path): checksum(path) for path in sources},
    }
    target = out / dataset
    target.mkdir()
    np.savez_compressed(target / "test_replay.npz", semantic_logits=semantic,
                        bias_only_logits=bias_logits, scaled_bias_relative_to_class0=offsets,
                        labels=labels, sample_keys=np.asarray(keys), decoded_predictions=full_export)
    report["replay_sha256"] = checksum(target / "test_replay.npz")
    (target / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    records = []
    for i, key in enumerate(keys):
        records.append({"sample_key": key, "true_class": int(labels[i]),
                        "frozen_class": int(base_export[i]),
                        "bias_only_class": int(bias_logits[i].argmax()),
                        "decoded_class": int(full_export[i])})
    with (target / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/bias_only_replay_20261018")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    report = {"new_training": False, "test_selection": False, "datasets": {}}
    table_rows = []
    for dataset in TRACES:
        try:
            item = replay(dataset, args.out_dir, args.device)
            report["datasets"][dataset] = item
            for variant, metrics in item["metrics"].items():
                table_rows.append({"dataset": dataset, "eta_s": item["eta_s"], "variant": variant, **metrics})
            print(dataset, {v: (m["n_correct"], m["n"], round(m["accuracy_pct"], 4))
                            for v, m in item["metrics"].items()}, flush=True)
        except ValueError as error:
            report["datasets"][dataset] = {"eligible": False, "reason": str(error)}
            print(dataset, "excluded:", error, flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if table_rows:
        with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
    else:
        raise RuntimeError("No trace could support a complete bias-only comparison.")


if __name__ == "__main__":
    main()
