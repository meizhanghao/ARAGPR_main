"""Post-hoc removal of shape coordinates from a validation-selected head."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def prediction_metrics(labels, baseline, predictions):
    labels, baseline, predictions = (np.asarray(x) for x in
                                     (labels, baseline, predictions))
    if labels.ndim != 1 or not len(labels) or not (
            labels.shape == baseline.shape == predictions.shape):
        raise ValueError("Expected equally sized, nonempty prediction vectors.")
    right = baseline == labels
    corrected = int(np.sum(~right & (predictions == labels)))
    harmed = int(np.sum(right & (predictions != labels)))
    correct = int(np.sum(predictions == labels))
    n = len(labels)
    if correct - int(right.sum()) != corrected - harmed:
        raise AssertionError("Paired correction/harm counts do not close.")
    return {
        "n": n, "n_correct": correct, "accuracy_pct": 100.0 * correct / n,
        "n_base_correct": int(right.sum()), "n_base_wrong": int((~right).sum()),
        "n_corrected": corrected, "n_harmed": harmed,
        "correction_pct": 100.0 * corrected / int((~right).sum()) if (~right).any() else None,
        "harm_pct": 100.0 * harmed / int(right.sum()) if right.any() else None,
        "consistency_pct": 100.0 * float(np.mean(predictions == baseline)),
    }


def evaluate_bias_only(semantic_logits, labels, scaled_shape_bias,
                       full_predictions=None):
    """Keep frozen semantic logits plus eta_s*b_s; fit/select nothing."""
    semantic_logits = np.asarray(semantic_logits)
    labels = np.asarray(labels)
    offsets = np.asarray(scaled_shape_bias)
    if semantic_logits.ndim != 2 or not len(semantic_logits) or semantic_logits.shape[1] < 2:
        raise ValueError("Expected nonempty N x C semantic logits with C >= 2.")
    n, c = semantic_logits.shape
    if offsets.shape != (c,) or not np.isfinite(offsets).all() or not np.isfinite(semantic_logits).all():
        raise ValueError("Bias must cover every class and all scores must be finite.")
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= c)):
        raise ValueError("Labels must be integer class indices in [0, C).")
    baseline = semantic_logits.argmax(axis=1)
    bias_logits = semantic_logits + offsets[None, :]
    bias_predictions = bias_logits.argmax(axis=1)
    metrics = {
        "frozen": prediction_metrics(labels, baseline, baseline),
        "bias_only": prediction_metrics(labels, baseline, bias_predictions),
    }
    if full_predictions is not None:
        full_predictions = np.asarray(full_predictions)
        if (full_predictions.shape != (n,) or full_predictions.dtype.kind not in "iu"
                or np.any((full_predictions < 0) | (full_predictions >= c))):
            raise ValueError("Full-model predictions must identify every test image and class.")
        metrics["decoded"] = prediction_metrics(labels, baseline, full_predictions)
    return metrics, bias_logits


def recover_scaled_bias(n_classes, constraints, atol=1e-6):
    """Recover eta*b up to one common shift from (base, final, difference).

    A connected graph covering ALL classes is required. Relative shifts of
    disconnected components cannot be inferred from pairwise exports.
    """
    if n_classes < 2 or not constraints:
        raise ValueError("At least two classes and exported constraints are required.")
    graph = [[] for _ in range(n_classes)]
    for base, final, difference in constraints:
        if not (0 <= base < n_classes and 0 <= final < n_classes) or not np.isfinite(difference):
            raise ValueError("Invalid class or nonfinite exported bias difference.")
        graph[base].append((final, float(difference)))
        graph[final].append((base, -float(difference)))
    offsets = np.full(n_classes, np.nan)
    offsets[0] = 0.0
    todo = [0]
    while todo:
        base = todo.pop()
        for final, difference in graph[base]:
            expected = offsets[base] + difference
            if np.isnan(offsets[final]):
                offsets[final] = expected
                todo.append(final)
            elif abs(offsets[final] - expected) > atol:
                raise ValueError("Exported bias differences disagree across cases.")
    if not np.isfinite(offsets).all():
        missing = np.flatnonzero(~np.isfinite(offsets)).tolist()
        raise ValueError(f"Bias differences do not identify all class offsets: {missing}")
    max_error = max(abs(offsets[f] - offsets[b] - d) for b, f, d in constraints)
    if max_error > atol:
        raise ValueError("Recovered bias does not close exported differences.")
    return offsets, {"n_constraints": len(constraints),
                     "classes_identified": n_classes,
                     "max_bias_difference_error": float(max_error),
                     "reference_class": 0, "reference_offset": 0.0}


def export_bias_only_control(rt, out_dir, shape_mode="decoded", config=None, bs=256):
    """Export exact full-split logits from the restored head, without training."""
    import torch

    if shape_mode not in ("decoded", "decoded_rel"):
        raise ValueError("This control supports decoded/decoded_rel inference only.")
    if getattr(rt, "sb", None) is None:
        raise ValueError("Restore a validation-selected shape head first.")
    if getattr(rt, "cb", None) is not None and float(getattr(rt, "eta_c", 0.0)) != 0.0:
        raise ValueError("Disable the cooccurrence branch for this shape-only control.")
    if bs < 1:
        raise ValueError("Batch size must be positive.")
    root = Path(out_dir) / rt.cbm.dataset / shape_mode
    if root.exists():
        raise FileExistsError(f"Preserve the existing control; use a new output directory: {root}")
    eta = float(rt.eta_s)
    bias = rt.sb.linear.bias.detach().cpu().numpy().copy()
    model_states = (rt.cbm.model.training, rt.sb.training)
    split_arrays = {}
    try:
        rt.cbm.model.eval()
        rt.sb.eval()
        with torch.no_grad():
            for split in ("val", "test"):
                semantic = getattr(rt.cbm, f"A_{split}_scaled")
                suffix = "_rel" if shape_mode == "decoded_rel" else ""
                shape = getattr(rt, f"Sclip_{split}{suffix}")
                labels = np.asarray(getattr(rt.cbm, f"y_{split}_oh")).argmax(axis=1)
                keys = np.asarray(list(getattr(rt.cbm, f"data_{split}_raw")))
                if not (len(semantic) == len(shape) == len(labels) == len(keys)):
                    raise ValueError(f"Unmatched {split} memberships.")
                z_sem, z_full = [], []
                for first in range(0, len(labels), bs):
                    sem_batch = torch.as_tensor(semantic[first:first+bs], dtype=torch.float32, device=rt.device)
                    shape_batch = torch.as_tensor(shape[first:first+bs], dtype=torch.float32, device=rt.device)
                    frozen = rt.cbm.model(sem_batch)
                    z_sem.append(frozen.cpu().numpy())
                    z_full.append((frozen + eta * rt.sb(shape_batch)).cpu().numpy())
                split_arrays[split] = (np.concatenate(z_sem), np.concatenate(z_full), labels, keys)
    finally:
        rt.cbm.model.train(model_states[0])
        rt.sb.train(model_states[1])
    root.mkdir(parents=True)
    report = {"dataset": rt.cbm.dataset, "shape_mode": shape_mode, "eta_s": eta,
              "selection_basis": "Original full-model validation selection; no refit or eta reselection.",
              "bias_definition": "eta_s times shape-head bias only; decoder bias is not added separately.",
              "config": config or {}, "splits": {}}
    rows = []
    for split, (semantic, full, labels, keys) in split_arrays.items():
        metrics, bias_logits = evaluate_bias_only(semantic, labels, eta * bias, full.argmax(axis=1))
        archive = root / f"{split}_logits.npz"
        np.savez_compressed(archive, semantic_logits=semantic, full_logits=full,
                            bias_only_logits=bias_logits, labels=labels,
                            sample_keys=keys, shape_head_bias=bias, eta_s=eta)
        report["splits"][split] = {"metrics": metrics,
                                  "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        rows.extend({"split": split, "variant": variant, **item} for variant, item in metrics.items())
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    torch.save({
        "shape_head_state": {k: v.detach().cpu().clone() for k, v in rt.sb.state_dict().items()},
        "semantic_model_state": {k: v.detach().cpu().clone() for k, v in rt.cbm.model.state_dict().items()},
        "eta_s": eta, "config": config or {},
    }, root / "selected_heads.pt")
    print(f"[bias-only] Paired full-split control saved to {root}")
    return report
