"""Pilot shape correction on a non-CBM frozen CLIP linear classifier.

Reuse the audited geo11 teacher cache and a saved, base-independent decoder.
No semantic concepts, semantic classifier, or DCBM checkpoint are loaded.
The cache's old frozen-prediction files supply image membership/labels only.
Fit the linear probe once, freeze it, and fit shape heads with separate seeds.
Selection consumes training/validation arrays only; evaluate test afterwards.
All outputs go to a new directory; existing experiments remain read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from experiments.controls.run_decoder_baselines import SPLITS, load_cached_inputs, sha256, write_csv, write_json
from utils.residual_controls import (
    TrainingConfig, apply_affine_map, apply_residual, fit_residual, prediction_metrics,
)

from aragpr.paths import repository_root

ROOT = repository_root()
def fit_linear_probe(train_x, train_y, val_x, val_y, *, seed=42, epochs=200,
                     patience=30, batch_size=256, lr=1e-3, weight_decay=1e-4,
                     device="cpu", progress=None):
    """Fit an ordinary affine classifier; test inputs are absent by design."""
    if (epochs < 1 or patience < 0 or batch_size < 1 or lr <= 0
            or weight_decay < 0 or not len(train_x) or not len(val_x)):
        raise ValueError("Invalid linear-probe fitting configuration.")
    if (len(train_x) != len(train_y) or len(val_x) != len(val_y)
            or train_x.shape[1] != val_x.shape[1]
            or not np.isfinite(train_x).all() or not np.isfinite(val_x).all()):
        raise ValueError("Misaligned or nonfinite linear-probe inputs.")
    classes = np.unique(train_y)
    if not np.array_equal(classes, np.arange(len(classes))) or not np.isin(val_y, classes).all():
        raise ValueError("Labels must use a contiguous, training-defined class order.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = nn.Linear(train_x.shape[1], len(classes)).to(device)
    tx, vx = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (train_x, val_x)]
    ty, vy = [torch.as_tensor(v, dtype=torch.long, device=device) for v in (train_y, val_y)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batches = torch.Generator(device="cpu").manual_seed(seed + 6000011)
    best, best_epoch, best_state, stale, history = -1, 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tx), generator=batches).to(device)
        loss_sum = torch.zeros((), device=device)
        for first in range(0, len(tx), batch_size):
            ix = order[first:first + batch_size]
            loss = F.cross_entropy(model(tx[ix]), ty[ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(ix)
        model.eval()
        with torch.no_grad():
            logits = model(vx)
            correct = int((logits.argmax(1) == vy).sum().item())
            val_loss = float(F.cross_entropy(logits, vy).item())
        if not np.isfinite(val_loss):
            raise ValueError("Linear probe diverged.")
        history.append({"epoch": epoch, "train_loss": float(loss_sum.item()) / len(tx),
                        "val_loss": val_loss, "val_correct": correct,
                        "val_accuracy_pct": 100 * correct / len(vx)})
        if correct > best:
            best, best_epoch, stale = correct, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        schedule.step()
        if progress and (epoch == 1 or epoch % 20 == 0):
            progress(f"LP epoch={epoch}/{epochs} val={100*correct/len(vx):.4f}% "
                     f"best={100*best/len(vx):.4f}%")
        if patience and stale >= patience:
            break
    return {"state": best_state, "best_epoch": best_epoch, "val_correct": best,
            "epochs_ran": epoch, "history": history}


def margin_replay(coordinates, base_logits, final_logits, state, eta):
    """Check every image's full coordinate sum, including unchanged predictions."""
    u = np.asarray(coordinates, dtype=np.float64)
    base, final = [np.asarray(v, dtype=np.float64) for v in (base_logits, final_logits)]
    weight = state["weight"].cpu().numpy().astype(np.float64)
    bias = state["bias"].cpu().numpy().astype(np.float64)
    b, f = base.argmax(1), final.argmax(1)
    ix = np.arange(len(u))
    q = eta * (weight[f] - weight[b]) * u
    qb = eta * (bias[f] - bias[b])
    m0, mf = base[ix, f] - base[ix, b], final[ix, f] - final[ix, b]
    reconstructed = m0 + q.sum(1) + qb
    error = float(np.max(np.abs(reconstructed - mf)))
    if not np.allclose(reconstructed, mf, rtol=2e-5, atol=5e-5):
        raise ValueError(f"Signed-margin replay failed: {error}")
    return {"base_class": b, "final_class": f, "base_margin": m0,
            "final_margin": mf, "contributions": q, "bias_contribution": qb,
            "reconstructed_margin": reconstructed}, error


def run_dataset(args, dataset, destination):
    start = time.monotonic()
    folder = destination / dataset
    folder.mkdir()
    print(f"[{dataset}] verifying cached features, membership and teacher provenance", flush=True)
    h, teacher, valid, keys, labels, original, sources = load_cached_inputs(args.cache_root, dataset)
    decoder_path = args.decoder_root / dataset / f"seed_{args.decoder_seed}" / "geometry/selected_model.pt"
    saved = torch.load(decoder_path, map_location="cpu", weights_only=False)
    decoder = saved["decoder_state"]
    if decoder["weight"].shape != (128, h["train"].shape[1]) or teacher["train"].shape[1] != 128:
        raise ValueError("This pilot requires the retained geo11 128-coordinate decoder.")
    if not saved.get("metadata", {}).get("geometry_supervision"):
        raise ValueError("A paired geometry-supervised decoder is required.")
    decoded = apply_affine_map(h, decoder, args.device)
    scaler = saved.get("coordinate_scaler")
    coordinates = ({s: scaler.transform(a).astype(np.float32) for s, a in decoded.items()}
                   if scaler is not None else decoded)
    sources["decoder_checkpoint"] = {"path": str(decoder_path), "sha256": sha256(decoder_path)}
    protocol = {"dataset": dataset, "status": "running", "sources": sources,
        "class_order": original["class_order"], "split_membership": original["split_membership"],
        "teacher_coverage": original["teacher_coverage"],
        "base": "Ordinary linear classifier of frozen standardized CLIP image embeddings; no concept bank.",
        "linear_probe": {"seed": args.base_seed, "epochs": args.base_epochs,
                         "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
                         "batch_size": 256, "patience": 30,
                         "selection": "Highest validation correct count, earliest epoch on ties."},
        "branch": {"decoder": "Reuse the same saved paired geo11 decoder and coordinate scaler.",
                   "decoder_seed": args.decoder_seed, "shape_dimension": 128,
                   "head_seeds": args.seeds, "head_config": asdict(TrainingConfig(epochs=args.head_epochs)),
                   "selection": "Highest validation accuracy; first epoch/eta on ties."},
        "evaluation": "All retained test images; fixed validation-selected checkpoints. "
                      "Head-seed variation conditional on a fixed LP, teacher bank and decoder. "
                      "Not a final-15-epoch mean or an end-to-end multi-seed estimate.",
        "semantic_concepts_or_DCBM_checkpoint_loaded": False}
    write_json(folder / "protocol.json", protocol)
    base_fit = fit_linear_probe(h["train"], labels["train"], h["val"], labels["val"],
        seed=args.base_seed, epochs=args.base_epochs, device=args.device,
        progress=lambda msg: print(f"[{dataset}] {msg}", flush=True))
    base_logits = apply_affine_map(h, base_fit["state"], args.device)
    torch.save({"model_state": base_fit["state"], "class_order": original["class_order"],
                "best_epoch": base_fit["best_epoch"],
                "input_scaler_source": sources["input_scaler.pt"]}, folder / "frozen_linear_probe.pt")
    write_csv(folder / "linear_probe_history.csv", base_fit["history"])
    protocol["linear_probe"]["selected_epoch"] = base_fit["best_epoch"]
    write_json(folder / "protocol.json", protocol)
    for split in SPLITS:
        np.savez_compressed(folder / f"{split}_linear_probe.npz", logits=base_logits[split],
                            labels=labels[split], sample_keys=keys[split], valid_teacher=valid[split])
    base_accuracy = 100 * float((base_logits["test"].argmax(1) == labels["test"]).mean())
    print(f"[{dataset}] LP frozen: test={base_accuracy:.4f}%; fitting shape correction", flush=True)
    rows = []
    for seed in args.seeds:
        branch = folder / f"seed_{seed}"
        branch.mkdir()
        fit = fit_residual(coordinates["train"], base_logits["train"], labels["train"],
            coordinates["val"], base_logits["val"], labels["val"],
            config=TrainingConfig(epochs=args.head_epochs), seed=seed, device=args.device,
            progress=lambda msg: print(f"[{dataset}/{seed}] {msg}", flush=True))
        selected = fit["selected"]
        torch.save({"head_state": selected["state"], "eta": selected["eta"],
                    "decoder_source": sources["decoder_checkpoint"],
                    "base_source": str(folder / "frozen_linear_probe.pt"),
                    "selected_epoch": selected["best_epoch"]}, branch / "selected_head.pt")
        write_csv(branch / "head_history.csv", fit["history"])
        write_csv(branch / "eta_selection.csv", fit["candidates"])
        reports = []
        for split in ("val", "test"):
            logits = apply_residual(coordinates[split], base_logits[split], fit, args.device)
            # Replay the serialized head independently of the fitter.
            replay_state = torch.load(branch / "selected_head.pt", map_location="cpu", weights_only=False)
            manual = (base_logits[split] + replay_state["eta"] * (
                coordinates[split] @ replay_state["head_state"]["weight"].numpy().T
                + replay_state["head_state"]["bias"].numpy()))
            np.testing.assert_allclose(logits, manual, rtol=2e-5, atol=5e-5)
            all_contributions, error = margin_replay(coordinates[split], base_logits[split], logits,
                                                   selected["state"], selected["eta"])
            np.savez_compressed(branch / f"{split}_predictions.npz", logits=logits,
                labels=labels[split], sample_keys=keys[split], valid_teacher=valid[split])
            np.savez_compressed(branch / f"{split}_contributions.npz", **all_contributions)
            for subset in ("all", "valid_teacher", "missing_teacher"):
                mask = (np.ones(len(labels[split]), dtype=bool) if subset == "all"
                        else valid[split] if subset == "valid_teacher" else ~valid[split])
                if not mask.any():
                    continue
                metrics = prediction_metrics(logits[mask], labels[split][mask], base_logits[split][mask])
                base_pct = 100 * float((base_logits[split][mask].argmax(1) == labels[split][mask]).mean())
                reports.append({"dataset": dataset, "seed": seed, "split": split, "subset": subset,
                    "base_accuracy_pct": base_pct, "eta": selected["eta"],
                    "head_selected_epoch": selected["best_epoch"], "margin_replay_max_error": error,
                    **metrics})
        rows.extend(reports)
        write_csv(branch / "metrics.csv", reports)
        write_csv(folder / "metrics.csv", rows)
        item = next(r for r in reports if r["split"] == "test" and r["subset"] == "all")
        print(f"[DONE {dataset}/{seed}] LP={item['base_accuracy_pct']:.4f}% "
              f"+shape={item['accuracy_pct']:.4f}% delta={item['delta_frozen_pp']:+.4f} pp", flush=True)
    values = [r["accuracy_pct"] for r in rows if r["split"] == "test" and r["subset"] == "all"]
    result = {"dataset": dataset, "base_accuracy_pct": base_accuracy,
              "shape_accuracy_mean_pct": statistics.mean(values),
              "shape_accuracy_head_seed_sd_pct": statistics.stdev(values) if len(values) > 1 else None,
              "delta_mean_pp": statistics.mean(values) - base_accuracy,
              "n_head_seeds": len(values), "fixed_base_seed": args.base_seed,
              "fixed_decoder_seed": args.decoder_seed, "n_test": len(labels["test"]),
              "elapsed_seconds": time.monotonic() - start}
    protocol["status"] = "complete"
    write_json(folder / "protocol.json", protocol)
    write_json(folder / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/residual_controls_20261018_paired01")
    parser.add_argument("--decoder-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", choices=["CUB_200_2011", "derm7pt"],
                        default=["CUB_200_2011", "derm7pt"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--decoder-seed", type=int, default=42)
    parser.add_argument("--base-epochs", type=int, default=200)
    parser.add_argument("--head-epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing results: --output must be a new directory.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Head seeds must be nonempty and unique.")
    args.cache_root = args.cache_root.resolve()
    args.decoder_root = (args.decoder_root or args.cache_root).resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    sources = [Path(__file__), ROOT / "experiments/controls/run_decoder_baselines.py", ROOT / "utils/residual_controls.py"]
    environment = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "code": [{"path": str(p), "sha256": sha256(p)} for p in sources]}
    write_json(args.output / "environment.json", environment)
    summaries = []
    with threadpool_limits(limits=args.threads):
        for dataset in args.datasets:
            summaries.append(run_dataset(args, dataset, args.output))
            write_csv(args.output / "summary.csv", summaries)
            write_json(args.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
