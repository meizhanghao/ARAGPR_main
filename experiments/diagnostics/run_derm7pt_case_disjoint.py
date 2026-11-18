"""Case-disjoint Derm7pt pilot from existing per-image/region caches.

Clinical and dermoscopic images from one recorded case stay together. All
dataset-fitted quantities (DCBM concepts/scalers/head, shape bank/scalers,
decoder and residual heads) are refitted on the new training split. No old
trained checkpoint or pooled teacher bank is reused. Test labels enter only
the final classification and fixed-head evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from experiments.controls.run_decoder_baselines import SPLITS, fidelity_metrics, sha256, write_csv, write_json
from experiments.controls.run_derm7pt_mechanism import case_metadata, cluster_bootstrap_difference
from experiments.diagnostics.run_fixed_head_within_class import evaluate_controls, write_summary, write_table
from experiments.controls.run_residual_controls import make_teacher
from utils.dcbm_1 import GlobalCBM
from utils.residual_controls import (
    TrainingConfig, apply_affine_map, fit_residual, fit_teacher_decoder,
    frozen_semantic_logits, prediction_metrics, standardize_splits,
)

from aragpr.paths import repository_root

ROOT = repository_root()
def split_cases(keys, labels, split_seed):
    """Stratify CASES, not images, before any learned preprocessing."""
    keys, labels = np.asarray(keys), np.asarray(labels)
    case_ids, modalities = case_metadata(keys)
    cases = np.unique(case_ids)
    case_labels = []
    for case in cases:
        values = np.unique(labels[case_ids == case])
        if len(values) != 1:
            raise ValueError(f"Conflicting binary labels in recorded case {case}.")
        case_labels.append(values[0])
    case_labels = np.asarray(case_labels)
    tr, held = train_test_split(np.arange(len(cases)), test_size=.4,
        stratify=case_labels, random_state=split_seed)
    va, te = train_test_split(held, test_size=.5,
        stratify=case_labels[held], random_state=split_seed + 1)
    assignment = dict(train=cases[tr], val=cases[va], test=cases[te])
    indices = {s: np.flatnonzero(np.isin(case_ids, assignment[s])) for s in SPLITS}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(assignment[left]) & set(assignment[right]):
            raise AssertionError("Recorded cases cross the new partitions.")
    merged = np.concatenate(list(indices.values()))
    np.testing.assert_array_equal(np.sort(merged), np.arange(len(keys)))
    return indices, case_ids, modalities


def fit_base(train_x, train_y, val_x, val_y, *, seed, epochs, patience, device):
    """DCBM semantic head: CE + L1, Adam; select by validation accuracy.

    This has no test argument, and retains the project's semantic-head defaults
    (lr=1e-4, L1=1e-4, batch size 32, constant learning rate).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = nn.Linear(train_x.shape[1], 2).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    tx, vx = [torch.as_tensor(x, dtype=torch.float32, device=device)
              for x in (train_x, val_x)]
    ty, vy = [torch.as_tensor(y, dtype=torch.long, device=device)
              for y in (train_y, val_y)]
    rng = torch.Generator(device="cpu").manual_seed(seed)
    best, stale, best_state, history, best_epoch = -1, 0, None, [], 0
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(tx), generator=rng).to(device)
        total = torch.zeros((), device=device)
        for first in range(0, len(tx), 32):
            idx = order[first:first + 32]
            loss = F.cross_entropy(head(tx[idx]), ty[idx]) + 1e-4 * head.weight.abs().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(idx)
        head.eval()
        with torch.no_grad():
            logits = head(vx)
            correct = int((logits.argmax(1) == vy).sum().item())
            val_loss = float(F.cross_entropy(logits, vy).item())
        history.append(dict(epoch=epoch, train_loss=float(total.item()) / len(tx),
            val_correct=correct, val_accuracy_pct=100 * correct / len(vy), val_loss=val_loss))
        if correct > best:
            best, best_epoch, stale = correct, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"[base] epoch={epoch} val={100*correct/len(vy):.2f}% "
                  f"best={100*best/len(vy):.2f}%", flush=True)
        if patience and stale >= patience:
            break
    return dict(state=best_state, best_epoch=best_epoch, val_correct=best,
                epochs_ran=epoch, history=history)


def score_row(method, split, subset, logits, base, labels, mask):
    selected, y = np.asarray(logits)[mask], np.asarray(labels)[mask]
    probs = torch.softmax(torch.as_tensor(selected), dim=1).numpy()[:, 1]
    return dict(method=method, split=split, subset=subset,
        **prediction_metrics(selected, y, np.asarray(base)[mask]),
        auroc=float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
        macro_f1=float(f1_score(y, selected.argmax(1), average="macro", zero_division=0)))


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", dict(status="running", stage="load_raw_caches"))
    paths = {s: ROOT / f"derm7pt_split/images_derm7pt_{s}_CLIP-ViT-L14.torch" for s in SPLITS}
    paths.update(metadata=ROOT / "dataset_all/derm7pt/meta/meta.csv",
        shape_csv=ROOT / "data/Segments/Seg_embs/shape_csv/derm7pt_geo11_features.csv",
        semantic_regions=ROOT / "data/Segments/Seg_embs/segcrop_derm7pt_SAM2_CLIP-ViT-L14.torch",
        classes=ROOT / "data/classes/derm7pt_classes.txt")
    sources = {k: dict(path=str(p), sha256=sha256(p)) for k, p in paths.items()}
    raw_pool, old_split = {}, {}
    for split in SPLITS:
        raw = torch.load(paths[split], map_location="cpu", weights_only=False)
        for key, value in raw.items():
            if key in raw_pool:
                raise ValueError(f"Duplicate image key in original caches: {key}")
            x = np.asarray(value[0].cpu()).reshape(-1)
            if x.shape != (768,) or not np.isfinite(x).all():
                raise ValueError(f"Nonfinite/wrong-size cached embedding: {key}")
            if value[1] not in ("benign", "malignant"):
                raise ValueError(f"Unknown class: {value[1]}")
            raw_pool[key] = value
            old_split[key] = split
    keys = np.asarray(list(raw_pool))
    labels_all = np.asarray([int(raw_pool[k][1] == "malignant") for k in keys])
    indices, cases_all, modalities_all = split_cases(keys, labels_all, args.split_seed)
    memberships, counts = [], []
    split_dir = output / "case_disjoint_embeddings"
    split_dir.mkdir()
    keys_by_split, labels, cases = {}, {}, {}
    for split in SPLITS:
        ids = indices[split]
        keys_by_split[split] = keys[ids]
        labels[split], cases[split] = labels_all[ids], cases_all[ids]
        torch.save({str(keys[j]): raw_pool[str(keys[j])] for j in ids},
                   split_dir / f"images_derm7pt_{split}_CLIP-ViT-L14.torch")
        for j in ids:
            memberships.append(dict(sample_key=str(keys[j]), label=int(labels_all[j]),
                case_id=str(cases_all[j]), modality=str(modalities_all[j]),
                split=split, original_split=old_split[str(keys[j])]))
        counts.append(dict(split=split, n_images=len(ids), n_cases=len(set(cases[split])),
            n_benign_images=int((labels[split] == 0).sum()),
            n_malignant_images=int((labels[split] == 1).sum())))
    write_csv(output / "split_membership.csv", memberships)
    write_csv(output / "split_counts.csv", counts)
    print("CASE-DISJOINT SPLIT " + json.dumps(counts), flush=True)

    cbm = GlobalCBM(str(split_dir), "derm7pt", "CLIP-ViT-L14", str(paths["classes"]), args.device)
    for split in SPLITS:
        np.testing.assert_array_equal(list(getattr(cbm, f"data_{split}_raw")), keys_by_split[split])
        np.testing.assert_array_equal(getattr(cbm, f"y_{split}_oh").argmax(1), labels[split])
    write_json(output / "status.json", dict(status="running", stage="rebuild_dcbm_concepts"))
    cbm.load_concepts(str(paths["semantic_regions"].parent), "SAM2", K=args.semantic_k,
                      use_pca=False, max_segments_per_class=args.max_regions_per_class)
    cbm.preprocess()
    base_fit = fit_base(cbm.A_train_scaled, labels["train"], cbm.A_val_scaled,
        labels["val"], seed=args.seed, epochs=args.base_epochs,
        patience=args.patience, device=args.device)
    write_csv(output / "base_training.csv", base_fit["history"])
    checkpoint = dict(concepts=cbm.concepts, K=cbm.K, scaler=cbm.scaler,
        model_state=base_fit["state"], num_classes=2, dataset="derm7pt",
        model_name="CLIP-ViT-L14", selected_epoch=base_fit["best_epoch"],
        concept_representatives=cbm.semantic_proto_representative,
        new_case_disjoint_training=True)
    torch.save(checkpoint, output / "frozen_dcbm.pt")
    raw_h = {s: getattr(cbm, f"X_{s}").astype(np.float32) for s in SPLITS}
    base = {s: frozen_semantic_logits(raw_h[s], checkpoint, args.device) for s in SPLITS}
    direct_base = apply_affine_map(
        {s: getattr(cbm, f"A_{s}_scaled") for s in SPLITS}, base_fit["state"], args.device)
    for s in SPLITS:
        np.testing.assert_allclose(base[s], direct_base[s], atol=3e-5, rtol=1e-5)
    h, input_scaler = standardize_splits(raw_h)
    torch.save(dict(h_scaler=input_scaler), output / "input_scaler.pt")

    write_json(output / "status.json", dict(status="running", stage="rebuild_shape_targets"))
    teacher, valid, _, _ = make_teacher(cbm, {"shape_csv": paths["shape_csv"]},
        output, SimpleNamespace(device=args.device, shape_dim=128))
    bank = torch.load(output / "teacher_bank.pt", map_location="cpu", weights_only=False)
    train_stems = {Path(str(k)).stem.lower() for k in keys_by_split["train"]}
    for rep in bank["prototype_representatives"].values():
        if Path(str(rep).split("_OUT_CROP_")[0]).name.lower() not in train_stems:
            raise AssertionError("A shape representative is not a new training image.")
    for rep in cbm.semantic_proto_representative.values():
        if Path(str(rep).split("_OUT_CROP_")[0]).name.lower() not in train_stems:
            raise AssertionError("A semantic representative is not a new training image.")
    for row in counts:
        s = row["split"]
        row["n_valid_teacher"] = int(valid[s].sum())
        row["teacher_coverage_pct"] = 100 * float(valid[s].mean())
    write_csv(output / "split_counts.csv", counts)
    decoder = fit_teacher_decoder(h["train"][valid["train"]], teacher["train"][valid["train"]],
        h["val"][valid["val"]], teacher["val"][valid["val"]], seed=args.seed,
        device=args.device, epochs=args.decoder_epochs, patience=args.patience,
        progress=lambda msg: print("[decoder] " + msg, flush=True))
    write_csv(output / "decoder_training.csv", decoder["history"])
    decoded = apply_affine_map(h, decoder["state"], args.device)
    u, coordinate_scaler = standardize_splits(decoded)
    torch.save(dict(decoder_state=decoder["state"], input_scaler=input_scaler,
        coordinate_scaler=coordinate_scaler, selected_epoch=decoder["best_epoch"]),
        output / "decoder.pt")
    fidelity = []
    for split in ("val", "test"):
        metrics, _ = fidelity_metrics(teacher[split][valid[split]], decoded[split][valid[split]])
        fidelity.append(dict(split=split, **metrics))
    write_csv(output / "decoder_fidelity.csv", fidelity)

    config = TrainingConfig(epochs=args.head_epochs, patience=args.patience)
    rows, fitted, model_predictions = [], {}, {}
    for mode in ("affine", "geometry"):
        x = h if mode == "affine" else u
        write_json(output / "status.json", dict(status="running", stage="fit_" + mode))
        print(f"FIT {mode}", flush=True)
        fit = fit_residual(x["train"], base["train"], labels["train"],
            x["val"], base["val"], labels["val"], config=config,
            seed=args.seed, device=args.device,
            progress=lambda msg: print("[" + mode + "] " + msg, flush=True))
        fitted[mode] = fit
        write_csv(output / f"{mode}_training.csv", fit["history"])
        torch.save(fit, output / f"{mode}_head.pt")
        selected = fit["selected"]
        w = selected["state"]["weight"].numpy().astype(np.float64)
        bias = selected["state"]["bias"].numpy().astype(np.float64)
        model_predictions[mode] = {s: base[s].astype(np.float64) + selected["eta"] *
            (x[s].astype(np.float64) @ w.T + bias) for s in SPLITS}
    # All model/scale choices are complete before accessing test classification.
    for split in SPLITS:
        predictions = dict(frozen=base[split], affine=model_predictions["affine"][split],
                           geometry=model_predictions["geometry"][split])
        for method, logits in predictions.items():
            for subset, mask in [("all", np.ones(len(logits), bool)),
                                 ("valid_teacher", valid[split]),
                                 ("missing_teacher", ~valid[split])]:
                if mask.any():
                    rows.append(score_row(method, split, subset, logits, base[split], labels[split], mask))
        np.savez_compressed(output / f"{split}_predictions.npz",
            sample_keys=keys_by_split[split], labels=labels[split], case_ids=cases[split],
            valid_teacher=valid[split], base_logits=base[split],
            affine_logits=predictions["affine"], decoded_logits=predictions["geometry"])
    write_csv(output / "classification.csv", rows)
    selected = fitted["geometry"]["selected"]
    w, bias = [selected["state"][k].numpy().astype(np.float64) for k in ("weight", "bias")]
    ut = coordinate_scaler.transform(teacher["test"][valid["test"]]).astype(np.float64)
    vi = valid["test"]
    vbase, vd = base["test"][vi], u["test"][vi]
    tfull = vbase + selected["eta"] * (ut @ w.T + bias)
    fixed_rows = [score_row(method, "test", "valid_teacher", logits, vbase,
        labels["test"][vi], np.ones(int(vi.sum()), bool)) for method, logits in
        [("frozen", vbase), ("decoded", model_predictions["geometry"]["test"][vi]),
         ("teacher_replacement", tfull)]]
    write_csv(output / "fixed_head_accuracy.csv", fixed_rows)
    controls = evaluate_controls(dataset="derm7pt_case_disjoint", seed=args.seed,
        labels=labels["test"][vi], keys=keys_by_split["test"][vi], base=vbase,
        decoded_logits=model_predictions["geometry"]["test"][vi],
        decoded_inputs=vd, teacher_inputs=ut, weight=w, bias=bias,
        eta=float(selected["eta"]), output=output / "fixed_head" / f"seed_{args.seed}",
        permutations=args.permutations, case_ids=cases["test"][vi])
    write_summary(output / "fixed_head", controls)
    comparisons = []
    yp_base = base["test"].argmax(1)
    for mode in ("affine", "geometry"):
        differences = ((model_predictions[mode]["test"].argmax(1) == labels["test"]).astype(float)
                       - (yp_base == labels["test"]).astype(float))
        comparisons.append(dict(comparison=mode + "-minus-frozen",
            **cluster_bootstrap_difference(differences, cases["test"], replicates=2000)))
    write_csv(output / "paired_case_bootstrap.csv", comparisons)
    for record in sources.values():
        if sha256(record["path"]) != record["sha256"]:
            raise AssertionError("An original cache changed during the experiment.")
    protocol = dict(status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed, split_seed=args.split_seed, split_unit="recorded_case_num_not_patient",
        train_val_test_case_overlap=0, sources=sources, counts=counts,
        semantic_k_requested=args.semantic_k, semantic_k_actual=cbm.K,
        semantic_max_regions_per_class=args.max_regions_per_class, shape_k=128,
        semantic_bank_new_train_only=True, shape_bank_new_train_only=True,
        all_fitted_scalers_new_train_only=True, old_trained_checkpoint_loaded=False,
        old_teacher_bank_reused=False, segmentation_regenerated=False,
        base_epochs=args.base_epochs, decoder_epochs=args.decoder_epochs,
        residual_training_config=asdict(config), selected_base_epoch=base_fit["best_epoch"],
        selected_decoder_epoch=decoder["best_epoch"],
        selected_heads={m: {k: v for k, v in f["selected"].items() if k != "state"}
                        for m, f in fitted.items()},
        uncertainty="Single split/seed pilot; bootstrap conditional on fitted models and test cases",
        runner_sha256=sha256(__file__), elapsed_seconds=time.monotonic() - started)
    write_json(output / "protocol.json", protocol)
    write_json(output / "status.json", dict(status="complete"))
    lines = ["# Derm7pt病例分组评估", "",
        "病例编号来自原始meta.csv的case_num；同一病例全部图像留在同一分区。这不是患者独立性认证。",
        "冻结CLIP特征及单区域描述子复用；DCBM概念库、形状库、所有scaler、基座head、decoder和残差head均重新拟合。",
        f"划分种子{args.split_seed}，训练种子{args.seed}；本次使用单划分、单训练种子，不报告多种子标准差。", "",
        "| 分区 | 病例数 | 图像数 | 有效Teacher/% |", "|---|---:|---:|---:|"]
    for r in counts:
        lines.append(f"| {r['split']} | {r['n_cases']} | {r['n_images']} | {r['teacher_coverage_pct']:.2f} |")
    lines += ["", "## 全测试集分类", "",
        "| 方法 | 准确率/% | 相对Frozen/pp | AUROC | Macro-F1 | 纠正/损害数量 |",
        "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["split"] == "test" and r["subset"] == "all":
            lines.append(f"| {r['method']} | {r['accuracy_pct']:.2f} | {r['delta_frozen_pp']:+.2f} | "
                f"{r['auroc']:.4f} | {r['macro_f1']:.4f} | {r['n_corrected']}/{r['n_harmed']} |")
    lines += ["", "## 固定头Teacher替换：同一有效Teacher测试子集", "",
              "| 条件 | 准确率/% |", "|---|---:|"]
    for r in fixed_rows:
        lines.append(f"| {r['method']} | {r['accuracy_pct']:.2f} |")
    lines += ["", "对应关系及两种置换见fixed_head/RESULTS_zh.md。",
        "结果对应本次病例分区及其重新拟合的模型，与保留图像分区的结果分别报告。",
        "病例分组解决本次分区重叠；单种子结果尚不能证明训练稳定性或临床泛化。", ""]
    (output / "RESULTS_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print("COMPLETE " + str(output), flush=True)
    for r in rows:
        if r["split"] == "test" and r["subset"] == "all":
            print(json.dumps(r, ensure_ascii=False), flush=True)
    for r in fixed_rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic-k", type=int, default=1000)
    parser.add_argument("--max-regions-per-class", type=int, default=500)
    parser.add_argument("--base-epochs", type=int, default=200)
    parser.add_argument("--decoder-epochs", type=int, default=300)
    parser.add_argument("--head-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)
