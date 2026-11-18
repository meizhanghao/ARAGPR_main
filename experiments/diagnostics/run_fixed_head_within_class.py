"""Evaluate original/global/within-class teachers with an unchanged classifier.

This runner replays saved diagnostic models with fixed parameters, evaluation
cohorts, and Decoded class contrasts.
``evaluate_controls`` is also used by the case-disjoint pilot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.controls.run_decoder_baselines import sha256, write_csv, write_json
from experiments.diagnostics.run_teacher_alignment_audit import (
    COHORTS, MEASURES, aggregate, summarize_alignment,
)

from aragpr.paths import repository_root

ROOT = repository_root()
CONDITIONS = ("original_teacher", "global_shuffle", "within_class_shuffle")


def write_table(path, rows):
    """Write heterogeneous metric records with an explicit union of columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(Path(path), [{key: row.get(key) for key in columns} for row in rows])


def make_permutation(labels, rng, within_class=False, exclude_self=False):
    """Permute complete rows, optionally within true-label strata.

    Ordinary permutations match the previously published local evaluation.
    Fixed points are recorded, not silently dropped. A stratum with one image
    cannot be deranged and stays fixed; the caller reports that fact.
    """
    labels = np.asarray(labels)
    order = np.arange(len(labels))
    groups = ([np.flatnonzero(labels == c) for c in np.unique(labels)]
              if within_class else [order.copy()])
    for members in groups:
        if len(members) < 2:
            continue
        for _ in range(10000):
            proposed = rng.permutation(members)
            if not exclude_self or np.all(proposed != members):
                break
        else:
            raise RuntimeError("Could not generate a fixed-point-free permutation.")
        order[members] = proposed
    if len(np.unique(order)) != len(order):
        raise AssertionError("The mapping must be a bijection.")
    if within_class:
        np.testing.assert_array_equal(labels[order], labels)
    return order


def evaluate_controls(*, dataset, seed, labels, keys, base, decoded_logits,
                      decoded_inputs, teacher_inputs, weight, bias, eta,
                      output, permutations=20, exclude_self=False,
                      case_ids=None):
    """Evaluate valid-target images using fixed original Decoded predictions.

    The caller must pass the SAME saved head-input transform for both response
    arrays. All arrays here already use that transform; it is not refitted.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    labels, keys = np.asarray(labels), np.asarray(keys).astype(str)
    base, full, ud, ut, w, bias = [np.asarray(a, dtype=np.float64) for a in
                                 (base, decoded_logits, decoded_inputs,
                                  teacher_inputs, weight, bias)]
    n = len(labels)
    if not n or len(set(keys)) != n or ud.shape != ut.shape or len(ud) != n:
        raise ValueError("Empty, duplicate or misaligned teacher cohort.")
    if not all(np.isfinite(a).all() for a in (base, full, ud, ut, w, bias)):
        raise ValueError("Nonfinite scores or coordinates.")
    np.testing.assert_allclose(full, base + eta * (ud @ w.T + bias),
                               atol=5e-5, rtol=2e-5)
    b, f = base.argmax(1), full.argmax(1)
    contrast = eta * (w[f] - w[b])
    qd, qt = contrast * ud, contrast * ut
    qb = eta * (bias[f] - bias[b])
    m0 = base[np.arange(n), f] - base[np.arange(n), b]
    reconstructed = m0 + qd.sum(1) + qb
    expected = full[np.arange(n), f] - full[np.arange(n), b]
    closure_error = float(np.max(np.abs(reconstructed - expected)))
    if closure_error > 2e-4:
        raise AssertionError(f"Contribution closure failed: {closure_error}")
    logits_t = base + eta * (ut @ w.T + bias)
    np.savez_compressed(output / "attributions.npz", sample_keys=keys,
        labels=labels, fixed_b=b, fixed_f=f, base_logits=base,
        decoded_logits=full, teacher_logits=logits_t,
        decoded_head_inputs=ud, teacher_head_inputs=ut,
        decoded_contributions=qd, teacher_contributions=qt,
        head_weight=w, head_bias=bias, eta=eta, bias_contribution=qb,
        base_margin=m0, case_ids=np.asarray(case_ids).astype(str)
        if case_ids is not None else np.full(n, ""))
    original_rows, permutation_rows, detail_rows = [], [], []
    for cohort in COHORTS:
        row, detail = summarize_alignment(labels, b, f, logits_t.argmax(1), qd, qt, cohort)
        original_rows.append(dict(dataset=dataset, seed=seed,
                                  condition="original_teacher", **row))
        if cohort == "all_changed":
            detail_index = {int(j): i for i, j in enumerate(detail["ids"])}
            for j in range(n):
                i = detail_index.get(j)
                detail_rows.append(dict(dataset=dataset, seed=seed,
                    sample_key=str(keys[j]), label=int(labels[j]),
                    case_id=str(case_ids[j]) if case_ids is not None else "",
                    frozen_b=int(b[j]), decoded_f_fixed=int(f[j]),
                    teacher_prediction=int(logits_t[j].argmax()),
                    changed=bool(b[j] != f[j]),
                    original_correction=bool(detail["corrections"][j]),
                    original_correction_retained=bool(detail["retained"][j]),
                    sign_eligible=int(detail["eligible"][i]) if i is not None else None,
                    sign_matches=int(detail["matches"][i]) if i is not None else None,
                    top3_intersection=int(detail["overlap"][i]) if i is not None else None,
                    decoded_top3=";".join(map(str, detail["top_decoded"][i]))
                    if i is not None else "",
                    teacher_top3=";".join(map(str, detail["top_teacher"][i]))
                    if i is not None else ""))
    saved_orders = {}
    for condition in CONDITIONS[1:]:
        # Preserve the old global stream exactly for ordinary permutations.
        rng = np.random.default_rng((100000 if condition == "global_shuffle" else 200000) + seed)
        orders = []
        for repeat in range(permutations):
            order = make_permutation(labels, rng,
                within_class=condition == "within_class_shuffle", exclude_self=exclude_self)
            orders.append(order)
            shuffled = ut[order]
            logits = base + eta * (shuffled @ w.T + bias)
            qnull = contrast * shuffled
            for cohort in COHORTS:
                row, _ = summarize_alignment(labels, b, f, logits.argmax(1), qd, qnull, cohort)
                permutation_rows.append(dict(dataset=dataset, seed=seed,
                    condition=condition, permutation=repeat,
                    self_pair_pct=100 * float((order == np.arange(n)).mean()),
                    same_class_pct=100 * float((labels[order] == labels).mean()),
                    same_case_pct=100 * float((np.asarray(case_ids)[order] == case_ids).mean())
                    if case_ids is not None else None, **row))
        saved_orders[condition] = np.asarray(orders)
    np.savez_compressed(output / "permutation_indices.npz", **saved_orders)
    means = []
    for condition in CONDITIONS[1:]:
        for cohort in COHORTS:
            members = [r for r in permutation_rows
                       if r["condition"] == condition and r["cohort"] == cohort]
            row = dict(dataset=dataset, seed=seed, condition=condition, cohort=cohort,
                       n_permutations=permutations, n_images=members[0]["n_images"],
                       n_original_corrections=members[0]["n_original_corrections"])
            for metric in MEASURES:
                values = [r[metric] for r in members if r[metric] is not None]
                row[metric] = float(np.mean(values)) if values else None
            row["mean_self_pair_pct"] = float(np.mean([r["self_pair_pct"] for r in members]))
            row["mean_same_class_pct"] = float(np.mean([r["same_class_pct"] for r in members]))
            means.append(row)
    write_csv(output / "original_metrics.csv", original_rows)
    write_csv(output / "permutation_metrics.csv", permutation_rows)
    write_csv(output / "permutation_mean.csv", means)
    write_csv(output / "all_valid_sample_details.csv", detail_rows)
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    write_json(output / "checks.json", dict(n_valid_images=n,
        n_changed_images=int((b != f).sum()),
        n_original_corrections=int(((b != labels) & (f == labels)).sum()),
        class_counts={str(k): int(v) for k, v in counts.items()},
        singleton_class_images=int(sum(v for v in counts.values() if v == 1)),
        max_contribution_closure_error=closure_error,
        head_scalers_eta_frozen=True, original_contrast_and_cohorts_fixed=True,
        permutation_policy="deranged" if exclude_self else "ordinary_with_recorded_fixed_points"))
    return original_rows + means


def write_summary(output, rows):
    output = Path(output)
    summary = aggregate(rows, ["dataset", "condition", "cohort"])
    write_table(output / "metrics_per_seed.csv", rows)
    write_csv(output / "metrics_summary.csv", summary)
    differences = []
    for original in rows:
        if original["condition"] != "original_teacher":
            continue
        for control in CONDITIONS[1:]:
            other = next(r for r in rows if r["dataset"] == original["dataset"]
                and r["seed"] == original["seed"] and r["cohort"] == original["cohort"]
                and r["condition"] == control)
            row = {k: original[k] for k in ["dataset", "seed", "cohort"]}
            row["control"] = control
            for metric in MEASURES:
                row[metric] = original[metric] - other[metric] if (
                    original[metric] is not None and other[metric] is not None) else None
            differences.append(row)
    write_csv(output / "paired_differences_per_seed.csv", differences)
    write_csv(output / "paired_differences_summary.csv",
              aggregate(differences, ["dataset", "control", "cohort"]))
    lines = ["# 固定头Teacher：全局与同类内打乱", "",
        "统计基于保存的固定模型及其输入缓存，模型来源与配置见protocol.json。",
        "只评估有效Teacher图像；样本集合、原Decoded类别对、head、偏置、eta和标准化固定。",
        "每个模型先对置换平均，再跨模型种子计算样本标准差；单个模型不报告种子标准差。", "",
        "| 数据集 | 条件 | 纠错保留/% | 方向一致/% | Top-3重叠/% |", "|---|---|---:|---:|---:|"]
    for row in summary:
        if row["cohort"] != "all_changed":
            continue
        vals = []
        for metric in MEASURES:
            mean, sd = row["mean_" + metric], row["std_" + metric]
            vals.append("N/A" if mean is None else f"{mean:.2f}" +
                        (f" ± {sd:.2f}" if sd is not None else ""))
        lines.append("| " + " | ".join([row["dataset"], row["condition"], *vals]) + " |")
    lines += ["", "指标均沿用现有论文定义。方向一致不要求幅度接近；Top-3重叠比较编号集合。",
        "原图Teacher优于类内打乱，才支持超出类别共同信息的逐图对应；它仍不分离轮廓、外观与背景。",
        "所有置换的自配对比例、分母和映射均已保存，不以测试结果选模型或更改实验设置。", ""]
    (output / "RESULTS_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    original = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if original["status"] != "complete":
        raise ValueError("The saved source diagnostic is incomplete.")
    verification = {(r["dataset"], r["seed"]): r for r in original["verification"]}
    rows, records = [], []
    for dataset in args.datasets:
        for seed in args.seeds:
            record = verification[dataset, seed]
            model_path = Path(record["model_path"])
            if sha256(model_path) != record["model_sha256"]:
                raise ValueError("Source checkpoint hash changed.")
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            head = model["head"]["selected"]
            w = head["state"]["weight"].detach().cpu().numpy()
            bias = head["state"]["bias"].detach().cpu().numpy()
            path = source / dataset / f"seed_{seed}/test_diagnostic_predictions.npz"
            source_hash = sha256(path)
            with np.load(path, allow_pickle=False) as saved:
                valid = saved["teacher_valid_indices"]
                labels = saved["labels"][valid]
                keys = saved["sample_keys"][valid].astype(str)
                if dataset == "derm7pt":
                    from experiments.controls.run_derm7pt_mechanism import case_metadata
                    case_ids = case_metadata(keys)[0]
                else:
                    case_ids = None
                new_rows = evaluate_controls(dataset=dataset, seed=seed,
                    labels=labels, keys=keys, base=saved["frozen_logits"][valid],
                    decoded_logits=saved["decoded_logits"][valid],
                    decoded_inputs=saved["decoded_head_inputs_valid"],
                    teacher_inputs=saved["teacher_head_inputs_valid"],
                    weight=w, bias=bias, eta=float(head["eta"]),
                    output=output / dataset / f"seed_{seed}",
                    permutations=args.permutations, exclude_self=args.exclude_self,
                    case_ids=case_ids)
            if source_hash != sha256(path) or sha256(model_path) != record["model_sha256"]:
                raise AssertionError("A source input changed during evaluation.")
            rows.extend(new_rows)
            records.append(dict(dataset=dataset, seed=seed, prediction_path=str(path),
                prediction_sha256=source_hash, model_path=str(model_path),
                model_sha256=record["model_sha256"]))
            print(f"[{dataset} seed={seed}] valid={len(labels)} complete", flush=True)
    summary = write_summary(output, rows)
    write_json(output / "protocol.json", dict(status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
        datasets=args.datasets, seeds=args.seeds, records=records,
        permutations=args.permutations, exclude_self=args.exclude_self,
        model_retrained=False, masks_regenerated=False,
        local_models_not_server_results=True, runner_sha256=sha256(__file__),
        elapsed_seconds=time.monotonic() - started))
    for row in summary:
        if row["cohort"] == "all_changed":
            print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
        default=ROOT / "outputs/posthoc_shape_diagnostics_20261022_final")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["CUB_200_2011", "derm7pt"],
                        default=["CUB_200_2011", "derm7pt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--exclude-self", action="store_true")
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("At least one permutation is required.")
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)
