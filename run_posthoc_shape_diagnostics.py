"""Read-only replay: complete affine constants and fixed-head teacher replacement.

This diagnostic does not train, segment, tune, or alter existing results. It
supports the serialized local pairing-control models, including their extra
training-fitted coordinate scaler. Results belong to those selected checkpoints,
not to manuscript/server last-15-epoch means.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch

from run_decoder_baselines import SPLITS, load_cached_inputs, sha256, write_csv, write_json
from utils.residual_controls import apply_affine_map, frozen_semantic_logits

ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = {
    "CUB_200_2011": ROOT / "outputs/cub_pairing_controls_20261020_pilot01",
    "derm7pt": ROOT / "outputs/derm7pt_mechanism_20261020_pilot01",
}


def array(value):
    return np.asarray(value.detach().cpu() if torch.is_tensor(value) else value,
                      dtype=np.float64)


def compose_affine(decoder_weight, decoder_bias, head_weight, head_bias,
                   coordinate_mean, coordinate_scale):
    """For v=(Bh+b_dec-mu)/s and r=Wv+b_head, return r=Ah+c.

    h is standardized by the SAVED training input scaler. Thus h=0 means the
    training-mean embedding, not a black image or absent object.
    """
    B, b_dec, W, b_head, mu, scale = map(array, (
        decoder_weight, decoder_bias, head_weight, head_bias,
        coordinate_mean, coordinate_scale))
    if B.ndim != 2 or W.ndim != 2 or W.shape[1] != B.shape[0]:
        raise ValueError("Decoder/head dimensions do not agree.")
    if any(x.shape != (B.shape[0],) for x in [b_dec, mu, scale]):
        raise ValueError("Invalid coordinate scaler or decoder bias dimensions.")
    if b_head.shape != (W.shape[0],) or np.any(scale <= 0):
        raise ValueError("Invalid head bias or coordinate scales.")
    if not all(np.isfinite(x).all() for x in [B, b_dec, W, b_head, mu, scale]):
        raise ValueError("Nonfinite model parameters.")
    scaled_weight = W / scale[None, :]
    return scaled_weight @ B, scaled_weight @ (b_dec - mu) + b_head


def compose_input_scaler(weight, bias, input_mean, input_scale):
    """Also propagate the input scaler for the supplementary raw-CLIP origin."""
    A,c,mu,scale=map(array,(weight,bias,input_mean,input_scale))
    if mu.shape!=(A.shape[1],) or scale.shape!=mu.shape or np.any(scale<=0):
        raise ValueError('Invalid CLIP input scaler dimensions or scales.')
    raw_weight=A/scale[None,:]
    return raw_weight,c-raw_weight@mu


def classification_row(dataset, seed, condition, subset, logits, base, labels, mask):
    n = int(mask.sum())
    if not n:
        return None
    predicted, frozen = logits.argmax(1)[mask], base.argmax(1)[mask]
    target = labels[mask]
    good, old_good = predicted == target, frozen == target
    corrected = int((~old_good & good).sum())
    harmed = int((old_good & ~good).sum())
    unchanged = predicted == frozen
    row = {"dataset": dataset, "seed": seed, "condition": condition, "subset": subset,
        "n_images": n, "accuracy_pct": float(100*good.mean()),
        "delta_frozen_pp": float(100*(good.mean()-old_good.mean())),
        "n_corrected": corrected, "n_harmed": harmed,
        "correction_pct": float(100*corrected/(~old_good).sum()) if (~old_good).any() else None,
        "harm_pct": float(100*harmed/old_good.sum()) if old_good.any() else None,
        "prediction_agreement_with_frozen_pct": float(100*unchanged.mean())}
    np.testing.assert_allclose(row["delta_frozen_pp"], 100*(corrected-harmed)/n, atol=1e-10)
    return row


def fixed_contributions(semantic, decoded_logits, decoded_inputs, teacher_inputs,
                        head_weight, head_bias, eta):
    """All contrasts are chosen ONCE from the original decoded/frozen outputs.

    Unchanged predictions have f=b and a trivial zero decomposition; callers
    report explanatory alignment only on changed predictions with valid teacher.
    """
    semantic, decoded_logits, vd, vt, W, bias = map(array, (
        semantic, decoded_logits, decoded_inputs, teacher_inputs, head_weight, head_bias))
    b, f = semantic.argmax(1), decoded_logits.argmax(1)
    difference = W[f] - W[b]
    q_dec = eta * difference * vd
    q_teacher = eta * difference * vt
    bias_margin = eta * (bias[f]-bias[b])
    rows = np.arange(len(b))
    semantic_margin = semantic[rows, f]-semantic[rows, b]
    decoded_margin = decoded_logits[rows, f]-decoded_logits[rows, b]
    teacher_logits = semantic + eta*(vt @ W.T+bias)
    teacher_margin = teacher_logits[rows, f]-teacher_logits[rows, b]
    np.testing.assert_allclose(semantic_margin+q_dec.sum(1)+bias_margin,
                               decoded_margin, atol=2e-9, rtol=2e-10)
    np.testing.assert_allclose(semantic_margin+q_teacher.sum(1)+bias_margin,
                               teacher_margin, atol=2e-9, rtol=2e-10)
    return {"b": b, "f": f, "q_decoded": q_dec, "q_teacher": q_teacher,
        "semantic_margin": semantic_margin, "decoded_margin": decoded_margin,
        "teacher_margin_same_pair": teacher_margin, "bias_margin": bias_margin,
        "teacher_logits": teacher_logits}


def ranked_alignment(q_dec, q_teacher, topk=3, epsilon=1e-10):
    """Compare fixed decoded top-k signs; tied rankings use prototype index order."""
    q_dec, q_teacher = array(q_dec), array(q_teacher)
    order_d = np.argsort(-np.abs(q_dec), kind="stable")[:topk]
    order_t = np.argsort(-np.abs(q_teacher), kind="stable")[:topk]
    eligible = (np.abs(q_dec[order_d]) > epsilon) & (np.abs(q_teacher[order_d]) > epsilon)
    matches = np.sign(q_dec[order_d][eligible]) == np.sign(q_teacher[order_d][eligible])
    rank_correlation = None
    if np.ptp(q_dec) > epsilon and np.ptp(q_teacher) > epsilon:
        value = float(spearmanr(q_dec, q_teacher).statistic)
        if np.isfinite(value):
            rank_correlation = value
    return {"decoded_top_ids": order_d.tolist(), "teacher_top_ids": order_t.tolist(),
        "sign_eligible": int(eligible.sum()), "sign_matches": int(matches.sum()),
        "top3_sign_agreement_pct": float(100*matches.mean()) if len(matches) else None,
        "top3_overlap_pct": 100*len(set(order_d)&set(order_t))/topk,
        "signed_contribution_spearman": rank_correlation}


def teacher_permutation_reference(dataset, seed, contrast, teacher_inputs, W, b_head, eta, labels,
                                  n_permutations=20):
    """Diagnostic null: preserve teacher marginals, break test-image pairing.

    These are evaluation-only permutations, not new trained seeds, not model
    selection, and not a class-conditional shuffle. Every comparison uses the
    original decoded b/f pair and original changed-image subset.
    """
    changed = np.flatnonzero(contrast['b'] != contrast['f'])
    differences = W[contrast['f'][changed]]-W[contrast['b'][changed]]
    rng = np.random.default_rng(100000+seed)
    q_dec = contrast['q_decoded'][changed]
    semantic = contrast['teacher_logits']-eta*(teacher_inputs@W.T+b_head)
    rows = []
    for permutation in range(n_permutations):
        order = rng.permutation(len(teacher_inputs))
        shuffled = teacher_inputs[order]
        logits = semantic+eta*(shuffled@W.T+b_head)
        q_teacher = eta*differences*shuffled[changed]
        align = [ranked_alignment(a,b) for a,b in zip(q_dec,q_teacher)]
        eligible = sum(r['sign_eligible'] for r in align)
        matches = sum(r['sign_matches'] for r in align)
        def mean(key):
            values = [r[key] for r in align if r[key] is not None]
            return float(np.mean(values)) if values else None
        rows.append({'dataset':dataset,'seed':seed,'permutation':permutation,
            'n_valid_images':len(labels),'n_original_changed_images':len(changed),
            'accuracy_pct':float(100*(logits.argmax(1)==labels).mean()),
            'top3_sign_agreement_pct':100*matches/eligible if eligible else None,
            'top3_overlap_pct':mean('top3_overlap_pct'),
            'signed_contribution_spearman':mean('signed_contribution_spearman')})
    return rows


def aggregate_rows(rows, keys, measurements):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for group, members in groups.items():
        result = dict(zip(keys, group))
        result["n_seeds"] = len(members)
        if all("n_images" in r for r in members):
            counts = [r["n_images"] for r in members]
            result["n_images_min"] = min(counts)
            result["n_images_max"] = max(counts)
        for key in measurements:
            values = [r[key] for r in members if r.get(key) is not None]
            result["mean_"+key] = float(np.mean(values)) if values else None
            result["std_"+key] = float(np.std(values, ddof=1)) if len(values)>1 else None
        out.append(result)
    return out


def diagnostic_seed(dataset, seed, run_root, out, h, teacher, valid, keys, labels, semantic):
    folder = run_root / f"seed_{seed}/paired"
    state_path = folder / "selected_model.pt"
    model_hash = sha256(state_path)
    state = torch.load(state_path, weights_only=False, map_location="cpu")
    if state.get("mode") != "paired" or state.get("seed") != seed:
        raise ValueError("Expected a true-pairing checkpoint with the specified seed.")
    head = state["head"]["selected"]
    eta = float(head["eta"])
    B, b_dec = map(array, [state["decoder_state"]["weight"], state["decoder_state"]["bias"]])
    W, b_head = map(array, [head["state"]["weight"], head["state"]["bias"]])
    if B.shape != (128,768):
        raise ValueError("Expected the saved 128-dimensional geo11 / CLIP-L14 branch.")
    scaler = state["coordinate_scaler"]
    mu, scale = array(scaler.mean_), array(scaler.scale_)
    A, constant = compose_affine(B, b_dec, W, b_head, mu, scale)
    raw_A,raw_constant=compose_input_scaler(A,constant,state['input_scaler'].mean_,
                                           state['input_scaler'].scale_)
    target_folder = out / dataset / f"seed_{seed}"
    target_folder.mkdir(parents=True)
    replay = []
    dec = apply_affine_map(h, state["decoder_state"], "cpu")
    for split in SPLITS:
        vd = scaler.transform(dec[split]).astype(np.float32)
        manual = semantic[split]+eta*(vd @ W.T+b_head)
        prediction_file = folder / f"{split}_predictions.npz"
        with np.load(prediction_file, allow_pickle=False) as saved:
            assert np.array_equal(saved["sample_keys"].astype(str), keys[split])
            assert np.array_equal(saved["labels"], labels[split])
            assert np.array_equal(saved["valid_teacher"], valid[split])
            np.testing.assert_allclose(saved["semantic_logits"], semantic[split], atol=2e-6, rtol=0)
            np.testing.assert_allclose(saved["decoded"], dec[split], atol=3e-5, rtol=2e-5)
            np.testing.assert_allclose(saved["logits"], manual, atol=3e-5, rtol=2e-5)
            assert np.array_equal(saved["logits"].argmax(1), manual.argmax(1))
            max_error = float(np.max(np.abs(saved["logits"]-manual)))
        replay.append({"split": split, "max_saved_logit_error": max_error,
                       "all_saved_predictions_match": True,
                       "source_predictions": str(prediction_file), "sha256": sha256(prediction_file)})
    # Compute in float64 for strict composition and contribution closure checks.
    input_x = array(h["test"])
    decoded_raw = input_x @ B.T+b_dec
    decoded_inputs = (decoded_raw-mu)/scale
    base = array(semantic["test"])
    full = base+eta*(decoded_inputs @ W.T+b_head)
    composed = base+eta*(input_x @ A.T+constant)
    np.testing.assert_allclose(full, composed, atol=2e-9, rtol=2e-10)
    # Same affine map in original cached CLIP coordinates, with every input
    # mean/scale propagated. Raw zero need not be a realizable image embedding.
    raw_x=input_x*array(state['input_scaler'].scale_)+array(state['input_scaler'].mean_)
    raw_composed=base+eta*(raw_x@raw_A.T+raw_constant)
    np.testing.assert_allclose(full,raw_composed,atol=2e-9,rtol=2e-10)
    with np.load(folder/"test_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(full, saved["logits"], atol=4e-5, rtol=2e-5)
        assert np.array_equal(full.argmax(1), saved["logits"].argmax(1))
    constant_logits = base+eta*constant
    head_bias_logits = base+eta*b_head
    vt_all = (array(teacher["test"])-mu)/scale
    all_mask = np.ones(len(input_x), dtype=bool)
    groups = {"all": all_mask, "valid_teacher": valid["test"], "missing_teacher": ~valid["test"]}
    table = []
    conditions = {"frozen":base, "head_bias_only":head_bias_logits,
                  "complete_constant":constant_logits,
                  "raw_clip_zero_constant":base+eta*raw_constant,"decoded_full":full}
    for condition, logits in conditions.items():
        for subset, mask in groups.items():
            row = classification_row(dataset, seed, condition, subset, logits, base, labels["test"], mask)
            if row:
                table.append(row)
    # A test teacher is only observed where valid=True. Never treat a placeholder
    # as a zero-shape observation or mix replacement with missing-teacher images.
    ids = np.flatnonzero(valid["test"])
    base_v, full_v = base[ids], full[ids]
    teacher_inputs = vt_all[ids]
    contrast = fixed_contributions(base_v, full_v, decoded_inputs[ids], teacher_inputs, W, b_head, eta)
    null_rows = teacher_permutation_reference(dataset,seed,contrast,teacher_inputs,W,b_head,eta,
                                               labels['test'][ids])
    write_csv(target_folder/'teacher_permutation_reference.csv',null_rows)
    replacement = contrast["teacher_logits"]
    delta = replacement-full_v
    for alpha in [0.25, 0.50, 0.75, 1.0]:
        interpolated = full_v+alpha*delta
        name = "teacher_replacement" if alpha==1 else f"teacher_interpolation_{alpha:.2f}"
        table.append(classification_row(dataset, seed, name, "valid_teacher", interpolated,
            base_v, labels["test"][ids], np.ones(len(ids),dtype=bool)))
    detailed, top_coordinates = [], []
    b, f = contrast["b"], contrast["f"]
    for local, original in enumerate(ids):
        if b[local]==f[local]:
            continue
        alignment = ranked_alignment(contrast["q_decoded"][local], contrast["q_teacher"][local])
        true_label = int(labels["test"][original])
        outcome = "correction" if f[local]==true_label else ("harm" if b[local]==true_label else "wrong_to_wrong")
        row = {"dataset":dataset,"seed":seed,"sample_key":str(keys["test"][original]),
            "outcome":outcome,"true_label":true_label,"frozen_b":int(b[local]),
            "decoded_f_fixed":int(f[local]),"teacher_argmax":int(replacement[local].argmax()),
            "semantic_margin":float(contrast["semantic_margin"][local]),
            "decoded_margin":float(contrast["decoded_margin"][local]),
            "teacher_margin_same_pair":float(contrast["teacher_margin_same_pair"][local]),
            "semantic_head_bias_margin":float(contrast["bias_margin"][local]),
            "sign_eligible":alignment["sign_eligible"],"sign_matches":alignment["sign_matches"],
            "top3_sign_agreement_pct":alignment["top3_sign_agreement_pct"],
            "top3_overlap_pct":alignment["top3_overlap_pct"],
            "signed_contribution_spearman":alignment["signed_contribution_spearman"],
            "decoded_top_ids":";".join(map(str,alignment["decoded_top_ids"])),
            "teacher_top_ids":";".join(map(str,alignment["teacher_top_ids"]))}
        detailed.append(row)
        for rank, proto in enumerate(alignment["decoded_top_ids"], start=1):
            top_coordinates.append({"dataset":dataset,"seed":seed,"sample_key":row["sample_key"],
                "outcome":outcome,"frozen_b":int(b[local]),"decoded_f_fixed":int(f[local]),
                "decoded_abs_rank":rank,"prototype_id":proto,
                "decoded_head_input":float(decoded_inputs[original,proto]),
                "teacher_head_input":float(teacher_inputs[local,proto]),
                "decoded_contribution":float(contrast["q_decoded"][local,proto]),
                "teacher_contribution":float(contrast["q_teacher"][local,proto])})
    alignments = []
    for outcome in ["all_changed", "correction", "harm", "wrong_to_wrong"]:
        rows = [r for r in detailed if outcome=="all_changed" or r["outcome"]==outcome]
        if not rows:
            continue
        eligible, matches = sum(r["sign_eligible"] for r in rows), sum(r["sign_matches"] for r in rows)
        def mean(key):
            values = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if values else None
        alignments.append({"dataset":dataset,"seed":seed,"outcome":outcome,
            "n_images":len(rows), "sign_eligible_coordinates":eligible,
            "top3_sign_agreement_pct":100*matches/eligible if eligible else None,
            "top3_overlap_pct":mean("top3_overlap_pct"),
            "signed_contribution_spearman":mean("signed_contribution_spearman"),
            "mean_decoded_margin":mean("decoded_margin"),
            "mean_teacher_margin_same_pair":mean("teacher_margin_same_pair")})
    magnitude = {"dataset":dataset,"seed":seed,"n_images":len(ids),
        "decoded_mean_abs_residual":float(np.abs(full_v-base_v).mean()),
        "teacher_mean_abs_residual":float(np.abs(replacement-base_v).mean()),
        "teacher_decoded_class_disagreement_pct":float(100*(replacement.argmax(1)!=full_v.argmax(1)).mean()),
        "replacement_additionally_corrected":int(((full_v.argmax(1)!=labels['test'][ids]) &
                                                  (replacement.argmax(1)==labels['test'][ids])).sum()),
        "replacement_additionally_harmed":int(((full_v.argmax(1)==labels['test'][ids]) &
                                               (replacement.argmax(1)!=labels['test'][ids])).sum())}
    np.savez_compressed(target_folder/"test_diagnostic_predictions.npz",
        sample_keys=keys["test"],labels=labels["test"],valid_teacher=valid["test"],
        frozen_logits=base,decoded_logits=full,head_bias_only_logits=head_bias_logits,
        complete_constant_logits=constant_logits,teacher_valid_indices=ids,
        raw_clip_zero_constant_logits=base+eta*raw_constant,
        teacher_replacement_valid_logits=replacement,
        decoded_head_inputs_valid=decoded_inputs[ids],teacher_head_inputs_valid=teacher_inputs,
        fixed_b=contrast["b"],fixed_f=contrast["f"],
        decoded_contributions=contrast["q_decoded"],teacher_contributions=contrast["q_teacher"],
        composed_constant=constant,head_bias=b_head,decoder_bias=b_dec,
        raw_clip_zero_constant=raw_constant,
        coordinate_mean=mu,coordinate_scale=scale)
    write_csv(target_folder/"changed_prediction_alignment.csv",detailed)
    write_csv(target_folder/"decoded_top3_contributions.csv",top_coordinates)
    verification = {"dataset":dataset,"seed":seed,"status":"complete",
        "model_path":str(state_path),"model_sha256":model_hash,"selected_eta":eta,
        "selected_epoch":head["best_epoch"],"saved_prediction_replay":replay,
        "affine_composition_max_error":float(np.max(np.abs(full-composed))),
        "raw_affine_composition_max_error":float(np.max(np.abs(full-raw_composed))),
        "complete_constant_uses_all_offsets":True,
        "coordinate_scaler":"Saved training-fitted post-decoder scaler, unchanged for teacher replacement.",
        "input_reference":"h_standardized=0, the training-mean CLIP embedding",
        "constant_l2":float(np.linalg.norm(eta*constant)),
        "head_bias_l2":float(np.linalg.norm(eta*b_head)),
        "complete_vs_head_bias_max_abs_logit_diff":float(np.max(np.abs(eta*(constant-b_head)))),
        "model_hash_unchanged":sha256(state_path)==model_hash,
        "teacher_interpolation_alphas":[0.0,0.25,0.5,0.75,1.0],
        "evaluation_teacher_permutations":20,
        "no_teacher_on_missing_images":True,"no_retraining":True}
    assert verification["model_hash_unchanged"]
    write_json(target_folder/"verification.json",verification)
    print(f"[{dataset}/seed{seed}] replay passed; constant/full/teacher-valid = " +
          "/".join(f"{r['accuracy_pct']:.4f}" for condition,subset in [
              ("complete_constant","all"),("decoded_full","all"),("teacher_replacement","valid_teacher")]
              for r in table if r['condition']==condition and r['subset']==subset),flush=True)
    return table,alignments,magnitude,verification,null_rows


def report(out, summary, alignment, magnitude, null_summary):
    def find(ds,condition,subset):
        return next(r for r in summary if r['dataset']==ds and r['condition']==condition and r['subset']==subset)
    def number(row):
        return f"{row['mean_accuracy_pct']:.2f} ± {row['std_accuracy_pct']:.2f}"
    datasets = list(dict.fromkeys(r['dataset'] for r in summary))
    lines = ["# 完整常数项与固定分类头 teacher 替换：本地诊断", "",
        "本次只回放已保存的真实配对模型（种子42、43、44），不重新训练、不生成mask、不修改论文。",
        "使用验证集已选定的单个检查点；均值±样本标准差来自三个已有模型，不是论文服务器末15轮均值。",
        "本地Frozen基线与论文不同；本地分类头前另有训练拟合的坐标标准化，已完整纳入计算。", "",
        "## 1. 完整常数项对照：全部测试图像", "",
        "设v=(Bh+b_dec-mu)/s，完整常数项为eta*[W*((b_dec-mu)/s)+b_head]。h=0表示训练均值嵌入。",
        "这是一项对已经拟合模型的输入依赖项消融，不是重新优化的常数分类器；特征中心不同会改变常数项定义。", "",
        "| 数据集 | 测试数 | Frozen | 仅head bias | 完整常数项 | Decoded完整分支 |",
        "|---|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        rs=[find(ds,c,'all') for c in ['frozen','head_bias_only','complete_constant','decoded_full']]
        lines.append(f"| {ds} | {rs[0]['n_images_min']} | " + " | ".join(number(r) for r in rs)+" |")
    lines += ["", "在当前本地模型中，后置scaler按训练decoded输出拟合，而输入已按训练均值中心化；因此mu约等于b_dec，传播的decoder常数几乎被抵消，完整常数项与head bias近乎相等。",
        "标准化原点对应训练均值，不是不存在对象。另保存raw_clip_zero_constant条件，将输入scaler的均值也展开，取原始缓存CLIP向量为零；这一参照依赖坐标原点，零向量通常不是实际图像嵌入，不能把它误当纯粹语义/几何分离。", "",
        "| 数据集 | 原始CLIP零向量处完整常数项（补充参照） |",
        "|---|---:|"]
    for ds in datasets:
        lines.append(f"| {ds} | {number(find(ds,'raw_clip_zero_constant','all'))} |")
    lines += ["", "## 2. 固定分类头 teacher 替换：仅有效teacher测试图像", "",
        "固定W、head bias、eta和所有训练scaler。只用原teacher替换decoder输出，再通过原坐标scaler。没有重新拟合head、调eta或选择最佳替换比例。",
        "固定头原先在decoded输入上训练；teacher与decoded分布可能不同，因此结果不是oracle上界，也不等同于另行训练head的Direct teacher。", "",
        "| 数据集 | 有效测试数 | Frozen | Decoded | Teacher替换（同一head） |",
        "|---|---:|---:|---:|---:|"]
    for ds in datasets:
        rs=[find(ds,c,'valid_teacher') for c in ['frozen','decoded_full','teacher_replacement']]
        lines.append(f"| {ds} | {rs[0]['n_images_min']} | " + " | ".join(number(r) for r in rs)+" |")
    lines += ["", "## 3. 原有改变预测样本的贡献对应关系", "",
        "只分析有teacher且decoded与Frozen预测不同的图像。固定原b=Frozen类别、f=Decoded类别，替换后不重选类别对。",
        "方向一致率在原Decoded的Top-3绝对贡献坐标上计算；两种贡献均非零才计入。Top-3重叠率=交集数/3。", "",
        "| 数据集 | 各种子改变预测图像数 | Top-3方向一致率/% | Top-3重叠率/% | 有符号贡献Spearman |",
        "|---|---:|---:|---:|---:|"]
    for ds in datasets:
        r=next((r for r in alignment if r['dataset']==ds and r['outcome']=='all_changed'),None)
        if r:
            lines.append(f"| {ds} | {r['n_images_min']}—{r['n_images_max']} | "
                f"{r['mean_top3_sign_agreement_pct']:.2f} ± {r['std_top3_sign_agreement_pct']:.2f} | "
                f"{r['mean_top3_overlap_pct']:.2f} ± {r['std_top3_overlap_pct']:.2f} | "
                f"{r['mean_signed_contribution_spearman']:.3f} ± {r['std_signed_contribution_spearman']:.3f} |")
    lines += ["", "贡献一致性为描述性统计，受同一分类头权重共享的影响，不单独证明因果形状机制。", "",
        "为校准上述统计，每个已有模型另作20次测试teacher向量全局置换，保留teacher边际分布，固定头及原有改变预测样本。",
        "这是评价阶段的无配对参照，不是重新训练的打乱监督模型，也不用于选择分类器；先对每种子20次置换求均值，再汇总三个种子。", "",
        "| 数据集 | 真实配对方向一致率/% | 无配对参照/% | 真实配对Top-3重叠/% | 无配对参照/% |",
        "|---|---:|---:|---:|---:|"]
    for ds in datasets:
        actual=next((r for r in alignment if r['dataset']==ds and r['outcome']=='all_changed'),None)
        null=next((r for r in null_summary if r['dataset']==ds),None)
        if actual and null:
            lines.append(f"| {ds} | {actual['mean_top3_sign_agreement_pct']:.2f} | "
                f"{null['mean_top3_sign_agreement_pct']:.2f} | {actual['mean_top3_overlap_pct']:.2f} | "
                f"{null['mean_top3_overlap_pct']:.2f} |")
    lines += ["",
        "其他输出包含逐种子全部/有效/无效teacher群体结果、correction/harm、全部固定替换比例、逐样本贡献及严格回放验证。",
        "不对共享病例的医学分区报告虚构的独立样本显著性结论。数据集不变，现有划分局限仍在。", ""]
    (out/"RESULTS_zh.md").write_text("\n".join(lines),encoding="utf-8")


def run(args):
    start = time.monotonic()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    protected = [ROOT.parents[1]/"lunwen/sn-article.tex",ROOT.parents[1]/"lunwen/sn-article.pdf"]
    before = {str(p):sha256(p) for p in protected if p.is_file()}
    all_rows,all_alignments,magnitudes,verifications,provenance = [],[],[],[],{}
    all_null_rows=[]
    run_roots = {"CUB_200_2011":args.cub_run_root,"derm7pt":args.derm_run_root}
    for dataset in args.datasets:
        print(f"[{dataset}] verify/read retained caches",flush=True)
        h,teacher,valid,keys,labels,manifest,sources = load_cached_inputs(args.cache_root,dataset)
        checkpoint_path=Path(manifest['sources']['checkpoint']['path'])
        assert sha256(checkpoint_path)==manifest['sources']['checkpoint']['sha256']
        sources['semantic_checkpoint']={"path":str(checkpoint_path),"sha256":sha256(checkpoint_path)}
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        input_scaler=torch.load(args.cache_root/dataset/'input_scaler.pt',map_location='cpu',weights_only=False)['h_scaler']
        semantic={}
        semantic_replay=[]
        for split in SPLITS:
            with np.load(args.cache_root/dataset/f'{split}_frozen.npz',allow_pickle=False) as z:
                semantic[split]=z['logits'].copy()
            # Use the original feature cache: inverse-transforming a float32
            # standardized array introduces avoidable round-trip error.
            raw_state=torch.load(Path(manifest['sources'][split]['path']),map_location='cpu',weights_only=False)
            raw=np.stack([np.asarray(raw_state[k][0].cpu()).reshape(-1)
                          for k in keys[split]]).astype(np.float32)
            del raw_state
            reproduced=frozen_semantic_logits(raw,checkpoint,'cpu')
            # Cache generation and CPU replay can use different float32 GEMM
            # kernels. Check every logit plus exact top-1 agreement separately.
            np.testing.assert_allclose(reproduced,semantic[split],atol=2e-4,rtol=2e-5)
            assert np.array_equal(reproduced.argmax(1),semantic[split].argmax(1))
            semantic_replay.append({'split':split,'all_predictions_match':True,
                'max_logit_error':float(np.max(np.abs(reproduced-semantic[split])))})
        for seed in args.seeds:
            st=torch.load(run_roots[dataset]/f'seed_{seed}/paired/selected_model.pt',map_location='cpu',weights_only=False)
            np.testing.assert_allclose(st['input_scaler'].mean_,input_scaler.mean_,atol=1e-10,rtol=0)
            np.testing.assert_allclose(st['input_scaler'].scale_,input_scaler.scale_,atol=1e-10,rtol=0)
            rows,align,mag,verification,null_rows=diagnostic_seed(dataset,seed,run_roots[dataset],out,
                h,teacher,valid,keys,labels,semantic)
            all_rows.extend(rows);all_alignments.extend(align);magnitudes.append(mag);verifications.append(verification)
            all_null_rows.extend(null_rows)
        for source in sources.values():
            assert sha256(Path(source['path']))==source['sha256']
        provenance[dataset]={"sources":sources,"class_order":manifest['class_order'],
            "test_n":len(labels['test']),"valid_teacher_test_n":int(valid['test'].sum()),
            "source_run":str(run_roots[dataset]),"semantic_replay":semantic_replay,
            "all_source_hashes_unchanged":True}
    summary=aggregate_rows(all_rows,['dataset','condition','subset'],[
        'accuracy_pct','delta_frozen_pp','n_corrected','n_harmed','correction_pct','harm_pct'])
    alignment_summary=aggregate_rows(all_alignments,['dataset','outcome'],[
        'top3_sign_agreement_pct','top3_overlap_pct','signed_contribution_spearman',
        'mean_decoded_margin','mean_teacher_margin_same_pair'])
    magnitude_summary=aggregate_rows(magnitudes,['dataset'],[
        'decoded_mean_abs_residual','teacher_mean_abs_residual',
        'teacher_decoded_class_disagreement_pct','replacement_additionally_corrected','replacement_additionally_harmed'])
    write_csv(out/'classification_per_seed.csv',all_rows)
    write_csv(out/'classification_summary.csv',summary)
    write_csv(out/'contribution_alignment_per_seed.csv',all_alignments)
    write_csv(out/'contribution_alignment_summary.csv',alignment_summary)
    write_csv(out/'replacement_shift_per_seed.csv',magnitudes)
    write_csv(out/'replacement_shift_summary.csv',magnitude_summary)
    write_csv(out/'teacher_permutation_reference.csv',all_null_rows)
    null_measurements=['accuracy_pct','top3_sign_agreement_pct','top3_overlap_pct',
                       'signed_contribution_spearman']
    null_per_seed=[]
    for dataset in args.datasets:
        for seed in args.seeds:
            rows=[r for r in all_null_rows if r['dataset']==dataset and r['seed']==seed]
            row={'dataset':dataset,'seed':seed,'n_permutations':len(rows)}
            for key in null_measurements:
                values=[r[key] for r in rows if r[key] is not None]
                row[key]=float(np.mean(values)) if values else None
                row['within_permutation_std_'+key]=float(np.std(values,ddof=1)) if len(values)>1 else None
            null_per_seed.append(row)
    null_summary=aggregate_rows(null_per_seed,['dataset'],null_measurements)
    write_csv(out/'teacher_permutation_per_seed_summary.csv',null_per_seed)
    write_csv(out/'teacher_permutation_summary.csv',null_summary)
    after={p:sha256(Path(p)) for p in before}
    assert before==after
    protocol={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),
        "datasets":args.datasets,"seeds":args.seeds,"source_models":"Existing true-pairing local pilots, validation-selected checkpoints",
        "fit_or_tune_on_test":False,"retrained_models":False,"generated_masks":False,
        "constant_definition":"At saved standardized CLIP input h=0: eta*[W*((b_dec-mu_coord)/s_coord)+b_head]",
        "teacher_replacement":"Use valid teachers only; keep the same fitted coordinate scaler, head, eta and original class contrast",
        "missing_teacher_placeholders_evaluated_as_teacher":False,
        "evaluation_only_teacher_permutations_per_model":20,
        "teacher_permutations_select_models":False,
        "reporting":"Mean and sample SD of three EXISTING selected-checkpoint models; not manuscript server epoch means",
        "original_paper_sha256":before,"paper_unchanged":before==after,
        "provenance":provenance,"verification":verifications,
        "python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,
        "elapsed_seconds":time.monotonic()-start,"runner_sha256":sha256(Path(__file__))}
    write_json(out/'protocol.json',protocol)
    report(out,summary,alignment_summary,magnitude_summary,null_summary)
    print(json.dumps({"output":str(out),"status":"complete","elapsed_seconds":protocol['elapsed_seconds'],
        "paper_unchanged":protocol['paper_unchanged']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,default=ROOT/'outputs/residual_controls_20261018_paired01')
    parser.add_argument('--cub-run-root',type=Path,default=DEFAULT_RUNS['CUB_200_2011'])
    parser.add_argument('--derm-run-root',type=Path,default=DEFAULT_RUNS['derm7pt'])
    parser.add_argument('--datasets',nargs='+',choices=list(DEFAULT_RUNS),default=list(DEFAULT_RUNS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):
        parser.error('Seeds must be distinct.')
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)


if __name__=='__main__':
    main()
