"""Auditable four-CUB review and exhaustive retained Derm7pt Top-7 replay.

This postprocessor neither fits a model nor changes segmentation/manuscript
assets. It reuses reviewed, exact-identity exemplar assets from the Top-5 scan.
Legacy trace coordinates remain a separate run; unexported legacy references
are marked unknown, never silently borrowed from the paired bank by number.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import html
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from tools.references.run_topk_reference_screen import (
    ROOT, WORK, CACHE, DATASETS, BODY, SourceLedger, digest, read_json,
    write_json, read_csv, write_csv, source_index, medical_metadata, norm,
    enrich, saved_run_arrays, checked_decomposition, export_case_sources,
    waterfall, font, paste_fit, POS, NEG,
)

PRIOR = WORK / '临时/Top5同类形状候选_20261024'
CUB_IDS = ('CUB-R42-1425', 'CUB-R44-0667', 'CUB-R43-1508', 'CUB-R44-0600')
STATUS = {'correction': '纠正：原错→现对', 'harm': '改错：原对→现错',
          'unchanged_correct': '保持正确', 'unchanged_error': '保持错误',
          'other_flip': '错误类别间改变'}
CLASS_ZH = {'benign': '良性', 'malignant': '恶性'}
CUB_NOTES = {
 'CUB-R42-1425': '四例中唯一的实际纠正。五项都是鸟体参考，但参考原图均非白颈渡鸦，适合说明跨鸟种共享形状参考，不宜写成同种部件一致支持。Other 与偏置也参与了最终翻转。',
 'CUB-R44-0667': '预测前后都正确。存在同种鸟体参考 P66，但它的标准化响应与权重差都为负，乘积才为正；不能描述为高相似度。整个残差使所选类别对的间隔缩小，不能当作提高分类间隔的实例。',
 'CUB-R43-1508': '预测前后都正确。同种参考 P58 对应喙部，响应与权重差均为正，局部展示较直观。Top-5 合计为负；最终间隔的小幅增加来自未显示项和偏置的共同作用。',
 'CUB-R44-0600': '预测前后都正确。同种参考 P95 的头部及身体前部贡献为正，但另两项正贡献来自背景，不能包装为纯鸟体形状解释。适合同时展示有意义参考与干扰参考。',
}
DERM_NOTES = {
 'Derm7pt-T-0214': '这三个数量达标的纠正例中，病灶项对最终正确类别的合计贡献为正（约 +0.653），较适合展示纠正的分解过程；但三项病灶参考均来自恶性原图，当前正确类别是良性，不能写成“同诊断的病灶参考支持正确类别”。它们是跨类别几何参考。',
 'Derm7pt-T-0346': '三项病灶参考合计约 −0.380，抵消而非推动此次正确翻转；偏置约 +0.470。适合展示分解的复杂性，不宜作为“病灶参考促成纠正”的主打实例。',
 'Derm7pt-T-0350': '病灶参考数量最多（四项），但病灶项合计约 −0.459，偏置约 +0.470。因此，它不能用来证明病灶贡献推动了纠正；不能只因有四个清楚病灶就优先选入正文。',
 'Derm7pt-T-0048': '这是唯一满足三项病灶参考的改错例，但恶性来源的两项病灶参考为错误的良性预测提供正贡献，另有病灶项为负。它不符合希望通过背景干扰解释改错的直观展示目标，建议保留比较而不优先采用。',
 'Derm7pt-R42-0117': '更适合展示非病灶关联的改错备选：七项均为正，六项来自标尺/网格相关训练区域，一项来自混合视野；没有清楚病灶参考。偏置仅约 +0.004。它不满足“三项病灶”门槛，但比数量达标的改错例更贴近希望展示的干扰参考现象。',
 'Derm7pt-T-0106': '适合与原始 T 纠正例配对比较：四项正贡献均来自标尺/网格相关参考，唯一清楚病灶参考为负，抵消错误的良性预测。需保留约 +0.470 的偏置，不能声称错误全部由标尺造成。',
}


def protected_snapshot():
    base = WORK / '临时/图1_R44-0877_20261024'
    return {str(p): digest(p) for name in ('图1', '图2')
            for p in (base / name).rglob('*') if p.is_file()}


def annotate(cases, refs):
    enrich(cases, refs)
    for c in cases:
        top = c['top']
        c['strict_three_lesion'] = c['dataset'] == 'Derm7pt' and c['body_count'] >= 3
        c['unknown_reference_count'] = sum(r.get('reference_missing', False) for r in top)
        c['readable_body_count'] = sum(r['is_body'] and (
            (r['q'] > 0 and r['same_final_class']) or (r['q'] < 0 and not r['same_final_class'])) for r in top)
        positive = sum(max(r['q'], 0) for r in top)
        c['positive_nonlesion_count'] = sum(r['q'] > 0 and r['category'] in ('artifact', 'skin', 'field', 'background') for r in top)
        c['positive_artifact_count'] = sum(r['q'] > 0 and r['category'] == 'artifact' for r in top)
        c['positive_nonlesion_share'] = sum(max(r['q'], 0) for r in top if r['category'] in ('artifact', 'skin', 'field', 'background')) / max(positive, 1e-12)
        c['positive_artifact_share'] = sum(max(r['q'], 0) for r in top if r['category'] == 'artifact') / max(positive, 1e-12)
        c['body_q_sum'] = math.fsum(r['q'] for r in top if r['is_body'])
        c['nonlesion_q_sum'] = math.fsum(r['q'] for r in top if r['category'] in ('artifact', 'skin', 'field', 'background'))
        c['counterintuitive_body_count'] = c['body_count'] - c['readable_body_count']
        c['full_residual_margin'] = c['final_margin'] - c['semantic_margin']
        c['strict_three_readable_body'] = c['dataset'] == 'Derm7pt' and c['readable_body_count'] >= 3
    return cases


def scan(out):
    if out.exists():
        raise FileExistsError('Use a new output directory, or --render-existing for this export.')
    out.mkdir(parents=True)
    write_json(out / 'selected_folders_before.json', protected_snapshot())
    ledger = SourceLedger()
    refs = {r['reference_id']: r for r in read_json(ledger.record(PRIOR / 'reviewed_references.json'))}
    raw = read_json(ledger.record(PRIOR / 'all_cases_scan.json'))
    cub = [copy.deepcopy(next(c for c in raw if c['candidate_id'] == cid)) for cid in CUB_IDS]
    for c in cub:
        c['split'] = 'test'
        ledger.record(c['model_path']); ledger.record(c['prediction_path'])
    medical, retained = medical_metadata()
    ledger.record(ROOT / 'dataset_all/derm7pt/meta/meta.csv')
    cfg = DATASETS['Derm7pt']
    root_cache = CACHE / cfg['dataset']
    manifest = read_json(ledger.record(root_cache / 'dataset.json'))
    classes = manifest['class_order']
    resolve, _ = source_index(ROOT / cfg['split'])
    lookup = {int(alias[3:]): r['reference_id'] for r in refs.values()
              if r['dataset'] == 'Derm7pt' for alias in r['aliases'] if alias.startswith('R/P')}
    import torch
    bank_path = ledger.record(root_cache / 'teacher_bank.pt')
    bank = torch.load(bank_path, map_location='cpu', weights_only=False)
    for pid in range(128):
        ref = refs[lookup[pid]]
        evidence = read_json(ledger.record(PRIOR / 'references' / ref['reference_id'] / 'evidence.json'))
        if bank['prototype_representatives'][pid] != ref['representative_key']:
            raise ValueError('Paired-bank representative identity changed')
        if bank['prototype_metadata'][pid]['raw_descriptor'] != evidence['metadata']['raw_descriptor']:
            raise ValueError('Paired-bank descriptor identity changed')
    cases, checks, runs = [], [], []
    for split in ('train', 'val', 'test'):
        frozen_path = ledger.record(root_cache / f'{split}_frozen.npz')
        with np.load(frozen_path, allow_pickle=False) as fz:
            frozen, labels, keys = (fz[k].copy() for k in ('logits', 'labels', 'sample_keys'))
        keys = keys.astype(str)
        for seed in (42, 43, 44):
            folder = ROOT / 'outputs' / cfg['models'] / f'seed_{seed}' / 'paired'
            model_path = ledger.record(folder / 'selected_model.pt')
            pred_path = ledger.record(folder / f'{split}_predictions.npz')
            saved, branch, shape, weight, bias = saved_run_arrays(str(model_path), str(pred_path))
            if not np.array_equal(keys, saved['sample_keys'].astype(str)) or not np.array_equal(labels, saved['labels']):
                raise ValueError('Split membership or labels differ from Frozen cache')
            np.testing.assert_allclose(frozen, saved['semantic_logits'], rtol=3e-5, atol=6e-5)
            final = saved['semantic_logits'] + shape
            np.testing.assert_allclose(final, saved['logits'], rtol=3e-5, atol=6e-5)
            if not np.array_equal(final.argmax(1), saved['logits'].argmax(1)):
                raise ValueError('Replayed top-1 differs')
            check = dict(run=f'Derm7pt-R{seed}', split=split, n_samples=len(keys),
                         max_logit_error=float(abs(final - saved['logits']).max()),
                         all_top1_predictions_reproduced=True, max_closure_error=0.)
            runs.append(dict(run=check['run'], split=split, model_path=str(model_path),
                             model_sha256=digest(model_path), prediction_path=str(pred_path),
                             prediction_sha256=digest(pred_path), teacher_bank_sha256=digest(bank_path)))
            for i, key in enumerate(keys):
                f, base, truth = int(final[i].argmax()), int(frozen[i].argmax()), int(labels[i])
                changed = f != base
                if changed:
                    b = base
                    category = 'correction' if f == truth else ('harm' if base == truth else 'other_flip')
                    rule = 'final_vs_Frozen_prediction'
                else:
                    other_logits = frozen[i].copy(); other_logits[base] = -np.inf
                    b = int(other_logits.argmax())
                    category = 'unchanged_correct' if f == truth else 'unchanged_error'
                    rule = 'same_prediction_vs_Frozen_runner_up'
                dw = weight[f] - weight[b]
                q = branch[i].astype('float64') * dw
                order = np.lexsort((np.arange(len(q)), -abs(q)))[:7]
                sem, end, qb = float(frozen[i, f] - frozen[i, b]), float(final[i, f] - final[i, b]), float(bias[f] - bias[b])
                closure = abs(sem + math.fsum(q) + qb - end)
                if closure > 1.2e-4:
                    raise ValueError('Full 128-coordinate decomposition failed')
                check['max_closure_error'] = max(check['max_closure_error'], closure)
                original = resolve(split, key)
                meta = medical[original.name.lower()]
                if norm(meta['source_class']) != norm(classes[truth]):
                    raise ValueError('Metadata diagnosis disagrees with saved label')
                cid = f'Derm7pt-R{seed}-' + ('' if split == 'test' else split + '-') + f'{i:04d}'
                cases.append(dict(candidate_id=cid, dataset='Derm7pt', run=f'R{seed}', split=split,
                    index=i, sample_key=str(key), input_path=str(original), category=category,
                    prediction_changed=changed, true_name=classes[truth], frozen_name=classes[base],
                    final_name=classes[f], contrast_name=classes[b], contrast_rule=rule,
                    semantic_margin=sem, final_margin=end, bias=qb, other=float(math.fsum(q) - math.fsum(q[order])),
                    topk_sum=float(math.fsum(q[order])), coverage=float(abs(q[order]).sum() / max(abs(q).sum(), 1e-12)),
                    closure_error=closure, test_teacher_valid=bool(saved['valid_teacher'][i]),
                    prediction_path=str(pred_path), model_path=str(model_path),
                    query_case_num=meta['case_num'], query_diagnosis=meta['diagnosis'], query_modality=meta['modality'],
                    query_case_other_splits=sorted({r['split'] for r in retained[meta['case_num']] if r['split'] != split}),
                    top=[dict(prototype_id=int(pid), rank=j+1, reference_id=lookup[int(pid)],
                              q=float(q[pid]), activation=float(branch[i, pid]),
                              effective_weight_difference=float(dw[pid])) for j, pid in enumerate(order)]))
            checks.append(check)
            print(f'Replayed {split}/R{seed}: {len(keys)} images', flush=True)
    # Extend complete original trace records to seven coordinates using their
    # OWN reference map. Unknown coordinate references are explicit omissions.
    tlookup = {int(alias[3:]): r['reference_id'] for r in refs.values()
               if r['dataset'] == 'Derm7pt' for alias in r['aliases'] if alias.startswith('T/P')}
    unknowns = []
    for old in raw:
        if old['dataset'] != 'Derm7pt' or old['run'] != 'T':
            continue
        c = copy.deepcopy(old)
        folder = Path(c['trace_case'])
        summary = read_json(ledger.record(folder / 'case_summary.json'))
        coords = read_csv(ledger.record(folder / 'shape_coordinates.csv'))
        logits = read_csv(ledger.record(folder / 'class_logits.csv'))
        top, other, closure = checked_decomposition(summary, coords, logits, topk=7)
        displayed = []
        for rank, row in enumerate(top, 1):
            pid = int(row['prototype_id'])
            rid = tlookup.get(pid)
            missing = rid is None
            if missing:
                rid = f'Derm7pt-T-unexported-P{pid}'
                refs.setdefault(rid, dict(reference_id=rid, dataset='Derm7pt',
                    representative_key='', original_path='', original_name='未导出参考',
                    category='unreviewed', label_zh='原始记录未导出此坐标的参考图', label_en='Unexported reference',
                    source_class='unknown', case_num='unknown', aliases=[f'T/P{pid}'],
                    context_status='not_available', geometry_status='not_available'))
                unknowns.append(dict(candidate_id=c['candidate_id'], prototype_id=pid))
            displayed.append(dict(prototype_id=pid, rank=rank, reference_id=rid,
                q=float(row['signed_margin_contribution']), activation=float(row['activation']),
                effective_weight_difference=float(row['weight_difference_f_minus_b']), reference_missing=missing))
        c.update(split='test', top=displayed, other=other, closure_error=closure,
                 topk_sum=math.fsum(r['q'] for r in displayed),
                 coverage=math.fsum(abs(r['q']) for r in displayed) / max(math.fsum(abs(float(r['signed_margin_contribution'])) for r in coords), 1e-12))
        cases.append(c)
    cases = annotate(cub + cases, refs)
    write_json(out / 'all_cases.json', cases)
    write_json(out / 'references.json', list(refs.values()))
    write_json(out / 'replay_checks.json', checks)
    write_json(out / 'runs.json', runs)
    write_json(out / 'legacy_unexported_references.json', unknowns)
    write_json(out / 'source_inputs.json', ledger.verify())
    print('Scan complete:', len(cases), 'records', flush=True)


def choose(cases):
    strict = [c for c in cases if c['strict_three_lesion']]
    # Supplementary examples are separately labeled, never counted as meeting
    # the three-lesion criterion. Prioritize test members and readable content.
    corrections = sorted([c for c in cases if c['dataset'] == 'Derm7pt' and c['split'] == 'test'
                          and c['category'] == 'correction' and not c['strict_three_lesion']
                          and not c['unknown_reference_count']], key=lambda c: (
        -c['readable_body_count'], -c['body_count'], c['counterintuitive_body_count'],
        not c['preferred_medical_case'], -c['coverage'], c['candidate_id']))
    harms = sorted([c for c in cases if c['dataset'] == 'Derm7pt' and c['split'] == 'test'
                    and c['category'] == 'harm' and not c['strict_three_lesion']
                    and c['positive_nonlesion_count'] >= 3 and not c['unknown_reference_count']], key=lambda c: (
        c['counterintuitive_body_count'] > 0, -c['positive_artifact_count'],
        -c['positive_artifact_share'], not c['preferred_medical_case'],
        -c['positive_nonlesion_share'], c['candidate_id']))
    def unique(items, limit):
        result, seen = [], set()
        for c in items:
            if c['sample_key'].lower() not in seen:
                result.append(c); seen.add(c['sample_key'].lower())
            if len(result) == limit:
                break
        return result
    supplementary = unique(corrections, 6) + unique(harms, 8)
    ids = set(CUB_IDS) | {c['candidate_id'] for c in strict + supplementary}
    rendered = [c for c in cases if c['candidate_id'] in ids]
    return strict, supplementary, rendered


def pure_panel(c, out):
    n = len(c['top'])
    image = Image.new('RGB', (330 + 300*n, 650), 'white')
    d = ImageDraw.Draw(image)
    paste_fit(image, c['input_path'], (10, 125, 280, 390))
    d.line((305, 12, 305, 635), fill='#CDD6DD', width=2)
    for j, row in enumerate(c['top']):
        base = out / 'references' / row['reference_id']; x = 322 + 300*j
        for name, y, height in (('training_context.png', 10, 245),
                                ('neutral_crop.png', 270, 140), ('verified_contour.png', 425, 150)):
            paste_fit(image, base / name, (x, y, 280, height))
        d.text((x+140, 610), f"{row['q']:+.3f}", anchor='mm', font=font(35), fill=POS if row['q'] > 0 else NEG)
    image.save(out / 'panels' / (c['candidate_id'] + '.png'), dpi=(300, 300))


def prediction_flow(c, out):
    image = Image.new('RGB', (1800, 290), 'white'); d = ImageDraw.Draw(image)
    labels = [('Frozen DCBM', c['frozen_name'], f"Margin {c['semantic_margin']:+.3f}"),
              ('Decoded shape residual', '', f"Total {c['full_residual_margin']:+.3f}"),
              ('Fused prediction', c['final_name'], f"Margin {c['final_margin']:+.3f}")]
    for j, (title, name, value) in enumerate(labels):
        x = 20 + 595*j
        d.rounded_rectangle((x, 35, x+540, 255), radius=15, fill=('#EEEEEE', '#DDEDF7', '#FAE9DD')[j], outline='#BAC6CE', width=2)
        for text, y, size in ((title, 69, 29), (name.replace('_', ' '), 125, 27), (value, 190, 29)):
            d.text((x+270, y), text, anchor='mm', font=font(size), fill='#263541')
        if j < 2:
            a = x+550
            d.line((a, 145, a+33, 145), fill='#607D8B', width=5)
            d.polygon([(a+35,145),(a+23,135),(a+23,155)], fill='#607D8B')
    image.save(out / 'flows' / (c['candidate_id']+'.png'), dpi=(300, 300))


STYLE = """body{font:16px/1.65 'Microsoft YaHei','Segoe UI',sans-serif;color:#263541;background:#f4f7f9;margin:0}main{max-width:1500px;margin:auto;padding:24px}a{color:#006c94}h1{font-size:27px}h2{font-size:22px}.box,article{background:white;border:1px solid #d7e1e8;border-radius:10px;padding:20px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d7e1e8;padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf0f4}.scroll{overflow-x:auto}img{max-width:100%;height:auto}.query{max-width:440px;max-height:420px;object-fit:contain}.ref-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.ref{border:1px solid #d7e1e8;border-radius:8px;padding:12px;min-width:0}.ref img{display:block;width:100%;height:200px;object-fit:contain;background:white}.ref .small{height:145px}.pos{color:#267d70}.neg{color:#b96050}.note{border-left:4px solid #d9a548;padding:12px 16px;background:#fffbef}.muted{color:#566877}input,select,button{font:inherit;padding:7px;margin:4px;border:1px solid #b7c8d1;border-radius:5px}code{overflow-wrap:anywhere}.stats{display:flex;flex-wrap:wrap;gap:20px}.tag{display:inline-block;background:#eaf0f4;border-radius:4px;padding:2px 8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.thumb{max-height:160px;max-width:220px;object-fit:contain}"""


def esc(value):
    return html.escape(str(value))


def cname(value):
    return CLASS_ZH.get(value, value.replace('_', ' '))


def page(title, body):
    return f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}\n.ref{{overflow-wrap:anywhere}}</style><main>{body}</main></html>'


def case_page(c, refs, out):
    cid = c['candidate_id']; n = len(c['top'])
    body = f'<a href="../index.html">← 返回总入口</a><h1>{cid} · {STATUS[c["category"]]}</h1>'
    body += f'<div class="box"><div class="stats"><img class="query" src="../cases/{cid}/input.jpg"><div><p>原图：<b>{esc(Path(c["input_path"]).name)}</b></p><p>真实类别：{esc(cname(c["true_name"]))}<br>Frozen：{esc(cname(c["frozen_name"]))}<br>最终：{esc(cname(c["final_name"]))}</p><p>分区：{c["split"]}；运行：{c["run"]}</p><p>全部贡献针对固定类别对：<b>{esc(cname(c["final_name"]))} − {esc(cname(c["contrast_name"]))}</b></p><p>对照规则：{"最终类别对原 Frozen 预测" if c["prediction_changed"] else "共同预测类别对 Frozen 第二名；不是纠正案例"}</p></div></div>'
    if c['dataset'] == 'Derm7pt':
        body += f'<p>病例编号 {c["query_case_num"]}；诊断 {esc(c["query_diagnosis"])}；模态 {esc(c["query_modality"])}。同病例保留在其他分区：{esc(", ".join(c["query_case_other_splits"]) or "未发现")}。</p><p>Top-7 清楚病灶/病灶内部参考 {c["body_count"]}/7；按来源类别和符号直观一致的病灶参考 {c["readable_body_count"]}/7；正贡献中非病灶参考占比 {c["positive_nonlesion_share"]:.1%}。</p>'
    body += '</div>'
    notes = CUB_NOTES.get(cid, '')
    if c['dataset'] == 'Derm7pt':
        notes = ('满足 Top-7 至少三项病灶参考的数量条件。' if c['strict_three_lesion'] else '这是放宽条件的补充备选，未满足三项病灶参考门槛。')
        if c['category'] == 'harm':
            notes += ' 正贡献推动错误的最终类别；背景、标尺、气泡等参考可显示分支中的非病灶关联，但不能据此断言模型在测试图像上定位到了这些物体。'
        if c['counterintuitive_body_count']:
            notes += f' 有 {c["counterintuitive_body_count"]} 项病灶参考不符合“正贡献同最终来源类别、负贡献另一类别”的直观模板；已保留原值和原图，不应筛去后声称全都一致。'
        if cid in DERM_NOTES:
            notes += ' ' + DERM_NOTES[cid]
    body += f'<div class="note">{esc(notes)}</div><h2>预测流程与完整瀑布图</h2><div class="box"><img src="../flows/{cid}.png"><img src="../waterfalls/{cid}_full.png"><p>从左到右：Sem；按 |q| 排列的 {n} 项；Other（其余 {128-n} 项）；Bias；Final。图中只保留数值，编号与来源见下表。</p><p>Sem {c["semantic_margin"]:+.6f}；显示项合计 {c["topk_sum"]:+.6f}；Other {c["other"]:+.6f}；Bias {c["bias"]:+.6f}；Final {c["final_margin"]:+.6f}。完整残差改变间隔 {c["full_residual_margin"]:+.6f}；显示项占全部 |q| 的 {c["coverage"]:.1%}。</p><details><summary>仅显示所选贡献的局部瀑布图（不等于最终间隔）</summary><img src="../waterfalls/{cid}_values.png"></details><p><a href="../waterfalls/{cid}_full.svg">完整 SVG</a> · <a href="../waterfalls/{cid}_values.svg">局部 SVG</a> · <a href="../cases/{cid}_waterfall.csv">瀑布 CSV</a> · <a href="../cases/{cid}/shape_coordinates.csv">全部128项</a> · <a href="../cases/{cid}/class_logits.csv">全部类别分数</a> · <a href="../cases/{cid}/candidate.json">溯源 JSON</a></p></div>'
    body += '<h2>全部显示项：内容、来源与符号</h2><div class="scroll"><table><tr><th>序</th><th>编号/参考内容</th><th>贡献 q</th><th>标准化响应</th><th>有效权重差</th><th>参考原图</th><th>参考来源类别</th><th>来源与最终类别相同？</th></tr>'
    for r in c['top']:
        css = 'pos' if r['q'] > 0 else 'neg'
        body += f'<tr><td>{r["rank"]}</td><td>P{r["prototype_id"]} · {esc(r["label"])}</td><td class="{css}">{r["q"]:+.6f}</td><td>{r["activation"]:+.6f}</td><td>{r["effective_weight_difference"]:+.6f}</td><td>{esc(r["original_name"])}</td><td>{esc(cname(r["source_class"]))}</td><td>{"是" if r["same_final_class"] else "否"}{"；同一病例！" if r["same_query_case"] else ""}</td></tr>'
    body += '</table></div><p class="muted">q = 标准化 decoded 响应 × 已含残差系数的类别权重差。q 的正负不是相似度，也不是参考原图类别的医学/鸟类概念标签。</p><div class="ref-grid">'
    for r in c['top']:
        ref = refs[r['reference_id']]; path = '../references/' + r['reference_id'] + '/'
        body += f'<section class="ref"><b>{r["rank"]}. P{r["prototype_id"]} · {esc(r["label"])}</b><p class="{"pos" if r["q"]>0 else "neg"}">{r["q"]:+.6f}</p><img src="{path}training_context.png"><img class="small" src="{path}neutral_crop.png"><img class="small" src="{path}verified_contour.png"><p>{esc(r["original_name"])}<br>{esc(cname(r["source_class"]))}</p><a href="{path}training_original.jpg">查看训练原图</a> · <a href="{path}context_detail.png">放大位置</a> · <a href="{path}region_provenance.json">定位依据</a>'
        if c['dataset'] == 'Derm7pt':
            body += f'<p>参考病例 {ref["case_num"]}；{esc(ref["diagnosis"])}；{esc(ref["modality"])}。<br>定位状态：{esc(ref["context_status"])}</p>'
        body += '</section>'
    body += f'</div><h2>可直接取用的无编号图件</h2><div class="box"><a href="../panels/{cid}.png"><img src="../panels/{cid}.png"></a><p>左为待解释图，右依次为训练参考的原图定位、RGB 局部和轮廓；没有 P 编号或 Top 字样。<a href="../panels/{cid}.png">下载 PNG</a>。</p></div>'
    body += '<p class="muted">本页基于所选检查点及原始trace进行定性回放。训练参考不定位测试证据，来源类别不定义几何原型；医学病例的跨分区情况单列报告。</p>'
    (out / 'pages' / (cid+'.html')).write_text(page(cid, body), encoding='utf-8')


def exports_and_pages(out):
    cases = read_json(out / 'all_cases.json')
    refs = {r['reference_id']: r for r in read_json(out / 'references.json')}
    strict, supplementary, rendered = choose(cases)
    for folder in ('references', 'cases', 'pages', 'panels', 'waterfalls', 'flows'):
        (out / folder).mkdir(exist_ok=True)
    ledger = SourceLedger()
    used = {r['reference_id'] for c in rendered for r in c['top']}
    for rid in used:
        source = PRIOR / 'references' / rid
        if not source.is_dir():
            raise ValueError('A rendered reference has no verified assets')
        target = out / 'references' / rid; target.mkdir(exist_ok=True)
        for p in source.iterdir():
            if p.is_file():
                ledger.record(p)
                shutil.copy2(p, target / p.name)
    for c in rendered:
        ledger.record(c['input_path'])
        export_case_sources(c, out)
        pure_panel(c, out); waterfall(c, out); prediction_flow(c, out)
        case_page(c, refs, out)
        print('Rendered:', c['candidate_id'], flush=True)
    table = []
    for c in cases:
        table.append({'案例编号': c['candidate_id'], '数据集': c['dataset'], '分区': c['split'], '运行': c['run'],
            '原图': Path(c['input_path']).name, '状态': STATUS[c['category']], '真实': c['true_name'],
            'Frozen': c['frozen_name'], '最终': c['final_name'], '对比类别': c['contrast_name'],
            '清楚目标参考数': c['body_count'], '类别符号直观一致参考数': c['readable_body_count'],
            '至少三项病灶参考': c['strict_three_lesion'], '未导出参考数': c['unknown_reference_count'],
            '正向非病灶参考数': c['positive_nonlesion_count'], '正向纯伪影参考数': c['positive_artifact_count'],
            '正向非病灶占比': c['positive_nonlesion_share'], 'Sem': c['semantic_margin'],
            '显示贡献合计': c['topk_sum'], 'Other': c['other'], 'Bias': c['bias'], 'Final': c['final_margin'],
            '同病例其他分区': ','.join(c.get('query_case_other_splits', [])), '原图路径': c['input_path']})
    write_csv(out / '全部案例统计.csv', table)
    write_csv(out / '病灶参考至少三项_全部候选.csv', [r for r in table if r['至少三项病灶参考']])
    write_csv(out / 'CUB四例详细统计.csv', [r for r in table if r['数据集'] == 'CUB'])
    write_csv(out / '全部Top项来源.csv', [dict(案例编号=c['candidate_id'], 分区=c['split'], 运行=c['run'],
        排名=r['rank'], 原型编号=r['prototype_id'], 贡献=r['q'], 标准化响应=r['activation'],
        有效权重差=r['effective_weight_difference'], 参考内容=r['label'], 参考原图=r['original_name'],
        参考类别=r['source_class'], 参考原图路径=r['original_path'], 清楚目标区域=r['is_body'],
        来源同最终类别=r['same_final_class'], 参考同待解释病例=r['same_query_case']) for c in cases for r in c['top']])
    summary = dict(retained_unique_images=2013, retained_split_sizes={'train': 1231, 'val': 391, 'test': 391},
        paired_replays=sum(c['dataset'] == 'Derm7pt' and c['run'] != 'T' for c in cases),
        original_complete_changed_traces=sum(c['dataset'] == 'Derm7pt' and c['run'] == 'T' for c in cases),
        strict_three_lesion_count=len(strict), strict_counts=[dict(split=s, run=r, category=k, count=n)
            for (s,r,k),n in sorted(Counter((c['split'],c['run'],c['category']) for c in strict).items())],
        all_status_counts=[dict(split=s, run=r, category=k, count=n)
            for (s,r,k),n in sorted(Counter((c['split'],c['run'],c['category']) for c in cases if c['dataset']=='Derm7pt').items())],
        rendered_ids=[c['candidate_id'] for c in rendered], supplementary_ids=[c['candidate_id'] for c in supplementary],
        legacy_unexported_reference_cases=len({r['candidate_id'] for r in read_json(out / 'legacy_unexported_references.json')}),
        protocol={'derm_topk':7,'cub_topk':5,'coordinates':128,
          'ranking':'descending absolute contribution; coordinate number breaks ties',
          'strict_rule':'at least 3 reviewed lesion/lesion_part exemplars among the exact Top-7',
          'source_class_rule':'reference source, not class definition of geometric prototype',
          'legacy_scope':'only complete saved changed-case records; no legacy full-population prediction export',
          'no_new_training':True,'no_new_segmentation':True})
    write_json(out / 'manifest.json', summary)
    write_json(out / 'asset_inputs.json', ledger.verify())
    if protected_snapshot() != read_json(out / 'selected_folders_before.json'):
        raise ValueError('Previously selected Figure 1/2 folders changed')
    write_json(out / 'protected_folders_check.json', {'unchanged':True,'file_count':len(protected_snapshot())})
    build_index(cases, strict, supplementary, rendered, out, summary)


def short_table(items):
    result = '<div class="scroll"><table><tr><th>案例/详细页</th><th>原图</th><th>分区与状态</th><th>Frozen → 最终（真实）</th><th>目标参考数 / 直观一致数</th><th>正贡献中非病灶占比</th></tr>'
    for c in items:
        result += f'<tr><td><a href="pages/{c["candidate_id"]}.html">{c["candidate_id"]}</a></td><td>{esc(Path(c["input_path"]).name)}</td><td>{c["split"]}<br>{STATUS[c["category"]]}</td><td>{esc(cname(c["frozen_name"]))} → {esc(cname(c["final_name"]))}<br>真实：{esc(cname(c["true_name"]))}</td><td>{c["body_count"]} / {c["readable_body_count"]}</td><td>{c["positive_nonlesion_share"]:.1%}</td></tr>'
    return result + '</table></div>'


def build_index(cases, strict, supplementary, rendered, out, summary):
    cub = [next(c for c in cases if c['candidate_id'] == cid) for cid in CUB_IDS]
    body = '<h1>CUB 四例详解与 Derm7pt Top-7 全量筛选</h1><div class="box"><p>三个正确配对保存模型逐一回放训练、验证、测试共 2,013 张图像（6,039 个模型×图像记录），另检查原始运行的 53 个完整改变预测记录。没有重新训练或分割。</p><p>纠正＝原错→现对；改错＝原对→现错。未改变预测的例子单独标为保持正确/错误。所有显示项都按 |q| 排列，不替换成更好看的参考。</p><p>“三个符合”的数量筛选：Top-7 中至少三项为清楚的病灶外轮廓或病灶内部区域。另列来源类别与符号是否直观一致；数量达标不自动表示解释完整或无干扰。</p></div>'
    body += '<h2>CUB：指定四例</h2>' + short_table(cub)
    for c in cub:
        body += f'<article><h2><a href="pages/{c["candidate_id"]}.html">{c["candidate_id"]}</a> · {STATUS[c["category"]]}</h2><p>{esc(CUB_NOTES[c["candidate_id"]])}</p><a href="pages/{c["candidate_id"]}.html"><img src="panels/{c["candidate_id"]}.png"></a><img src="waterfalls/{c["candidate_id"]}_full.png"></article>'
    body += f'<h2>Derm7pt：病灶参考至少三项的全部 {len(strict)} 个候选</h2><div class="note">三个保存配对模型中满足条件的都是保持正确的例子；原始 T 记录补充了三个纠正和一个改错。T 与 R42/R43/R44 使用各自记录及原型对应，不混用数值。这些案例仍有不同程度的非病灶参考或类别与符号不直观的项目，需逐例审阅。</div>'
    body += short_table(sorted(strict, key=lambda c:(c['split']!='test',c['category']!='correction',c['category']!='harm',c['candidate_id'])))
    body += '<h2>额外比较：纠正候选与伪影/背景相关改错候选</h2><p>以下是明确放宽“三项病灶”条件的补充备选。改错候选优先选择至少三个正贡献来自非病灶参考、且病灶参考的来源类别与符号没有直观冲突的例子。这里的“伪影相关”描述训练参考内容，并非测试图像的定位或因果结论。</p>' + short_table(supplementary)
    body += '<h2>选图建议：数量与实际贡献分开看</h2><div class="box">'
    for cid in ('Derm7pt-T-0214','Derm7pt-T-0346','Derm7pt-T-0350','Derm7pt-T-0048','Derm7pt-R42-0117','Derm7pt-T-0106'):
        body += f'<p><a href="pages/{cid}.html"><b>{cid}</b></a>：{esc(DERM_NOTES[cid])}</p>'
    body += '<p>没有找到同时满足“三项清楚病灶参考”及“所有病灶来源类别都与期望符号一致”的测试纠正/改错配对。若优先展示纠正与背景相关改错，可先比较 T-0214 与 R42-0117；若要求来自同一原始运行，可比较 T-0214 与 T-0106，并保留上述限制。</p></div>'
    body += '<h2>扫描记录与图件</h2><div class="box"><p><a href="全部案例统计.csv">全部案例统计 CSV</a> · <a href="全部Top项来源.csv">全部 Top 项来源 CSV</a> · <a href="病灶参考至少三项_全部候选.csv">数量达标候选 CSV</a> · <a href="CUB四例详细统计.csv">CUB 四例 CSV</a> · <a href="manifest.json">范围与筛选协议</a> · <a href="replay_checks.json">预测和分解核验</a></p><p>每个详细页包含原图、流程图、完整/局部瀑布图、128项贡献、全部类别 logits、参考原图名、区域位置、RGB 图和轮廓。可选图件无 P 编号或 Top 字样。</p><p>原始 T 记录中六例的第6/7项涉及五个未导出参考的坐标，保持未知；这些例子即使把未知项都计为病灶，也不足三项，不影响数量达标清单。详见 <a href="legacy_unexported_references.json">未导出项</a>。原始记录另有三个不完整束，不用于 Top-7 解释；其测试图像均已在三个配对模型中覆盖。</p></div>'
    body += '<h2>完整记录索引</h2><div class="box"><input id="search" placeholder="编号、原图名、类别"><select id="split"><option value="">全部分区</option><option>test</option><option>val</option><option>train</option></select><select id="status"><option value="">全部状态</option>' + ''.join(f'<option value="{k}">{v}</option>' for k,v in STATUS.items()) + '</select><span id="count"></span><div id="catalog"></div><button id="prev">上一页</button><button id="next">下一页</button></div>'
    simple = [dict(id=c['candidate_id'],split=c['split'],status=c['category'],file=Path(c['input_path']).name,
        truth=c['true_name'],base=c['frozen_name'],final=c['final_name'],body=c['body_count'],
        readable=c['readable_body_count'],rendered=c in rendered) for c in cases]
    (out / 'catalog.js').write_text('const CASES='+json.dumps(simple,ensure_ascii=False)+';\nconst STATUS='+json.dumps(STATUS,ensure_ascii=False)+';\n'+"""
const s=document.querySelector('#search'),p=document.querySelector('#split'),t=document.querySelector('#status');let start=0,filtered=[];
function e(v){return String(v).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function show(reset=true){if(reset)start=0;const q=s.value.toLowerCase();filtered=CASES.filter(c=>(!p.value||c.split===p.value)&&(!t.value||c.status===t.value)&&JSON.stringify(c).toLowerCase().includes(q));document.querySelector('#count').textContent=`共 ${filtered.length} 条；第 ${Math.floor(start/40)+1} 页`;
document.querySelector('#catalog').innerHTML='<div class="scroll"><table><tr><th>编号</th><th>原图</th><th>状态</th><th>Frozen → 最终</th><th>病灶/鸟体参考</th></tr>'+filtered.slice(start,start+40).map(c=>`<tr><td>${c.rendered?`<a href="pages/${c.id}.html">${c.id}</a>`:e(c.id)}</td><td>${e(c.file)}</td><td>${e(c.split)} / ${STATUS[c.status]}</td><td>${e(c.base)} → ${e(c.final)}</td><td>${c.body}（直观一致 ${c.readable}）</td></tr>`).join('')+'</table></div>'}
[s,p,t].forEach(x=>x.addEventListener('input',()=>show()));document.querySelector('#prev').onclick=()=>{start=Math.max(0,start-40);show(false)};document.querySelector('#next').onclick=()=>{if(start+40<filtered.length)start+=40;show(false)};show();
""",encoding='utf-8')
    body += '<script src="catalog.js"></script>'
    (out / 'index.html').write_text(page('案例详解与 Top-7 筛选', body), encoding='utf-8')
    lines = ['# CUB 四例与 Derm7pt Top-7 筛选', '',
        '三个配对保存模型覆盖全部保留图像；原始 T 记录只作为单独补充。', '',
        f'配对回放：{summary["paired_replays"]}；完整原始改变记录：{summary["original_complete_changed_traces"]}；至少三项病灶参考：{len(strict)}。', '',
        '|案例|状态|分区|病灶参考数|来源类别与符号直观一致数|', '|---|---|---|---:|---:|']
    lines += [f'|{c["candidate_id"]}|{STATUS[c["category"]]}|{c["split"]}|{c["body_count"]}|{c["readable_body_count"]}|' for c in strict]
    lines += ['', 'CUB 四例：'] + [f'- {cid}：{CUB_NOTES[cid]}' for cid in CUB_IDS]
    lines += ['', 'Derm7pt 选图分析：'] + [f'- {cid}：{note}' for cid,note in DERM_NOTES.items()]
    (out / '筛选说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-existing', action='store_true', help='Rebuild only this marked export, without rescanning.')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.render_existing:
        if not (out/'selected_folders_before.json').is_file() or not (out/'all_cases.json').is_file():
            raise ValueError('Not a prepared case review directory')
    else:
        scan(out)
    exports_and_pages(out)
    print('Review ready:', out / 'index.html', flush=True)


if __name__ == '__main__':
    main()
