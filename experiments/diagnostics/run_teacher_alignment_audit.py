"""Read-only audit of correction retention and fixed-contrast contribution alignment.

Reuses saved fixed-head teacher replacement outputs. The classifier, class
contrast, and evaluation cohort remain fixed throughout the audit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from experiments.controls.run_decoder_baselines import sha256, write_csv, write_json


from aragpr.paths import repository_root

ROOT = repository_root()
COHORTS=('all_changed','original_corrections')
MEASURES=('correction_retention_pct','top3_sign_agreement_pct','top3_overlap_pct')


def summarize_alignment(labels, frozen, decoded, teacher_prediction, q_decoded,
                        q_teacher, cohort='all_changed', topk=3, epsilon=1e-10):
    """Retain original images, b/f contrasts, and decoded top-k throughout.

    Retention conditions on ORIGINAL Frozen-wrong / Decoded-correct images.
    Direction agreement counts nonzero terms at the ORIGINAL decoded top-k.
    Overlap is |TopK_decoded intersect TopK_teacher|/k, averaged over images.
    """
    labels,frozen,decoded,teacher_prediction=map(np.asarray,(labels,frozen,decoded,teacher_prediction))
    qd,qt=map(lambda a:np.asarray(a,dtype=np.float64),(q_decoded,q_teacher))
    if cohort not in COHORTS:
        raise ValueError('Unknown evaluation cohort.')
    if qd.shape!=qt.shape or qd.ndim!=2 or len(qd)!=len(labels):
        raise ValueError('Contribution arrays and membership are not aligned.')
    if not 1<=topk<=qd.shape[1] or not np.isfinite(qd).all() or not np.isfinite(qt).all():
        raise ValueError('Invalid top-k or contribution values.')
    corrections=(frozen!=labels)&(decoded==labels)
    changed=frozen!=decoded
    chosen=changed if cohort=='all_changed' else corrections
    ids=np.flatnonzero(chosen)
    order_d=np.argsort(-np.abs(qd[ids]),axis=1,kind='stable')[:,:topk]
    order_t=np.argsort(-np.abs(qt[ids]),axis=1,kind='stable')[:,:topk]
    val_d=np.take_along_axis(qd[ids],order_d,axis=1)
    val_t=np.take_along_axis(qt[ids],order_d,axis=1)
    eligible=(np.abs(val_d)>epsilon)&(np.abs(val_t)>epsilon)
    matches=(np.sign(val_d)==np.sign(val_t))&eligible
    overlap=(order_d[:,:,None]==order_t[:,None,:]).any(axis=2).sum(axis=1)
    ncorrect=int(corrections.sum())
    retained=corrections&(teacher_prediction==labels)
    row={'cohort':cohort,'n_valid_images':len(labels),'n_images':len(ids),
        'n_original_corrections':ncorrect,'n_retained_corrections':int(retained.sum()),
        'correction_retention_pct':100*float(retained.sum())/ncorrect if ncorrect else None,
        'sign_eligible_coordinates':int(eligible.sum()),
        'sign_matches':int(matches.sum()),
        'top3_sign_agreement_pct':100*float(matches.sum())/eligible.sum() if eligible.any() else None,
        'top3_overlap_pct':100*float(overlap.mean())/topk if len(ids) else None}
    detail={'ids':ids,'eligible':eligible.sum(1),'matches':matches.sum(1),
        'overlap':overlap,'top_decoded':order_d,'top_teacher':order_t,
        'corrections':corrections,'retained':retained}
    return row,detail


def aggregate(rows,keys,measures=MEASURES):
    groups={}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    output=[]
    for values,members in groups.items():
        row=dict(zip(keys,values));row['n_seeds']=len(members)
        for measure in measures:
            data=[r[measure] for r in members if r[measure] is not None]
            row['mean_'+measure]=float(np.mean(data)) if data else None
            row['std_'+measure]=float(np.std(data,ddof=1)) if len(data)>1 else None
        output.append(row)
    return output


def make_report(out,rows,summary,differences):
    def find(dataset,condition,cohort):
        return next(r for r in summary if r['dataset']==dataset and r['condition']==condition and r['cohort']==cohort)
    def number(row,metric):
        mean,sd=row['mean_'+metric],row['std_'+metric]
        return 'N/A' if mean is None else f'{mean:.2f} ± {sd:.2f}' if sd is not None else f'{mean:.2f}'
    datasets=list(dict.fromkeys(r['dataset'] for r in rows))
    text=['# Teacher替换后的纠错保留与贡献对应','',
        '统计使用已保存模型及其验证集选定检查点，仅包含有效teacher测试图像。各数据集、种子和评价分母见下表及逐样本记录。',
        '所有对比复用同一Frozen基准、分类头和原有类别对；指标描述这些固定模型上的对应关系。','',
        '## 指标与评价范围','',
        '- 原有纠错保留率：原来Frozen错误、Decoded正确的图像中，固定头teacher替换后仍然正确的比例。',
        '- 贡献方向一致率：固定原b=Frozen类别、f=Decoded类别，选原Decoded绝对贡献最大的3个坐标；两种贡献均非零的坐标中，符号一致的比例。',
        '- Top-3重叠率：两种响应各自绝对贡献最大的3个坐标的交集数除以3，再按图像平均。它不是Jaccard，后者的分母不同。',
        '主表的贡献指标统计全部原有改变预测图像；附表只统计原有正确纠错图像。f=b的图像具有平凡零贡献，不加入贡献一致性。所有范围在替换前固定，不依据替换结果筛选。','',
        '每个已有模型另对有效测试图像的完整teacher向量做20次全局置换，固定原分类头、eta、标准化及类别对；每种子先对20次置换取均值，再对三个种子求均值±样本标准差。20次置换不是20个训练种子，不用于选模型。','',
        '## 主结果：真实配对与置换参照','',
        '| 数据集 | 原有纠错保留率/% | 置换参照/% | 方向一致率/% | 置换参照/% | Top-3重叠率/% | 置换参照/% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for ds in datasets:
        a=find(ds,'paired','all_changed');n=find(ds,'permuted_mean','all_changed')
        text.append('| '+ds+' | '+' | '.join(number(r,m) for m in MEASURES for r in [a,n])+' |')
    text+=['','## 每个种子的纠错数量','',
        '| 数据集 | 种子 | 原有改变预测数 | 原有正确纠错数 | Teacher保留纠错数 | 保留率/% |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['condition']=='paired' and r['cohort']=='all_changed':
            rate='N/A' if r['correction_retention_pct'] is None else f"{r['correction_retention_pct']:.2f}"
            text.append(f"| {r['dataset']} | {r['seed']} | {r['n_images']} | {r['n_original_corrections']} | {r['n_retained_corrections']} | {rate} |")
    text+=['','纠错保留率对较小的分母较敏感，需结合各个种子的纠错数量阅读。不同种子的子集可能不同，同一测试图像的跨种子重复不构成独立病例。','',
        '## 仅原有正确纠错图像的贡献对应','',
        '| 数据集 | 方向一致率/% | 置换参照/% | Top-3重叠率/% | 置换参照/% |',
        '|---|---:|---:|---:|---:|']
    for ds in datasets:
        a=find(ds,'paired','original_corrections');n=find(ds,'permuted_mean','original_corrections')
        text.append('| '+ds+' | '+' | '.join(number(r,m) for m in MEASURES[1:] for r in [a,n])+' |')
    text+=['','## 指标解释','',
        '三个指标的理想上限都是100%。更有支持力的模式是：真实配对的纠错保留高，重要贡献同向，Top-3原型较多重合；三者相对同一模型的置换参照均有稳定优势。',
        '没有统一的“达到70%或80%即合格”阈值。纠错保留和Top-3重叠尤其需要置换参照，因为共享语义分支及固定头的权重结构本身也会产生保留或重叠。不能用均匀随机抽3个原型的理论重叠率作为这里的实际零假设。',
        '符号一致没有约束贡献幅度：即使每项正负号多半相同，幅度变化、其他坐标及偏置仍可使分类结果改变。方向一致率较高不自动意味着原有纠错保留率高。',
        '均值、种子间标准差和配对差值是条件于所评估模型的描述性统计；共享病例及有限重复次数限制了统计推断和临床推广。','',
        '## 文件','',
        '- metrics_summary.csv：主表与纠错子集的三种子统计。',
        '- paired_metrics_per_seed.csv：逐种子真实配对统计及原始分母。',
        '- permutation_metrics.csv：每次置换、每种子的完整结果。',
        '- permutation_mean_per_seed.csv：每个种子的置换均值。',
        '- paired_minus_permutation_per_seed.csv / paired_minus_permutation_summary.csv：每种子配对差值及三种子汇总。',
        '- changed_image_details.csv：固定原类别对的逐图像纠错保留、符号和Top-3明细。',
        '- protocol.json：输入哈希、匹配核验及未修改文件记录。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(text),encoding='utf-8')


def run(args):
    start=time.monotonic();source=args.source.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    protocol_path=source/'protocol.json'
    import json
    input_records=[{'path':str(protocol_path),'sha256':sha256(protocol_path)}]
    original=json.loads(protocol_path.read_text(encoding='utf-8'))
    if original['status']!='complete':raise ValueError('Source replay was not complete.')
    paired_rows,null_rows,details,verifications=[],[],[],[]
    by_model={(v['dataset'],v['seed']):v for v in original['verification']}
    protected=[ROOT.parents[1]/'lunwen/sn-article.tex',ROOT.parents[1]/'lunwen/sn-article.pdf']
    paper_before={str(p):sha256(p) for p in protected if p.is_file()}
    old_metrics_path=source/'contribution_alignment_per_seed.csv'
    old_null_path=source/'teacher_permutation_reference.csv'
    input_records.extend({'path':str(p),'sha256':sha256(p)} for p in [old_metrics_path,old_null_path])
    old_metrics=list(csv.DictReader(old_metrics_path.open(encoding='utf-8-sig')))
    old_null=list(csv.DictReader(old_null_path.open(encoding='utf-8-sig')))
    for dataset in original['datasets']:
        for seed in original['seeds']:
            entry=by_model[dataset,seed];state_path=Path(entry['model_path'])
            assert sha256(state_path)==entry['model_sha256']
            input_records.append({'path':str(state_path),'sha256':sha256(state_path)})
            state=torch.load(state_path,map_location='cpu',weights_only=False)
            head=state['head']['selected'];eta=float(head['eta'])
            W=head['state']['weight'].detach().cpu().numpy().astype(np.float64)
            bias=head['state']['bias'].detach().cpu().numpy().astype(np.float64)
            pred_path=source/dataset/f'seed_{seed}/test_diagnostic_predictions.npz'
            csv_path=source/dataset/f'seed_{seed}/changed_prediction_alignment.csv'
            input_records.extend({'path':str(p),'sha256':sha256(p)} for p in [pred_path,csv_path])
            with np.load(pred_path,allow_pickle=False) as z:
                ids=z['teacher_valid_indices'];labels=z['labels'][ids]
                semantic=z['frozen_logits'][ids];full=z['decoded_logits'][ids]
                replacement=z['teacher_replacement_valid_logits'];vt=z['teacher_head_inputs_valid']
                vd=z['decoded_head_inputs_valid']
                qd=z['decoded_contributions'];qt=z['teacher_contributions']
                b=z['fixed_b'];f=z['fixed_f'];keys=z['sample_keys'][ids]
                np.testing.assert_array_equal(ids,np.flatnonzero(z['valid_teacher']))
                np.testing.assert_array_equal(b,semantic.argmax(1))
                np.testing.assert_array_equal(f,full.argmax(1))
                np.testing.assert_allclose(full,semantic+eta*(vd@W.T+bias),atol=2e-9,rtol=2e-10)
                np.testing.assert_allclose(replacement,semantic+eta*(vt@W.T+bias),atol=2e-9,rtol=2e-10)
                np.testing.assert_allclose(qd,eta*(W[f]-W[b])*vd,atol=2e-9,rtol=2e-10)
                np.testing.assert_allclose(qt,eta*(W[f]-W[b])*vt,atol=2e-9,rtol=2e-10)
            prediction=replacement.argmax(1)
            for cohort in COHORTS:
                row,detail=summarize_alignment(labels,b,f,prediction,qd,qt,cohort)
                row={'dataset':dataset,'seed':seed,'condition':'paired',**row}
                paired_rows.append(row)
                old_outcome='all_changed' if cohort=='all_changed' else 'correction'
                old=next(r for r in old_metrics if r['dataset']==dataset and int(r['seed'])==seed and r['outcome']==old_outcome)
                for metric in MEASURES[1:]:
                    np.testing.assert_allclose(row[metric],float(old[metric]),atol=1e-10)
                if cohort=='all_changed':
                    for i,j in enumerate(detail['ids']):
                        details.append({'dataset':dataset,'seed':seed,'sample_key':str(keys[j]),
                            'true_label':int(labels[j]),'frozen_b':int(b[j]),'decoded_f_fixed':int(f[j]),
                            'teacher_prediction':int(prediction[j]),
                            'original_correction':bool(detail['corrections'][j]),
                            'original_correction_retained':bool(detail['retained'][j]),
                            'sign_eligible':int(detail['eligible'][i]),'sign_matches':int(detail['matches'][i]),
                            'top3_intersection_size':int(detail['overlap'][i]),
                            'top3_overlap_pct':float(100*detail['overlap'][i]/3),
                            'decoded_top3':';'.join(map(str,detail['top_decoded'][i])),
                            'teacher_top3':';'.join(map(str,detail['top_teacher'][i]))})
            old_detail=list(csv.DictReader(csv_path.open(encoding='utf-8-sig')))
            ncorrect=sum(r['outcome']=='correction' for r in old_detail)
            nkept=sum(r['outcome']=='correction' and r['teacher_argmax']==r['true_label'] for r in old_detail)
            current=paired_rows[-2]
            assert current['n_original_corrections']==ncorrect and current['n_retained_corrections']==nkept
            rng=np.random.default_rng(100000+seed)
            for repeat in range(args.permutations):
                order=rng.permutation(len(labels));shuffled=vt[order]
                logits=semantic+eta*(shuffled@W.T+bias)
                qnull=eta*(W[f]-W[b])*shuffled
                for cohort in COHORTS:
                    row,_=summarize_alignment(labels,b,f,logits.argmax(1),qd,qnull,cohort)
                    null_rows.append({'dataset':dataset,'seed':seed,'condition':'permuted',
                                      'permutation':repeat,**row})
                    if repeat<20 and cohort=='all_changed':
                        old=next(r for r in old_null if r['dataset']==dataset and int(r['seed'])==seed and int(r['permutation'])==repeat)
                        for metric in MEASURES[1:]:
                            np.testing.assert_allclose(row[metric],float(old[metric]),atol=1e-10)
            verifications.append({'dataset':dataset,'seed':seed,
                'original_contribution_summaries_match':True,'correction_counts_match_original_csv':True,
                'existing_20_permutation_metrics_match':args.permutations>=20,
                'valid_only_and_original_contrast_verified':True})
            print(f'[{dataset}/seed{seed}] retained {nkept}/{ncorrect} corrections; original outputs verified.',flush=True)
    null_means=[]
    for dataset in original['datasets']:
        for seed in original['seeds']:
            for cohort in COHORTS:
                rows=[r for r in null_rows if r['dataset']==dataset and r['seed']==seed and r['cohort']==cohort]
                row={'dataset':dataset,'seed':seed,'condition':'permuted_mean','cohort':cohort,
                     'n_permutations':len(rows),'n_images':rows[0]['n_images'],
                     'n_original_corrections':rows[0]['n_original_corrections']}
                for measure in MEASURES:
                    data=[r[measure] for r in rows if r[measure] is not None]
                    row[measure]=float(np.mean(data)) if data else None
                null_means.append(row)
    differences=[]
    for r in paired_rows:
        n=next(v for v in null_means if all(v[k]==r[k] for k in ['dataset','seed','cohort']))
        row={k:r[k] for k in ['dataset','seed','cohort']}
        for measure in MEASURES:
            row[measure]=r[measure]-n[measure] if r[measure] is not None and n[measure] is not None else None
        differences.append(row)
    summary=aggregate(paired_rows+null_means,['dataset','condition','cohort'])
    diff_summary=aggregate(differences,['dataset','cohort'])
    write_csv(out/'paired_metrics_per_seed.csv',paired_rows)
    write_csv(out/'permutation_metrics.csv',null_rows)
    write_csv(out/'permutation_mean_per_seed.csv',null_means)
    write_csv(out/'metrics_summary.csv',summary)
    write_csv(out/'paired_minus_permutation_per_seed.csv',differences)
    write_csv(out/'paired_minus_permutation_summary.csv',diff_summary)
    write_csv(out/'changed_image_details.csv',details)
    paper_after={p:sha256(p) for p in paper_before}
    assert paper_before==paper_after
    # Recheck every input after all outputs have been generated.
    for r in input_records:assert sha256(r['path'])==r['sha256']
    write_json(out/'protocol.json',{'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
        'source':str(source),'seeds':original['seeds'],'datasets':original['datasets'],
        'local_models_not_server_results':True,'permutations_per_model':args.permutations,
        'permutation_rng':'numpy default_rng(100000+model_seed); complete valid-test teacher row permutations',
        'classification_head_eta_scalers_fixed':True,'original_b_f_cohort_fixed':True,
        'summary':'Mean and sample SD across three saved training seeds; null averaged within each seed first',
        'retrained':False,'generated_masks':False,'test_selected_models':False,
        'paper_sha256':paper_before,'paper_unchanged':paper_before==paper_after,
        'inputs':input_records,'inputs_unchanged':True,'verification':verifications,
        'runner_sha256':sha256(Path(__file__)),'python':platform.python_version(),
        'elapsed_seconds':time.monotonic()-start})
    make_report(out,paired_rows,summary,diff_summary)
    print(f'COMPLETE {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/posthoc_shape_diagnostics_20261022_final')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--permutations',type=int,default=20)
    args=parser.parse_args()
    if args.permutations<1:parser.error('At least one permutation is required.')
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):run(args)
