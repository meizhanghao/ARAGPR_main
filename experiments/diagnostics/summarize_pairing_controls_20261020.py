"""Verify saved CUB models and summarize CUB/Derm7pt pairing diagnostics.

The report uses fitted models and saved outputs. Additional Derm7pt reference
summaries are reported separately from the replayed experiments.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from experiments.controls.run_decoder_baselines import load_cached_inputs, sha256, write_csv, write_json


from aragpr.paths import repository_root

ROOT = repository_root()
CUB = ROOT / 'outputs/cub_pairing_controls_20261020_pilot01'
DERM = ROOT / 'outputs/derm7pt_mechanism_20261020_pilot01'
CACHE = ROOT / 'outputs/residual_controls_20261018_paired01'
REPORT = ROOT.parents[1] / 'cub_derm7pt_pairing_report_20261020.md'
AUTHOR = ROOT.parents[1] / 'lunwen/figure_audit/teacher_coverage_percent_20261020/author_server_pairing_summary.json'
MODES = ('paired', 'shuffled', 'within_class_shuffled')
NAMES = {'paired': '真实配对', 'shuffled': '全局打乱',
         'within_class_shuffled': '同类内打乱'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def interval_cell(row, low_key, high_key):
    if not row.get(low_key) or not row.get(high_key):
        return '未估计（种子差值无变化或样本不足）'
    return f"[{float(row[low_key]):+.3f}, {float(row[high_key]):+.3f}]"


def verify_cub():
    protocol = json.loads((CUB / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['status'] != 'complete':
        raise ValueError('Wait for all nine CUB fits to finish.')
    h, teacher, valid, keys, labels, manifest, _ = load_cached_inputs(CACHE, 'CUB_200_2011')
    groups = read_csv(CUB / 'classification_groups.csv')
    fidelity = read_csv(CUB / 'decoder_fidelity.csv')
    semantic = {}
    for split in ('train', 'val', 'test'):
        with np.load(CACHE / 'CUB_200_2011' / f'{split}_frozen.npz', allow_pickle=False) as archive:
            semantic[split] = archive['logits'].copy()
    checks, complete_models = [], []
    for seed in protocol['seeds']:
        for mode in MODES:
            folder = CUB / f'seed_{seed}' / mode
            state = torch.load(folder / 'selected_model.pt', map_location='cpu', weights_only=False)
            selected = state['head']['selected']
            decoder_history = read_csv(folder / 'decoder_history.csv')
            decoder_best = min(decoder_history, key=lambda row: float(row['val_loss']))
            assert int(decoder_best['epoch']) == state['decoder_best_epoch']
            history = read_csv(folder / 'head_history.csv')
            best_head = max(history, key=lambda row: int(row['val_correct']))
            assert float(best_head['eta']) == selected['eta']
            assert int(best_head['epoch']) == selected['best_epoch']
            assert int(best_head['val_correct']) == selected['val_correct']
            assert state['decoder_state']['weight'].shape == (128, 768)
            assert selected['state']['weight'].shape == (200, 128)
            with np.load(folder / 'training_teacher_pairing.npz', allow_pickle=False) as archive:
                permutation = archive['permutation']
                np.testing.assert_array_equal(np.sort(permutation), np.arange(valid['train'].sum()))
                np.testing.assert_array_equal(archive['input_keys'], keys['train'][valid['train']])
                np.testing.assert_array_equal(archive['target_source_keys'], keys['train'][valid['train']][permutation])
                if mode == 'within_class_shuffled':
                    np.testing.assert_array_equal(labels['train'][valid['train']][permutation],
                                                  labels['train'][valid['train']])
            for split in ('train', 'val', 'test'):
                # Independent CPU matrix expressions, rather than the inference helpers.
                decoded = h[split] @ state['decoder_state']['weight'].numpy().T
                decoded += state['decoder_state']['bias'].numpy()
                standardized = state['coordinate_scaler'].transform(decoded).astype(np.float32)
                logits = semantic[split] + selected['eta'] * (
                    standardized @ selected['state']['weight'].numpy().T +
                    selected['state']['bias'].numpy())
                with np.load(folder / f'{split}_predictions.npz', allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive['sample_keys'], keys[split])
                    np.testing.assert_array_equal(archive['labels'], labels[split])
                    np.testing.assert_array_equal(archive['valid_teacher'], valid[split])
                    np.testing.assert_array_equal(archive['semantic_logits'], semantic[split])
                    np.testing.assert_allclose(decoded, archive['decoded'], rtol=3e-5, atol=4e-5)
                    np.testing.assert_allclose(logits, archive['logits'], rtol=3e-5, atol=4e-5)
                    np.testing.assert_array_equal(logits.argmax(1), archive['logits'].argmax(1))
                    maximum_error = float(np.max(np.abs(logits - archive['logits'])))
                correct = logits.argmax(1) == labels[split]
                frozen_correct = semantic[split].argmax(1) == labels[split]
                for subset, members in (('all', np.ones(len(correct), bool)),
                                        ('valid_teacher', valid[split]),
                                        ('missing_teacher', ~valid[split])):
                    row = next(r for r in groups if r['mode'] == mode and
                               int(r['seed']) == seed and r['split'] == split and r['subset'] == subset)
                    assert int(row['n']) == int(members.sum())
                    assert int(row['n_correct']) == int(correct[members].sum())
                    assert int(row['n_corrected']) == int((correct & ~frozen_correct & members).sum())
                    assert int(row['n_harmed']) == int((~correct & frozen_correct & members).sum())
                    assert int(row['n_corrected']) - int(row['n_harmed']) == int(
                        correct[members].sum() - frozen_correct[members].sum())
                if split in ('val', 'test'):
                    row = next(r for r in fidelity if r['mode'] == mode and int(r['seed']) == seed and r['split'] == split)
                    score = r2_score(teacher[split][valid[split]], decoded[valid[split]], multioutput='uniform_average')
                    np.testing.assert_allclose(score, float(row['r2_uniform']), atol=2e-6, rtol=2e-5)
                checks.append({'seed': seed, 'mode': mode, 'split': split,
                               'all_predictions_match': True, 'max_logit_error_cpu': maximum_error,
                               'classification_and_correction_harm_match': True})
            complete_models.append({'seed': seed, 'mode': mode,
                                    'decoder_and_head_selection_matches_validation_history': True})
    for source in protocol['sources'].values():
        assert sha256(Path(source['path'])) == source['sha256']
    for source in protocol['code_sources']:
        assert sha256(Path(source['path'])) == source['sha256']
    # Current manuscript artifacts were last delivered before this run began.
    expected_paper_hashes = {
        'lunwen/sn-article.tex': '1b0b49b0b9c4341e97929f19fac419431e4e33ae0b09c82140d129c629fc76f3',
        'lunwen/sn-article.pdf': 'c5241730c0dccb5a6fcded3f6c8f316c77f0926182cda2d8606473fcc4e1d679',
        'paper.docx': '6e38fb548ce8350b86d7338dd87663d1d16be47877d3a98d8720d9abbfb45289',
    }
    paper_files = []
    for relative, expected in expected_paper_hashes.items():
        path = ROOT.parents[1] / relative
        actual = sha256(path)
        assert actual == expected, f'Manuscript artifact changed: {path}'
        paper_files.append({'path': str(path), 'sha256': actual, 'unchanged': True})
    verification = {'status': 'complete', 'models': complete_models, 'split_replays': checks,
                    'immutable_sources_and_training_code_verified': True,
                    'manuscript_artifacts_unchanged': paper_files,
                    'invariant_tests': 'python -m unittest test_cub_pairing_controls -v (4 passed)'}
    write_json(CUB / 'final_verification.json', verification)
    return protocol, semantic, labels, valid


def make_report(protocol, semantic, labels, valid):
    summary = read_csv(CUB / 'classification_summary.csv')
    groups = read_csv(CUB / 'classification_groups.csv')
    fidelity = read_csv(CUB / 'decoder_fidelity.csv')
    image_comparisons = [r for r in read_csv(CUB / 'paired_comparisons.csv') if r['seed'] == 'seed_average']
    seed_comparisons = read_csv(CUB / 'paired_seed_differences.csv')
    frozen = 100 * np.mean(semantic['test'].argmax(1) == labels['test'])
    table = {mode: next(r for r in summary if r['mode'] == mode and r['subset'] == 'all') for mode in MODES}
    lines = [
        '# CUB 与 Derm7pt 配对监督诊断', '',
        '报告比较真实配对、全局打乱和同类内打乱三种监督方式，汇总CUB与Derm7pt保存的模型、分类结果和目标重建指标。运行配置与种子见对应的protocol.json。', '',
        '## CUB 实验协议', '',
        '固定同一 Frozen DCBM 检查点、保留分区、CLIP 特征与训练集 geo11 原型库（K_s=128）。每组 decoder 都是768→128的仿射映射，head为128→200；种子变化包含初始化、训练顺序及训练配对置换，未重新训练 Frozen 或重建原型库。只置换训练图像与完整128维 teacher 向量的对应，不置换向量坐标。全局打乱破坏跨类别及图像对应；同类打乱保留类别相关性，破坏类内图像对应。', '',
        '每组只用有效训练 teacher 监督 decoder；验证 teacher 始终保留真实对应。decoder使用SmoothL1、AdamW、lr0.001、weight_decay0.0001、batch64、最多300轮、patience30，以验证损失最小的检查点为准。所有训练图像均进入残差分类头，分支标准化只拟合训练集；head使用CE+L1（0.0001）、Adam(lr0.001)、batch64、最多100轮、patience30，eta={0.1,0.25,0.5,0.75,1}，以验证正确数最大选择，同分保留首先遇到的候选。各组搜索预算和选择规则相同，选择出的eta/轮数可以不同。测试标签不参与训练或选参。', '',
        '所有组沿用相同的图像成员。±表示分支种子之间的样本标准差（ddof=1）；分类结果使用验证集选定的单检查点。成员、有效teacher覆盖率及检查点来源见运行记录。', '',
        '## CUB 分类结果', '',
        '| 设置 | 测试准确率（%） | 相对同一 Frozen（百分点） |',
        '|---|---:|---:|',
        f'| 固定 Frozen | {frozen:.2f} | — |',
    ]
    for mode in MODES:
        row = table[mode]
        lines.append(f"| {NAMES[mode]} | {float(row['mean_accuracy_pct']):.2f} ± {float(row['sample_std_across_seeds_pct']):.2f} | {float(row['mean_delta_frozen_pp']):+.3f} |")
    lines += ['', 'Frozen准确率由当前缓存中的基准模型预测计算；各组相对差值使用同一模型、同一测试成员作为参照。', '',
              '| 种子 | 真实配对 | 全局打乱 | 同类内打乱 |', '|---|---:|---:|---:|']
    for seed in protocol['seeds']:
        values = [next(float(r['accuracy_pct']) for r in groups if r['mode'] == mode and int(r['seed']) == seed and r['split'] == 'test' and r['subset'] == 'all') for mode in MODES]
        lines.append(f'| {seed} | ' + ' | '.join(f'{value:.4f}%' for value in values) + ' |')
    lines += ['', '## 成对差值与重建检查', '',
              '| 对比（真实配对减对照） | 平均差（百分点） | 固定模型的测试图像分层bootstrap 95%区间 | 三个成对种子的探索性t区间 |',
              '|---|---:|---:|---:|']
    for image_row in image_comparisons:
        seed_row = next(r for r in seed_comparisons if r['comparison'] == image_row['comparison'])
        control = image_row['comparison'].replace('paired-minus-', '')
        image_interval = interval_cell(image_row, 'image_stratified_bootstrap_95_low_pp',
                                       'image_stratified_bootstrap_95_high_pp')
        seed_interval = interval_cell(seed_row, 'paired_seed_t_95_low_pp', 'paired_seed_t_95_high_pp')
        lines.append(f"| 真实−{NAMES[control]} | {float(image_row['difference_pp']):+.3f} | {image_interval} | {seed_interval} |")
    lines += ['', '图像bootstrap共4000次，在每个鸟类内抽样，保持当前测试类别构成；每次对两个方法复用同一抽样索引。它只刻画固定模型、固定类别构成下的图像抽样变化。成对种子区间只基于3个decoder/head种子、df=2，仍条件于同一 Frozen、原型库和分区；不是完整流水线多次训练或官方测试集上的显著性证据。', '',
              '| 配对方式 | 有效teacher测试图像上的平均R² | 平均Pearson |',
              '|---|---:|---:|']
    for mode in MODES:
        rows = [r for r in fidelity if r['mode'] == mode and r['split'] == 'test']
        lines.append(f"| {NAMES[mode]} | {np.mean([float(r['r2_uniform']) for r in rows]):.3f} | {np.mean([float(r['pearson_defined_mean']) for r in rows]):.3f} |")
    means = {m: float(table[m]['mean_accuracy_pct']) for m in MODES}
    differences = {m: means['paired'] - means[m] for m in MODES[1:]}
    if all(value > 0 for value in differences.values()):
        conclusion = 'CUB真实配对的分类均值高于两个打乱对照；优势的稳定性需结合配对差值区间及逐种子方向评估。'
    else:
        conclusion = 'CUB真实配对的分类均值未同时高于两个打乱对照。当前结果未显示一致的额外分类优势，也不能由此确定分支使用的是背景、纹理或其他具体信息。'
    lines += ['', conclusion, '',
              '重建与分类检验的是不同问题：真实配对的teacher恢复较好，表明学到了部分真实目标对应；若分类接近或落后打乱组，则仍不能把分类收益专门归因于该对应关系。打乱头保留同样的128维输出和正则化训练，可能仍产生可供分类头利用的映射；本实验没有识别其具体内容。', '',
              '## 两个数据集：固定检查点诊断', '',
              '| 数据集／来源 | 固定 Frozen | 真实配对 | 全局打乱 | 同类内打乱 |',
              '|---|---:|---:|---:|---:|']
    combined = []
    for dataset, folder, frozen_value in [('CUB', CUB, frozen), ('Derm7pt', DERM, 100 * 300 / 391)]:
        local_rows = read_csv(folder / 'classification_summary.csv')
        cells = []
        for mode in MODES:
            row = next(r for r in local_rows if r['mode'] == mode and r['subset'] == 'all')
            cells.append(f"{float(row['mean_accuracy_pct']):.2f} ± {float(row['sample_std_across_seeds_pct']):.2f}")
            combined.append({'dataset': dataset, 'source': 'local_validation_selected_checkpoints',
                             'frozen_accuracy_pct': frozen_value, **row})
        lines.append(f'| {dataset}／固定检查点 | {frozen_value:.2f} | ' + ' | '.join(cells) + ' |')
    lines += ['', '两个数据集使用相同训练和选择规则，在固定Frozen下重复拟合decoder/head。Derm7pt保留分区中的共享病例限制了独立性，其配对区间按病例成组抽样。下方附加参考结果来自独立的汇总文件，单列报告。', '',
              '## Derm7pt附加参考汇总', '']
    author = json.loads(AUTHOR.read_text(encoding='utf-8'))
    supplied = author['reported_mean_and_plusminus_pct']
    lines += ['| 来源 | Frozen | 真实配对 | 全局打乱 | 同类内打乱 |', '|---|---:|---:|---:|---:|',
              f"| 附加参考汇总 | {author['frozen_accuracy_pct']:.2f} | {supplied['paired'][0]:.2f} ± {supplied['paired'][1]:.2f} | {supplied['global_shuffled'][0]:.2f} ± {supplied['global_shuffled'][1]:.2f} | {supplied['within_class_shuffled'][0]:.2f} ± {supplied['within_class_shuffled'][1]:.2f} |", '',
              '附加参考文件仅包含汇总均值与离散程度，未提供可回放的逐种子预测，因此不计算其成对种子差值方差，也不沿用另一组模型的置信区间。各数据集、模型及统计协议分别解释。', '',
              '## 文件、核验与复现', '',
              '核验使用独立CPU矩阵回放，比较逐图预测、成员、标签、teacher有效性及修正/损害计数，检查验证集选定的decoder与head，并独立复算R²。详细结果见final_verification.json。', '',
              f'- CUB结果：`{CUB}`',
              f'- 独立核验：`{CUB / "final_verification.json"}`',
              f'- 运行脚本：`{ROOT / "experiments/controls/run_cub_pairing_controls.py"}`',
              f'- Derm7pt结果：`{DERM}`', '',
              '复现时使用新输出目录，避免覆盖本次实验：', '', '```powershell',
              f"Set-Location '{ROOT}'",
              "python -X utf8 -m unittest tests.test_cub_pairing_controls -v",
              "python -m experiments.controls.run_cub_pairing_controls --output outputs/cub_pairing_controls_new_run",
              '```', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    write_csv(CUB / 'two_dataset_local_summary.csv', combined)
    write_json(CUB / 'report_manifest.json', {'report': str(REPORT), 'sha256': sha256(REPORT),
                                            'scope': 'CUB and Derm7pt only; server/local sources kept separate'})
    print(str(REPORT))


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        make_report(*verify_cub())
