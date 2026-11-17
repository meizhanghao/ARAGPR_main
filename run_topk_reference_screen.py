"""Screen saved CUB/Derm7pt predictions and exact Top-K training references.

No training, segmentation, checkpoint selection, or replacement of manuscript
assets.  Preparation writes only a new output directory.  Finalization can be
repeated after editing that directory's reference_reviews.json.  The complete
test population is scanned for each of three saved correctly paired models;
existing paper traces are retained separately.  A stable prediction is compared
with the Frozen runner-up and is NEVER described as a correction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from run_cub_figure_candidates import (
    SourceLedger, bundle_asset, checked_decomposition, digest, fit_text, font,
    paste_fit, read_csv, read_json, replay_logits, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
CACHE = ROOT / 'outputs/residual_controls_20261018_paired01'
OLD = WORK / '临时/图1_CUB候选_20261023/最终候选图集'
POS, NEG, INK = '#267D70', '#B96050', '#263541'
DATASETS = {
    'CUB': dict(dataset='CUB_200_2011', split='CUB_200_2011_split',
                assets='CUB_200_2011_SAM2_binary_v3',
                models='cub_pairing_controls_20261020_pilot01',
                trace='explanation_trace_cub_binary_v3/CUB_200_2011/decoded'),
    'Derm7pt': dict(dataset='derm7pt', split='derm7pt_split', assets='derm7pt_SAM2',
                    models='derm7pt_mechanism_20261020_pilot01',
                    trace='explanation_trace/derm7pt/decoded'),
}


def norm(value):
    return re.sub('[^a-z]', '', value.lower())


def relative(path, out):
    return str(Path(path).relative_to(out)).replace('\\', '/')


def original_name(key):
    return re.sub(r'_OUT_CROP_\d+_X_\.jpg$', '.jpg', key, flags=re.I)


def source_index(split_root):
    """Require an unambiguous basename when old exports have unqualified keys."""
    full, base = {}, defaultdict(list)
    for split in ('train', 'val', 'test'):
        for p in (split_root / split).glob('*/*'):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                full[split, str(p.relative_to(split_root / split)).lower()] = p
                base[split, p.name.lower()].append(p)
    def resolve(split, key):
        key = str(key).replace('\\', '/')
        if (split, key.lower()) in full:
            return full[split, key.lower()]
        matches = base[split, Path(key).name.lower()]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous/missing {split} image: {key}')
        return matches[0]
    return resolve, base


def medical_metadata():
    index = {}
    for row in read_csv(ROOT / 'dataset_all/derm7pt/meta/meta.csv'):
        diagnosis = row['diagnosis'].lower()
        binary = 'malignant' if ('melanoma' in diagnosis or 'basal cell carcinoma' in diagnosis) else 'benign'
        for modality in ('clinic', 'derm'):
            if not row[modality]:
                continue
            name = Path(row[modality]).name.lower()
            item = {'case_num': row['case_num'], 'diagnosis': row['diagnosis'],
                    'source_class': binary, 'modality': modality}
            if name in index and index[name]['case_num'] != item['case_num']:
                raise ValueError('Conflicting medical case identifier')
            if name in index and index[name]['modality'] != modality:
                item['modality'] = 'clinic_and_derm_metadata'
            index[name] = item
    retained = defaultdict(list)
    for split in ('train', 'val', 'test'):
        for p in (ROOT / 'derm7pt_split' / split).glob('*/*'):
            if p.is_file() and p.name.lower() in index:
                retained[index[p.name.lower()]['case_num']].append({'split': split, 'path': str(p)})
    return index, retained


def training_context(original, key, base, dataset, assets, out, ledger):
    dest = out / 'references' / base.name
    if dataset == 'CUB':
        stem, region = re.fullmatch(r'(.+)_OUT_CROP_(\d+)_X_\.jpg', key).groups()
        manifest_path = ledger.record(assets / 'masks' / (stem + '_MASKS_DONE.json'))
        manifest = read_json(manifest_path)
        rr, = [r for r in manifest['regions'] if int(r['region_id']) == int(region)]
        for kind in ('crop', 'mask'):
            suffix = '.jpg' if kind == 'crop' else '.png'
            if rr[kind + '_sha256'] != digest(base / (kind + suffix)):
                raise ValueError('Manifest/crop/mask hash mismatch')
        if manifest['source'].get('sha256') and digest(original) != manifest['source']['sha256']:
            raise ValueError('Original image hash mismatch')
        processed = manifest['processed_size_wh']
        with Image.open(original) as im:
            image = im.convert('RGB').resize(tuple(processed), Image.Resampling.LANCZOS)
        bbox, status = rr['bbox_xywh'], 'hash_verified_segmentation_manifest'
        extra = {'manifest': str(manifest_path), 'processed_size_wh': processed}
    else:
        with Image.open(original) as im:
            rgb = np.asarray(im.convert('RGB'))
        with Image.open(base / 'crop.jpg') as im:
            crop = np.asarray(im.convert('RGB'))
        with Image.open(base / 'mask.png') as im:
            support = (np.asarray(im.convert('L')) > 127).astype('uint8')
        # The saved medical crop is at the original resolution. Match only its
        # masked pixels; black surroundings in the saved cutout are not evidence.
        if crop.shape[0] > rgb.shape[0] or crop.shape[1] > rgb.shape[1] or not support.any():
            raise ValueError('Invalid medical crop dimensions or empty support')
        scores = cv2.matchTemplate(rgb.astype('float32'), crop.astype('float32'),
            cv2.TM_SQDIFF, mask=np.repeat(support[:, :, None], 3, axis=2))
        scores = np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=np.inf)
        y, x = np.unravel_index(np.argmin(scores), scores.shape)
        h, w = crop.shape[:2]
        denom = int(support.sum()) * 3
        mse = float(scores[y, x]) / denom
        scores[max(0, y-h):y+h+1, max(0, x-w):x+w+1] = np.inf
        alt = float(np.min(scores)) / denom
        # Less restrictive than the old six-reference audit, with the actual
        # error and ambiguity retained. Failed matches are never called verified.
        status = ('rgb_masked_match_verified' if mse < 25 and
                  (not np.isfinite(alt) or alt > max(25, mse*10)) else 'rgb_match_requires_review')
        bbox = [int(x), int(y), int(w), int(h)]
        image = Image.fromarray(rgb)
        extra = {'mask_support_mse': mse,
                 'nonoverlapping_alternative_mse': alt if np.isfinite(alt) else None}
    x, y, w, h = bbox
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x+w, y+h), outline='#0072B2', width=3)
    image.save(dest / 'training_context.png')
    # A magnified contextual view makes 10-pixel branches/eye patches checkable.
    pad = max(20, min(120, max(w, h)))
    box = (max(0, x-pad), max(0, y-pad), min(image.width, x+w+pad), min(image.height, y+h+pad))
    image.crop(box).save(dest / 'context_detail.png')
    result = {'source': str(original), 'source_sha256': digest(original),
              'bbox_xywh': bbox, 'status': status, **extra,
              'scope': 'training reference display only; not a test localization or clinical annotation'}
    write_json(dest / 'region_provenance.json', result)
    return result


def prepare(out, topk, resume=False):
    if out.exists() and not (resume and (out/'selected_folders_before.json').is_file()
                             and not (out/'all_cases_scan.json').exists()):
        raise FileExistsError('Preparation requires a new output directory; use --finalize for the prepared gallery.')
    for folder in ('references', 'sheets', 'cases', 'panels', 'waterfalls', 'figure1_previews'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    ledger = SourceLedger()
    # Fingerprint the user-selected folders. They are not output destinations.
    protected = WORK / '临时/图1_R44-0877_20261024'
    selection = {str(p): digest(p) for folder in ('图1', '图2')
                 for p in (protected / folder).rglob('*') if p.is_file()}
    if (out/'selected_folders_before.json').is_file():
        if read_json(out/'selected_folders_before.json') != selection:
            raise ValueError('Selected folders changed since preparation began')
    else:
        write_json(out / 'selected_folders_before.json', selection)
    medical, retained = medical_metadata()
    ledger.record(ROOT / 'dataset_all/derm7pt/meta/meta.csv')
    import torch
    from cancha_cub import _preprocess_mask, _extract_contour_raw, _extract_geometric_features
    from utils.sdr_cbm_visualize import SDRCBMVisualizer

    def geometry(path):
        clean = _preprocess_mask(path)
        contour = _extract_contour_raw(clean)
        return clean.shape, contour, _extract_geometric_features(contour)

    old_refs = {}
    for p in (OLD / 'references').glob('*/review.json'):
        r = read_json(p)
        old_refs[r['representative_key']] = (p.parent, r)
    all_refs, lookups, runs, cases, replay_checks, exclusions = {}, {}, [], [], [], []
    review_overrides = {
        '79ef29d5f5af65cf': ('background', 'Branch fragment', '枝条碎片'),
        '0c897f008835ae19': ('background', 'Blurred background', '模糊背景'),
        '98c44bbf00631767': ('artifact', 'Copyright logo', '版权图案'),
    }

    def materialize(dataset, pid, source, key, resolve, assets, alias):
        evidence = read_json(ledger.record(source / 'evidence.json'))
        if evidence['representative_key'] != key or evidence['geometry_status'] != 'verified_against_training_descriptor':
            raise ValueError(f'Unverified descriptor: {source}')
        if evidence['metadata'].get('source_split') != 'train':
            raise ValueError('A reference is not from training')
        rid = dataset + '-' + hashlib.sha256((key+'\n'+str(evidence['metadata']['raw_descriptor'])).encode()).hexdigest()[:16]
        if rid in all_refs:
            if alias not in all_refs[rid]['aliases']:
                all_refs[rid]['aliases'].append(alias)
            return rid
        dest = out / 'references' / rid
        dest.mkdir(exist_ok=True)
        for name in ('crop.jpg', 'mask.png', 'neutral_crop.png', 'verified_contour.png', 'overlay.png', 'evidence.json'):
            p = ledger.record(source / name)
            shutil.copy2(p, dest / name)
        original = ledger.record(resolve('train', original_name(key)))
        shutil.copy2(original, dest / 'training_original.jpg')
        context = training_context(original, key, dest, dataset, assets, out, ledger)
        previous = old_refs.get(key, (None, {}))[1] if dataset == 'CUB' else {}
        category = previous.get('category', 'unreviewed')
        label_en, label_zh = previous.get('label_en', 'Unreviewed region'), previous.get('label_zh', '待复核')
        old_id = previous.get('identity')
        if old_id in review_overrides:
            category, label_en, label_zh = review_overrides[old_id]
        ref = {'reference_id': rid, 'dataset': dataset, 'representative_key': key,
               'original_path': str(original), 'original_name': original.name,
               'category': category, 'label_en': label_en, 'label_zh': label_zh,
               'context_status': context['status'], 'aliases': [alias],
               'geometry_status': evidence['geometry_status'],
               'review_scope': 'displayed training region, not the entire cluster or test-image localization'}
        if dataset == 'CUB':
            ref['source_class'] = key.split('/')[0].split('.', 1)[-1]
        else:
            ref.update(medical[original.name.lower()])
            ref['retained_case_images'] = retained[ref['case_num']]
        all_refs[rid] = ref
        return rid

    for dataset, cfg in DATASETS.items():
        print('Prepare references:', dataset, flush=True)
        resolve, _ = source_index(ROOT / cfg['split'])
        assets = ROOT / 'data/Segments/Seg_embs' / cfg['assets']
        bank_path = ledger.record(CACHE / cfg['dataset'] / 'teacher_bank.pt')
        bank = torch.load(bank_path, map_location='cpu', weights_only=False)
        manifest = read_json(ledger.record(CACHE / cfg['dataset'] / 'dataset.json'))
        classes = [c.split('.', 1)[-1] if dataset == 'CUB' else c for c in manifest['class_order']]
        cbm = SimpleNamespace(dataset=cfg['dataset'], class_names=classes, K=2048)
        rt = SimpleNamespace(cbm=cbm, shape_proto_representative=bank['prototype_representatives'],
                             shape_proto_metadata=bank['prototype_metadata'])
        viz = SDRCBMVisualizer(rt, out_dir=str(out / '_reference_build' / dataset),
            dataset=cfg['dataset'], image_root=str(ROOT / cfg['split']),
            shape_asset_root=str(assets), device='cpu', shape_geometry_fn=geometry)
        bank_lookup = {}
        for pid in range(128):
            key = bank['prototype_representatives'][pid]
            old = old_refs.get(key) if dataset == 'CUB' else None
            if old and read_json(old[0] / 'evidence.json')['metadata']['raw_descriptor'] == bank['prototype_metadata'][pid]['raw_descriptor']:
                source = old[0]
            else:
                evidence = viz._prepare_shape_evidence(pid, str(out / '_reference_build' / dataset))
                source = Path(evidence['evidence_json_path']).parent
            bank_lookup[pid] = materialize(dataset, pid, source, key, resolve, assets, 'R/P'+str(pid))
            if pid % 32 == 31:
                print(f'  {pid+1}/128 references', flush=True)
        lookups[dataset] = bank_lookup
        frozen_path = ledger.record(CACHE / cfg['dataset'] / 'test_frozen.npz')
        with np.load(frozen_path, allow_pickle=False) as a:
            frozen, labels, keys = a['logits'].copy(), a['labels'].copy(), a['sample_keys'].astype(str)
        for seed in (42, 43, 44):
            folder = ROOT / 'outputs' / cfg['models'] / f'seed_{seed}' / 'paired'
            model_path = ledger.record(folder / 'selected_model.pt')
            pred_path = ledger.record(folder / 'test_predictions.npz')
            state = torch.load(model_path, map_location='cpu', weights_only=False)
            if state['mode'] != 'paired' or int(state['seed']) != seed:
                raise ValueError('Expected a correctly paired saved model')
            with np.load(pred_path, allow_pickle=False) as a:
                if not np.array_equal(keys, a['sample_keys'].astype(str)) or not np.array_equal(labels, a['labels']):
                    raise ValueError('Prediction membership differs from frozen cache')
                np.testing.assert_allclose(frozen, a['semantic_logits'], rtol=3e-5, atol=6e-5)
                branch, shape_logits = replay_logits(a['decoded'], state['coordinate_scaler'], state['head'])
                final = frozen + shape_logits
                np.testing.assert_allclose(final, a['logits'], rtol=3e-5, atol=6e-5)
                if not np.array_equal(final.argmax(1), a['logits'].argmax(1)):
                    raise ValueError('Replay changed a predicted label')
                error = float(np.max(abs(final-a['logits'])))
                teacher_valid = a['valid_teacher'].astype(bool)
            selected = state['head']['selected']
            weights = np.asarray(selected['state']['weight'], dtype='float64') * float(selected['eta'])
            bias = np.asarray(selected['state']['bias'], dtype='float64') * float(selected['eta'])
            base_pred, final_pred = frozen.argmax(1), final.argmax(1)
            run = f'R{seed}'
            run_id = f'{dataset}-{run}'
            replay_checks.append({'run': run_id, 'n_samples': len(keys), 'max_logit_error': error,
                                  'all_top1_predictions_reproduced': True})
            runs.append({'run': run_id, 'model_path': str(model_path), 'model_sha256': digest(model_path),
                         'prediction_path': str(pred_path), 'prediction_sha256': digest(pred_path),
                         'teacher_bank_path': str(bank_path), 'teacher_bank_sha256': digest(bank_path),
                         'eta': float(selected['eta']), 'scope': 'saved local validation-selected single checkpoint'})
            for i, key in enumerate(keys):
                f, base, true = int(final_pred[i]), int(base_pred[i]), int(labels[i])
                if base != f:
                    b, rule = base, 'final_vs_Frozen_prediction'
                    category = 'correction' if f == true else ('harm' if base == true else 'other_flip')
                else:
                    candidates = frozen[i].copy()
                    candidates[base] = -np.inf
                    b, rule = int(candidates.argmax()), 'same_prediction_vs_Frozen_runner_up'
                    category = 'unchanged_correct' if f == true else 'unchanged_error'
                dw = weights[f] - weights[b]
                q = branch[i].astype('float64') * dw
                order = np.lexsort((np.arange(len(q)), -abs(q)))[:topk]
                sem = float(frozen[i, f]-frozen[i, b])
                end = float(final[i, f]-final[i, b])
                qb = float(bias[f]-bias[b])
                closure = abs(sem + math.fsum(q) + qb - end)
                if closure > 1.2e-4:
                    raise ValueError(f'Full contribution closure failed: {run_id}/{i} {closure}')
                original = resolve('test', key)
                item = {'candidate_id': f'{run_id}-{i:04d}', 'dataset': dataset, 'run': run,
                        'index': i, 'sample_key': key, 'input_path': str(original),
                        'category': category, 'prediction_changed': base != f,
                        'true_name': classes[true], 'frozen_name': classes[base],
                        'final_name': classes[f], 'contrast_name': classes[b], 'contrast_rule': rule,
                        'semantic_margin': sem, 'final_margin': end, 'bias': qb,
                        'other': float(math.fsum(q)-math.fsum(q[order])),
                        'topk_sum': float(math.fsum(q[order])),
                        'coverage': float(abs(q[order]).sum()/max(abs(q).sum(), 1e-12)),
                        'closure_error': closure, 'test_teacher_valid': bool(teacher_valid[i]),
                        'prediction_path': str(pred_path), 'model_path': str(model_path),
                        'top': [{'prototype_id': int(pid), 'rank': j+1, 'reference_id': bank_lookup[int(pid)],
                                 'q': float(q[pid]), 'activation': float(branch[i, pid]),
                                 'effective_weight_difference': float(dw[pid])}
                                for j, pid in enumerate(order)]}
                if dataset == 'Derm7pt':
                    meta = medical[original.name.lower()]
                    if norm(meta['source_class']) != norm(classes[true]):
                        raise ValueError('Medical metadata disagrees with saved test class')
                    item.update(query_case_num=meta['case_num'], query_diagnosis=meta['diagnosis'],
                                query_modality=meta['modality'],
                                query_case_other_splits=sorted({r['split'] for r in retained[meta['case_num']] if r['split'] != 'test'}))
                cases.append(item)
            print(f'Scanned {run_id}: {len(keys)} samples, replay error {error:.3g}', flush=True)
        # Retain original manuscript trace separately; its coordinates/head may
        # differ from the correctly paired control checkpoints above.
        trace = ROOT / 'outputs' / cfg['trace']
        info = read_json(ledger.record(trace / 'trace_run.json'))
        runs.append({'run': dataset+'-T', 'trace_root': str(trace), 'trace_run': info,
                     'scope': 'existing local trace only; all saved changed cases'})
        for p in sorted(trace.glob('*test_*/case_summary.json')):
            required=('shape_coordinates.csv','class_logits.csv','shape_prototypes.csv')
            missing=[name for name in required if not (p.parent/name).is_file()]
            if missing:
                exclusions.append({'dataset':dataset,'path':str(p.parent),
                                   'reason':'incomplete_original_trace','missing':missing})
                continue
            s = read_json(ledger.record(p))
            coords = read_csv(ledger.record(p.parent / 'shape_coordinates.csv'))
            classrows = read_csv(ledger.record(p.parent / 'class_logits.csv'))
            top, other, closure = checked_decomposition(s, coords, classrows, topk=topk)
            refs = {int(r['prototype_id']): r for r in read_csv(ledger.record(p.parent / 'shape_prototypes.csv'))}
            displayed = []
            for rank, row in enumerate(top, 1):
                pid = int(row['prototype_id'])
                source_row = refs[pid]
                ep = bundle_asset(p.parent, source_row['evidence_json_relative_path'], trace)
                rid = materialize(dataset, pid, ep.parent, source_row['representative_key'],
                                  resolve, assets, 'T/P'+str(pid))
                displayed.append({'prototype_id': pid, 'rank': rank, 'reference_id': rid,
                                  'q': float(row['signed_margin_contribution']),
                                  'activation': float(row['activation']),
                                  'effective_weight_difference': float(row['weight_difference_f_minus_b'])})
            original = resolve('test', s['sample_key'])
            asum = math.fsum(abs(float(r['signed_margin_contribution'])) for r in coords)
            item = {'candidate_id': f'{dataset}-T-{int(s["index"]):04d}', 'dataset': dataset, 'run': 'T',
                    'index': int(s['index']), 'sample_key': s['sample_key'], 'input_path': str(original),
                    'category': s['category'], 'prediction_changed': True,
                    'true_name': s['true_name'], 'frozen_name': s['dcbm_name'],
                    'final_name': s['sdrcbm_name'], 'contrast_name': s['dcbm_name'],
                    'contrast_rule': 'final_vs_Frozen_prediction',
                    'semantic_margin': s['semantic_margin_f_minus_b'], 'final_margin': s['final_margin_f_minus_b'],
                    'bias': s['shape_bias_margin_f_minus_b'], 'other': other,
                    'topk_sum': sum(r['q'] for r in displayed),
                    'coverage': sum(abs(r['q']) for r in displayed)/max(asum, 1e-12),
                    'closure_error': closure, 'trace_case': str(p.parent), 'top': displayed}
            if dataset == 'Derm7pt':
                meta = medical[original.name.lower()]
                item.update(query_case_num=meta['case_num'], query_diagnosis=meta['diagnosis'],
                            query_modality=meta['modality'],
                            query_case_other_splits=sorted({r['split'] for r in retained[meta['case_num']] if r['split'] != 'test'}))
            cases.append(item)
    # Save a compact exact input scan, separate from editable visual annotations.
    write_json(out / 'references.json', list(all_refs.values()))
    write_json(out / 'reference_reviews.json', {rid: {'category': r['category'], 'label_en': r['label_en'],
               'label_zh': r['label_zh'], 'review_note': ''} for rid, r in all_refs.items()})
    write_json(out / 'all_cases_scan.json', cases)
    write_json(out / 'runs.json', runs)
    write_json(out / 'replay_checks.json', replay_checks)
    write_json(out / 'excluded_incomplete_traces.json', exclusions)
    write_json(out / 'protocol.json', {'topk': topk, 'n_coordinates': 128,
        'ranking': 'descending absolute signed contribution; original coordinate IDs break ties',
        'unchanged_contrast': 'final class versus the Frozen runner-up; NOT a prediction correction',
        'changed_contrast': 'final class versus the actual Frozen top-1 class',
        'new_training': False, 'new_segmentation': False,
        'scope': 'qualitative case screening; not a representative interpretability evaluation or server table update'})
    sheets(list(all_refs.values()), out)
    inputs = ledger.verify()
    write_json(out / 'source_integrity.json', {'all_recorded_inputs_unchanged': True, 'files': inputs})
    print('Preparation complete:', len(cases), 'run-sample cases;', len(all_refs), 'reference regions', flush=True)


def sheets(refs, out):
    index = []
    for dataset in DATASETS:
        group = sorted([r for r in refs if r['dataset'] == dataset], key=lambda r: r['aliases'][0])
        for page, start in enumerate(range(0, len(group), 16), 1):
            subset = group[start:start+16]
            image = Image.new('RGB', (1840, 370*math.ceil(len(subset)/4)), 'white')
            draw = ImageDraw.Draw(image)
            for j, r in enumerate(subset):
                x, y = j%4*460, j//4*370
                base = out/'references'/r['reference_id']
                fit_text(draw, r['reference_id'][-16:]+' '+','.join(r['aliases']), (x+8,y+5), 442, 17)
                paste_fit(image, base/'training_context.png', (x+7,y+35,220,205))
                paste_fit(image, base/'context_detail.png', (x+233,y+35,220,100))
                paste_fit(image, base/'neutral_crop.png', (x+233,y+141,104,100))
                paste_fit(image, base/'verified_contour.png', (x+345,y+141,104,100))
                fit_text(draw, r['original_name'], (x+8,y+246), 442, 18)
                fit_text(draw, r['source_class'], (x+8,y+294), 442, 20)
                fit_text(draw, r['category']+': '+r['label_en'], (x+8,y+325), 442, 17)
                draw.rectangle((x,y,x+459,y+369), outline='#CDD6DD')
            dest = out/'sheets'/f'{dataset}_{page:02d}.jpg'
            image.save(dest, quality=93)
            index.append({'dataset': dataset, 'page': page, 'path': relative(dest,out),
                          'references': [r['reference_id'] for r in subset]})
    write_json(out/'sheets/index.json', index)


BODY = {'CUB': {'bird_part','whole_bird'}, 'Derm7pt': {'lesion','lesion_part'}}


def enrich(cases, refs):
    for c in cases:
        positive, negative = [], []
        counts = Counter()
        for row in c['top']:
            r = refs[row['reference_id']]
            body = r['category'] in BODY[c['dataset']]
            same = norm(r['source_class']) == norm(c['final_name'])
            opposite = norm(r['source_class']) == norm(c['contrast_name'])
            same_query_case = c['dataset'] == 'Derm7pt' and r['case_num'] == c['query_case_num']
            row.update(is_body=body, same_final_class=same, same_contrast_class=opposite,
                       same_query_case=same_query_case,
                       source_class=r['source_class'], original_name=r['original_name'],
                       original_path=r['original_path'], category=r['category'], label=r['label_zh'])
            counts['body_count'] += body
            counts['body_sameclass_count'] += body and same
            counts['positive_sameclass_body_count'] += body and same and row['q'] > 0 and not same_query_case
            counts['positive_otherclass_body_count'] += body and not same and row['q'] > 0
            counts['negative_otherclass_body_count'] += body and not same and row['q'] < 0
            counts['negative_sameclass_body_count'] += body and same and row['q'] < 0
            counts['artifact_background_count'] += r['category'] in ('background','artifact','skin','field')
            counts['unreviewed_count'] += r['category'] == 'unreviewed'
            counts['same_query_case_count'] += same_query_case
            (positive if row['q'] > 0 else negative).append(row)
        for key in ('body_count','body_sameclass_count','positive_sameclass_body_count',
                    'positive_otherclass_body_count','negative_otherclass_body_count',
                    'negative_sameclass_body_count','artifact_background_count','unreviewed_count',
                    'same_query_case_count'):
            c[key] = int(counts[key])
        c['mixed_signs'] = bool(positive and negative)
        # A strict requested visual pattern requires actual regions AND the
        # reference-source labels. A coordinate's sign alone is insufficient.
        c['has_requested_positive'] = c['positive_sameclass_body_count'] > 0
        c['all_topk_body'] = c['body_count'] == len(c['top'])
        c['complete_sign_class_pattern'] = bool(positive and negative) and all(
            r['is_body'] and (r['same_final_class'] if r['q'] > 0 else not r['same_final_class'])
            and not r['same_query_case'] for r in c['top'])
        c['all_positive_sameclass_body'] = bool(positive) and all(
            r['is_body'] and r['same_final_class'] and not r['same_query_case'] for r in positive)
        c['preferred_medical_case'] = (c['dataset'] != 'Derm7pt' or
                                      (not c['query_case_other_splits'] and not c['same_query_case_count']))
    return cases


def case_order(c):
    return (c['category'] != 'correction', not c['has_requested_positive'],
            not c['complete_sign_class_pattern'], not c['preferred_medical_case'],
            -c['positive_sameclass_body_count'], -c['negative_otherclass_body_count'],
            -c['body_count'], c['unreviewed_count'], -c['coverage'], c['candidate_id'])


def pure_panel(c, out):
    """Only source photographs, crops, contours and real values: no P labels."""
    image = Image.new('RGB', (1830, 630), 'white')
    draw = ImageDraw.Draw(image)
    paste_fit(image, c['input_path'], (10, 125, 280, 390))
    draw.line((304,10,304,615), fill='#CDD6DD', width=2)
    for j, row in enumerate(c['top']):
        base = out/'references'/row['reference_id']
        x = 322+300*j
        for name, y, height in (('training_context.png',10,240),('neutral_crop.png',265,138),
                                ('verified_contour.png',418,145)):
            paste_fit(image, base/name, (x,y,280,height))
        draw.text((x+140,597), f"{row['q']:+.3f}", anchor='mm', font=font(35),
                  fill=POS if row['q'] > 0 else NEG)
    path = out/'panels'/(c['candidate_id']+'.png')
    image.save(path, dpi=(300,300))
    return relative(path,out)


@lru_cache(maxsize=1)
def waterfall_runtime():
    """Reuse the verified Figure 4 renderer, including its cumulative bars."""
    source=WORK/'lunwen/tools/export_selected_waterfall.py'
    spec=importlib.util.spec_from_file_location('paper_waterfall_export',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    style=module.figure4_style()
    module.plt.rcParams.update({'font.family':'serif','font.serif':[style['font']],
        'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    return module,style


def waterfall(c, out):
    module,style=waterfall_runtime()
    compact=[]
    level=0.
    for row in c['top']:
        q=row['q']
        compact.append({'component_type':'shape_coordinate','component_id':str(row['prototype_id']),
                        'delta':q,'cumulative_before':level,'cumulative_after':level+q})
        level+=q
    full=[{'component_type':'semantic_total','component_id':'','delta':c['semantic_margin'],
           'cumulative_before':0.,'cumulative_after':c['semantic_margin']}]
    level=c['semantic_margin']
    for row in compact:
        full.append({**row,'cumulative_before':level,'cumulative_after':level+row['delta']})
        level+=row['delta']
    for kind,value in (('shape_other',c['other']),('shape_bias',c['bias'])):
        full.append({'component_type':kind,'component_id':'','delta':value,
                     'cumulative_before':level,'cumulative_after':level+value})
        level+=value
    full.append({'component_type':'final_total','component_id':'','delta':c['final_margin'],
                 'cumulative_before':0.,'cumulative_after':c['final_margin']})
    if abs(level-c['final_margin'])>1.2e-4:raise ValueError('Rendered waterfall does not close')
    records={}
    for name,rows in (('values',compact),('full',full)):
        stem=out/'waterfalls'/(c['candidate_id']+'_'+name)
        records[name]=module.render(rows,stem,style,'',values_only=True,formats=('png','svg'))
        records[name]['partial']=(name=='values')
        if any('P' in text or 'Top' in text for text in records[name]['visible_text']):
            raise ValueError('Identifier lettering unexpectedly entered pure plot')
    write_csv(out/'cases'/(c['candidate_id']+'_waterfall.csv'),full)
    write_json(out/'cases'/(c['candidate_id']+'_waterfall_render.json'),records)


def figure1_preview(c,out):
    """DCBM versus SDR-CBM comparison, retaining actual prediction behavior."""
    image=Image.new('RGB',(1830,870),'white')
    d=ImageDraw.Draw(image)
    fit_text(d,'Frozen DCBM', (350,20),680,34)
    fit_text(d,'Shape-decoded residual CBM', (1050,20),740,34)
    paste_fit(image,c['input_path'],(15,180,285,415))
    d.line((320,80,320,830),fill='#CDD6DD',width=2)
    fit_text(d,c['frozen_name'],(350,80),640,29)
    fit_text(d,c['final_name'],(1050,80),740,29,POS if c['final_name']==c['true_name'] else NEG)
    fit_text(d,'Same prediction' if not c['prediction_changed'] else 'Prediction changed',
             (350,125),640,23)
    # Reuse exact saved semantic exemplars where a trace is available.
    trace=Path(c['trace_case']) if c.get('trace_case') else None
    if trace is None and c['dataset']=='CUB' and c['prediction_changed']:
        root=WORK/'临时/图1_CUB候选_20261023/运行缓存/replayed_traces'/c['run']/'CUB_200_2011/decoded'
        maybe=root/f"{c['category']}_test_{c['index']:05d}"
        if maybe.is_dir():trace=maybe
    semantic=[]
    if trace and (trace/'semantic_prototypes.csv').is_file():
        boundary=trace.parent
        for row in read_csv(trace/'semantic_prototypes.csv')[:3]:
            p=bundle_asset(trace,row.get('bundle_crop_relative_path'),boundary)
            if p:semantic.append(p)
    if semantic:
        for j,p in enumerate(semantic):paste_fit(image,p,(350+j*215,225,200,205))
        fit_text(d,'Saved semantic training references',(350,470),650,25)
    else:
        fit_text(d,'Frozen semantic prediction',(350,310),650,30)
        fit_text(d,'No geometric reference readout in this branch',(350,380),650,23)
    for j,row in enumerate(c['top']):
        x=1050+151*j
        base=out/'references'/row['reference_id']
        paste_fit(image,base/'neutral_crop.png',(x,220,140,150))
        paste_fit(image,base/'verified_contour.png',(x,390,140,150))
        d.text((x+70,580),f"{row['q']:+.3f}",anchor='mm',font=font(27),fill=POS if row['q']>0 else NEG)
    fit_text(d,'Training shape references',(1050,650),740,27)
    fit_text(d,'Contrast: '+c['final_name']+' versus '+c['contrast_name'],(350,745),1440,25)
    fit_text(d,f"Semantic {c['semantic_margin']:+.3f}   Other {c['other']:+.3f}   Bias {c['bias']:+.3f}   Final {c['final_margin']:+.3f}",
             (350,795),1440,25)
    path=out/'figure1_previews'/(c['candidate_id']+'.png')
    image.save(path,dpi=(300,300))
    return relative(path,out)


@lru_cache(maxsize=6)
def saved_run_arrays(model_path,prediction_path):
    import torch
    model=torch.load(model_path,map_location='cpu',weights_only=False)
    with np.load(prediction_path,allow_pickle=False) as source:
        saved={k:source[k].copy() for k in source.files}
    branch,shape=replay_logits(saved['decoded'],model['coordinate_scaler'],model['head'])
    selected=model['head']['selected']
    return (saved,branch,shape,np.asarray(selected['state']['weight'],dtype='float64')*float(selected['eta']),
            np.asarray(selected['state']['bias'],dtype='float64')*float(selected['eta']))


def export_case_sources(c,out):
    """Keep full logits/128 coordinates beside every rendered candidate."""
    dest=out/'cases'/c['candidate_id']
    dest.mkdir(exist_ok=True)
    source=Path(c['input_path'])
    shutil.copy2(source,dest/'input.jpg')
    if c.get('trace_case'):
        for name in ('case_summary.json','shape_coordinates.csv','shape_prototypes.csv',
                     'semantic_prototypes.csv','class_logits.csv'):
            p=Path(c['trace_case'])/name
            if p.is_file():shutil.copy2(p,dest/name)
    else:
        saved,branch,shape,weight,bias=saved_run_arrays(c['model_path'],c['prediction_path'])
        i=c['index']
        semantic=saved['semantic_logits']
        final=saved['logits']
        f=int(final[i].argmax())
        if c['prediction_changed']:b=int(semantic[i].argmax())
        else:
            z=semantic[i].copy();z[f]=-np.inf;b=int(z.argmax())
        dw=weight[f]-weight[b]
        q=branch[i].astype('float64')*dw
        order=np.lexsort((np.arange(len(q)),-abs(q)))
        if list(order[:len(c['top'])]) != [r['prototype_id'] for r in c['top']]:
            raise ValueError('Rendered candidate no longer matches exact top coordinates')
        if max(abs(q[r['prototype_id']]-r['q']) for r in c['top'])>1e-8:
            raise ValueError('Rendered contribution changed')
        ranks={int(pid):rank+1 for rank,pid in enumerate(order)}
        write_csv(dest/'shape_coordinates.csv',[{'prototype_id':pid,'absolute_rank':ranks[pid],
            'activation':float(branch[i,pid]),'weight_difference_f_minus_b':float(dw[pid]),
            'signed_margin_contribution':float(q[pid]),'contrast_f':f,'contrast_b':b}
            for pid in range(len(q))])
        cfg=DATASETS[c['dataset']]
        manifest=read_json(CACHE/cfg['dataset']/'dataset.json')
        classes=[x.split('.',1)[-1] if c['dataset']=='CUB' else x for x in manifest['class_order']]
        write_csv(dest/'class_logits.csv',[{'class_id':k,'class_name':name,
            'semantic_logit':float(semantic[i,k]),'shape_logit':float(shape[i,k]),
            'final_logit':float(final[i,k])} for k,name in enumerate(classes)])
    write_json(dest/'candidate.json',{**c,'input_sha256':digest(source),
        'display_scope':'training references, not test localization',
        'all_128_coordinates_saved':True})


def finalize(out, render_limit, requested):
    refs={r['reference_id']:r for r in read_json(out/'references.json')}
    reviews=read_json(out/'reference_reviews.json')
    for rid,r in refs.items():r.update(reviews[rid])
    cases=enrich(read_json(out/'all_cases_scan.json'),refs)
    cases.sort(key=case_order)
    topk=read_json(out/'protocol.json')['topk']
    shortlist=[]
    for dataset in DATASETS:
        group=[c for c in cases if c['dataset']==dataset]
        selected=[]
        seen=set()
        # Prioritize true corrections; stable correct examples have their own
        # explicitly labeled lane so a cleaner picture cannot imply a flip.
        for category in ('correction','unchanged_correct','harm'):
            pool=[c for c in group if c['category']==category and c['has_requested_positive']]
            pool.sort(key=lambda c:(not c['complete_sign_class_pattern'],not c['preferred_medical_case'],
                -c['body_count'],-c['positive_sameclass_body_count'],
                -c['negative_otherclass_body_count'],-c['coverage'],c['candidate_id']))
            for c in pool:
                identity=c['sample_key'].lower()
                if identity in seen:continue
                seen.add(identity)
                selected.append(c)
                if sum(s['category']==category for s in selected)>=render_limit:break
        # A second lane gives cross-class body references when strict same-class
        # requested cases are scarce; it is never mixed into the strict count.
        for c in sorted([x for x in group if x['category']=='correction'],
                        key=lambda c:(-c['body_count'],-c['coverage'])):
            if c['sample_key'].lower() in seen:continue
            seen.add(c['sample_key'].lower());selected.append(c)
            if sum(s['category']=='correction' for s in selected)>=render_limit:break
        shortlist+=selected
    byid={c['candidate_id']:c for c in cases}
    for cid in requested:
        if cid not in byid:raise ValueError('Unknown --case '+cid)
        if byid[cid] not in shortlist:shortlist.append(byid[cid])
    for i,c in enumerate(shortlist):
        c['panel']=pure_panel(c,out)
        waterfall(c,out)
        c['figure1_preview']=figure1_preview(c,out)
        export_case_sources(c,out)
        write_json(out/'cases'/(c['candidate_id']+'.json'),c)
        if (i+1)%10==0:print(f'Rendered {i+1}/{len(shortlist)}',flush=True)
    shortlist_ids={c['candidate_id'] for c in shortlist}
    for c in cases:c['shortlisted']=c['candidate_id'] in shortlist_ids
    flat=[]
    for c in cases:
        flat.append({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
                     for k,v in c.items() if k not in ('top','panel','figure1_preview')})
    # Optional rendering fields differ by row; keep the CSV schema fixed.
    fields=list(dict.fromkeys(k for row in flat for k in row))
    write_csv(out/'all_cases_ranking.csv',[{k:row.get(k,'') for k in fields} for row in flat])
    write_csv(out/'all_topk_provenance.csv',[{'candidate_id':c['candidate_id'],'dataset':c['dataset'],**r}
              for c in cases for r in c['top']])
    write_csv(out/'matching_case_summary_zh.csv',[
        {'案例编号':c['candidate_id'],'数据集':c['dataset'],'状态':c['category'],
         '测试原图':c['sample_key'],'真实类别':c['true_name'],'Frozen预测':c['frozen_name'],
         '最终预测':c['final_name'],'对比类别':c['contrast_name'],
         '本体参考数量':c['body_count'],'正贡献同类本体数量':c['positive_sameclass_body_count'],
         '负贡献异类本体数量':c['negative_otherclass_body_count'],
         'Top5绝对贡献覆盖率':round(100*c['coverage'],3),
         '完整符合正同类负异类':c['complete_sign_class_pattern'],
         '医学病例未跨分区':c['preferred_medical_case'] if c['dataset']=='Derm7pt' else '',
         '五项原图名称':' | '.join(r['original_name'] for r in c['top']),
         '五项来源类别':' | '.join(r['source_class'] for r in c['top']),
         '五项区域内容':' | '.join(r['label'] for r in c['top']),
         '五项贡献':' | '.join(f"{r['q']:+.6f}" for r in c['top'])}
        for c in cases if c['has_requested_positive'] or c['all_topk_body'] or c['shortlisted']])
    write_json(out/'candidates.json',cases)
    write_json(out/'reviewed_references.json',list(refs.values()))
    stats={}
    for dataset in DATASETS:
        group=[c for c in cases if c['dataset']==dataset]
        stats[dataset]={'run_sample_count':len(group),'unique_test_images':len({c['sample_key'] for c in group}),
            'category_counts':dict(Counter(c['category'] for c in group)),
            'positive_sameclass_body_by_category':dict(Counter(c['category'] for c in group if c['has_requested_positive'])),
            'complete_sign_class_pattern_by_category':dict(Counter(c['category'] for c in group if c['complete_sign_class_pattern'])),
            'all_five_body_by_category':dict(Counter(c['category'] for c in group if c['all_topk_body'])),
            'reference_categories':dict(Counter(r['category'] for r in refs.values() if r['dataset']==dataset))}
        stats[dataset]['positive_sameclass_body_unique_correct_queries']=len({c['sample_key'] for c in group
            if c['has_requested_positive'] and c['category'] in ('correction','unchanged_correct')})
    protected=read_json(out/'selected_folders_before.json')
    unchanged=all(Path(p).is_file() and digest(p)==sha for p,sha in protected.items())
    current={str(p) for folder in ('图1','图2') for p in (WORK/'临时/图1_R44-0877_20261024'/folder).rglob('*') if p.is_file()}
    if not unchanged or current!=set(protected):raise ValueError('Existing selected folder content changed')
    recorded=read_json(out/'source_integrity.json')
    if any(digest(Path(r['path']))!=r['sha256'] for r in recorded['files']):
        raise ValueError('A recorded source changed after preparation')
    write_json(out/'manifest.json',{'status':'complete','topk':topk,'statistics':stats,
            'shortlist':[c['candidate_id'] for c in shortlist],
        'max_margin_closure_error':max(c['closure_error'] for c in cases),
        'selected_fig1_fig2_folders_unchanged':True,'new_training':False,'new_segmentation':False})
    sheets(list(refs.values()),out)
    gallery(cases,refs,stats,out,topk)
    write_report(cases,refs,stats,out,topk)
    recommendations(cases,out,topk)
    print(json.dumps({'statistics':stats,'shortlist':[c['candidate_id'] for c in shortlist],
                      'output':str(out),'old_selected_folders_unchanged':True},ensure_ascii=False,indent=2))


def gallery(cases,refs,stats,out,topk):
    # All selected evidence lives locally. A paged DOM keeps thousands of
    # full-resolution photographs from loading before the reader opens them.
    refdata={rid:{**r,'base':'references/'+rid} for rid,r in refs.items()}
    ui_fields=('candidate_id','dataset','category','input_path','sample_key','true_name','frozen_name',
        'final_name','contrast_name','prediction_changed','body_count','positive_sameclass_body_count',
        'negative_otherclass_body_count','coverage','query_case_num','query_diagnosis','query_case_other_splits',
        'same_query_case_count','semantic_margin','topk_sum','other','bias','final_margin','shortlisted',
        'preferred_medical_case','has_requested_positive','complete_sign_class_pattern','all_topk_body',
        'panel','figure1_preview')
    top_fields=('reference_id','prototype_id','rank','q','activation','effective_weight_difference',
                'is_body','same_final_class')
    ui_cases=[{**{k:c[k] for k in ui_fields if k in c},
               'top':[{k:r[k] for k in top_fields} for r in c['top']]} for c in cases]
    payload=json.dumps({'cases':ui_cases,'refs':refdata},ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CUB / Derm7pt 形状参考筛选</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;color:#263541;margin:24px auto;max-width:1580px;padding:0 18px;line-height:1.65}h1{font-size:27px}h2{font-size:21px}button,select,input{padding:8px;margin:4px;border:1px solid #aebbc3;border-radius:4px}button{cursor:pointer;background:#edf4f5}a{color:#0072b2}.note{padding:14px;background:#eef3f7}.controls{position:sticky;top:0;background:#fff;z-index:3;border-bottom:2px solid #cdd6dd;padding:8px}article{margin:25px 0;border:1px solid #cdd6dd;border-radius:6px;padding:16px}.badge{display:inline-block;margin:3px 6px 3px 0;padding:3px 9px;background:#edf4f5}.good{background:#e0f0e8}.bad{background:#f8e9e3}.query{display:grid;grid-template-columns:250px 1fr;gap:20px}.query img{width:245px;height:220px;object-fit:contain}.refs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:15px}.ref{min-width:0;border-top:4px solid #267d70;background:#fafcfd;padding:8px}.ref.neg{border-color:#b96050}.ref img{width:100%;height:160px;object-fit:contain}.ref .local{height:110px}.ref p{font-size:13px;overflow-wrap:anywhere;margin:5px 0}.ref .q{font-family:"Times New Roman",serif;font-size:24px;font-weight:500}.pure{max-width:100%;max-height:580px;object-fit:contain}summary{cursor:pointer}.small{font-size:13px;color:#60707c}#count{font-weight:bold}@media(max-width:900px){.refs{grid-template-columns:repeat(2,minmax(0,1fr))}.query{grid-template-columns:1fr}.controls{position:static}}</style>
<h1>CUB / Derm7pt：真实 Top-__K__ 形状参考候选</h1>
<p class="note">分别核对参考区域内容与来源类别。形状簇没有鸟种/诊断标签；正贡献也不等于高相似度。这里的同类仅指所展示训练参考原图的标签。<b>纠错</b>使用最终类与 Frozen 原预测的对比；<b>预测未变</b>使用共同预测类与 Frozen 第二候选的对比，不能冒充纠错。所有数值来自本地保存的单检查点，不能替代论文服务器平均准确率。</p>
<p><a href="筛选说明.md">筛选说明与推荐</a> · <a href="all_cases_ranking.csv">全部案例统计</a> · <a href="all_topk_provenance.csv">每一项的原图与贡献</a> · <a href="reference_index.html">全部训练参考</a> · <a href="manifest.json">处理范围</a></p>
<div class="controls"><select id="ds"><option value="CUB">CUB</option><option value="Derm7pt">Derm7pt</option><option value="">全部数据集</option></select><select id="cat"><option value="correction">纠错</option><option value="unchanged_correct">预测未变且正确</option><option value="harm">误改</option><option value="">全部状态</option></select><select id="req"><option value="positive">至少一项：正贡献＋同类＋本体</option><option value="complete">五项完整符合正同类/负异类</option><option value="body">五项均为本体</option><option value="">全部案例</option></select><label><input type="checkbox" id="independent">Derm7pt 病例不跨分区</label><label><input type="checkbox" id="short">仅已生成纯图</label><input id="search" placeholder="案例编号、原图名、类别" size="28"><div><button id="prev">上一页</button><span id="count"></span><button id="next">下一页</button></div></div><main id="main"></main>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));let page=0;const per=12;const labels={correction:'纠错',harm:'误改',unchanged_correct:'预测未变且正确',unchanged_error:'预测未变且错误',other_flip:'其他改变'};
function uri(p){return 'file:///'+String(p).replace(/\\\\/g,'/').split('/').map((v,i)=>i===0?v:encodeURIComponent(v)).join('/')}
function filters(){const q=$('search').value.toLowerCase();return DATA.cases.filter(c=>(!$('ds').value||c.dataset===$('ds').value)&&(!$('cat').value||c.category===$('cat').value)&&(!$('short').checked||c.shortlisted)&&(!$('independent').checked||c.preferred_medical_case)&&($('req').value!=='positive'||c.has_requested_positive)&&($('req').value!=='complete'||c.complete_sign_class_pattern)&&($('req').value!=='body'||c.all_topk_body)&&(!q||JSON.stringify(c).toLowerCase().includes(q)))}
function image(p,cls=''){return `<a href="${esc(p)}" target="_blank"><img loading="lazy" src="${esc(p)}" class="${cls}"></a>`}
function render(){const all=filters(),n=Math.ceil(all.length/per);page=Math.max(0,Math.min(page,n-1));$('count').textContent=` ${all.length} 个运行—案例；第 ${all.length?page+1:0}/${n} 页 `;$('main').innerHTML=all.slice(page*per,(page+1)*per).map(c=>{let a=`<article id="${c.candidate_id}"><h2>${esc(c.candidate_id)} <span class="badge ${c.category==='correction'?'good':c.category==='harm'?'bad':''}">${labels[c.category]}</span></h2><div class="query">${image(uri(c.input_path))}<div><p>测试原图：<a target="_blank" href="${esc(uri(c.input_path))}">${esc(c.sample_key)}</a></p><p>真实类别：${esc(c.true_name)}<br>Frozen：${esc(c.frozen_name)} → SDR：${esc(c.final_name)}</p><p>本图对比：${esc(c.final_name)} 相对 ${esc(c.contrast_name)} ${c.prediction_changed?'':'（Frozen 第二候选）'}</p><p><span class="badge">本体参考 ${c.body_count}/__K__</span><span class="badge">正同类本体 ${c.positive_sameclass_body_count}</span><span class="badge">负异类本体 ${c.negative_otherclass_body_count}</span><span class="badge">绝对贡献覆盖 ${(100*c.coverage).toFixed(1)}%</span></p>${c.dataset==='Derm7pt'?`<p>病例 ${esc(c.query_case_num)}；${esc(c.query_diagnosis)}；其他分区：${esc(c.query_case_other_splits.join(', ')||'无')}；参考同病例数 ${c.same_query_case_count}</p>`:''}<p class="small">Sem ${c.semantic_margin.toFixed(3)}；五项合计 ${c.topk_sum.toFixed(3)}；Other ${c.other.toFixed(3)}；Bias ${c.bias.toFixed(3)}；Final ${c.final_margin.toFixed(3)}</p></div></div><div class="refs">`;a+=c.top.map(t=>{const r=DATA.refs[t.reference_id],base=r.base;return `<div class="ref ${t.q<0?'neg':''}"><div class="q" style="color:${t.q>=0?'#267d70':'#b96050'}">${t.q>=0?'+':''}${t.q.toFixed(3)}</div><p>${esc(r.label_zh)} · ${t.is_body?'本体':'未计入本体'} · ${t.same_final_class?'同最终类':'异最终类'}</p>${image(base+'/training_context.png')}${image(base+'/context_detail.png','local')}${image(base+'/neutral_crop.png','local')}${image(base+'/verified_contour.png','local')}<p>原图：<a target="_blank" href="${esc(uri(r.original_path))}">${esc(r.original_name)}</a><br>来源类别：${esc(r.source_class)}${r.diagnosis?'<br>诊断：'+esc(r.diagnosis):''}<br>位置核查：${esc(r.context_status)}<br>溯源编号 P${t.prototype_id}；实际排名 ${t.rank}</p><p>头输入 ${t.activation.toFixed(5)} × 有效权重差 ${t.effective_weight_difference.toFixed(5)}</p><a href="${base}/evidence.json">原始证据</a></div>`}).join('');a+='</div>';if(c.panel)a+=`<details><summary>纯图、瀑布图与图1比较预览</summary>${image(c.panel,'pure')}${image('waterfalls/'+c.candidate_id+'_values.png','pure')}${image('waterfalls/'+c.candidate_id+'_full.png','pure')}${image(c.figure1_preview,'pure')}<p><a href="cases/${c.candidate_id}.json">完整案例记录</a></p></details>`;return a+'</article>'}).join('')||'<p>没有符合当前筛选的案例。可更换状态或放宽筛选查看原因。</p>'}
for(const id of ['ds','cat','req','independent','short','search'])$(id).addEventListener('input',()=>{page=0;render()});$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};function openHash(){if(location.hash){const cid=decodeURIComponent(location.hash.slice(1));$('ds').value=cid.startsWith('CUB-')?'CUB':'Derm7pt';$('cat').value='';$('req').value='';$('search').value=cid;page=0}render()}window.addEventListener('hashchange',openHash);openHash();</script></html>'''
    (out/'index.html').write_text(template.replace('__DATA__',payload).replace('__K__',str(topk)),encoding='utf-8')
    sections=['<!doctype html><meta charset="utf-8"><title>全部训练形状参考</title>',
              '<style>body{font-family:Microsoft YaHei;max-width:1800px;margin:20px auto}img{max-width:100%}p{margin:30px 0}</style>',
              '<h1>全部训练参考：原图位置、放大位置、区域、轮廓</h1>']
    for sheet in read_json(out/'sheets/index.json'):
        sections.append(f'<p>{sheet["dataset"]} · {sheet["page"]}</p><a href="{sheet["path"]}"><img loading="lazy" src="{sheet["path"]}"></a>')
    (out/'reference_index.html').write_text('\n'.join(sections),encoding='utf-8')


def write_report(cases,refs,stats,out,topk):
    lines=[f'# Top-{topk} 本体与来源类别筛选\n',
           '现有图1/图2文件夹保持完整；等作者选定新案例后再替换。没有重新训练、生成分割或更改论文数据。\n',
           '## 范围与口径\n',
           'CUB、Derm7pt 各自扫描三个已保存真实配对模型的完整测试预测，并保留原论文导出的预测改变案例。跨运行重复的测试图会分别保留，但推荐列表优先去重。\n',
           '形状簇是跨类几何聚类，来源类名只是当前训练参考原图的标签，不能将同类来源直接视为已识别的物种/诊断概念。正负号由响应与分类权重差共同决定。\n',
           '预测改变案例比较最终类与 Frozen 原预测；未改变案例比较共同预测与 Frozen 第二候选，明确标为“预测未变”，不作为纠错展示。保留全部128项分解，Other现在是其余123项。\n',
           '## 筛选统计\n',json.dumps(stats,ensure_ascii=False,indent=2),
           '\n## 供挑选的已生成图\n']
    for dataset in DATASETS:
        lines.append(f'\n### {dataset}\n')
        for c in cases:
            if c['dataset']==dataset and c.get('panel'):
                same=[f"P{r['prototype_id']} {r['q']:+.3f} {r['original_name']}" for r in c['top']
                      if r['is_body'] and r['same_final_class'] and r['q']>0]
                lines.append(f"- **{c['candidate_id']}**，{c['category']}，{c['sample_key']}；本体 {c['body_count']}/{topk}；正同类本体 {c['positive_sameclass_body_count']}；负异类本体 {c['negative_otherclass_body_count']}；同类正项：{'; '.join(same) or '无'}。")
    lines+=['\n## 如何继续\n',
            '打开 index.html；先看纠错，再看“预测未变且正确”，也可筛“五项完整符合正同类/负异类”。若严格条件无结果，应保留这一事实，不替换真实前五项。图1若要展示改正预测，必须从纠错通道选择；图2可展示正确预测的分解，但须如实写明 Frozen 也正确。\n',
            'panels 为只含图片/数值的 Top-5 组合；waterfalls 为图4同色系的纯数值五项图及闭合完整版；figure1_previews 为比较预览。纯图中不含 P编号或排名字样。\n',
            '每一项原图、区域、轮廓、来源类别、响应、有效权重差见 all_topk_provenance.csv；整体排名见 all_cases_ranking.csv；图像原位置及SHA-256见 references/*/region_provenance.json。\n']
    exclusions=read_json(out/'excluded_incomplete_traces.json')
    lines+=['\n## 已知边界\n',
            '本次“全部处理”指上述保存模型的全部测试预测及全部128个形状坐标的训练代表，未重新生成任何分割，也未为现有模型替换代表。CUB共135个跨bank不同代表，Derm7pt共129个。\n',
            '原Derm7pt目录另有3个不完整旧trace缺少全128项CSV，已单列 excluded_incomplete_traces.json，没有用前五项反推其余贡献；这3张测试图仍包含在三个配对模型的完整测试扫描中。\n',
            '区域内容来自图像复核，不是专家临床标注；模糊或归属不清的区域不计为本体。照片标签相同不等于该几何簇只表示这个类别。\n',
            '现有参考库中许多代表为背景、气泡或测量标尺，这说明当前聚类与区域选择未限定目标本体。筛选展示不能消除这一来源问题，也不能用改名称把它们变成病灶或鸟体。\n']
    (out/'筛选说明.md').write_text('\n'.join(lines),encoding='utf-8')


def recommendations(cases,out,topk):
    choices=[
        ('CUB-R42-1425','优先用于图1的纠错比较，也可沿用作图2',
         '从 Groove-billed Ani 改正为 White-necked Raven。前五项全部为鸟体：头喙、全鸟、前躯、全鸟、尾羽。但来源均为其他鸟种，适合讲跨类别共享形状参考，不能讲“同种支持、异种排斥”。'),
        ('CUB-R42-0199','同鸟种部件正贡献的纠错候选',
         'Belted Kingfisher 的喙部参考同属 Belted Kingfisher，贡献为正；但其他四项包含树叶、枝条和全幅视野，因此本体参考仅一项。'),
        ('CUB-R44-0667','优先查看：更接近所希望的正负来源关系',
         'Myrtle Warbler 测试图与正项全鸟参考属于同一鸟种；三个负项来自其他鸟种的鸟体，另一负项为背景。两模型预测均正确，残差总效应降低该类间隔，不能用于声称“改正了预测”。'),
        ('CUB-R43-1508','同鸟种喙部更清楚的图2备选',
         'Cape Glossy Starling 喙部正贡献；另有两个异类鸟体负项和两个背景负项。两模型预测均正确，五项只有三项为鸟体。'),
        ('CUB-R44-0600','同鸟种头部与前躯的备选',
         'Tennessee Warbler 的头部与前躯是同鸟种正项，两个异类鸟体是负项，但另外两个正项为背景。属于预测未变的案例。'),
        ('Derm7pt-R43-0167','可查看，但不作为理想医学形态展示首选',
         '从 benign 改正为 malignant。有同为 malignant 来源的病灶正项，但也有 malignant 病灶负项，另三项为标尺、气泡和含病灶的整幅视野；测试病例还跨现有分区。'),
        ('Derm7pt-R43-0291','病例未跨分区的医学备选',
         '两模型均正确预测 malignant，Top-5 有一项同诊断大类的完整病灶正贡献，其余项仍为标尺、气泡或皮肤，不能作为纯病灶机制示例。'),
    ]
    byid={c['candidate_id']:c for c in cases}
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>图1、图2 Top-5 候选总览</title>',
        '<style>body{font-family:Microsoft YaHei,sans-serif;color:#263541;max-width:1500px;margin:26px auto;padding:0 20px;line-height:1.8}h1{font-size:28px}h2{font-size:21px}.note{padding:16px;background:#eef3f7}section{border-top:2px solid #cdd6dd;margin-top:28px;padding-top:12px}img{max-width:100%;height:auto}a{color:#0072b2}.tag{display:inline-block;background:#edf4f5;padding:3px 10px;margin:4px 8px 4px 0}</style>',
        f'<h1>图1、图2：Top-{topk} 候选总览</h1>',
        f'<p class="note">共扫描{len(cases):,}个运行—测试图像组合、复核{len(read_json(out/"references.json"))}个不同训练参考。<b>没有找到五项同时完整符合“正项同类、负项异类、全为目标本体”的案例。</b>下列展示各自最接近的候选与实际不足。现有图1/图2目录保持完整，选定后再替换。</p>',
        '<p><a href="index.html">打开全部案例筛选</a> · <a href="matching_case_summary_zh.csv">中文候选汇总表</a> · <a href="reference_index.html">全部训练参考</a> · <a href="筛选说明.md">范围、复核口径与来源说明</a></p>']
    for cid,title,note in choices:
        c=byid[cid]
        parts.extend([f'<section><h2>{cid}：{title}</h2><p>{note}</p>',
            f'<p><span class="tag">本体 {c["body_count"]}/5</span><span class="tag">正同类本体 {c["positive_sameclass_body_count"]}</span><span class="tag">负异类本体 {c["negative_otherclass_body_count"]}</span><span class="tag">绝对贡献覆盖 {100*c["coverage"]:.1f}%</span></p>',
            f'<p>测试原图：{html.escape(c["sample_key"])}。<a href="index.html#{cid}">查看五项原图名称、来源类别、局部与轮廓</a></p>'])
        if c.get('panel'):
            parts.extend([f'<a href="{c["panel"]}"><img src="{c["panel"]}"></a>',
                f'<p><a href="{c["figure1_preview"]}">图1比较预览</a> · <a href="waterfalls/{cid}_values.png">五项纯瀑布图PNG</a> · <a href="waterfalls/{cid}_values.svg">SVG</a> · <a href="waterfalls/{cid}_full.png">完整分解</a> · <a href="cases/{cid}/shape_coordinates.csv">全部128项</a></p>'])
        parts.append('</section>')
    parts.append('</html>')
    (out/'推荐入口.html').write_text('\n'.join(parts),encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--topk',type=int,default=5)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume-prepare',action='store_true',help='Resume only a marked incomplete preparation directory')
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--refresh-gallery',action='store_true',help='Regenerate HTML/report only from an already finalized scan')
    parser.add_argument('--render-limit',type=int,default=6,help='Per dataset/status shortlist, distinct query images')
    parser.add_argument('--case',action='append',default=[],help='Additionally render an exact candidate ID')
    args=parser.parse_args()
    if not 1<=args.topk<=5:parser.error('topk must be in [1,5], matching the available original trace evidence')
    if not args.prepare and not args.finalize and not args.refresh_gallery:
        parser.error('Choose --prepare, --finalize, or --refresh-gallery')
    if args.prepare:prepare(args.output.resolve(),args.topk,args.resume_prepare)
    if args.finalize:finalize(args.output.resolve(),args.render_limit,args.case)
    if args.refresh_gallery:
        out=args.output.resolve()
        manifest=read_json(out/'manifest.json')
        if manifest['status']!='complete':raise ValueError('Finalize the scan first')
        cases=read_json(out/'candidates.json')
        refs={r['reference_id']:r for r in read_json(out/'reviewed_references.json')}
        stats=manifest['statistics'];topk=manifest['topk']
        gallery(cases,refs,stats,out,topk)
        write_report(cases,refs,stats,out,topk)
        recommendations(cases,out,topk)


if __name__=='__main__':
    main()
