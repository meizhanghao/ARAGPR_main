import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from aragpr.training import _preprocess_mask, _extract_contour_raw, _extract_geometric_features
from utils.dcbm_1 import ResidualTrainer
from utils.sdr_cbm_visualize import SDRCBMVisualizer


def geometry(path):
    clean = _preprocess_mask(path)
    contour = _extract_contour_raw(clean)
    return clean.shape, contour, _extract_geometric_features(contour)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = torch.nn.Linear(3, 3)
        head = torch.nn.Linear(5, 3)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
            model.bias.zero_()
            head.weight.zero_()
            head.bias.zero_()
            head.weight[1, 0] = 2
            head.weight[2, 1] = 4
            head.weight[1, 4] = 3
        cbm = SimpleNamespace(
            model=model, dataset='synthetic', class_names=['base', 'new', 'third'],
            concept_names=['part_a', 'part_b', 'part_c'],
            A_test_scaled=np.tile([2, 1, 0], (5, 1)).astype(np.float32),
            y_test=np.array([1, 0, 0, 1, 2]),
            data_test_raw={f'class_{i}/sample.jpg': None for i in range(5)},
            semantic_proto_representative={i: f'train/part{i}_OUT_CROP_0_X_.jpg' for i in range(3)},
        )
        self.rt = SimpleNamespace(cbm=cbm, sb=head, eta_s=1.0,
                                  Sclip_test=np.eye(5, dtype=np.float32))
        self.viz = SDRCBMVisualizer(self.rt, str(self.root / 'out'), device='cpu',
                                   shape_asset_root=str(self.root / 'sam'),
                                   shape_geometry_fn=geometry)

    def test_all_categories_and_stable_ids(self):
        rows = self.viz._case_rows(self.viz._compute_split_state('test', 'decoded'))
        self.assertEqual([r['category'] for r in rows],
                         ['correction', 'harm', 'unchanged_correct', 'unchanged_error', 'other_flip'])
        self.assertEqual(len({r['sample_id'] for r in rows}), 5)
        self.assertEqual(rows[4]['sample_key'], 'class_4/sample.jpg')
        again = self.viz._case_rows(self.viz._compute_split_state('test', 'decoded'))
        self.assertEqual([r['sample_id'] for r in rows], [r['sample_id'] for r in again])

    def test_decoded_rel_is_not_weighted_twice(self):
        weights = np.array([.5, 2, 0, .3, 1.2], dtype=np.float32)
        self.rt.Sclip_test_rel = self.rt.Sclip_test * weights
        self.rt.shape_reliability = weights
        self.rt.eta_s = .75
        state = self.viz._compute_split_state('test', 'decoded_rel')
        with torch.no_grad():
            expected = .75 * self.rt.sb(torch.from_numpy(self.rt.Sclip_test_rel)).numpy()
        np.testing.assert_allclose(state['z_shape'], expected)

    def test_bad_sample_mapping_fails_instead_of_truncating_ids(self):
        self.rt.cbm.data_test_raw.pop('class_4/sample.jpg')
        with self.assertRaisesRegex(ValueError, 'Sample-key'):
            self.viz.export_case_study(cases_per_type=0)

    def test_exhaustive_manifests_without_rendering(self):
        result = self.viz.export_case_study(cases_per_type=0)
        root = Path(result['root'])
        self.assertEqual(len(read_csv(root / 'all_cases.csv')), 5)
        for category, index in [('correction', '0'), ('harm', '1'), ('other_flip', '4')]:
            self.assertEqual(read_csv(root / f'{category}_cases.csv')[0]['index'], index)
        with (root / 'trace_run.json').open() as f:
            run = json.load(f)
        self.assertTrue(run['export_complete'])
        self.assertEqual(run['accuracy_delta'], run['final_accuracy'] - run['baseline_accuracy'])

    def test_empty_category_has_header(self):
        with torch.no_grad():
            self.rt.sb.weight.zero_()
        root = Path(self.viz.export_case_study(cases_per_type=0)['root'])
        with (root / 'harm_cases.csv').open(encoding='utf-8-sig') as f:
            self.assertIn('sample_key', f.readline())
        self.assertEqual(read_csv(root / 'harm_cases.csv'), [])

    def test_export_all_and_exact_selection(self):
        result = self.viz.export_case_study(cases_per_type=-1, render_plots=False)
        self.assertEqual(len(result['selected']), 2)
        for case in result['selected']:
            case_dir = Path(case['case_dir'])
            self.assertEqual(len(read_csv(case_dir / 'shape_coordinates.csv')), 5)
            self.assertLess(case['waterfall_closure_error'], 5e-5)
            steps = read_csv(case_dir / 'waterfall.csv')
            self.assertEqual(sum(r['component_type'] == 'shape_coordinate' for r in steps), 3)
        selected = self.viz.export_case_study(indices=[1, 1], render_plots=False)['selected']
        self.assertEqual([r['index'] for r in selected], [1])
        with self.assertRaisesRegex(ValueError, 'not correction/harm'):
            self.viz.export_case_study(indices=[4])

    def test_all_and_explicit_selection_record_full_category_median(self):
        self.rt.cbm.y_test[4] = 1
        for options in ({'cases_per_type': -1}, {'indices': [0]}):
            root = Path(self.viz.export_case_study(render_plots=False, **options)['root'])
            rows = read_csv(root / 'selected_cases.csv')
            for row in rows:
                expected = 2.5 if row['category'] == 'correction' else 4.0
                self.assertEqual(float(row['category_shape_margin_median']), expected)

    def test_summary_closure_includes_context_and_reports_small_error(self):
        ctx = torch.nn.Linear(1, 3, bias=False)
        with torch.no_grad():
            ctx.weight.copy_(torch.tensor([[0.0], [.3], [0.0]]))
        self.rt.cb = ctx
        self.rt.Actx_test = np.ones((5, 1), dtype=np.float32)
        self.rt.eta_c = 1.0
        state = self.viz._compute_split_state('test', 'decoded')
        row = self.viz._case_rows(state)[0]
        row.update(selection_rule='test', selection_rank=1)
        exp = self.viz._build_margin_explanation(state, row, topk=5)
        exp['display_topk'] = 3
        exp['final_margin_f_minus_b'] += 1e-5
        result = self.viz._export_trace_case(exp, str(self.root), 'decoded', False, render_plots=False)
        self.assertAlmostEqual(result['waterfall_closure_error'], 1e-5, delta=1e-6)
        rows = read_csv(Path(result['case_dir']) / 'waterfall.csv')
        self.assertEqual(rows[1]['component_type'], 'context_total')
        self.assertEqual(rows[-2]['component_type'], 'shape_bias')

    def test_waterfall_keeps_negative_top3_other_and_bias(self):
        exp = {'index': 0, 'semantic_margin_f_minus_b': -1.0,
               'context_margin_f_minus_b': 0.0, 'top_shape_abs_ids': [0, 1, 2, 3, 4],
               'shape_coordinate_margin': np.array([4, -2, 1, .5, -.25]),
               'shape_bias_margin_f_minus_b': .2, 'final_margin_f_minus_b': 2.45}
        rows = self.viz._waterfall_rows(exp, display_topk=3)
        self.assertEqual([r['delta'] for r in rows if r['component_type'] == 'shape_coordinate'], [4, -2, 1])
        self.assertEqual(next(r['delta'] for r in rows if r['component_type'] == 'shape_other'), .25)
        self.assertAlmostEqual(rows[-2]['cumulative_after'], 2.45)
        exp['final_margin_f_minus_b'] += .1
        with self.assertRaisesRegex(RuntimeError, 'does not close'):
            self.viz._waterfall_rows(exp, 3)

    def test_places_hierarchy_never_falls_back_to_other_class(self):
        source = self.root / 'sam/crops/indoor/shop/shared_OUT_CROP_0_X_.jpg'
        source.parent.mkdir(parents=True)
        Image.new('RGB', (8, 8)).save(source)
        self.assertIsNone(self.viz._resolve_asset_path('outdoor/shop/shared_OUT_CROP_0_X_.jpg', 'crop'))
        self.assertEqual(self.viz._resolve_asset_path('indoor/shop/shared_OUT_CROP_0_X_.jpg', 'crop'), str(source))

    def make_shape_asset(self):
        mask_path = self.root / 'sam/masks/train/member_OUT_MASK_0_X_.png'
        crop_path = self.root / 'sam/crops/train/member_OUT_CROP_0_X_.jpg'
        for path in (mask_path, crop_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((64, 80), dtype=np.uint8)
        cv2.ellipse(mask, (40, 32), (27, 17), 20, 0, 360, 255, -1)
        Image.fromarray(mask).save(mask_path)
        Image.new('RGB', (80, 64), (135, 70, 100)).save(crop_path)
        key = 'train/member_OUT_CROP_0_X_.jpg'
        self.rt.shape_proto_representative = {0: key}
        self.rt.shape_proto_nearest = {0: ['wrong/exemplar_OUT_CROP_0_X_.jpg']}
        self.rt.shape_proto_metadata = {0: {'representative_key': key,
            'representative_role': 'exact_selected_member',
            'raw_descriptor': geometry(str(mask_path))[2].tolist()}}
        return crop_path, mask_path

    def test_paired_geometry_uses_exact_selected_training_member(self):
        self.make_shape_asset()
        evidence = self.viz._prepare_shape_evidence(0, str(self.root / 'bundle'))
        self.assertEqual(evidence['geometry_status'], 'verified_against_training_descriptor')
        self.assertEqual(evidence['representative_key'], self.rt.shape_proto_representative[0])
        for field in ('pair_path', 'overlay_path', 'verified_contour_path', 'neutral_crop_path'):
            self.assertTrue(Path(evidence[field]).is_file())
        with Image.open(evidence['neutral_crop_path']) as im:
            self.assertEqual(im.getpixel((0, 0)), (224, 224, 224))

    def test_mismatched_descriptor_or_size_not_named(self):
        crop_path, _ = self.make_shape_asset()
        self.rt.shape_proto_metadata[0]['raw_descriptor'][0] += 1
        evidence = self.viz._prepare_shape_evidence(0, str(self.root / 'bundle'))
        self.assertEqual(evidence['geometry_status'], 'descriptor_mismatch')
        self.assertIsNone(evidence['pair_path'])
        self.rt.shape_proto_metadata[0]['raw_descriptor'][0] -= 1
        self.viz._shape_evidence_cache.clear()
        Image.new('RGB', (10, 10)).save(crop_path)
        evidence = self.viz._prepare_shape_evidence(0, str(self.root / 'bundle'))
        self.assertEqual(evidence['geometry_status'], 'crop_mask_size_mismatch')
        self.assertEqual(evidence['naming']['status'], 'skipped_unverified_geometry')

    def test_six_decimal_descriptors_pass_geometry_verification(self):
        self.make_shape_asset()
        meta = self.rt.shape_proto_metadata[0]
        meta['raw_descriptor'] = [float(f'{v:.6f}') for v in meta['raw_descriptor']]
        evidence = self.viz._prepare_shape_evidence(0, str(self.root / 'bundle'))
        self.assertEqual(evidence['geometry_status'], 'verified_against_training_descriptor')

    def test_shape_names_encode_rgb_not_geo11(self):
        self.make_shape_asset()
        vocab = self.root / 'vocab.txt'
        vocab.write_text('wing\nleaf\n', encoding='utf-8')
        self.viz.shape_vocab_path = str(vocab)
        class FakeClip:
            def encode_text(self, tokens):
                return torch.eye(2)
            def encode_image(self, tensor):
                self.tensor_shape = tuple(tensor.shape)
                return torch.tensor([[1., 0.]])
        model = FakeClip()
        runtime = (model, lambda im: torch.from_numpy(np.array(im)).permute(2, 0, 1), 'test')
        with patch.object(self.viz, '_clip_encoder', return_value=runtime), \
             patch('utils.sdr_cbm_visualize.clip',
                   SimpleNamespace(tokenize=lambda *args, **kwargs: torch.zeros(2, 1))):
            evidence = self.viz._prepare_shape_evidence(0, str(self.root / 'bundle'))
        self.assertEqual(model.tensor_shape, (1, 3, 64, 80))
        self.assertEqual(evidence['naming']['candidates'][0]['phrase'], 'wing')
        self.assertEqual(evidence['naming']['status'], 'candidate_unreviewed')

    def test_representative_metadata_preserves_clustering_values_and_ties(self):
        csv_path = self.root / 'teacher.csv'
        features = np.array([[1, 2], [1, 2], [6, 1], [7, 1]], dtype=np.float32)
        region_ids = ['003', '04', '0010', '1']
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_image', 'region_id', 'a', 'b'])
            for i, values in enumerate(features):
                writer.writerow([f'train/img{i}', region_ids[i], *values])
                for folder, tag, ext in (('crops', 'CROP', 'jpg'), ('masks', 'MASK', 'png')):
                    path = self.root / f'sam/{folder}/train/img{i}_OUT_{tag}_{region_ids[i]}_X_.{ext}'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (8, 8)).save(path)
        cbm = SimpleNamespace(data_train_raw={f'train/img{i}': None for i in range(4)},
                              data_val_raw={}, data_test_raw={})
        for method in ('median_nearest', 'mean_nearest', 'mean_center'):
            rt = ResidualTrainer.__new__(ResidualTrainer)
            rt.cbm = cbm
            rt.setup_shape_teacher(str(csv_path), K_shape=2, fd_dim=2,
                                   shape_prototype=method)
            transformed = rt.shape_feat_scaler.transform(features)
            normed = transformed / (np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-8)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(normed)
            expected = []
            for k in range(2):
                members = np.flatnonzero(km.labels_ == k)
                target = np.median(normed[members], axis=0) if method == 'median_nearest' else normed[members].mean(axis=0)
                chosen = members[np.argmax(normed[members] @ target)]
                vector = km.cluster_centers_[k] if method == 'mean_center' else normed[chosen]
                expected.append(vector / (np.linalg.norm(vector) + 1e-8))
                if method != 'mean_center':
                    key = f'train/img{chosen}_OUT_CROP_{region_ids[chosen]}_X_.jpg'
                    self.assertEqual(rt.shape_proto_representative[k], key)
                    np.testing.assert_array_equal(rt.shape_proto_metadata[k]['raw_descriptor'], features[chosen])
                    self.assertIsNotNone(self.viz._resolve_asset_path(key, 'crop'))
                    self.assertIsNotNone(self.viz._resolve_asset_path(key, 'mask'))
            np.testing.assert_allclose(rt.sp, expected, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
