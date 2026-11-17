import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from cancha_cub import _preprocess_mask, _extract_contour_raw, _extract_geometric_features, get_shape_csv
from audit_cub_shape_assets import audit_bank
from data.Segments.scripts.utilities.segment import (
    asset_sha256, create_crops_masks, load_paired_mask_records,
)
from main_segment_1 import load_data, segmentation_paths
from utils.dcbm_1 import ResidualTrainer
from utils.sdr_cbm_visualize import SDRCBMVisualizer


class PairedShapeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank = self.root / 'shape'
        self.image = np.full((90, 100, 3), (10, 40, 90), np.uint8)
        self.mask = np.zeros((90, 100), bool)
        self.mask[15:65, 20:80] = True

    def write_pair(self, name='bird', mask_only=False):
        return create_crops_masks(
            [{'bbox': [10, 10, 80, 60], 'segmentation': self.mask}],
            self.image, self.bank, '001.Bird', name, mask_only=mask_only,
            source_metadata={'path': 'original.jpg', 'size_wh': [100, 90]},
        )

    def test_pair_has_real_bbox_and_checksums(self):
        self.write_pair()
        records = load_paired_mask_records(self.bank / 'masks')
        self.assertEqual(len(records), 1)
        mask_path, record = next(iter(records.items()))
        self.assertEqual(record['mask_sha256'], asset_sha256(mask_path))
        manifest = json.loads((mask_path.parent / 'bird_MASKS_DONE.json').read_text())
        self.assertEqual(manifest['regions'][0]['bbox_xywh'], [10, 10, 80, 60])
        crop = cv2.imread(str(self.bank / record['crop_relpath']))
        self.assertEqual(crop.shape[:2], (60, 80))
        np.testing.assert_allclose(crop[20, 20], [10, 40, 90], atol=3)
        with self.assertRaisesRegex(ValueError, 'binary mask'):
            _preprocess_mask(str(self.bank / record['crop_relpath']), strict_binary=True)

    def test_modified_same_size_mask_is_rejected(self):
        self.write_pair()
        mask = next((self.bank / 'masks').rglob('*.png'))
        changed = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        changed[20, 20] = 0
        cv2.imwrite(str(mask), changed)
        with self.assertRaisesRegex(ValueError, 'changed after segmentation'):
            load_paired_mask_records(self.bank / 'masks')

    def test_mask_only_and_mixed_versions_are_rejected(self):
        self.write_pair()
        mask = next((self.bank / 'masks').rglob('*.png'))
        cv2.imwrite(str(mask.with_suffix('.jpg')), np.zeros((60, 80), np.uint8))
        with self.assertRaisesRegex(ValueError, 'invalid paired outputs'):
            load_paired_mask_records(self.bank / 'masks')
        mask.with_suffix('.jpg').unlink()
        self.write_pair(mask_only=True)
        with self.assertRaisesRegex(ValueError, 'invalid paired outputs'):
            load_paired_mask_records(self.bank / 'masks')

    def test_strict_csv_and_teacher_retain_exact_source(self):
        for i in range(4):
            self.mask[:] = False
            self.mask[15:55 + i, 20:60 + i * 4] = True
            self.write_pair(f'bird{i}')
        csv_path, dim, _ = get_shape_csv(
            str(self.bank / 'masks'), str(self.bank / 'shape_csv/cub_fd32_features.csv'),
            'geo', strict_assets=True)
        rt = ResidualTrainer.__new__(ResidualTrainer)
        rt.cbm = SimpleNamespace(
            data_train_raw={f'001.Bird/bird{i}': None for i in range(4)},
            data_val_raw={}, data_test_raw={})
        rt.setup_shape_teacher(csv_path, K_shape=2, fd_dim=dim)
        for meta in rt.shape_proto_metadata.values():
            self.assertTrue((self.bank / meta['mask_relpath']).is_file())
            self.assertEqual(asset_sha256(self.bank / meta['mask_relpath']), meta['mask_sha256'])
        with open(csv_path, newline='') as handle:
            reader = csv.DictReader(handle)
            rows, fields = list(reader), reader.fieldnames
        rows[0]['source_image'] = '001.Bird/wrong'
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'source identity mismatch'):
            get_shape_csv(str(self.bank / 'masks'),
                          str(self.bank / 'shape_csv/cub_fd32_features.csv'), 'geo', strict_assets=True)

    def test_separate_semantic_bank_and_unique_input_resolution(self):
        self.write_pair()
        semantic_bank = self.root / 'semantic'
        semantic_crop = semantic_bank / 'crops/001.Bird/bird_OUT_CROP_0_X_.jpg'
        semantic_crop.parent.mkdir(parents=True)
        cv2.imwrite(str(semantic_crop), self.image)
        model = torch.nn.Linear(1, 1)
        rt = SimpleNamespace(cbm=SimpleNamespace(model=model, class_names=['bird']))
        viz = SDRCBMVisualizer(rt, str(self.root / 'trace'), device='cpu',
                              shape_asset_root=str(self.bank),
                              semantic_asset_root=str(semantic_bank),
                              image_root=str(self.root / 'split'))
        key = '001.Bird/bird_OUT_CROP_0_X_.jpg'
        self.assertEqual(viz._resolve_asset_path(key, 'crop', viz.semantic_asset_root), str(semantic_crop.resolve()))
        self.assertNotEqual(viz._resolve_asset_path(key, 'crop'), str(semantic_crop.resolve()))
        for category in ['001.Bird', '002.Bird']:
            path = self.root / 'split/test' / category / 'bird.jpg'
            path.parent.mkdir(parents=True)
            cv2.imwrite(str(path), self.image)
        self.assertIsNone(viz._resolve_input_path('test', 'bird.jpg'))
        self.assertIsNotNone(viz._resolve_input_path('test', '001.Bird/bird.jpg'))
        viz._input_path_indexes.clear()
        (self.root / 'split/test/002.Bird/bird.jpg').unlink()
        self.assertIsNotNone(viz._resolve_input_path('test', 'bird.jpg'))

    def test_custom_segmentation_paths_and_cub_source_root(self):
        source = self.root / 'split/train/001.Bird/bird.jpg'
        source.parent.mkdir(parents=True)
        cv2.imwrite(str(source), self.image)
        paths, dataset = load_data('CUB_200_2011', 'SAM2', None, False,
                                   image_root=str(self.root / 'split'), output_dir=str(self.bank))
        self.assertEqual(dataset, 'CUB_200_2011')
        self.assertEqual(paths, [source.as_posix()])
        root, stats = segmentation_paths(dataset, output_dir=str(self.bank))
        self.assertEqual(Path(root), self.bank)
        self.assertEqual(Path(stats), self.bank / 'stats.torch')

    def test_audit_checks_complete_source_coverage(self):
        source = self.root / 'split/train/001.Bird/bird.jpg'
        source.parent.mkdir(parents=True)
        cv2.imwrite(str(source), self.image)
        create_crops_masks(
            [{'bbox': [10, 10, 80, 60], 'segmentation': self.mask}],
            self.image, self.bank, '001.Bird', 'bird',
            source_metadata={'sha256': asset_sha256(source)})
        report = audit_bank(self.bank, self.root / 'split')
        self.assertTrue(report['asset_provenance_passed'])
        self.assertEqual(report['valid_geo11_regions'], 1)
        cv2.imwrite(str(source.with_name('unprocessed.jpg')), self.image)
        with self.assertRaisesRegex(ValueError, 'Incomplete image coverage'):
            audit_bank(self.bank, self.root / 'split')

    def test_export_checks_recorded_asset_hash(self):
        self.write_pair()
        records = load_paired_mask_records(self.bank / 'masks')
        path, source = next(iter(records.items()))

        def geometry(mask_path):
            clean = _preprocess_mask(mask_path, strict_binary=True)
            contour = _extract_contour_raw(clean)
            return clean.shape, contour, _extract_geometric_features(contour)

        meta = dict(source, representative_key='001.Bird/bird_OUT_CROP_0_X_.jpg',
                    representative_role='exact_selected_member',
                    raw_descriptor=geometry(str(path))[2].tolist())
        rt = SimpleNamespace(cbm=SimpleNamespace(model=torch.nn.Linear(1, 1), class_names=['bird']),
                             shape_proto_metadata={0: meta},
                             shape_proto_representative={0: meta['representative_key']})
        viz = SDRCBMVisualizer(rt, str(self.root / 'trace'), device='cpu',
                              shape_asset_root=str(self.bank), shape_geometry_fn=geometry)
        evidence = viz._prepare_shape_evidence(0, str(self.root / 'trace'))
        self.assertEqual(evidence['geometry_status'], 'verified_against_training_descriptor')
        cv2.imwrite(str(self.bank / source['crop_relpath']), np.zeros((60, 80, 3), np.uint8))
        viz._shape_evidence_cache.clear()
        evidence = viz._prepare_shape_evidence(0, str(self.root / 'trace'))
        self.assertEqual(evidence['geometry_status'], 'geometry_error')
        self.assertIn('training asset hash', evidence['geometry_error'])
        self.assertIsNone(evidence['pair_path'])


if __name__ == '__main__':
    unittest.main()
