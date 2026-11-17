"""Scientific invariants for the multiclass pairing-control summaries."""
import unittest

import numpy as np

from experiments.controls.run_cub_pairing_controls import (bootstrap_contrast,
                                      stratified_bootstrap_indices)
from experiments.controls.run_derm7pt_mechanism import training_permutation
from utils.residual_controls import prediction_metrics


class CUBPairingInvariants(unittest.TestCase):
    def test_resampling_preserves_rare_and_singleton_classes(self):
        # Unequal, non-contiguous labels expose accidental global resampling.
        labels = np.array([17, 2, 17, 99, 2, 17, 17])
        indices = stratified_bootstrap_indices(labels, replicates=300, seed=7)
        for row in indices:
            unique, counts = np.unique(labels[row], return_counts=True)
            np.testing.assert_array_equal(unique, [2, 17, 99])
            np.testing.assert_array_equal(counts, [2, 4, 1])
            self.assertEqual(np.sum(row == 3), 1)
        np.testing.assert_array_equal(
            indices, stratified_bootstrap_indices(labels, 300, 7))
        self.assertFalse(np.array_equal(
            indices, stratified_bootstrap_indices(labels, 300, 8)))

    def test_pairwise_interval_zero_and_swapping_symmetry(self):
        indices = stratified_bootstrap_indices([0, 0, 1, 1, 1, 2], 500, 11)
        zero = bootstrap_contrast(np.zeros(6), indices)
        self.assertEqual(zero['difference_pp'], 0)
        self.assertEqual(zero['image_stratified_bootstrap_95_low_pp'], 0)
        self.assertEqual(zero['image_stratified_bootstrap_95_high_pp'], 0)
        difference = np.array([1., -1., 0., 1., 0., -1.])
        forward = bootstrap_contrast(difference, indices)
        reverse = bootstrap_contrast(-difference, indices)
        self.assertAlmostEqual(forward['difference_pp'], -reverse['difference_pp'])
        self.assertAlmostEqual(forward['image_stratified_bootstrap_95_low_pp'],
                               -reverse['image_stratified_bootstrap_95_high_pp'])
        self.assertAlmostEqual(forward['image_stratified_bootstrap_95_high_pp'],
                               -reverse['image_stratified_bootstrap_95_low_pp'])

    def test_multiclass_shuffle_preserves_complete_teacher_population(self):
        labels = np.repeat(np.arange(200), 4)
        targets = np.arange(800 * 128).reshape(800, 128)
        for mode in ('paired', 'shuffled', 'within_class_shuffled'):
            permutation = training_permutation(labels, mode, 42)
            np.testing.assert_array_equal(np.sort(permutation), np.arange(800))
            # Whole rows, including all 128 coordinates, must remain intact.
            np.testing.assert_array_equal(
                targets[permutation][np.argsort(permutation)], targets)
            if mode == 'within_class_shuffled':
                np.testing.assert_array_equal(labels[permutation], labels)
            if mode == 'shuffled':
                self.assertGreater(np.sum(labels[permutation] != labels), 700)

    def test_correction_harm_distinguishes_wrong_to_wrong_changes(self):
        labels = np.array([0, 1, 2, 3, 0, 1])
        baseline = np.array([0, 0, 1, 3, 0, 2])
        updated = np.array([0, 1, 3, 2, 1, 1])
        metrics = prediction_metrics(np.eye(4)[updated], labels,
                                     np.eye(4)[baseline])
        self.assertEqual(metrics['n_corrected'], 2)
        self.assertEqual(metrics['n_harmed'], 2)
        self.assertEqual(metrics['n_correct'], 3)
        self.assertEqual(metrics['delta_frozen_pp'], 0)
        self.assertEqual(np.sum(baseline != updated), 5)


if __name__ == '__main__':
    unittest.main()
