"""Validate fidelity definitions against independent reference calculations."""
import unittest
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from run_decoder_baselines import fidelity_metrics


class DecoderBaselineMetricsTests(unittest.TestCase):
    def test_nonunit_test_variance_matches_reference_r2(self):
        y = np.array([[1., 2.], [3., 12.], [5., 22.], [7., 32.]])
        p = y + np.array([[.2, 1.], [.5, -3.], [-.1, 2.], [.4, 1.]])
        metrics, coordinates = fidelity_metrics(y, p)
        self.assertAlmostEqual(metrics["mse"], mean_squared_error(y, p), places=12)
        self.assertAlmostEqual(metrics["r2_uniform"], r2_score(y, p), places=12)
        self.assertNotAlmostEqual(metrics["r2_uniform"], 1 - metrics["mse"], places=3)
        expected = [np.corrcoef(y[:, i], p[:, i])[0, 1] for i in range(2)]
        np.testing.assert_allclose(coordinates["pearson"], expected, atol=1e-12)

    def test_train_mean_is_not_assumed_to_be_test_mean(self):
        train = np.array([[0., 1.], [2., 3.]])
        test = np.array([[8., 10.], [10., 12.], [12., 14.]])
        prediction = np.broadcast_to(train.mean(0), test.shape)
        metrics, _ = fidelity_metrics(test, prediction)
        self.assertLess(metrics["r2_uniform"], 0)
        self.assertIsNone(metrics["pearson_defined_mean"])
        self.assertEqual(metrics["pearson_undefined_coordinates"], 2)
        self.assertEqual(metrics["pearson_paper_zero_undefined"], 0)

    def test_constant_targets_use_explicit_finite_r2_convention(self):
        y = np.array([[1., 2.], [1., 4.], [1., 6.]])
        metrics, coordinates = fidelity_metrics(y, y.copy())
        np.testing.assert_array_equal(coordinates["r2"], [1., 1.])
        self.assertEqual(metrics["r2_uniform"], 1)
        self.assertEqual(metrics["pearson_defined_coordinates"], 1)
        with self.assertRaises(ValueError):
            fidelity_metrics(y[:1], y[:1])


if __name__ == "__main__":
    unittest.main()
