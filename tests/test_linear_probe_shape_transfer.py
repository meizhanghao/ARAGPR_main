"""Check validation selection and complete attribution replay for a non-CBM base."""
import unittest

import numpy as np
import torch

from run_linear_probe_shape_transfer import fit_linear_probe, margin_replay
from utils.residual_controls import apply_affine_map


class LinearProbeTransferTests(unittest.TestCase):
    def test_probe_selects_the_first_validation_maximum(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(9)
        x = rng.normal(size=(80, 3)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        result = fit_linear_probe(x[:60], y[:60], x[60:], y[60:],
                                  epochs=30, patience=0, lr=.04, batch_size=16)
        expected = max(result['history'], key=lambda row: row['val_correct'])
        self.assertEqual(result['best_epoch'], expected['epoch'])
        score = apply_affine_map({'val': x[60:]}, result['state'])['val']
        self.assertEqual(int((score.argmax(1) == y[60:]).sum()), expected['val_correct'])

    def test_contributions_close_with_bias_and_unchanged_predictions(self):
        u = np.array([[1., -2.], [-1., 3.], [0., 0.]])
        base = np.array([[3., 1.], [2., 1.], [3., 1.]])
        state = {'weight': torch.tensor([[.3, -.4], [-.2, .6]]),
                 'bias': torch.tensor([.1, -.5])}
        eta = .75
        final = base + eta * (u @ state['weight'].numpy().T + state['bias'].numpy())
        values, error = margin_replay(u, base, final, state, eta)
        self.assertLess(error, 1e-7)
        np.testing.assert_allclose(values['final_margin'], values['reconstructed_margin'])
        self.assertTrue(np.any(values['base_class'] != values['final_class']))
        self.assertTrue(np.any(values['base_class'] == values['final_class']))


if __name__ == '__main__':
    unittest.main()
