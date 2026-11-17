"""Check experiment validity: frozen replay, train-only scaling and selection."""
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from utils.residual_controls import (
    TrainingConfig, apply_affine_map, apply_residual, fit_residual,
    fit_teacher_decoder, frozen_semantic_logits, prediction_metrics,
    random_projection, standardize_splits,
)


class ResidualControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scaling_and_projection_never_fit_validation_or_test(self):
        train = np.array([[0., 2.], [2., 2.]], dtype=np.float32)
        arrays = {"train": train, "val": np.array([[100., 2.]]),
                  "test": np.array([[-50., 2.]])}
        transformed, scaler = standardize_splits(arrays)
        np.testing.assert_allclose(scaler.mean_, [1., 2.])
        np.testing.assert_allclose(transformed["train"], [[-1., 0.], [1., 0.]])
        np.testing.assert_allclose(transformed["val"], [[99., 0.]])
        a, b = random_projection(16, 5, 42), random_projection(16, 5, 42)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a @ a.T, np.eye(5), atol=1e-6)
        self.assertFalse(np.array_equal(a, random_projection(16, 5, 43)))

    def test_frozen_checkpoint_replay_with_zero_norm_prototype(self):
        concepts = np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.float32)
        x = np.array([[1., 0.], [0., 2.], [-1., 0.]], dtype=np.float32)
        valid = np.exp(x - x.max(1, keepdims=True))
        expected_activation = np.column_stack((valid / valid.sum(1, keepdims=True), np.zeros(3)))
        scaler = StandardScaler().fit(expected_activation)
        weight = np.array([[1., 2., -3.], [-2., 1., .4]], dtype=np.float32)
        bias = np.array([.3, -.8], dtype=np.float32)
        checkpoint = {"concepts": concepts, "scaler": scaler,
                      "model_state": {"weight": torch.from_numpy(weight), "bias": torch.from_numpy(bias)}}
        actual = frozen_semantic_logits(x, checkpoint, batch_size=1)
        expected = scaler.transform(expected_activation).astype(np.float32) @ weight.T + bias
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_real_fitting_selects_validation_max_and_keeps_frozen_scores(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(100, 4)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        sem = np.zeros((100, 2), dtype=np.float32)
        before = sem.copy()
        config = TrainingConfig(epochs=20, patience=5, batch_size=16, lr=.03,
                                l1=1e-4, eta_values=(.5, 1.))
        fitted = fit_residual(x[:70], sem[:70], y[:70], x[70:], sem[70:], y[70:],
                              config=config, seed=42)
        self.assertEqual(fitted["selected"]["val_correct"],
                         max(row["val_correct"] for row in fitted["candidates"]))
        for candidate in fitted["candidates"]:
            history = [r for r in fitted["history"] if r["eta"] == candidate["eta"]]
            first_max = max(history, key=lambda r: r["val_correct"])
            self.assertEqual(candidate["best_epoch"], first_max["epoch"])
        logits = apply_residual(x[70:], sem[70:], fitted, batch_size=3)
        metrics = prediction_metrics(logits, y[70:], sem[70:])
        self.assertGreater(metrics["accuracy_pct"], 85.)
        self.assertEqual(metrics["n_correct"] - int((sem[70:].argmax(1) == y[70:]).sum()),
                         metrics["n_corrected"] - metrics["n_harmed"])
        np.testing.assert_array_equal(sem, before)

    def test_teacher_decoder_can_recover_a_known_affine_map(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(100, 3)).astype(np.float32)
        target = (x @ np.array([[.5, .2], [0., -.7], [.3, .1]], dtype=np.float32)
                  + np.array([.2, -.1], dtype=np.float32))
        fitted = fit_teacher_decoder(x[:70], target[:70], x[70:], target[70:], seed=42,
                                      epochs=80, patience=15, batch_size=32, lr=.03)
        prediction = apply_affine_map({"val": x[70:]}, fitted["state"])["val"]
        self.assertLess(np.mean((prediction - target[70:])**2), 1e-3)
        self.assertEqual(fitted["best_epoch"],
                         min(fitted["history"], key=lambda r: r["val_loss"])["epoch"])

    def test_existing_output_is_rejected_before_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "experiments.controls.run_residual_controls", "--output-root", directory,
                                     "--dry-run"], cwd=root, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Preserve existing results", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
