import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.bias_only import evaluate_bias_only, export_bias_only_control, recover_scaled_bias


class BiasOnlyTests(unittest.TestCase):
    def test_multiclass_decisions_and_conditional_counts(self):
        sem = np.array([[2., 1., 0.], [0., 1., 2.], [0., 2., 1.], [3., 1., 0.]])
        before = sem.copy()
        metrics, logits = evaluate_bias_only(sem, np.array([1, 2, 0, 0]),
                                             np.array([0., 2., 0.]))
        self.assertEqual(logits.argmax(1).tolist(), [1, 1, 1, 0])
        result = metrics["bias_only"]
        self.assertEqual((result["n_corrected"], result["n_harmed"]), (1, 1))
        self.assertEqual((result["correction_pct"], result["harm_pct"]), (50., 50.))
        np.testing.assert_array_equal(sem, before)

    def test_common_shift_does_not_change_predictions(self):
        sem = np.array([[0., 1., 2.], [4., 2., 1.]])
        labels = np.array([1, 0])
        offsets, diagnostics = recover_scaled_bias(3, [(0, 1, .5), (1, 2, -.2), (0, 2, .3)])
        np.testing.assert_allclose(offsets, [0., .5, .3])
        a, _ = evaluate_bias_only(sem, labels, offsets)
        b, _ = evaluate_bias_only(sem, labels, offsets + 10.)
        self.assertEqual(a, b)
        self.assertEqual(diagnostics["classes_identified"], 3)

    def test_incomplete_or_inconsistent_exports_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "all class offsets"):
            recover_scaled_bias(3, [(0, 1, .2)])
        with self.assertRaisesRegex(ValueError, "disagree"):
            recover_scaled_bias(2, [(0, 1, .2), (0, 1, .4)])
        with self.assertRaisesRegex(ValueError, "every class"):
            evaluate_bias_only(np.zeros((2, 3)), np.array([0, 1]), np.zeros(2))

    def test_export_uses_restored_head_and_preserves_state(self):
        model = torch.nn.Linear(2, 3)
        class ToyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 3)

            def forward(self, features):
                return self.linear(features)

        head = ToyHead()
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
            model.bias.zero_()
            head.linear.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., -1.]]))
            head.linear.bias.copy_(torch.tensor([0., .8, -.2]))
        features = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        cbm = SimpleNamespace(dataset="toy", model=model)
        rt = SimpleNamespace(cbm=cbm, sb=head, cb=None, eta_s=.5, eta_c=0., device="cpu")
        for split in ("val", "test"):
            setattr(cbm, f"A_{split}_scaled", features)
            setattr(cbm, f"y_{split}_oh", np.array([[1., 0., 0.], [0., 1., 0.]]))
            setattr(cbm, f"data_{split}_raw", {"a": None, "b": None})
            setattr(rt, f"Sclip_{split}", features)
        parameters = [p.detach().clone() for m in (model, head) for p in m.parameters()]
        model.train()
        head.eval()
        with tempfile.TemporaryDirectory() as directory:
            export_bias_only_control(rt, directory, bs=1)
            with np.load(Path(directory) / "toy/decoded/test_logits.npz") as data:
                np.testing.assert_allclose(
                    data["bias_only_logits"] - data["semantic_logits"],
                    np.array([[0., .4, -.1], [0., .4, -.1]]), atol=1e-7)
            with self.assertRaises(FileExistsError):
                export_bias_only_control(rt, directory)
        self.assertTrue(model.training)
        self.assertFalse(head.training)
        for original, current in zip(parameters, (p for m in (model, head) for p in m.parameters())):
            torch.testing.assert_close(original, current, rtol=0., atol=0.)


if __name__ == "__main__":
    unittest.main()
