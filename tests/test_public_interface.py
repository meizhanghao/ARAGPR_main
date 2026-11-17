import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from aragpr.attribution import class_margin_attribution
from aragpr.paths import resolve_feature_file


class PublicInterfaceTests(unittest.TestCase):
    def test_margin_matches_direct_logits_with_explicit_contrast(self):
        rng = np.random.default_rng(91)
        base = rng.normal(size=(9, 4))
        u, w, b = rng.normal(size=(9, 7)), rng.normal(size=(4, 7)), rng.normal(size=4)
        ref, target = np.arange(9) % 4, (np.arange(9) + 2) % 4
        out = class_margin_attribution(base, u, w, b, .3,
            reference_classes=ref, target_classes=target)
        direct = base + .3 * (u @ w.T + b)
        expected = direct[np.arange(9), target] - direct[np.arange(9), ref]
        np.testing.assert_allclose(out["base_margin"] + out["coordinate_contributions"].sum(1)
                                   + out["bias_contribution"], expected)

    def test_same_class_has_zero_margin_and_zero_contributions(self):
        out = class_margin_attribution([[2., 0.]], [[1., 2.]], [[.1, .2], [0., .1]], [0., 0.])
        np.testing.assert_array_equal(out["coordinate_contributions"], [[0., 0.]])
        self.assertEqual(out["final_margin"][0], 0.)

    def test_rejects_noninteger_contrast(self):
        with self.assertRaises(ValueError):
            class_margin_attribution([[0., 1.]], [[1.]], [[1.], [0.]], [0., 0.],
                                     reference_classes=np.array([.5]))

    def test_windows_manifest_can_be_relocated_without_editing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = root / "derm7pt_split" / "features.torch"
            f.parent.mkdir()
            f.write_bytes(b"synthetic feature fixture")
            with patch.dict(os.environ, {"ARAGPR_DATA_ROOT": str(root)}):
                actual = resolve_feature_file(r"Z:\old_project\derm7pt_split\features.torch",
                    anchor=root / "cache" / "derm7pt", dataset="derm7pt")
            self.assertEqual(actual, f.resolve())


if __name__ == "__main__":
    unittest.main()
