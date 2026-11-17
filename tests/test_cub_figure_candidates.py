import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

from tools.references.run_cub_figure_candidates import (
    PART_VOCAB, SourceLedger, bundle_asset, checked_decomposition, clip_screen,
    make_shortlist, rank_key, replay_logits, write_csv, write_json,
)


class CandidateSelectionTests(unittest.TestCase):
    def decomposition(self):
        summary = dict(eta_s=.5, semantic_margin_f_minus_b=-.2,
            context_margin_f_minus_b=0., shape_bias_margin_f_minus_b=.1,
            final_margin_f_minus_b=.35, dcbm_class=0, sdrcbm_class=1)
        rows = [dict(prototype_id=str(i), activation=str(a),
                    weight_difference_f_minus_b=str(.5*w),
                    signed_margin_contribution=str(.5*a*w))
                for i, (a, w) in enumerate([(1, .8), (1, -.3), (2, .1), (1, .2)])]
        logits = [dict(class_id="0", semantic_logit="1", final_logit="1"),
                  dict(class_id="1", semantic_logit=".8", final_logit="1.35")]
        return summary, rows, logits

    def test_real_absolute_ranking_and_closure_keep_a_negative_term(self):
        summary, rows, logits = self.decomposition()
        top, other, error = checked_decomposition(summary, rows, logits)
        self.assertEqual([r["prototype_id"] for r in top], ["0", "1", "2"])
        self.assertAlmostEqual(other, .1)
        self.assertLess(error, 1e-12)
        self.assertEqual(rows[0]["signed_margin_contribution"], "0.4")

    def test_corrupted_coordinates_and_wrong_full_vector_predictions_fail(self):
        summary, rows, logits = self.decomposition()
        rows[1]["signed_margin_contribution"] = ".3"
        with self.assertRaisesRegex(ValueError, "activation times weight"):
            checked_decomposition(summary, rows, logits)
        summary, rows, logits = self.decomposition()
        logits.append(dict(class_id="2", semantic_logit="0", final_logit="2"))
        with self.assertRaisesRegex(ValueError, "complete logits"):
            checked_decomposition(summary, rows, logits)

    def test_missing_coordinate_is_rejected(self):
        summary, rows, logits = self.decomposition()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            checked_decomposition(summary, rows[1:], logits)

    def test_scaled_saved_head_replay_includes_bias_and_eta(self):
        x = np.array([[1, 4], [3, 8], [5, 12]], dtype=np.float32)
        scaler = StandardScaler().fit(x)
        head = dict(selected=dict(eta=.25, state=dict(
            weight=np.array([[1, 2], [-2, 3]], dtype=np.float32),
            bias=np.array([.4, -.3], dtype=np.float32))))
        features, logits = replay_logits(x, scaler, head)
        expected = .25 * (scaler.transform(x) @ head["selected"]["state"]["weight"].T + [.4, -.3])
        np.testing.assert_allclose(logits, expected, atol=1e-7)
        self.assertFalse(np.array_equal(features, x))

    def test_linux_portable_asset_and_escape_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "case").mkdir()
            (root / "prototype_assets").mkdir()
            asset = root / "prototype_assets" / "mask.png"
            asset.write_bytes(b"fixture")
            self.assertEqual(bundle_asset(root / "case", "../prototype_assets/mask.png", root), asset.resolve())
            with self.assertRaisesRegex(ValueError, "escapes"):
                bundle_asset(root / "case", "../../outside.png", root)
            with self.assertRaisesRegex(ValueError, "relative"):
                bundle_asset(root / "case", "/home/server/mask.png", root)

    def test_presentation_ranking_prioritizes_corrections_and_clear_parts(self):
        def row(cid, category, parts, birds):
            return dict(candidate_id=cid, category=category, top3_usable_parts=parts,
                        top3_bird_references=birds, positive_part_abs_sum=.1, top3_coverage=.2)
        examples = [row("harm", "harm", 3, 3), row("background", "correction", 0, 0),
                    row("head", "correction", 1, 1), row("head-body", "correction", 2, 3)]
        self.assertEqual([r["candidate_id"] for r in sorted(examples, key=rank_key)],
                         ["head-body", "head", "background", "harm"])

    def test_integrity_ledger_detects_input_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.txt"
            path.write_text("original")
            ledger = SourceLedger()
            ledger.record(path)
            self.assertEqual(len(ledger.verify()), 1)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed during"):
                ledger.verify()

    def test_shortlist_keeps_part_variety_and_distinct_images(self):
        def case(cid, index, part, contribution):
            return dict(candidate_id=cid, index=index, category="correction",
                top3_usable_parts=1, top3_bird_references=1,
                positive_part_abs_sum=contribution, top3_coverage=.2,
                top=[dict(reference=dict(identity=part, usable_part=True))])
        examples = [case("head-1", 1, "head", .5), case("head-2", 2, "head", .4),
                    case("leg-1", 3, "leg", .1), case("duplicate-image", 1, "bill", .05)]
        self.assertEqual([r["candidate_id"] for r in make_shortlist(examples, 3)],
                         ["head-1", "leg-1", "head-2"])

    def test_cached_clip_reuse_checks_actual_reference_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cached, current, out = root / "cached", root / "current", root / "out"
            previous = cached / "references" / "ref1"
            previous.mkdir(parents=True)
            current.mkdir()
            out.mkdir()
            (previous / "neutral_crop.png").write_bytes(b"same-image")
            (current / "neutral_crop.png").write_bytes(b"same-image")
            write_csv(cached / "clip_reference_rankings.csv", [dict(reference_id="ref1",
                representative_key="bird/part.jpg", rank=i+1, candidate=candidate)
                for i, candidate in enumerate(PART_VOCAB)])
            write_json(cached / "clip_vocabulary.json", dict(phrases=PART_VOCAB))
            args = SimpleNamespace(clip_rankings=cached / "clip_reference_rankings.csv")
            refs = dict(ref1=dict(representative_key="bird/part.jpg", path=current))
            self.assertEqual(clip_screen(refs, args, out, SourceLedger())["ref1"], PART_VOCAB[0])
            (current / "neutral_crop.png").write_bytes(b"different-image")
            with self.assertRaisesRegex(ValueError, "different reference pixels"):
                clip_screen(refs, args, out, SourceLedger())


if __name__ == "__main__":
    unittest.main()
