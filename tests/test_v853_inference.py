"""Exact sign-flip test, bootstrap determinism and fail-closed behavior."""
import copy
import json
import statistics
import unittest
from pathlib import Path

from analysis.inference import infer_paired_fixture
from analysis.stat_contracts import PedagogicalPairedFixture

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "v85_paired_fixture.json"


def fixture_doc():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class InferenceTests(unittest.TestCase):
    def test_exact_enumeration_and_effect_size(self):
        fixture = PedagogicalPairedFixture.model_validate(fixture_doc())
        result = infer_paired_fixture(fixture)
        diffs = [p.variant - p.baseline for p in fixture.pairs]
        self.assertEqual(result.sign_flips_enumerated, 1024)
        self.assertEqual(result.n_pairs, 10)
        self.assertAlmostEqual(result.mean_difference, 1.9)
        self.assertAlmostEqual(result.paired_standardized_effect_dz,
                               statistics.fmean(diffs) / statistics.stdev(diffs))
        self.assertGreater(result.exact_two_sided_p, 0)
        self.assertLessEqual(result.exact_two_sided_p, 1)
        self.assertFalse(result.scientific_evidence)

    def test_exact_p_for_balanced_opposing_differences(self):
        doc = fixture_doc()
        for n, pair in enumerate(doc["pairs"]):
            pair["variant"] = pair["baseline"] + (1 if n % 2 == 0 else -1)
        result = infer_paired_fixture(PedagogicalPairedFixture.model_validate(doc))
        self.assertEqual(result.exact_two_sided_p, 1.0)
        self.assertEqual(result.mean_difference, 0.0)

    def test_deterministic_bootstrap_and_export(self):
        fixture = PedagogicalPairedFixture.model_validate(fixture_doc())
        a = infer_paired_fixture(fixture)
        b = infer_paired_fixture(fixture)
        self.assertEqual(a, b)
        self.assertLess(a.bootstrap_percentile_ci_95[0], a.bootstrap_percentile_ci_95[1])
        changed = infer_paired_fixture(fixture, bootstrap_seed=23)
        self.assertNotEqual(a.bootstrap_seed, changed.bootstrap_seed)

    def test_stale_validity_refused(self):
        from analysis.validity import assess_paired_fixture
        fixture = PedagogicalPairedFixture.model_validate(fixture_doc())
        altered = fixture.model_copy(update={"dataset_id": "another-dataset"})
        with self.assertRaisesRegex(ValueError, "Stale or forged"):
            infer_paired_fixture(altered, validity=assess_paired_fixture(fixture))

    def test_unvarying_offsets_block(self):
        doc = fixture_doc()
        for pair in doc["pairs"]:
            pair["variant"] = pair["baseline"] + 2
        with self.assertRaisesRegex(ValueError, "no measurable variation"):
            infer_paired_fixture(PedagogicalPairedFixture.model_validate(doc))

    def test_parameter_boundaries(self):
        fixture = PedagogicalPairedFixture.model_validate(fixture_doc())
        for opts in ({"bootstrap_resamples": 50}, {"bootstrap_seed": -1},
                     {"bootstrap_resamples": True}, {"bootstrap_seed": True}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                infer_paired_fixture(fixture, **opts)


if __name__ == "__main__":
    unittest.main()
