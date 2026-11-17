"""V8.5.2 descriptive statistics must respect the observation unit."""
import json
import math
import unittest
from pathlib import Path

from analysis.contracts import AnalysisDataset, MetricObservation, QualityAssessment
from analysis.descriptive import describe_paired_fixture, describe_verified_pilot
from analysis.stat_contracts import PedagogicalPairedFixture

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "v85_paired_fixture.json"
HASH = "a" * 64


def example_fixture():
    return PedagogicalPairedFixture.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))


def pilot_aggregate():
    records = [MetricObservation(
        observation_id=f"pilot-{i}-mean", run_id=f"run{i}", job_id=f"pilot-{i}",
        job_sha256=HASH, artifacts_sha256=HASH, value=v,
        input_value_count=5, simulated=False,
    ) for i, v in enumerate((3.0, 6.0))]
    dataset = AnalysisDataset(campaign_id="pilot", protocol_sha256=HASH,
                              source_report_sha256=HASH, observations=records)
    quality = QualityAssessment(
        campaign_id="pilot", dataset_sha256=dataset.fingerprint(),
        observation_count=2, verified_job_count=2, simulated_observation_count=0,
        blockers=["Not independent experiments"],
    )
    return dataset, quality


class DescriptiveTests(unittest.TestCase):
    def test_paired_descriptive_known_values(self):
        output = describe_paired_fixture(example_fixture())
        self.assertEqual(output.n_independent_pairs, 10)
        self.assertEqual(output.groups["paired_difference"].n, 10)
        self.assertAlmostEqual(output.groups["paired_difference"].mean, 1.9)
        self.assertEqual(output.groups["paired_difference"].minimum, 0)
        self.assertEqual(output.groups["paired_difference"].maximum, 4)
        self.assertIsNotNone(output.groups["paired_difference"].sample_std_dev)
        self.assertEqual(output.mean_difference_variant_minus_baseline, 1.9)

    def test_pilot_does_not_invent_replicates(self):
        result = describe_verified_pilot(*pilot_aggregate())
        self.assertEqual(result.kind, "job_aggregates")
        self.assertIsNone(result.n_independent_pairs)
        self.assertEqual(result.mean_difference_variant_minus_baseline, 3)
        self.assertEqual([stat.n for stat in result.groups.values()], [1, 1])
        self.assertTrue(all(stat.sample_std_dev is None for stat in result.groups.values()))

    def test_quality_checksum_rejected(self):
        dataset, quality = pilot_aggregate()
        bad_quality = quality.model_copy(update={"dataset_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "SHA-256 differ"):
            describe_verified_pilot(dataset, bad_quality)

    def test_bad_input_pair_is_rejected(self):
        doc = example_fixture().model_dump(mode="json")
        doc["pairs"][2]["variant"] = float("nan")
        with self.assertRaises(ValueError):
            PedagogicalPairedFixture.model_validate(doc)


if __name__ == "__main__":
    unittest.main()
