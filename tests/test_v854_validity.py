"""V8.5.4: validity gates and defensible report construction."""
import json
import tempfile
import unittest
from pathlib import Path

from analysis.pipeline import build_fixture_analysis, export_report, inspect_report
from analysis.stat_contracts import AnalysisReport, PedagogicalPairedFixture
from analysis.validity import assess_paired_fixture
from test_v852_descriptive import pilot_aggregate
from analysis.validity import assess_verified_pilot

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "v85_paired_fixture.json"


def base_doc():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class ValidityTests(unittest.TestCase):
    def test_fixture_is_never_scientific_evidence(self):
        f = PedagogicalPairedFixture.model_validate(base_doc())
        report = build_fixture_analysis(f)
        self.assertTrue(report.validity.ready_for_pedagogical_inference)
        self.assertFalse(report.validity.ready_for_scientific_inference)
        self.assertFalse(report.scientific_evidence)
        self.assertIsNotNone(report.inference)

    def test_small_fixture_blocks_inference(self):
        doc = base_doc()
        doc["pairs"] = doc["pairs"][:6]
        report = build_fixture_analysis(PedagogicalPairedFixture.model_validate(doc))
        self.assertFalse(report.validity.ready_for_pedagogical_inference)
        self.assertIsNone(report.inference)
        self.assertTrue(report.validity.blockers)

    def test_duplicate_pair_is_rejected(self):
        doc = base_doc()
        doc["pairs"][1]["pair_id"] = doc["pairs"][0]["pair_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            PedagogicalPairedFixture.model_validate(doc)

    def test_no_fake_science_flag(self):
        report = build_fixture_analysis(PedagogicalPairedFixture.model_validate(base_doc()))
        document = report.model_dump(mode="json")
        document["scientific_evidence"] = True
        with self.assertRaises(ValueError):
            AnalysisReport.model_validate(document)

    def test_checksum_source_binding(self):
        report = build_fixture_analysis(PedagogicalPairedFixture.model_validate(base_doc()))
        document = report.model_dump(mode="json")
        document["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source digest"):
            AnalysisReport.model_validate(document)

    def test_pilot_never_inference_ready(self):
        dataset, quality = pilot_aggregate()
        report = assess_verified_pilot(dataset, quality)
        self.assertFalse(report.ready_for_pedagogical_inference)
        self.assertFalse(report.ready_for_scientific_inference)
        self.assertTrue(report.blockers)

    def test_report_export_idempotent_and_inspection(self):
        report = build_fixture_analysis(PedagogicalPairedFixture.model_validate(base_doc()))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(export_report(report, tmp), export_report(report, tmp))
            self.assertEqual(inspect_report(tmp), report)
            path = Path(tmp) / "analysis_report.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["descriptive"]["groups"]["baseline"]["mean"] = -999
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                inspect_report(tmp)
            with self.assertRaises(FileExistsError):
                export_report(report, tmp)

    def test_cannot_forge_inference_on_pilot(self):
        source = build_fixture_analysis(PedagogicalPairedFixture.model_validate(base_doc()))
        doc = source.model_dump(mode="json")
        doc["source_kind"] = "verified_pilot_aggregates"
        with self.assertRaises(ValueError):
            AnalysisReport.model_validate(doc)


if __name__ == "__main__":
    unittest.main()
