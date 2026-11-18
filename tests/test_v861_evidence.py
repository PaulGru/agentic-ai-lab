"""Trust boundary: rebuild sources, retain provenance, never mutate source jobs."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from analysis.pipeline import build_fixture_analysis
from analysis.stat_contracts import AnalysisReport, PedagogicalPairedFixture
from execution.backends.local import LocalDockerBackend
from execution.job_manager import DurableJobManager
from research.critique.contracts import EvidenceBundle, ReviewInput
from research.critique.evidence import (
    build_evidence, load_input, read_document, resolve_ref, validate_context, verify_evidence,
)
from tests.test_v851_ingestion import SimulatedTransport, sample_protocol
from tests.v86_helpers import design_context, fixture_document, sample_bundle


class EvidenceTests(unittest.TestCase):
    def test_fixture_recomputed_and_context_explicitly_unlinked(self):
        bundle = sample_bundle()
        self.assertEqual(bundle.analysis, build_fixture_analysis(
            PedagogicalPairedFixture.model_validate(fixture_document())))
        self.assertEqual(bundle.design_execution_link, "unavailable")
        self.assertFalse(bundle.scientific_evidence)
        self.assertEqual(resolve_ref(bundle, "/analysis/descriptive/groups/paired_difference/n"), 10)
        verify_evidence(bundle)

    def test_self_consistent_but_forged_analysis_is_rejected(self):
        bundle = sample_bundle(False)
        doc = bundle.analysis.model_dump(mode="json")
        doc["descriptive"]["groups"]["baseline"]["mean"] = 999.0
        forged = AnalysisReport.model_validate(doc)  # valid schema/digests alone do not prove calculations
        with self.assertRaisesRegex(ValueError, "recomputed"):
            build_evidence(bundle.source_kind, bundle.source_document, expected_analysis=forged)
        altered = EvidenceBundle.model_validate({**bundle.model_dump(mode="json"), "analysis": doc})
        with self.assertRaisesRegex(ValueError, "recomputed"):
            verify_evidence(altered)

    def test_no_science_or_fake_verified_link(self):
        for field, value in (("scientific_evidence", True), ("design_execution_link", "verified")):
            doc = sample_bundle().model_dump(mode="json")
            doc[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                EvidenceBundle.model_validate(doc)

    def test_parent_and_matrix_drift_rejected(self):
        for mutation in ("matrix", "design", "mission", "literature"):
            context = design_context()
            if mutation == "matrix":
                context.matrix_document["cells"] = []
            elif mutation == "design":
                context.design_document["design"]["title"] = "Changed parent title"
            elif mutation == "mission":
                context.mission_document["mission"]["title"] = "Changed mission title"
            else:
                context.literature_document["status"] = "rejected"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_context(context)

    def test_refs_cannot_escape_bundle(self):
        bundle = sample_bundle(False)
        for ref in ("../secrets", "/analysis/missing", "/source_document/pairs/-1",
                    "/source_document/pairs/01", "/analysis/~9", "/scientific_evidence/x"):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                resolve_ref(bundle, ref)

    def test_bounded_json_and_windows_bom_unicode_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "révision avec espaces.json"
            path.write_text(json.dumps(fixture_document()), encoding="utf-8-sig")
            request = ReviewInput(review_id="example", source_kind="pedagogical_paired_fixture", source_path=str(path))
            self.assertEqual(load_input(request), sample_bundle(False))
            for data in ('{"a":1,"a":2}', '{"a":NaN}', '[]', ' ' * 262_145):
                path.write_text(data, encoding="utf-8")
                with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                    read_document(path)
        for path in ("https://example.org/data.json", "\\\\server\\data.json", "//server/data.json"):
            with self.assertRaises(ValueError):
                read_document(path)

    def test_input_requires_explicit_ledger_and_llm_settings(self):
        for changes in ({"source_kind": "verified_pilot_aggregates"}, {"assist_critique": True},
                        {"max_cycles": 3}, {"max_cycles": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ReviewInput.model_validate({"review_id": "test", "source_kind": "pedagogical_paired_fixture",
                                            "source_path": "fixture.json", **changes})


class ReadOnlyLedgerTests(unittest.TestCase):
    def test_missing_database_does_not_create_parent_or_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing"
            with self.assertRaises(FileNotFoundError):
                DurableJobManager(path=root / "jobs.db", read_only=True)
            self.assertFalse(root.exists())

    def test_real_read_only_sql_and_api_with_simulation_metadata_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger avec espaces é.db"
            backend = LocalDockerBackend(execution_authorized=True, transport=SimulatedTransport())
            manager = DurableJobManager(backend, path)
            protocol = sample_protocol()
            for job in protocol.jobs():
                record = manager.submit(job)
                backend.wait(record.handle, timeout=3)
                self.assertEqual(manager.refresh(job.job_id).phase, "succeeded")
            before = path.read_bytes()
            reader = DurableJobManager(path=path, read_only=True)
            bundle = build_evidence("verified_pilot_aggregates", protocol.model_dump(mode="json"), manager=reader)
            self.assertTrue(all(x["simulated"] for x in bundle.dataset_document["observations"]))
            self.assertEqual([x["input_value_count"] for x in bundle.dataset_document["observations"]], [5, 5])
            verify_evidence(bundle, reader)
            with self.assertRaises(PermissionError):
                build_evidence("verified_pilot_aggregates", protocol.model_dump(mode="json"), manager=manager)
            for operation in (lambda: reader.submit(protocol.jobs()[0]),
                              lambda: reader.refresh(protocol.jobs()[0].job_id),
                              lambda: reader.cancel(protocol.jobs()[0].job_id),
                              lambda: reader._update(protocol.jobs()[0].job_id, "failed", "test")):
                with self.assertRaises(PermissionError):
                    operation()
            with reader._connect() as db:
                with self.assertRaises(sqlite3.OperationalError):
                    db.execute("DELETE FROM jobs")
            self.assertEqual(path.read_bytes(), before)
            with manager._connect() as db:
                db.execute("UPDATE jobs SET artifacts_sha256=?", ("0" * 64,))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_evidence(bundle, reader)


if __name__ == "__main__":
    unittest.main()
