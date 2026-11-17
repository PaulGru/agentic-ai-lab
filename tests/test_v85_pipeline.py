"""End-to-end V8.5 analysis FROM an actual V8.4.6-shaped durable ledger.

Transport is simulated in tests, not a Docker daemon. We assert the report
honestly identifies observations as aggregate engineering-only data.
"""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from analysis.pipeline import build_verified_pilot_analysis, export_report, inspect_report
from execution.backends.local import LocalDockerBackend
from execution.campaign import PilotProtocol
from execution.job_manager import DurableJobManager
from test_v851_ingestion import SimulatedTransport

PROTOCOL = Path(__file__).resolve().parents[1] / "examples" / "v846_protocol.json"


class VerifiedLedgerPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = PilotProtocol.model_validate_json(PROTOCOL.read_text(encoding="utf-8"))
        self.backend = LocalDockerBackend(execution_authorized=True, transport=SimulatedTransport())
        self.manager = DurableJobManager(self.backend, self.root / "jobs.db")

    def complete(self):
        for job in self.protocol.jobs():
            rec = self.manager.submit(job)
            self.assertEqual(self.backend.wait(rec.handle, timeout=3).status.value, "succeeded")
            self.assertEqual(self.manager.refresh(job.job_id).phase, "succeeded")

    def test_verified_pipeline_requires_completion(self):
        with self.assertRaises(RuntimeError):
            build_verified_pilot_analysis(self.protocol, self.manager)

    def test_reconstructs_safe_descriptives_from_ledger(self):
        self.complete()
        readonly_manager = DurableJobManager(LocalDockerBackend(execution_authorized=False,
                                           transport=SimulatedTransport()), self.manager.path)
        report = build_verified_pilot_analysis(self.protocol, readonly_manager)
        self.assertEqual(report.source_kind, "verified_pilot_aggregates")
        self.assertEqual(report.descriptive.mean_difference_variant_minus_baseline, 3.0)
        self.assertEqual(report.descriptive.groups["baseline"].mean, 3.0)
        self.assertEqual(report.descriptive.groups["variant"].mean, 6.0)
        self.assertEqual(report.descriptive.groups["baseline"].n, 1)
        self.assertIsNone(report.descriptive.groups["baseline"].sample_std_dev)
        self.assertIsNone(report.inference)
        self.assertFalse(report.scientific_evidence)
        self.assertFalse(report.validity.ready_for_pedagogical_inference)
        self.assertEqual(inspect_report(self._export(report)), report)

    def _export(self, report):
        out = self.root / "export"
        export_report(report, out)
        return out

    def test_corrupt_db_never_becomes_analysis(self):
        self.complete()
        with closing(sqlite3.connect(self.manager.path)) as db:
            with db:
                db.execute("UPDATE jobs SET artifacts_sha256=? WHERE job_id=?",
                           ("0" * 64, self.protocol.jobs()[0].job_id))
        with self.assertRaises(ValueError):
            build_verified_pilot_analysis(self.protocol, self.manager)


if __name__ == "__main__":
    unittest.main()
