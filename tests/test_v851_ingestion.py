"""V8.5.1 contract tests; no Docker daemon, LLM API or Postgres required."""
from __future__ import annotations

import json
import sqlite3
import statistics
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from pydantic import ValidationError

from analysis.contracts import AnalysisDataset, MetricObservation, QualityAssessment, fingerprint, verify_export
from analysis.ingest import export_analysis, ingest_verified_pilot, inspect_export
from execution.backends.docker_transport import DockerOutcome
from execution.backends.local import LocalDockerBackend
from execution.campaign import PilotProtocol, build_report
from execution.job_manager import DurableJobManager

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "examples" / "v846_protocol.json"


class SimulatedTransport:
    simulated = True

    def run(self, job, code, cancel):
        vals = job.workload.parameters["values"]
        response = {
            "kind": "agentic-v842-artifacts", "sha256": job.fingerprint(),
            "files": {
                "metrics.json": {
                    "count": len(vals), "mean": statistics.fmean(vals),
                    "minimum": min(vals), "maximum": max(vals),
                },
                "run_metadata.json": {
                    "job_id": job.job_id, "job_sha256": job.fingerprint(),
                    "seed": job.seed, "workload": job.workload.name,
                    "execution": "docker_sandbox", "python_version": "3.12.0",
                },
            },
        }
        return DockerOutcome(status="success", exit_code=0, stdout=json.dumps(response),
                             stderr="", duration_seconds=0.01)


def sample_protocol():
    return PilotProtocol.model_validate_json(PROTOCOL_PATH.read_text(encoding="utf-8"))


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = sample_protocol()
        self.backend = LocalDockerBackend(execution_authorized=True, transport=SimulatedTransport())
        self.manager = DurableJobManager(self.backend, self.root / "jobs.db")

    def complete(self):
        for job in self.protocol.jobs():
            record = self.manager.submit(job)
            result = self.backend.wait(record.handle, timeout=3)
            self.assertEqual(result.status.value, "succeeded")
            self.assertEqual(self.manager.refresh(job.job_id).phase, "succeeded")

    def test_normalize_complete_verified_ledger(self):
        self.complete()
        dataset, quality = ingest_verified_pilot(self.protocol, self.manager)
        self.assertEqual(dataset.schema_version, "8.5.1")
        self.assertEqual(dataset.provenance, "verified_v846_sqlite_ledger")
        self.assertEqual([r.value for r in dataset.observations], [3.0, 6.0])
        self.assertEqual([r.input_value_count for r in dataset.observations], [5, 5])
        self.assertEqual(quality.observation_count, 2)
        self.assertEqual(quality.simulated_observation_count, 2)
        self.assertFalse(quality.ready_for_scientific_inference)
        self.assertFalse(dataset.scientific_evidence)
        self.assertIn("replication", " ".join(quality.blockers))
        self.assertEqual(dataset.source_report_sha256,
                         fingerprint(build_report(self.protocol, self.manager)))

    def test_refuse_no_completed_jobs(self):
        with self.assertRaises(RuntimeError):
            ingest_verified_pilot(self.protocol, self.manager)

    def test_refuse_partially_completed_campaign(self):
        job = self.protocol.jobs()[0]
        record = self.manager.submit(job)
        self.backend.wait(record.handle, timeout=3)
        self.manager.refresh(job.job_id)
        with self.assertRaises(RuntimeError):
            ingest_verified_pilot(self.protocol, self.manager)

    def test_refuse_corrupt_ledger_artifact_hash(self):
        self.complete()
        with closing(sqlite3.connect(self.manager.path)) as db:
            with db:
                db.execute("UPDATE jobs SET artifacts_sha256=? WHERE job_id=?",
                           ("0" * 64, self.protocol.jobs()[0].job_id))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            ingest_verified_pilot(self.protocol, self.manager)

    def test_refuse_protocol_drift_even_when_campaign_id_unchanged(self):
        self.complete()
        doc = self.protocol.model_dump(mode="json")
        doc["runs"][0]["values"][0] = 99
        changed = PilotProtocol.model_validate(doc)
        with self.assertRaises(ValueError):
            ingest_verified_pilot(changed, self.manager)

    def test_export_and_inspect_idempotent(self):
        self.complete()
        dataset, quality = ingest_verified_pilot(self.protocol, self.manager)
        out = self.root / "analysis_output"
        paths = export_analysis(dataset, quality, out)
        self.assertTrue(all(x.is_file() for x in paths))
        self.assertEqual(paths, export_analysis(dataset, quality, out))
        reconstructed, assessment = inspect_export(out)
        self.assertEqual(reconstructed, dataset)
        self.assertEqual(assessment, quality)
        manager_again = DurableJobManager(LocalDockerBackend(execution_authorized=False,
                                          transport=SimulatedTransport()), self.manager.path)
        self.assertEqual(ingest_verified_pilot(self.protocol, manager_again), (dataset, quality))

    def test_refuse_overwrite_different_export(self):
        self.complete()
        dataset, quality = ingest_verified_pilot(self.protocol, self.manager)
        out = self.root / "analysis"
        export_analysis(dataset, quality, out)
        (out / "quality.json").write_text('{"different":true}', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export_analysis(dataset, quality, out)
        # The dataset is still intact; neither output was overwritten.
        self.assertEqual(json.loads((out / "observations.json").read_text(encoding="utf-8")),
                         dataset.model_dump(mode="json"))

    def test_export_modified_dataset_is_detected(self):
        self.complete()
        dataset, quality = ingest_verified_pilot(self.protocol, self.manager)
        out = self.root / "analysis"
        export_analysis(dataset, quality, out)
        path = out / "observations.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["observations"][0]["value"] = 123.0
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 differ"):
            inspect_export(out)

    def test_forbid_fake_research_eligibility(self):
        self.complete()
        dataset, quality = ingest_verified_pilot(self.protocol, self.manager)
        document = quality.model_dump(mode="json")
        document["ready_for_scientific_inference"] = True
        with self.assertRaises(ValidationError):
            QualityAssessment.model_validate(document)
        document2 = dataset.model_dump(mode="json")
        document2["scientific_evidence"] = True
        with self.assertRaises(ValidationError):
            AnalysisDataset.model_validate(document2)

    def test_forbid_duplicate_run_observations(self):
        self.complete()
        dataset, _ = ingest_verified_pilot(self.protocol, self.manager)
        doc = dataset.model_dump(mode="json")
        doc["observations"][1]["run_id"] = doc["observations"][0]["run_id"]
        with self.assertRaises(ValidationError):
            AnalysisDataset.model_validate(doc)

    def test_reject_nonfinite_metric(self):
        self.complete()
        dataset, _ = ingest_verified_pilot(self.protocol, self.manager)
        doc = dataset.observations[0].model_dump(mode="json")
        doc["value"] = float("nan")
        with self.assertRaises(ValidationError):
            MetricObservation.model_validate(doc)

    def test_quality_binding_must_match_dataset(self):
        self.complete()
        dataset, quality = ingest_verified_pilot(self.protocol, self.manager)
        doc = quality.model_dump(mode="json")
        doc["observation_count"] = 3
        with self.assertRaisesRegex(ValueError, "Observation count"):
            verify_export(dataset, QualityAssessment.model_validate(doc))


if __name__ == "__main__":
    unittest.main()
