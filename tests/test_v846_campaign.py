import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contextlib import closing
import sqlite3

from pydantic import ValidationError

from execution.campaign import PilotProtocol, build_report, save_report
from execution.backends.local import LocalDockerBackend
from execution.backends.docker_transport import DockerOutcome
from execution.job_manager import DurableJobManager

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "v846_protocol.json"


def protocol():
    return PilotProtocol.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))


class DeterministicTestTransport:
    # Does NOT run Docker; simulated outputs are labelled accordingly.
    simulated = True

    def __init__(self):
        self.calls = []

    def run(self, job, code, cancel):
        import statistics
        self.calls.append(job.job_id)
        vals = job.workload.parameters["values"]
        envelope = {
            "kind": "agentic-v842-artifacts", "sha256": job.fingerprint(),
            "files": {
                "metrics.json": {"count": len(vals), "mean": statistics.fmean(vals),
                                 "minimum": min(vals), "maximum": max(vals)},
                "run_metadata.json": {"job_id": job.job_id, "job_sha256": job.fingerprint(),
                                      "seed": job.seed, "workload": job.workload.name,
                                      "python_version": "3.12.0", "execution": "docker_sandbox"},
            },
        }
        return DockerOutcome(status="success", exit_code=0,
                             stdout=json.dumps(envelope), stderr="", duration_seconds=0.001)


class CampaignTests(unittest.TestCase):
    def test_jobs_are_allowlisted_and_provider_independent(self):
        p = protocol()
        self.assertEqual(len(p.jobs()), 2)
        self.assertTrue(all(j.workload.name == "synthetic.mean" for j in p.jobs()))
        self.assertTrue(all(j.source_design_sha256 == p.fingerprint() for j in p.jobs()))
        self.assertNotIn("openai", p.model_dump_json().lower())
        self.assertNotIn("anthropic", p.model_dump_json().lower())

    def test_protocol_digest_stable(self):
        p = protocol()
        self.assertEqual(p.fingerprint(), PilotProtocol.model_validate_json(p.model_dump_json()).fingerprint())

    def test_duplicate_run_id_rejected(self):
        doc = protocol().model_dump(mode="json")
        doc["runs"][1]["run_id"] = doc["runs"][0]["run_id"]
        with self.assertRaises(ValidationError):
            PilotProtocol.model_validate(doc)

    def test_resource_escape_rejected(self):
        doc = protocol().model_dump(mode="json")
        doc["resources"]["gpus"] = 1
        with self.assertRaises(ValidationError):
            PilotProtocol.model_validate(doc)

    def test_unknown_workload_fields_rejected(self):
        doc = protocol().model_dump(mode="json")
        doc["runs"][0]["command"] = "rm -rf /"
        with self.assertRaises(ValidationError):
            PilotProtocol.model_validate(doc)

    def test_nonfinite_values_rejected(self):
        doc = protocol().model_dump(mode="json")
        doc["runs"][0]["values"][0] = float("nan")
        with self.assertRaises(ValidationError):
            PilotProtocol.model_validate(doc)

    def test_no_report_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalDockerBackend(execution_authorized=True, transport=DeterministicTestTransport())
            manager = DurableJobManager(backend, Path(directory) / "jobs.db")
            with self.assertRaises(RuntimeError):
                build_report(protocol(), manager)

    def test_end_to_end_ledger_collect_and_report(self):
        p = protocol()
        with tempfile.TemporaryDirectory() as directory:
            transport = DeterministicTestTransport()
            backend = LocalDockerBackend(execution_authorized=True, transport=transport)
            manager = DurableJobManager(backend, Path(directory) / "jobs.db")
            for job in p.jobs():
                record = manager.submit(job)
                final = backend.wait(record.handle, timeout=3)
                self.assertEqual(final.status.value, "succeeded")
                self.assertEqual(manager.refresh(job.job_id).phase, "succeeded")
            report = build_report(p, manager)
            self.assertEqual(report["analysis"]["last_minus_first_mean"], 3.0)
            self.assertTrue(report["simulated"])
            self.assertFalse(report["scientific_evidence"])
            self.assertEqual(len(transport.calls), 2)
            report_path = Path(directory) / "report.json"
            save_report(report, report_path)
            save_report(report, report_path)
            self.assertEqual(json.loads(report_path.read_text()), report)
            with self.assertRaises(FileExistsError):
                save_report({"different": True}, report_path)
            # After process reconstruction, report is rebuilt solely from ledger.
            fresh_manager = DurableJobManager(LocalDockerBackend(execution_authorized=True,
                                           transport=DeterministicTestTransport()), Path(directory) / "jobs.db")
            rebuilt = build_report(p, fresh_manager)
            self.assertEqual(rebuilt, report)

    def test_corruption_detected_by_ledger_checksum(self):
        p = protocol()
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalDockerBackend(execution_authorized=True, transport=DeterministicTestTransport())
            manager = DurableJobManager(backend, Path(directory) / "jobs.db")
            for job in p.jobs():
                record = manager.submit(job)
                backend.wait(record.handle, timeout=3)
                manager.refresh(job.job_id)
            import sqlite3
            with closing(sqlite3.connect(manager.path)) as db:
                with db:
                    db.execute(
                        "UPDATE jobs SET artifacts_sha256=? WHERE job_id=?",
                        ("0" * 64, p.jobs()[0].job_id)
                    )
            with self.assertRaises(ValueError):
                build_report(p, manager)


if __name__ == "__main__":
    unittest.main()
