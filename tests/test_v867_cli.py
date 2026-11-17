import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from execution.backends.local import LocalDockerBackend
from execution.job_manager import DurableJobManager
from langgraph.checkpoint.memory import InMemorySaver

from research.critique.contracts import ReviewInput
from research.critique.graph import ReviewWorkflow, config_for
from research.critique.reporting import inspect_export
from research.critique.store import ReviewStore
from scripts.demo_v86 import verify_recorded_review
from tests.test_v851_ingestion import SimulatedTransport, sample_protocol

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def run_cli(self, *args, input=""):
        return subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.demo_v86", *args],
                              cwd=ROOT, capture_output=True, input=input, text=True, encoding="utf-8", timeout=90)

    def test_preview_never_creates_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-created.db"
            result = self.run_cli("--store", str(path), "preview", "--input", "examples/v86_review_input.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(path.exists())

    def test_offline_cli_approves_new_version_then_exports_and_reverifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "reviews.db"
            out = root / "export avec espaces é"
            result = self.run_cli("--store", str(store), "offline", "--input", "examples/v86_review_input.json",
                                  "--draft", "examples/v86_revision_draft.json", "--out-dir", str(out), input="APPROVE\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("revision_approved", result.stdout)
            package = inspect_export(out)
            approved = [x for x in package["artifacts"] if x["kind"] == "approved_revision"]
            self.assertEqual(len(approved), 1)
            self.assertEqual(approved[0]["document"]["execution_status"], "not_executed")
            self.assertEqual(package["calls"], [])
            result = self.run_cli("--store", str(store), "verify", "--review", "v86-demo-001")
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_cli("inspect", "--out-dir", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NON", result.stdout)

    def test_offline_llm_configuration_is_rejected_before_store_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = json.loads((ROOT / "examples" / "v86_review_input.json").read_text(encoding="utf-8"))
            doc.update(assist_critique=True, llm={"provider": "openai", "model": "fictional-test-model"})
            source = root / "request.json"
            source.write_text(json.dumps(doc), encoding="utf-8")
            store = root / "absent.db"
            result = self.run_cli("--store", str(store), "offline", "--input", str(source))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(store.exists())

    def test_full_review_from_existing_simulated_pilot_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = sample_protocol()
            path = root / "jobs.db"
            backend = LocalDockerBackend(execution_authorized=True, transport=SimulatedTransport())
            jobs = DurableJobManager(backend, path)
            for job in protocol.jobs():
                record = jobs.submit(job)
                backend.wait(record.handle, timeout=3)
                jobs.refresh(job.job_id)
            before = path.read_bytes()
            source = root / "protocol.json"
            source.write_text(protocol.model_dump_json(), encoding="utf-8")
            request = ReviewInput(review_id="pilot-review", source_kind="verified_pilot_aggregates",
                                  source_path=str(source), jobs_db=str(path))
            store = ReviewStore(root / "reviews.db")
            graph = ReviewWorkflow(store).build(InMemorySaver())
            result = graph.invoke({"input_document": request.model_dump(mode="json")}, config_for(request.review_id))
            self.assertEqual(result["status"], "review_complete")
            self.assertIn("simulated_transport", [x["issue_id"] for x in store.require("review", request.review_id)["issues"]])
            verify_recorded_review(store, request.review_id)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
