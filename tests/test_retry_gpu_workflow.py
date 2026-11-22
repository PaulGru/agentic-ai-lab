import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.models import LLMResponse
from analysis.contracts import fingerprint
from analysis.retry_pilot import QUESTION
from execution.job_manager import DurableJobManager
from execution.retry_gpu import MINUTES, RetryGPUBackend
from execution.retry_gpu_worker import LOOKUP, MODEL, REVISION, run_trials
from providers.replay_provider import FakeProvider
from research.acl_pdf import render_acl
from research.brief_workflow import RETRY_GPU_STAGE, BriefWorkflow, Decision, Revision
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from research.critique.store import ReviewStore
from research.literature_search import OpenAlexClient
from research.manuscript import verify_manuscript_draft
from tests.test_ood_gpu_workflow import CONFIG as CALIBRATION_CONFIG
from tests.test_v82_literature import FakeOpenAlex, mission_data, valid_review


ROOT = Path(__file__).resolve().parents[1]
BRIEF = (ROOT / "tests/fixtures/research_brief.md").read_text(encoding="utf-8")
CONFIG = {**CALIBRATION_CONFIG, "worker_path": "/home/operator/lab-gpu/retry_gpu_worker.py"}


def worker_fixture(request):
    parameters = request["parameters"]
    tasks = {task["product"]: task for task in parameters["tasks"].values()}

    def generate(instructions, prompt):
        if instructions == LOOKUP:
            product = re.search(r"[A-Z]{2}-[0-9]{3}", prompt).group()
            answer = {"product": product, "field": tasks[product]["field"]}
        else:
            result = json.loads(prompt.splitlines()[-1].split(" : ", 1)[-1])
            answer = {"answer": result.get("value")}
        return {"text": json.dumps(answer), "input_tokens": 39, "output_tokens": 12, "latency_ms": 1}

    return {
        **{key: request[key] for key in ("job_id", "job_sha256", "worker_sha256")},
        "model": MODEL, "revision": REVISION, "dtype": "bfloat16",
        "responses": run_trials(parameters, generate),
        "runtime": {
            "gpu_index": request["gpu_index"], "gpu_name": "Fixture RTX A5000", "cuda_version": "12.4",
            "torch_version": "2.6.0", "transformers_version": "4.57.1", "elapsed_ms": 70,
            "peak_vram_bytes": 9_000_000_000,
        },
    }


class RetryGPUWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "workflow"
        self.pilot = self.root / "pilot-01"
        self.ledger = self.pilot / "jobs.sqlite3"
        self.corpus = read_document(ROOT / "examples/v9_retry_pilot.json")
        self.corpus["tasks"][0]["prompt"] = "Pour le pilote relance, indique le stock actuel AX-103."
        self.corpus["catalog"]["AX-103"]["stock"] = "79"
        self.settings = LLMSettings(provider="anthropic", model="planner-test")
        mission = {**mission_data(), "research_question": QUESTION}
        provider = FakeProvider([LLMResponse(text=json.dumps(data)) for data in (mission, valid_review())])
        self.flow = BriefWorkflow(self.root, provider_factory=lambda _: provider,
                                  search_factory=lambda: OpenAlexClient(api_key="", transport=FakeOpenAlex()))

    def approve(self, stage):
        gate = self.flow.status()["pending"]
        self.assertEqual(gate["stage"], stage)
        return self.flow.approve(Decision(actor="test", target_sha256=gate["target_sha256"]))

    def ready(self):
        self.flow.start("retry-study", BRIEF, self.settings, "retry_gpu", self.corpus, gpu_config=CONFIG)
        for stage in ("mission", "review_mission", "catalog", "literature", "review_literature", "review_protocol"):
            self.approve(stage)
        return self.flow.status()["pending"]

    def fixture_transport(self, command, **options):
        request = json.loads(options["input"])
        plan = read_document(self.pilot / "action-plan.json")
        job = plan["jobs"][0]
        self.assertEqual(self.flow.store.current()["action"], "reserve")
        record = DurableJobManager(path=self.ledger, read_only=True).get(request["job_id"])
        self.assertEqual(record.phase, "submission_reserved")
        self.assertEqual(request["job_sha256"], job["sha256"])
        self.assertEqual(request["parameters"], job["spec"]["workload"]["parameters"])
        self.assertEqual(request["parameters"]["tasks"]["T01"]["value"], "79")
        self.assertEqual(request["parameters"]["tasks"]["T01"]["prompt"], self.corpus["tasks"][0]["prompt"])
        self.assertIn("retry_gpu_worker.py", command[-1])
        self.assertIn(f"{MINUTES * 60}s", command[-1])
        response = worker_fixture(request)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(response).encode(), stderr=b"")

    def test_approved_protocol_to_catalog_job_and_paired_report(self):
        with patch("execution.ood_gpu.subprocess.run", side_effect=self.fixture_transport) as ssh:
            gate = self.ready()
            ssh.assert_not_called()
            self.assertFalse(self.ledger.exists())
            plan = gate["details"]
            study = read_document(self.pilot / "study.json")
            self.assertEqual(plan, read_document(self.pilot / "action-plan.json"))
            self.assertEqual(plan["protocol_sha256"], fingerprint(study["protocol"]))
            self.assertEqual(study["protocol"]["measured_question"], QUESTION)
            self.assertEqual(plan["jobs"][0]["spec"]["source_design_sha256"], plan["protocol_sha256"])
            self.assertEqual(plan["budget"]["trials"], 32)
            self.assertEqual(plan["budget"]["generations"], 64)
            status = self.approve(RETRY_GPU_STAGE)
            ssh.assert_called_once()
        self.assertEqual(status["stage"], "review_manuscript", status)
        report = read_document(status["report"]["path"])
        self.assertEqual(report["action_plan_sha256"], plan["plan_sha256"])
        self.assertEqual(report["approval"]["target_sha256"], gate["target_sha256"])
        self.assertEqual(report["comparison"]["clean"]["gain_points"], 0)
        self.assertEqual(report["comparison"]["transient"]["retry_only"], 8)
        self.assertEqual(report["comparison"]["transient"]["gain_points"], 100)
        self.assertFalse(report["scientific_evidence"])
        draft_dir = Path(status["draft"]["draft"]).parent
        manuscript = verify_manuscript_draft(ReviewStore(draft_dir.parent / "review.sqlite3", read_only=True),
                                            status["draft"]["review_id"], draft_dir)
        self.assertEqual(manuscript.schema_version, "9.0-retry-gpu")
        self.assertEqual([row.mean for row in manuscript.figure], [100, 100, 0, 100])
        expected_pointer = "/analysis/comparison/transient/gain_points"
        self.assertTrue(any(s.reference.pointer == expected_pointer for s in manuscript.statements))
        self.assertIn("pilote sur GPU", render_acl(manuscript, approved=False))

    def test_tampering_with_exported_corpus_or_plan_blocks_before_ssh(self):
        gate = self.ready()
        for name in ("study.json", "action-plan.json"):
            with self.subTest(name=name):
                path = self.pilot / name
                original = path.read_bytes()
                document = read_document(path)
                if name == "study.json":
                    document["corpus"]["tasks"][0]["prompt"] = "Une autre tâche"
                else:
                    document["jobs"][0]["spec"]["workload"]["parameters"]["trials"].pop()
                path.write_text(json.dumps(document), encoding="utf-8")
                try:
                    with patch("execution.ood_gpu.subprocess.run") as ssh:
                        with self.assertRaises(ValueError):
                            self.flow.approve(Decision(actor="test", target_sha256=gate["target_sha256"]))
                        ssh.assert_not_called()
                        self.assertFalse(self.ledger.exists())
                finally:
                    path.write_bytes(original)

    def test_wrong_worker_or_absent_catalog_cannot_use_the_calibration(self):
        with self.assertRaisesRegex(ValueError, "explicit catalog corpus"):
            self.flow.start("retry-study", BRIEF, self.settings, "retry_gpu", gpu_config=CONFIG)
        with self.assertRaisesRegex(ValueError, "retry_gpu_worker.py"):
            self.flow.start("retry-study", BRIEF, self.settings, "retry_gpu", self.corpus,
                            gpu_config=CALIBRATION_CONFIG)
        self.assertFalse(self.root.exists())

    def test_protocol_revision_invalidates_approval_and_changes_compiled_schedule(self):
        old = self.ready()
        self.corpus["order_seed"] += 1
        self.flow.revise(Revision(actor="test", stage="protocol", reason="Contrebalancer dans un nouvel ordre.",
                                  target_sha256=self.flow.status()["checkpoint_sha256"], corpus=self.corpus,
                                  gpu_config={**CONFIG, "gpu_index": 0}))
        self.approve("review_protocol")
        new = self.flow.status()["pending"]
        with patch("execution.ood_gpu.subprocess.run") as ssh:
            with self.assertRaises(PermissionError):
                self.flow.approve(Decision(actor="test", target_sha256=old["target_sha256"]))
            with self.assertRaises(PermissionError):
                RetryGPUBackend(new["details"], approved_plan_sha256=old["details"]["plan_sha256"])
            ssh.assert_not_called()
        self.assertNotEqual(new["details"]["schedule_sha256"], old["details"]["schedule_sha256"])
        self.assertEqual(new["details"]["gpu_config"]["gpu_index"], 0)

    def test_uncertain_or_corrupted_gpu_batch_is_never_replayed(self):
        gate = self.ready()

        def corrupted(command, **options):
            receipt = worker_fixture(json.loads(options["input"]))
            receipt["responses"][1]["id"] = receipt["responses"][0]["id"]
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(receipt).encode(), stderr=b"")

        with patch("execution.ood_gpu.subprocess.run", side_effect=corrupted) as ssh:
            status = self.approve(RETRY_GPU_STAGE)
            self.assertEqual(status["status"], "blocked")
            self.assertTrue(status["block"]["uncertain"])
            job_id = gate["details"]["jobs"][0]["spec"]["job_id"]
            self.assertEqual(DurableJobManager(path=self.ledger, read_only=True).get(job_id).phase,
                             "needs_reconciliation")
            reopened = BriefWorkflow(self.root)
            with self.assertRaises(PermissionError):
                reopened.recover(Decision(actor="test", target_sha256=status["checkpoint_sha256"]))
            with self.assertRaises(PermissionError):
                reopened.approve(Decision(actor="test", target_sha256=gate["target_sha256"]))
            ssh.assert_called_once()

    def test_recovery_attaches_complete_receipt_without_a_second_gpu_job(self):
        self.ready()
        with (patch("execution.ood_gpu.subprocess.run", side_effect=self.fixture_transport) as ssh,
              patch.object(self.flow, "_complete_retry_gpu", side_effect=KeyboardInterrupt)):
            with self.assertRaises(KeyboardInterrupt):
                self.approve(RETRY_GPU_STAGE)
            reopened = BriefWorkflow(self.root)
            status = reopened.status()
            complete = reopened.recover(Decision(actor="test", target_sha256=status["checkpoint_sha256"]))
            self.assertEqual(complete["stage"], "review_manuscript", complete)
            self.assertTrue(Path(complete["report"]["path"]).is_file())
            ssh.assert_called_once()
