import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from execution.job_manager import DurableJobManager
from execution.ood_gpu import OODGPUBackend, OODGPUConfig, run_ood_gpu, ssh_command
from execution.ood_gpu_worker import MODEL, REVISION, TRIAL_IDS, validate_request, worker_sha256
from research.brief_workflow import BriefWorkflow, Decision, Revision
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from research.ood_study import check_ood_study
from tests.test_ood_replay_workflow import BRIEF, offline_workflow

CONFIG = {
    "ssh_host": "gpu-test",
    "python_path": "/home/operator/lab-gpu/.venv/bin/python",
    "worker_path": "/home/operator/lab-gpu/ood_gpu_worker.py",
    "snapshot_path": f"/home/operator/model/snapshots/{REVISION}",
    "gpu_index": 1,
}


def fake_receipt(request):
    words = {"stock-reference": " 7 ", "zone-ood": "B-1", "zone-reference": "C-2", "stock-ood": "7"}
    return {
        "job_id": request["job_id"], "job_sha256": request["job_sha256"], "worker_sha256": request["worker_sha256"],
        "model": MODEL, "revision": REVISION, "dtype": "bfloat16",
        "responses": [
            {"id": trial["id"], "text": words[trial["id"]], "input_tokens": 34, "output_tokens": 4, "latency_ms": 11}
            for trial in request["trials"]
        ],
        "runtime": {
            "gpu_index": request["gpu_index"], "gpu_name": "Fixture RTX A5000", "cuda_version": "12.4",
            "torch_version": "2.6.0", "transformers_version": "4.57.1", "elapsed_ms": 60,
            "peak_vram_bytes": 9_000_000_000,
        },
    }


class OODGPUWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "workflow"
        self.flow = offline_workflow(self.root)
        self.brief = (BRIEF.replace("ood_replay", "ood_gpu")
                      .replace("et deux modèles fictifs", "et une révision de Qwen")
                      .replace("Utiliser des réponses préenregistrées dans un parcours sans appel réel ni matériel GPU.",
                               "Utiliser une seule carte et quatre générations bornées."))
        self.settings = LLMSettings(provider="openai", model="gpt-6.2-astra")

    def approve(self, stage):
        status = self.flow.status()
        self.assertEqual(status["stage"], stage, status)
        target = status["pending"]["target_sha256"]
        return self.flow.approve(Decision(actor="test", target_sha256=target))

    def ready(self):
        self.flow.start("gpu-calibration", self.brief, self.settings, gpu_config=CONFIG)
        for stage in ("mission", "review_mission", "catalog", "literature", "review_literature", "review_protocol"):
            self.approve(stage)
        return self.flow.status()["pending"]

    def test_gpu_receipt_reaches_the_critique_and_manuscript_through_the_real_ssh_adapter(self):
        gate = self.ready()
        self.assertEqual(gate["stage"], "ood_gpu_execution")
        self.assertEqual(gate["details"]["gpu_config"], CONFIG)
        job = gate["details"]["jobs"][0]
        self.assertEqual(job["spec"]["resources"]["gpus"], 1)
        self.assertEqual(len(job["spec"]["workload"]["parameters"]["trials"]), 4)
        ledger = self.root / "pilot-01" / "jobs.sqlite3"
        self.assertFalse(ledger.exists())

        def ssh(command, **options):
            request = json.loads(options["input"])
            self.assertEqual(command, ssh_command(OODGPUConfig.model_validate(CONFIG)))
            self.assertEqual(request["job_sha256"], job["sha256"])
            self.assertEqual(request["trials"], job["spec"]["workload"]["parameters"]["trials"])
            self.assertTrue(all(set(trial) == {"id", "prompt"} for trial in request["trials"]))
            self.assertEqual(DurableJobManager(path=ledger, read_only=True).get(request["job_id"]).phase,
                             "submission_reserved")
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(fake_receipt(request)).encode(), stderr=b"")

        with patch("execution.ood_gpu.subprocess.run", side_effect=ssh) as transport:
            status = self.approve("ood_gpu_execution")
            self.assertEqual(status["stage"], "review_manuscript", status)
            transport.assert_called_once()
            draft = Path(status["draft"]["draft"])
            report = read_document(draft.parent.parent / "analysis.json")
            self.assertEqual(report["degradation_points"], {"qwen3-4b": 50.0})
            self.assertEqual(report["runtime"]["gpu_index"], 1)
            self.assertEqual(report["model_revision"], REVISION)
            self.assertTrue(report["reference_calibration_passed"])
            self.assertFalse(report["simulated"])
            self.assertFalse(report["scientific_evidence"])
            self.assertEqual({trial["job_sha256"] for trial in report["trials"]}, {job["sha256"]})
            self.assertIn("Calibration OOD sur GPU", draft.read_text(encoding="utf-8"))
            review = (draft.parent.parent / "review" / "review.md").read_text(encoding="utf-8")
            self.assertIn("Réponses mesurées sur GPU. Calibration descriptive sur des tâches fictives.", review)
            self.assertEqual(self.approve("review_manuscript")["status"], "complete")
            transport.assert_called_once()

    def test_gpu_receipt_with_a_different_worker_is_blocked(self):
        self.ready()

        def mismatched_receipt(config, request):
            return {**fake_receipt(request), "worker_sha256": "0" * 64}

        with patch("execution.ood_gpu.ssh_receipt", side_effect=mismatched_receipt) as transport:
            blocked = self.approve("ood_gpu_execution")
            self.assertEqual(blocked["status"], "blocked")
            self.assertTrue(blocked["block"]["uncertain"])
            self.assertIsNone(blocked["draft"])
            transport.assert_called_once()

    def test_gpu_recovery_attaches_the_saved_receipt_without_ssh(self):
        self.ready()

        def interrupt_after_receipt(protocol, ledger):
            run_ood_gpu(protocol, ledger)
            raise KeyboardInterrupt

        with (patch("execution.ood_gpu.ssh_receipt", side_effect=lambda _, request: fake_receipt(request)) as transport,
              patch("research.brief_workflow.run_ood_gpu", side_effect=interrupt_after_receipt)):
            with self.assertRaises(KeyboardInterrupt):
                self.approve("ood_gpu_execution")
            self.flow = BriefWorkflow(self.root)
            before = self.flow.status()
            rebuilt = self.flow.recover(Decision(actor="test", target_sha256=before["checkpoint_sha256"]))
            self.assertEqual(rebuilt["stage"], "review_manuscript")
            transport.assert_called_once()

    def test_gpu_config_rejects_shell_options_and_requires_a_pinned_snapshot(self):
        self.assertFalse(self.root.exists())
        with self.assertRaisesRegex(ValueError, "explicit|GPU configuration"):
            self.flow.start("gpu-calibration", self.brief, self.settings)
        for invalid in ({"ssh_host": "-Fconfig"}, {"worker_path": "/tmp/script.py"},
                        {"python_path": "/tmp/python;id"}, {"snapshot_path": "/tmp/main"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                OODGPUConfig.model_validate({**CONFIG, **invalid})
        protocol = self.ready()
        self.assertEqual(protocol["details"]["worker_sha256"], worker_sha256())
        source = read_document(self.root / "pilot-01" / "study.json")
        study = check_ood_study(source)
        with patch("execution.ood_gpu.ssh_receipt") as transport:
            with self.assertRaises(PermissionError):
                OODGPUBackend(study).submit(study.job())
            transport.assert_not_called()

    def test_gpu_transport_timeout_is_durable_and_never_replayed(self):
        gate = self.ready()
        with patch("execution.ood_gpu.subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 1)) as transport:
            blocked = self.approve("ood_gpu_execution")
            self.assertEqual(blocked["status"], "blocked")
            self.assertTrue(blocked["block"]["uncertain"])
            self.assertIsNone(blocked["pending"])
            ledger = DurableJobManager(path=self.root / "pilot-01" / "jobs.sqlite3", read_only=True)
            record = ledger.get(gate["details"]["jobs"][0]["spec"]["job_id"])
            self.assertEqual(record.phase, "needs_reconciliation")
            self.flow = BriefWorkflow(self.root)
            recovery = Decision(actor="test", target_sha256=blocked["checkpoint_sha256"])
            with self.assertRaisesRegex(PermissionError, "No complete OOD receipts"):
                self.flow.recover(recovery)
            with self.assertRaises(PermissionError):
                self.flow.approve(Decision(actor="test", target_sha256=gate["target_sha256"]))
            transport.assert_called_once()

    def test_revision_of_gpu_index_blocks_the_old_approval_before_ssh(self):
        prior = self.ready()
        self.flow.revise(Revision(
            actor="test", target_sha256=self.flow.status()["checkpoint_sha256"], stage="protocol",
            reason="Réserver la carte zéro pour ce test.", gpu_config={**CONFIG, "gpu_index": 0},
        ))
        current = self.approve("review_protocol")["pending"]
        self.assertEqual(current["details"]["gpu_config"]["gpu_index"], 0)
        transport = Mock()
        with patch("execution.ood_gpu.ssh_receipt", transport), self.assertRaises(PermissionError):
            self.flow.approve(Decision(actor="test", target_sha256=prior["target_sha256"]))
        transport.assert_not_called()

    def test_gpu_worker_rejects_unknown_worker_or_trial_order_before_loading_a_model(self):
        snapshot = self.root / REVISION
        snapshot.mkdir(parents=True)
        request = {
            "job_id": "calibration-test", "job_sha256": "c" * 64, "worker_sha256": worker_sha256(),
            "model": MODEL, "revision": REVISION, "snapshot": str(snapshot), "gpu_index": 0,
            "trials": [{"id": trial, "prompt": "test"} for trial in TRIAL_IDS],
        }
        self.assertEqual(validate_request(request), snapshot)
        with self.assertRaisesRegex(ValueError, "worker source"):
            validate_request({**request, "worker_sha256": "c" * 64})
        with self.assertRaisesRegex(ValueError, "Invalid calibration trial"):
            validate_request({**request, "trials": list(reversed(request["trials"]))})


if __name__ == "__main__":
    unittest.main()
