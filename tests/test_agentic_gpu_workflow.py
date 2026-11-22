import json
from pathlib import Path
import re
from subprocess import CompletedProcess, TimeoutExpired
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.models import LLMResponse
from execution.agentic_gpu import MODELS, POLICIES, PROFILES, TASKS, asset_manifest
from execution.agentic_gpu_worker import ACTION, run_trials
from execution.job_manager import DurableJobManager
from providers.replay_provider import FakeProvider
from research.agentic_study import EVALUATIONS, design_constraints, validate_capability
from research.brief_autonomy import AutonomyBudget, authorize_autonomy, autonomy_plan, run_autonomously
from research.brief_workflow import BriefWorkflow, Decision
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from research.critique.store import ReviewStore
from research.literature_search import OpenAlexClient
from research.manuscript import verify_manuscript_draft
from research.resource_catalog import AGENTIC_ID
from tests.test_v82_literature import FakeOpenAlex, mission_data, valid_review
from tests.test_v83_design import valid_design

QUESTION = "Comment varient les erreurs d'agents avec outils entre checkpoints, relance et perturbations ?"
BRIEF = f"""## Question
{QUESTION}
## Contexte
Pilote descriptif de trois checkpoints sur des lectures appariées.
## Périmètre
Comparer les politiques de relance de catalogue et plusieurs perturbations.
## Contraintes
Utiliser le même outil local et une révision fixe par checkpoint.
## Critères de réussite
Comparer les taux de réussite depuis les reçus complets des trois lots.
"""
CONFIG = {
    "ssh_host": "fixture-gpu", "python_path": "/opt/pilot/python", "worker_path": "/opt/pilot/agentic_gpu_worker.py",
    "snapshot_path": f"/opt/existing/{MODELS['qwen4'][1]}", "model_cache": "/opt/pilot/models",
    "gpu_index": 1, "allow_model_download": True,
}


def agentic_design():
    data = valid_design()
    data["factors"] = [
        {**factor, "operationalization": "Appliquer à chaque paire les conditions mesurables du corpus contrôlé."}
        for factor in design_constraints()["factors"]
    ]
    data["baselines"] = [{"name": "Qwen sans relance en contrôle",
                           "factor_levels": {"checkpoint": "qwen4", "policy": "no_retry", "shift": "clean"},
                           "rationale": "Ce contrôle fixe le modèle et la politique en l'absence de perturbation."}]
    data["hypothesis_tests"][0].update(
        factor_name="policy", fixed_levels={"checkpoint": "qwen4", "shift": "transient"},
        contrast="Comparer la relance à la lecture unique lorsqu'une erreur transitoire est injectée.",
    )
    data["resource_limits"].update(max_tokens_per_llm_call=560, max_total_llm_calls=2 * EVALUATIONS,
                                   max_total_tokens=2 * EVALUATIONS * 560)
    return data


def generation(instructions, prompt):
    if instructions == ACTION:
        answer = {"product": re.search(r"[A-Z]{2}-[0-9]{3}", prompt).group(), "field": "stock"}
    else:
        result = json.loads(prompt.rsplit("Catalog result: ", 1)[1])
        answer = {"answer": result.get("value")}
    return {"text": json.dumps(answer), "input_tokens": 38, "output_tokens": 12, "elapsed_ns": 1000}


class AgenticGPUWorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "agentic"
        mission = {**mission_data(), "research_question": QUESTION}
        choice = {"resource_id": AGENTIC_ID, "question_excerpt": QUESTION,
                  "rationale": "Le pilote croise les checkpoints, les politiques et les perturbations demandés."}
        agent = FakeProvider([LLMResponse(text=json.dumps(data)) for data in
                              (mission, valid_review(), choice, agentic_design())])
        self.flow = BriefWorkflow(self.root, provider_factory=lambda _: agent,
                                  search_factory=lambda: OpenAlexClient(api_key="", transport=FakeOpenAlex()))
        self.flow.start("agentic", BRIEF, LLMSettings(provider="openai", model="offline-fixture"), gpu_config=CONFIG)
        budget = AutonomyBudget(llm_calls=4)
        consent = autonomy_plan(self.flow, budget)
        self.assertEqual(len(consent["details"]["scope"]["execution"]["capabilities"]["gpu.agentic_lookup.v1"]["models"]), 3)
        authorize_autonomy(self.flow, budget, Decision(actor="operator", target_sha256=consent["target_sha256"]))
        self.operations = []

    def transport(self, command, **options):
        request = json.loads(options["input"])
        key, phase = request["model_key"], request["phase"]
        self.operations.append((key, phase))
        fields = {k: request[k] for k in ("job_id", "job_sha256", "worker_sha256", "model_key")}
        manifest = asset_manifest(key)
        self.assertIn("agentic_gpu_worker.py", command[-1])
        ledger = self.root / "pilot-01" / "jobs.sqlite3"
        if phase != "measurement":
            self.assertFalse(ledger.exists())
            self.assertEqual(request["trials"], [])
            reply = {**fields, "ready": phase == "provision" or key == "qwen4",
                     "installed": phase == "provision", "manifest": manifest}
        else:
            self.assertEqual(DurableJobManager(path=ledger, read_only=True).get(request["job_id"]).phase,
                             "submission_reserved")
            self.assertEqual(len(request["trials"]), 48)
            reply = {**fields, "model": manifest, "responses": run_trials(request["trials"], generation),
                     "runtime": {"gpu_index": 1, "gpu_name": "Fixture RTX A5000", "cuda_version": "12.6",
                                 "torch_version": "2.6", "transformers_version": "4.57",
                                 "elapsed_ms": 30, "peak_vram_bytes": 1000}}
        return CompletedProcess(command, 0, stdout=json.dumps(reply).encode(), stderr=b"")

    def test_single_delegation_prepares_models_then_runs_and_drafts(self):
        with patch("execution.ood_gpu.subprocess.run", side_effect=self.transport):
            status = run_autonomously(self.flow)
        self.assertEqual(status["stage"], "review_manuscript", status)
        self.assertEqual(self.operations, [
            ("qwen17", "preflight"), ("qwen17", "provision"), ("qwen4", "preflight"),
            ("olmoe", "preflight"), ("olmoe", "provision"),
            ("qwen17", "measurement"), ("qwen4", "measurement"), ("olmoe", "measurement"),
        ])
        report = read_document(status["report"]["path"])
        self.assertEqual(len(report["trials"]), EVALUATIONS)
        self.assertEqual(len(report["cells"]), len(MODELS) * len(POLICIES) * len(PROFILES))
        self.assertEqual(report["shift_gaps"]["qwen17-no_retry-transient"]["drop_points"], 100)
        self.assertIn("abstained", {row["outcome"] for row in report["trials"]})
        self.assertEqual(report["shift_gaps"]["olmoe-retry_once-transient"]["drop_points"], 0)
        self.assertEqual(report["shift_gaps"]["olmoe-retry_once-transient"]["distinct_tasks"], len(TASKS))
        self.assertEqual(report["models"]["olmoe"]["architecture"], "moe")
        self.assertFalse(report["scientific_evidence"])
        draft = Path(status["draft"]["draft"]).parent
        manuscript = verify_manuscript_draft(ReviewStore(draft.parent / "review.sqlite3", read_only=True),
                                            status["draft"]["review_id"], draft)
        self.assertEqual(manuscript.schema_version, "9.0-agentic-gpu")

    def test_malformed_preparation_blocks_before_every_gpu_job(self):
        def corrupted(command, **options):
            result = self.transport(command, **options)
            content = json.loads(result.stdout)
            content["job_sha256"] = "0" * 64
            return CompletedProcess(command, 0, stdout=json.dumps(content).encode(), stderr=b"")

        with patch("execution.ood_gpu.subprocess.run", side_effect=corrupted) as ssh:
            status = run_autonomously(self.flow)
        self.assertEqual(status["block"]["reason"], "gpu_preflight_failed")
        self.assertFalse((self.root / "pilot-01/jobs.sqlite3").exists())
        ssh.assert_called_once()

    def test_incomplete_or_mislabelled_factor_grid_is_refused(self):
        for kind, levels in (("architecture", list(MODELS)), ("other", ["olmoe", "qwen4"])):
            design = agentic_design()
            design["factors"][0].update(kind=kind, levels=levels)
            with self.assertRaises(ValueError):
                validate_capability(design)

    def test_ambiguous_second_checkpoint_keeps_first_receipt_without_replaying(self):
        def interrupted(command, **options):
            request = json.loads(options["input"])
            if request["model_key"] == "qwen4" and request["phase"] == "measurement":
                self.operations.append(("qwen4", "measurement"))
                raise TimeoutExpired(command, 1)
            return self.transport(command, **options)

        with patch("execution.ood_gpu.subprocess.run", side_effect=interrupted):
            status = run_autonomously(self.flow)
            before = list(self.operations)
            self.assertTrue(status["block"]["uncertain"])
            self.assertEqual(run_autonomously(self.flow)["status"], "blocked")
            self.assertEqual(self.operations, before)
        p = self.root / "pilot-01"
        plan = read_document(p / "action-plan.json")
        ledger = DurableJobManager(path=p / "jobs.sqlite3", read_only=True)
        jobs = {job["spec"]["workload"]["parameters"]["model_key"]: job["spec"]["job_id"] for job in plan["jobs"]}
        self.assertEqual(ledger.get(jobs["qwen17"]).phase, "succeeded")
        self.assertEqual(ledger.get(jobs["qwen4"]).phase, "needs_reconciliation")
        self.assertIsNone(ledger.get(jobs["olmoe"]))
