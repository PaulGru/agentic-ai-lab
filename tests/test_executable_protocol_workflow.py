import hashlib
from io import BytesIO
import json
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import Request, build_opener

from agent.models import LLMResponse
from execution.job_manager import DurableJobManager
from execution.protocol_gpu_worker import (
    MODEL, REVISION, _HuggingFaceRedirect, _install, execute as worker_execute, manifest_sha256, worker_sha256,
)
from experiments.protocol_program import RestrictedProgram
from providers.replay_provider import FakeProvider
from research.acl_pdf import render_acl
from research.brief_autonomy import AutonomyBudget, authorize_autonomy, autonomy_plan, run_autonomously
from research.brief_workflow import (
    PROGRAM_STAGE, IMPLEMENTATION_STAGE, RESOLUTION_STAGE, RESOURCE_STAGE, BriefWorkflow, Decision, Revision,
)
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from research.critique.store import ReviewStore
from research.literature_search import OpenAlexClient
from research.manuscript import verify_manuscript_draft
from research.resource_catalog import ARITHMETIC_ID, RANKING_ID
from tests.test_v82_literature import FakeOpenAlex, mission_data, valid_review
from tests.test_v83_design import valid_design

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Le score pondéré classe-t-il mieux les documents pertinents que le simple recouvrement de termes ?"
BRIEF = (ROOT / "examples/v9_new_protocol_brief.md").read_text(encoding="utf-8")


def evaluation_set():
    return read_document(ROOT / "examples/v9_new_protocol_evaluation.json")


def ranking_design():
    design = valid_design()
    design.update(
        title="Comparer deux heuristiques de classement",
        objective="Estimer par documents appariés la différence de réussite entre recouvrement et score pondéré.",
        factors=[{"name": "method", "kind": "other", "levels": ["overlap", "weighted"],
                  "operationalization": "Choisir le document dont le score du type sélectionné est le plus élevé."}],
        baselines=[{"name": "Recouvrement simple", "factor_levels": {"method": "overlap"},
                    "rationale": "Le nombre de termes communs fournit la méthode de référence."}],
        controlled_variables=["Les deux heuristiques voient exactement les mêmes documents et caractéristiques.",
                              "La règle détermine les égalités dans le même ordre."],
        perturbation_and_recovery_procedure="Toutes les tâches utilisent les mêmes caractéristiques définies à l'avance.",
        literature_links=[{"work_id": "W101",
                           "design_rationale": "La notice aide à décrire le principe de comparaison en milieu contrôlé.",
                           "limitation": "Cette notice ne traite pas du classement testé dans le pilote."}],
    )
    design["hypothesis_tests"][0].update(
        factor_name="method", level_a="overlap", level_b="weighted",
        comparison_scope="marginal_paired", fixed_levels={},
        contrast="Comparer les deux heuristiques sur les mêmes paires de documents.",
    )
    return design


def response(code):
    return {"hypothesis": "Le score pondéré classe mieux les documents.",
            "protocol": ["Choisir une réponse par condition et paire."],
            "seed": 13, "dependencies": [], "code": code, "expected_output": "Indice entier du document."}


PROGRAM = '''def predict(features, levels, seed):
    return (1 if features["weighted_a"] >= features["weighted_b"] else 2) if levels["method"] == "weighted" else (1 if features["overlap_a"] >= features["overlap_b"] else 2)
'''
GPU_PROGRAM = 'def predict(features, levels, seed): return 1 if levels["method"] == "weighted" else 0'
GPU_CONFIG = {
    "ssh_host": "fixture-gpu", "python_path": "/opt/pilot/python",
    "worker_path": "/opt/pilot/protocol_gpu_worker.py", "gpu_index": 0,
    "snapshot_path": f"/opt/pilot/models/{REVISION}",
}
MISSION = {**mission_data(), "research_question": QUESTION,
           "hypotheses": [{"id": "H1", "claim": "La pondération améliore la précision du classement.",
                           "falsification_test": "Vérifier que le différentiel apparié se situe au-dessus de zéro."}]}
REPAIR_QUESTION = "Le maximum ou le minimum prédit-il mieux la plus grande valeur de deux entiers ?"
REPAIR_BRIEF = f"""# Comparer deux règles de calcul

## Question
{REPAIR_QUESTION}

## Contexte
Chaque tâche présente deux entiers dont on cherche la plus grande valeur.

## Périmètre
Comparer le minimum et le maximum dans un pilote numérique synthétique.

## Contraintes
Évaluer les deux règles sur les mêmes entrées, avec les graines 13 et 42.

## Critères de réussite
Comparer les proportions de réponses exactes et vérifier la provenance du résultat.
"""


def repair_replies():
    mission = {**MISSION, "title": "Comparer deux règles pour prédire le maximum",
               "objective": "Mesurer l'exactitude de deux règles sur des paires comparables.",
               "research_question": REPAIR_QUESTION, "independent_variables": ["Règle de calcul"],
               "dependent_metrics": ["Taux de réponses exactes"], "controls": ["Mêmes paires d'entiers"],
               "baselines": ["Prendre la plus petite valeur"],
               "hypotheses": [{"id": "H1", "claim": "Le maximum prédit mieux la plus grande valeur de deux entiers.",
                               "falsification_test": "Comparer les deux proportions sur les mêmes entrées."}]}
    gap = {"resource_id": None, "question_excerpt": REPAIR_QUESTION,
           "rationale": "Le comparateur d'entiers est absent du catalogue de ressources initial."}
    repair = {"kind": "local_recipe", "question_excerpt": REPAIR_QUESTION,
              "rationale": "Construire des paires numériques à référence indépendante.",
              "next_requirement": None,
              "recipe": {"method_a": "minimum", "method_b": "maximum", "target": "maximum"}}
    return mission, gap, repair


def repair_design():
    design = ranking_design()
    design.update(title="Comparer les deux règles de calcul",
                  objective="Comparer minimum et maximum pour retrouver la plus grande valeur.",
                  factors=[{"name": "method", "kind": "other", "levels": ["minimum", "maximum"],
                            "operationalization": "Choisir l'entier indiqué par la règle de la cellule."}],
                  baselines=[{"name": "Règle du minimum", "factor_levels": {"method": "minimum"},
                              "rationale": "Le minimum fournit une règle de référence opposée."}],
                  controlled_variables=["Garder les mêmes paires et graines pour les deux méthodes.",
                                        "Utiliser la même métrique entière à chaque essai."],
                  perturbation_and_recovery_procedure="Évaluer les deux règles sur les mêmes paires.")
    design["hypothesis_tests"][0].update(level_a="minimum", level_b="maximum",
                                         contrast="Comparer les deux règles sur les mêmes couples.")
    design["resource_limits"]["tasks_per_cell"] = 6
    return design


REPAIR_PROGRAM = '''def predict(features, levels, seed):
    return features["left"] if (features["left"] >= features["right"]) == (levels["method"] == "maximum") else features["right"]
'''


class ProtocolCPUWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "workflow"
        self.provider = FakeProvider([])
        self.workflow = BriefWorkflow(self.root, provider_factory=lambda _: self.provider,
                                      search_factory=lambda: OpenAlexClient(api_key="", transport=FakeOpenAlex()))
        self.settings = LLMSettings(provider="anthropic", model="planner-fixture")

    def approve(self, stage, reply=None):
        gate = self.workflow.status()["pending"]
        self.assertEqual(gate["stage"], stage)
        if reply:
            self.provider.responses.append(LLMResponse(text=json.dumps(reply)))
        return self.workflow.approve(Decision(actor="operator", target_sha256=gate["target_sha256"]))

    def through_protocol(self, evaluation=None, design=None, gpu_config=None):
        self.workflow.start("ranking", BRIEF, self.settings, evaluation=evaluation or evaluation_set(),
                            gpu_config=gpu_config)
        for stage, reply in [("mission", MISSION), ("review_mission", None), ("catalog", None),
                             ("literature", valid_review()), ("review_literature", None),
                             ("protocol", design or ranking_design())]:
            self.approve(stage, reply)
        return self.approve("review_protocol")

    def test_new_question_repairs_code_then_produces_verified_acl_manuscript(self):
        self.assertEqual(self.through_protocol()["stage"], IMPLEMENTATION_STAGE)
        invalid = self.approve(IMPLEMENTATION_STAGE, response("def predict(features, levels, seed): return 2"))
        self.assertEqual(invalid["block"]["reason"], "invalid_implementation")
        self.assertIn("witness 2 failed", invalid["block"]["message"])
        self.assertFalse((self.root / "pilot-01/jobs.sqlite3").exists())
        self.approve(IMPLEMENTATION_STAGE)  # accepts the bounded return to code preparation
        self.assertEqual(len(self.provider.start_calls), 4)
        status = self.approve(IMPLEMENTATION_STAGE, response(PROGRAM))
        request = json.loads(self.provider.start_calls[-1]["prompt"])
        self.assertIn("witness 2 failed", request["repair_feedback"])
        self.assertNotIn("cases", request)
        plan = status["pending"]["details"]
        self.assertEqual(status["stage"], PROGRAM_STAGE)
        self.assertEqual(plan["evaluations"], 12)
        self.assertFalse(plan["network"])
        study = read_document(self.root / "pilot-02/study.json")
        self.assertEqual(study["protocol"]["version"], 1)
        self.assertEqual(study["version"], 2)
        job = plan["jobs"][0]["spec"]
        self.assertEqual(job["source_design_sha256"], plan["protocol_sha256"])
        self.assertNotIn("expected", job["workload"]["parameters"]["cases"][0])
        status = self.approve(PROGRAM_STAGE)
        self.assertEqual(status["stage"], "review_manuscript", status)
        report = read_document(status["report"]["path"])
        self.assertEqual(report["contrasts"]["H1"]["successes_a"], 2)
        self.assertEqual(report["contrasts"]["H1"]["successes_b"], 6)
        self.assertEqual(report["contrasts"]["H1"]["pairs"], 6)
        self.assertFalse(report["scientific_evidence"])
        draft = Path(status["draft"]["draft"]).parent
        paper = verify_manuscript_draft(ReviewStore(draft.parent / "review.sqlite3", read_only=True),
                                        status["draft"]["review_id"], draft)
        self.assertTrue(any(row.reference.pointer == "/analysis/contrasts/H1" for row in paper.statements))
        self.assertIn("pilote numérique sur CPU", render_acl(paper, approved=False))
        finished = self.approve("review_manuscript")
        self.assertEqual(finished["status"], "complete")

    def test_incompatible_protocol_returns_to_design_before_calling_code_agent(self):
        design = ranking_design()
        design["task_suite"]["source_type"] = "external_unverified"
        status = self.through_protocol(design=design)
        self.assertEqual(status["block"]["reason"], "incompatible_evaluation")
        self.assertEqual(status["pending"]["details"]["proposal"]["stage"], "protocol")
        self.assertEqual(len(self.provider.start_calls), 3)

    def test_one_delegation_covers_code_repair_and_measurement_to_draft(self):
        self.workflow.start("ranking", BRIEF, self.settings, evaluation=evaluation_set())
        replies = (MISSION, valid_review(), ranking_design(),
                   response("def predict(features, levels, seed): return 2"), response(PROGRAM))
        self.provider.responses.extend(LLMResponse(text=json.dumps(reply)) for reply in replies)
        budget = AutonomyBudget(llm_calls=5)
        plan = autonomy_plan(self.workflow, budget)
        authorize_autonomy(self.workflow, budget, Decision(actor="operator", target_sha256=plan["target_sha256"]))
        status = run_autonomously(self.workflow)
        self.assertEqual(status["stage"], "review_manuscript", status)
        self.assertTrue(status["draft"])
        self.assertIn("validation éditoriale", status["autonomy_stop"])
        history = self.workflow.store.history()
        self.assertEqual(sum(event["action"] == "authorize_autonomy" for event in history), 1)
        reservations = [event["detail"]["gate"]["stage"] for event in history if event["action"] == "reserve"]
        self.assertEqual(reservations.count(IMPLEMENTATION_STAGE), 2)
        self.assertEqual(reservations.count(PROGRAM_STAGE), 1)

    def test_modified_plan_blocks_submission(self):
        self.through_protocol()
        gate = self.approve(IMPLEMENTATION_STAGE, response(PROGRAM))["pending"]
        path = self.root / "pilot-01/action-plan.json"
        plan = read_document(path)
        plan["evaluations"] += 1
        path.write_text(json.dumps(plan), encoding="utf-8")
        with patch("research.brief_workflow.run_program_plan") as run:
            with self.assertRaises(ValueError):
                self.workflow.approve(Decision(actor="operator", target_sha256=gate["target_sha256"]))
            run.assert_not_called()
        self.assertFalse(path.with_name("jobs.sqlite3").exists())

    def test_resume_uses_completed_receipts_without_resubmitting(self):
        self.through_protocol()
        gate = self.approve(IMPLEMENTATION_STAGE, response(PROGRAM))["pending"]
        with patch.object(self.workflow, "_complete_program", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.approve(PROGRAM_STAGE)
        job_id = gate["details"]["jobs"][0]["spec"]["job_id"]
        ledger = DurableJobManager(path=self.root / "pilot-01/jobs.sqlite3", read_only=True)
        self.assertEqual(ledger.get(job_id).phase, "succeeded")
        workflow = BriefWorkflow(self.root)
        with patch("research.brief_workflow.run_program_plan") as run:
            status = workflow.recover(Decision(actor="operator", target_sha256=workflow.status()["checkpoint_sha256"]))
            self.assertEqual(status["stage"], "review_manuscript", status)
            run.assert_not_called()

    def test_restricted_program_rejects_imports_calls_and_unknown_inputs(self):
        for source in ("import time\n" + PROGRAM,
                       "def predict(features, levels, seed): return int(1)",
                       "def predict(features, levels, seed): return features['missing']"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                RestrictedProgram(source, features=["overlap_a"], levels=["method"])
        bounded = RestrictedProgram("def predict(features, levels, seed): return seed % 2", features=(), levels=())
        self.assertEqual(bounded.predict({}, {}, 2**32 - 1), 1)

    def ready_gpu(self):
        evaluation = evaluation_set()
        evaluation["schema_version"] = "gpu.integer_prompt.v1"
        for case in evaluation["cases"]:
            case["prompts"] = [f"Variante A : classe les documents du cas {case['id']}.",
                               f"Variante B : classe les documents du cas {case['id']}."]
        for idx, witness in enumerate(evaluation["witnesses"]):
            witness["expected"] = idx
        self.through_protocol(evaluation=evaluation, gpu_config=GPU_CONFIG)
        status = self.approve(IMPLEMENTATION_STAGE, response(GPU_PROGRAM))
        self.assertEqual(status["stage"], PROGRAM_STAGE, status)
        return status["pending"]

    def test_same_transition_prepares_pinned_model_and_runs_gpu_plan(self):
        gate = self.ready_gpu()
        plan = gate["details"]
        self.assertEqual(plan["model"]["revision"], REVISION)
        self.assertEqual(plan["worker_sha256"], worker_sha256())
        job = plan["jobs"][0]
        expected = [trial["prompt"] for trial in job["spec"]["workload"]["parameters"]["trials"]]
        calls = []

        def transport(command, **options):
            request = json.loads(options["input"])
            calls.append(request["phase"])
            self.assertEqual(request["prompts"], expected if request["phase"] == "measurement" else [])
            self.assertEqual(request["job_sha256"], job["sha256"])
            self.assertIn("protocol_gpu_worker.py", command[-1])
            base = {key: request[key] for key in ("job_id", "job_sha256", "worker_sha256")}
            ledger = self.root / "pilot-01/jobs.sqlite3"
            if request["phase"] == "preflight":
                self.assertFalse(ledger.exists())
                reply = {**base, "snapshot": request["snapshot"], "ready": True}
            else:
                self.assertEqual(DurableJobManager(path=ledger, read_only=True).get(request["job_id"]).phase,
                                 "submission_reserved")
                reply = {**base, "model": MODEL, "revision": REVISION, "responses": [
                    {"index": idx, "text": "2" if "second" in prompt else "1",
                     "input_tokens": 25, "output_tokens": 2, "elapsed_ns": 1000}
                    for idx, prompt in enumerate(request["prompts"])],
                    "runtime": {"gpu_index": 0, "gpu_name": "Fixture A5000", "cuda_version": "12.6", "torch_version": "2.6",
                                "transformers_version": "4.57", "elapsed_ms": 100, "peak_vram_bytes": 9000}}
            return CompletedProcess(command, 0, json.dumps(reply).encode(), b"")

        with patch("execution.ood_gpu.subprocess.run", side_effect=transport):
            status = self.approve(PROGRAM_STAGE)
        self.assertEqual(calls, ["preflight", "measurement"])
        self.assertEqual(status["stage"], "review_manuscript", status)
        report = read_document(status["report"]["path"])
        self.assertEqual(report["gpu_model"], MODEL)
        self.assertEqual(len(report["trials"]), 12)
        self.assertEqual(report["trials"][0]["model_tokens"], 27)
        self.assertEqual(report["contrasts"]["H1"]["gain_points"], 0)

    def test_missing_snapshot_returns_to_protocol_without_submitting(self):
        self.ready_gpu()

        def transport(command, **options):
            request = json.loads(options["input"])
            self.assertEqual(request["phase"], "preflight")
            reply = {key: request[key] for key in ("job_id", "job_sha256", "worker_sha256", "snapshot")}
            return CompletedProcess(command, 0, json.dumps({**reply, "ready": False}).encode(), b"")

        with patch("execution.ood_gpu.subprocess.run", side_effect=transport) as ssh:
            status = self.approve(PROGRAM_STAGE)
            ssh.assert_called_once()
        self.assertEqual(status["block"]["reason"], "gpu_snapshot_unavailable")
        self.assertFalse(status["block"]["uncertain"])
        self.assertEqual(status["pending"]["details"]["proposal"]["stage"], "protocol")
        self.assertFalse((self.root / "pilot-01/jobs.sqlite3").exists())

    def test_standalone_worker_checks_snapshot_without_cuda_for_preflight(self):
        snapshot = self.root / REVISION
        snapshot.mkdir(parents=True)
        request = {"phase": "preflight", "job_id": "fixture-001", "job_sha256": "a" * 64,
                   "worker_sha256": worker_sha256(), "snapshot": str(snapshot), "gpu_index": 0, "prompts": []}
        payload = b"fixture"
        manifest = {"config.json": (len(payload), hashlib.sha256(payload).hexdigest())}
        with patch("execution.protocol_gpu_worker.MODEL_FILES", manifest):
            self.assertFalse(worker_execute(request)["ready"])
            (snapshot / "config.json").write_bytes(payload)
            self.assertTrue(worker_execute(request)["ready"])
            (snapshot / "config.json").write_bytes(b"altered")
            self.assertFalse(worker_execute(request)["ready"])

    def test_hugging_face_cdn_redirect_stays_on_trusted_https(self):
        handler = _HuggingFaceRedirect()
        request = Request("https://huggingface.co/approved-model/resolve/pinned/weights.safetensors")
        signed_cdn = "https://us.aws.cdn.hf.co/pinned/weights.safetensors"
        result = handler.redirect_request(request, None, 302, "", {}, signed_cdn)
        self.assertEqual(result.full_url, signed_cdn)
        for url in (signed_cdn.replace("https:", "http:"), signed_cdn.replace(".hf.co", ".hf.co.example.org")):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "Unexpected model download redirect"):
                handler.redirect_request(request, None, 302, "", {}, url)

    def test_brief_only_selects_a_versioned_corpus_in_one_delegation(self):
        self.workflow.start("ranking", BRIEF, self.settings)
        choice = {"resource_id": RANKING_ID, "question_excerpt": QUESTION,
                  "rationale": "Le corpus compare les deux méthodes de classement demandées."}
        self.provider.responses.extend(LLMResponse(text=json.dumps(reply)) for reply in (
            MISSION, valid_review(), choice, ranking_design(), response(PROGRAM)))
        budget = AutonomyBudget(llm_calls=5)
        plan = autonomy_plan(self.workflow, budget)
        self.assertEqual([row["id"] for row in plan["details"]["scope"]["resource_catalog"]], [RANKING_ID])
        authorize_autonomy(self.workflow, budget, Decision(actor="operator", target_sha256=plan["target_sha256"]))
        status = run_autonomously(self.workflow)
        self.assertEqual(status["stage"], "review_manuscript", status)
        chosen = read_document(self.root / "versions/v01/resource-selection.json")
        execution = read_document(self.root / "pilot-01/action-plan.json")
        self.assertEqual(execution["resource_id"], RANKING_ID)
        self.assertEqual(execution["evaluation_sha256"], chosen["resource"]["evaluation_sha256"])
        request = json.loads(self.provider.start_calls[2]["prompt"])
        self.assertNotIn("cases", request["resources"][0])
        self.assertNotIn("witnesses", request["resources"][0])

    def test_no_resource_preserves_question_and_accepts_a_supplied_evaluation(self):
        self.workflow.start("ranking", BRIEF, self.settings)
        for stage, reply in [("mission", MISSION), ("review_mission", None), ("catalog", None),
                             ("literature", valid_review()), ("review_literature", None)]:
            self.approve(stage, reply)
        status = self.approve(RESOURCE_STAGE, {"resource_id": None, "question_excerpt": QUESTION,
                                              "rationale": "Le sélecteur s'abstient dans cette fixture de retour."})
        self.assertEqual(status["stage"], RESOLUTION_STAGE)
        self.assertIsNone(status["pending"]["details"]["gap"]["decision"]["resource_id"])
        self.assertFalse((self.root / "pilot-01/jobs.sqlite3").exists())
        status = self.workflow.revise(Revision(stage="protocol", actor="operator", reason="Ajouter le corpus de classement.",
                                               evaluation=evaluation_set(), target_sha256=status["checkpoint_sha256"]))
        self.assertEqual(status["stage"], "protocol")

    def start_repair(self):
        mission, gap, repair = repair_replies()
        self.workflow.start("comparison", REPAIR_BRIEF, self.settings)
        for stage, reply in [("mission", mission), ("review_mission", None), ("catalog", None),
                             ("literature", valid_review()), ("review_literature", None), (RESOURCE_STAGE, gap)]:
            self.approve(stage, reply)
        return repair

    def test_autonomy_resolves_a_missing_resource_and_resumes_to_the_verified_draft(self):
        mission, gap, repair = repair_replies()
        self.workflow.start("comparison", REPAIR_BRIEF, self.settings)
        self.provider.responses.extend(LLMResponse(text=json.dumps(reply)) for reply in (
            mission, valid_review(), gap, repair, repair_design(), response(REPAIR_PROGRAM)))
        budget = AutonomyBudget(llm_calls=6)
        plan = autonomy_plan(self.workflow, budget)
        authorize_autonomy(self.workflow, budget, Decision(actor="operator", target_sha256=plan["target_sha256"]))
        status = run_autonomously(self.workflow)
        self.assertEqual(status["stage"], "review_manuscript", status)
        self.assertEqual(len(self.provider.start_calls), 6)
        self.assertEqual(sum(event["action"] == "reserve" and event["detail"]["gate"]["stage"] == RESOLUTION_STAGE
                             for event in self.workflow.store.history()), 1)
        version = self.root / "versions/v01"
        observed_gap = read_document(version / "resource-selection.json")
        resolution = read_document(version / "capability-resolution.json")
        self.assertEqual(resolution["gap"], observed_gap)
        self.assertEqual(resolution["decision"], repair)
        report = read_document(status["report"]["path"])
        self.assertFalse(report["scientific_evidence"])
        self.assertEqual(report["contrasts"]["H1"]["successes_a"], 2)
        self.assertEqual(report["contrasts"]["H1"]["successes_b"], 12)
        execution = read_document(self.root / "pilot-01/action-plan.json")
        self.assertEqual(execution["capability_resolution"], resolution)
        self.assertFalse(execution["network"])
        self.assertFalse(execution["gpu"])
        implementation = json.loads(self.provider.start_calls[-1]["prompt"])
        self.assertNotIn("cases", implementation)
        self.assertEqual(len(implementation["witnesses"]), 6)

    def test_unresolved_prerequisite_is_reported_after_restart(self):
        repair = self.start_repair()
        requirement = "Obtenir les droits de lecture du corpus requis pour le pilote."
        status = self.approve(RESOLUTION_STAGE, dict(repair, kind="external_prerequisite",
                                                    next_requirement=requirement, recipe=None))
        self.assertEqual(status["block"]["reason"], "capability_unresolved")
        self.assertEqual(status["block"]["message"], requirement)
        workflow = BriefWorkflow(self.root, provider_factory=lambda _: self.fail("unexpected provider"))
        self.assertIsNone(workflow.status()["pending"])
        self.assertEqual(workflow.status()["block"], status["block"])
        self.assertFalse((self.root / "pilot-01/jobs.sqlite3").exists())

    def test_resolution_respects_the_delegated_llm_budget(self):
        mission, gap, _ = repair_replies()
        self.workflow.start("comparison", REPAIR_BRIEF, self.settings)
        self.provider.responses.extend(LLMResponse(text=json.dumps(reply)) for reply in (mission, valid_review(), gap))
        budget = AutonomyBudget(llm_calls=3)
        plan = autonomy_plan(self.workflow, budget)
        authorize_autonomy(self.workflow, budget, Decision(actor="operator", target_sha256=plan["target_sha256"]))
        status = run_autonomously(self.workflow)
        self.assertEqual(status["stage"], RESOLUTION_STAGE)
        self.assertIn("Budget", status["autonomy_stop"])
        self.assertEqual(len(self.provider.start_calls), 3)
        self.assertFalse((self.root / "versions/v01/capability-resolution.json").exists())

    def test_invalid_recipe_does_not_create_jobs(self):
        repair = self.start_repair()
        repair["recipe"]["method_a"] = "download_corpus"
        status = self.approve(RESOLUTION_STAGE, repair)
        self.assertEqual(status["block"]["reason"], "invalid_capability_resolution")
        self.assertIsNone(status["pending"])
        self.assertFalse((self.root / "versions/v01/resolved-resource.json").exists())

    def test_altered_repaired_corpus_blocks_protocol_after_restart(self):
        repair = self.start_repair()
        self.assertEqual(self.approve(RESOLUTION_STAGE, repair)["stage"], "protocol")
        path = self.root / "versions/v01/resolved-evaluation.json"
        corpus = read_document(path)
        corpus["cases"][0]["expected"] = 0
        path.write_text(json.dumps(corpus), encoding="utf-8")
        workflow = BriefWorkflow(self.root, provider_factory=lambda _: self.fail("unexpected provider"))
        self.assertIsNone(workflow.status()["pending"])
        self.assertEqual(workflow.status()["status"], "blocked")

    def test_snapshot_is_published_only_after_every_registered_byte_is_verified(self):
        snapshot = self.root / REVISION
        self.root.mkdir()
        payload = b"approved artifact"
        manifest = {"config.json": (len(payload), hashlib.sha256(payload).hexdigest())}
        with patch("execution.protocol_gpu_worker.MODEL_FILES", manifest), patch("execution.protocol_gpu_worker.disk_usage") as disk:
            disk.return_value.free = len(payload)
            transport = build_opener()
            with patch("urllib.request.OpenerDirector.open", return_value=BytesIO(b"corrupt")):
                with self.assertRaisesRegex(ValueError, "approved revision"):
                    _install(snapshot, transport)
                self.assertFalse(snapshot.exists())
            with patch("urllib.request.OpenerDirector.open", return_value=BytesIO(payload)) as opener:
                self.assertTrue(_install(snapshot, transport))
                self.assertTrue((snapshot / "config.json").is_file())
                self.assertFalse(_install(snapshot, transport))
                opener.assert_called_once()
            tokenizer = b"fixed tokenizer"
            manifest["tokenizer.json"] = (len(tokenizer), hashlib.sha256(tokenizer).hexdigest())
            disk.return_value.free = len(tokenizer)
            with patch("urllib.request.OpenerDirector.open", return_value=BytesIO(tokenizer)) as opener:
                self.assertTrue(_install(snapshot, transport))
                opener.assert_called_once()
                self.assertTrue(opener.call_args.args[0].full_url.endswith("/tokenizer.json"))
            self.assertEqual((snapshot / "config.json").read_bytes(), payload)
            self.assertEqual((snapshot / "tokenizer.json").read_bytes(), tokenizer)

    def test_auto_gpu_selects_model_and_provisions_before_measurement(self):
        question = "La mise en forme structurée des consignes améliore-t-elle l'exactitude des additions de Qwen ?"
        brief = BRIEF.replace(QUESTION, question)
        mission = {**MISSION, "research_question": question}
        design = ranking_design()
        design["factors"][0].update(name="prompt_style", levels=["direct", "structured"])
        design["baselines"][0]["factor_levels"] = {"prompt_style": "direct"}
        design["hypothesis_tests"][0].update(factor_name="prompt_style", level_a="direct", level_b="structured")
        candidate = response('def predict(features, levels, seed): return 1 if levels["prompt_style"] == "structured" else 0')
        choice = {"resource_id": ARITHMETIC_ID, "question_excerpt": question,
                  "rationale": "Les deux formats de consigne et le calcul d'addition correspondent au test."}
        config = {key: value for key, value in GPU_CONFIG.items() if key != "snapshot_path"}
        config.update(model_cache="/opt/pilot/models", allow_model_download=True)
        self.workflow.start("addition", brief, self.settings, gpu_config=config)
        self.provider.responses.extend(LLMResponse(text=json.dumps(reply)) for reply in (
            mission, valid_review(), choice, design, candidate))
        budget = AutonomyBudget(llm_calls=5)
        delegation = autonomy_plan(self.workflow, budget)
        gpu_limits = delegation["details"]["scope"]["execution"]["capabilities"]["gpu.integer_prompt.v1"]
        self.assertEqual(gpu_limits["model_provisioning"]["attempts"], 1)
        authorize_autonomy(self.workflow, budget, Decision(actor="operator", target_sha256=delegation["target_sha256"]))
        calls = []

        def transport(command, **options):
            request = json.loads(options["input"])
            phase = request["phase"]
            calls.append(phase)
            binding = {key: request[key] for key in ("job_id", "job_sha256", "worker_sha256")}
            if phase in {"preflight", "provision"}:
                self.assertFalse((self.root / "pilot-01/jobs.sqlite3").exists())
                self.assertEqual(request["prompts"], [])
                reply = {**binding, "snapshot": f"/opt/pilot/models/{REVISION}", "ready": phase == "provision"}
                if phase == "provision":
                    reply.update(installed=True, manifest_sha256=manifest_sha256())
            else:
                self.assertEqual(len(request["prompts"]), 12)
                reply = {**binding, "model": MODEL, "revision": REVISION, "responses": [
                    {"index": idx, "text": "46", "input_tokens": 25, "output_tokens": 2, "elapsed_ns": 1000}
                    for idx in range(12)],
                    "runtime": {"gpu_index": 0, "gpu_name": "Fixture A5000", "cuda_version": "12.6", "torch_version": "2.6",
                                "transformers_version": "4.57", "elapsed_ms": 100, "peak_vram_bytes": 9000}}
            return CompletedProcess(command, 0, json.dumps(reply).encode(), b"")

        with patch("execution.ood_gpu.subprocess.run", side_effect=transport):
            status = run_autonomously(self.workflow)
        self.assertEqual(calls, ["preflight", "provision", "measurement"])
        self.assertEqual(status["stage"], "review_manuscript", status)
        self.assertTrue(read_document(self.root / "pilot-01/model-preparation.json")["installed"])
        self.assertIn("model_preparation_sha256", read_document(status["report"]["path"]))
