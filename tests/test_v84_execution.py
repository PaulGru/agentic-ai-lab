"""V8.4 deterministic tests: never call external APIs."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from research.execution_contract import (
    binding_template, prepare_canary_plan, verify_approved_design,
)
from research.canary_runtime import (
    LLMCallResult, arithmetic_task, execute_canary, extract_integer_answer,
)
from research.experimental_design import ExperimentalDesign
from research.literature import digest_json
def valid_design():
    return {
        "title": "Pilote comparatif récupération d'erreurs agents LLM",
        "objective": "Estimer par tâches appariées le différentiel de réussite sous erreur d'outil entre politiques avec et sans récupération.",
        "stage": "exploratory_pilot",
        "hypothesis_tests": [{
            "hypothesis_id": "H1",
            "factor_name": "strategy",
            "primary_metric_id": "task_success",
            "contrast": "Comparer retry_once à no_retry sous tool_error, à tâches et graines identiques.",
            "level_a": "no_retry",
            "level_b": "retry_once",
            "comparison_scope": "conditional_paired",
            "fixed_levels": {"noise": "tool_error"},
            "null_hypothesis": "Le différentiel de réussite moyen de retry_once moins no_retry est inférieur ou égal à zéro.",
            "falsification_rule": "Si l'estimation et son intervalle compatible ne soutiennent pas un gain, ne pas retenir la prédiction.",
        }],
        "factors": [
            {
                "name": "strategy", "kind": "orchestration", "levels": ["no_retry", "retry_once"],
                "operationalization": "Même agent et outils, seule la politique d'une reprise contrôlée après erreur varie.",
            },
            {
                "name": "noise", "kind": "perturbation", "levels": ["clean", "tool_error"],
                "operationalization": "Injecter une réponse d'outil erronée prédéfinie en condition tool_error ; laisser intact en clean.",
            },
        ],
        "baselines": [
            {
                "name": "Agent sans récupération nominal",
                "factor_levels": {"strategy": "no_retry", "noise": "clean"},
                "rationale": "Condition nominale nécessaire pour estimer le taux de réussite de base.",
            },
            {
                "name": "Agent sans récupération perturbé",
                "factor_levels": {"strategy": "no_retry", "noise": "tool_error"},
                "rationale": "Référence perturbée pour calculer l'effet additionnel d'une reprise.",
            },
        ],
        "task_suite": {
            "name": "Mini suite locale de tâches à outils",
            "source_type": "synthetic_local",
            "provenance_or_construction": "Trois tâches synthétiques validables par assertions Python standard, à examiner avant exécution.",
            "exact_scoring_rule": "Réussite 1 uniquement si la sortie finale correspond à la valeur de référence déterministe.",
            "eligibility_and_exclusion": "Tâches sans données personnelles ni dépendance externe ; exclusions définies avant analyse.",
            "licensing_verified": False,
            "access_verified": False,
        },
        "metrics": [
            {
                "id": "task_success", "operational_definition": "Nombre de sorties finales correctes divisé par le nombre total de tâches assignées.",
                "unit": "binary", "aggregation": "Moyenne par condition et différence de moyennes appariées.",
                "missing_or_failed_run_rule": "Tout échec d'appel ou timeout est compté comme tâche échouée.",
            },
            {
                "id": "total_tokens", "operational_definition": "Somme des tokens entrée et sortie remontés par les providers pour une évaluation.",
                "unit": "tokens", "aggregation": "Médiane et distribution par condition, avec tâches appariées.",
                "missing_or_failed_run_rule": "Tracer les tokens manquants comme manquants sans les imputer à zéro.",
            },
            {
                "id": "latency_ms", "operational_definition": "Durée murale depuis le début jusqu'à la réponse finale ou à l'échec.",
                "unit": "milliseconds", "aggregation": "Médiane et quantiles sur toutes les évaluations de chaque condition.",
                "missing_or_failed_run_rule": "Les timeouts sont censurés au seuil fixé et déclarés séparément.",
            },
        ],
        "controlled_variables": [
            "Conserver exactement le même modèle, version, prompts et outils pour toutes les conditions.",
            "Répéter les mêmes tâches et graines par condition et comparer leurs résultats par paires.",
        ],
        "perturbation_and_recovery_procedure": "Une erreur d'outil injectée au premier appel, avec au plus une reprise autorisée dans retry_once.",
        "randomization_and_counterbalancing": "Ordre des cellules permuté de façon déterministe par graine avant les exécutions.",
        "analysis_plan": {
            "pairing_strategy": "Apparier les observations par tâche, graine et perturbation identiques.",
            "primary_estimand": "Différence absolue de taux de réussite entre retry_once et no_retry sous tool_error.",
            "uncertainty_method": "Bootstrap apparié par tâche et graine, avec limites d'un pilote de faible effectif.",
            "multiple_testing_policy": "H1 est le contraste primaire unique ; toutes les autres métriques sont descriptives.",
            "missing_data_policy": "Compter les échecs comme non-réussites ; séparer les coûts et latences manquants.",
            "stopping_rule": "Arrêter au plafond de tâches/graines/appels/tokens pré-déclaré, sans regarder les résultats.",
            "power_limitation": "Ce pilote ne fournit pas de justification de puissance pour une publication ACL.",
        },
        "resource_limits": {
            "tasks_per_cell": 3, "seeds": [13, 42],
            "max_llm_calls_per_evaluation": 2,
            "max_tokens_per_llm_call": 512,
            "max_total_llm_calls": 48,
            "max_total_tokens": 24576,
            "max_api_cost_usd": 5.0,
        },
        "literature_links": [{
            "work_id": "W101",
            "design_rationale": "L'extrait récupéré mentionne les erreurs d'outils, justifiant une condition tool_error.",
            "limitation": "Le résumé ne valide pas l'effet de la politique retry_once ni la méthodologie complète.",
        }],
        "feasibility_checks": [
            "Vérifier disponibilité et version exactes des modèles/providers avant le pilote.",
            "Vérifier les licences, la conformité et les coûts unitaires réels avant toute exécution.",
        ],
        "limitations": [
            "Petit jeu synthétique : aucune généralisation à des tâches NLP réelles ne peut être présumée.",
            "Deux graines seulement ; confiance limitée, protocole sujet à validation humaine.",
        ],
        "results_status": "not_executed",
    }


def inputs():
    design = ExperimentalDesign.model_validate(valid_design())
    raw = design.model_dump()
    document = {
        "schema_version": "8.3", "status": "approved", "execution_status": "not_executed",
        "mission_sha256": "a" * 64,
        "literature_sha256": "b" * 64,
        "design_sha256": digest_json(raw),
        "design": raw,
        "planned_cells": design.design_cells,
        "planned_evaluations": design.total_evaluations,
        "planned_worst_case_llm_calls": design.worst_case_llm_calls,
    }
    matrix = {
        "schema_version": "8.3", "design_sha256": document["design_sha256"],
        "execution_status": "not_executed",
        "cells": design.factor_matrix(),
        "tasks_per_cell": design.resource_limits.tasks_per_cell,
        "seeds": design.resource_limits.seeds,
    }
    bindings = {
        "schema_version": "8.4", "fixed_model": {"provider": "openai", "model": "test-model-no-network"},
        "factor_bindings": {
            "strategy": {"no_retry": "one_shot", "retry_once": "retry_once"},
            "noise": {"clean": "clean", "tool_error": "tool_error"},
        },
        "pricing": {"usd_per_million_input": 1.0, "usd_per_million_output": 5.0},
    }
    return document, matrix, bindings


class FakeModel:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, *, max_output_tokens):
        self.prompts.append(prompt)
        if "TOOL_OK:" in prompt:
            number = int(prompt.split("TOOL_OK: ")[-1].splitlines()[0].split()[0])
        elif "TOOL_OK_AFTER_RETRY:" in prompt:
            number = int(prompt.split("TOOL_OK_AFTER_RETRY: ")[-1].splitlines()[0].split()[0])
        else:
            number = -99999
        return LLMCallResult(text=json.dumps({"answer": number}), input_tokens=38, output_tokens=12)


class V84Tests(unittest.TestCase):
    def test_canary_preflight_and_lineage(self):
        document, matrix, bindings = inputs()
        plan = prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=2)
        self.assertEqual(plan["mode"], "engineering_canary_not_scientific_results")
        self.assertEqual(plan["worst_case_calls"], 2)
        self.assertEqual(plan["treatment"]["perturbation"], "clean")
        self.assertEqual(len(plan["task_seed_pairs"]), 2)
        self.assertLess(plan["conservative_estimated_usd"], 5)

    def test_rejects_unapproved_or_altered_design_matrix(self):
        document, matrix, bindings = inputs()
        document["status"] = "draft"
        with self.assertRaisesRegex(ValueError, "approved"):
            prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)
        document, matrix, bindings = inputs()
        matrix["cells"][0]["factor_levels"]["noise"] = "tool_error"
        with self.assertRaisesRegex(ValueError, "Matrix"):
            prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)

    def test_template_placeholder_fails_closed(self):
        document, matrix, bindings = inputs()
        template = binding_template(verify_approved_design(document, matrix))
        with self.assertRaises(ValidationError):
            prepare_canary_plan(document, matrix, template, cell_id="C001", evaluations=1)

    def test_unknown_or_collapsed_levels_rejected(self):
        document, matrix, bindings = inputs()
        bindings["factor_bindings"]["strategy"]["retry_once"] = "one_shot"
        with self.assertRaisesRegex(ValueError, "distinct treatments"):
            prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)
        document, matrix, bindings = inputs()
        bindings["factor_bindings"]["strategy"]["retry_once"] = "magic_agent"
        with self.assertRaisesRegex(ValueError, "Unsupported treatment"):
            prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)

    def test_budget_preflight_blocks_before_api(self):
        document, matrix, bindings = inputs()
        bindings["pricing"]["usd_per_million_output"] = 1000.0
        document["design"]["resource_limits"]["max_api_cost_usd"] = 0.01
        document["design_sha256"] = digest_json(document["design"])
        matrix["design_sha256"] = document["design_sha256"]
        with self.assertRaisesRegex(ValueError, "cost budget"):
            prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=2)

    def test_model_level_bindings_required(self):
        document, matrix, bindings = inputs()
        other = valid_design()
        other["factors"][0]["kind"] = "model_scale"
        other["hypothesis_tests"][0]["fixed_levels"] = {"noise": "tool_error"}
        d = ExperimentalDesign.model_validate(other)
        document["design"] = d.model_dump()
        document["design_sha256"] = digest_json(document["design"])
        matrix["design_sha256"] = document["design_sha256"]
        matrix["cells"] = d.factor_matrix()
        bindings["fixed_model"] = None
        bindings["factor_bindings"]["strategy"] = {
            "no_retry": {"provider": "openai", "model": "small-model"},
            "retry_once": {"provider": "anthropic", "model": "big-model"},
        }
        p = prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)
        self.assertEqual(p["treatment"]["model"], {"provider": "openai", "model": "small-model"})

    def test_deterministic_paired_task_and_strict_scoring(self):
        a = arithmetic_task(2, 42)
        self.assertEqual(a, arithmetic_task(2, 42))
        self.assertNotEqual(a["expression"], arithmetic_task(2, 13)["expression"])
        self.assertIsNone(extract_integer_answer("Here is the answer: 42"))
        self.assertIsNone(extract_integer_answer('{"answer": true}'))
        self.assertEqual(extract_integer_answer('{"answer": 42}'), 42)

    def test_real_execution_path_with_fake_provider_and_no_duplicates(self):
        document, matrix, bindings = inputs()
        plan = prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=2)
        fake = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            receipt = execute_canary(plan, path, model_factory=lambda spec: fake)
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual([r["task_success"] for r in receipt["observations"]], [1, 1])
            self.assertEqual(receipt["total_calls"], 2)
            self.assertEqual(len(fake.prompts), 2)
            again = execute_canary(plan, path, model_factory=lambda spec: fake)
            self.assertEqual(again["total_calls"], 2)
            self.assertEqual(len(fake.prompts), 2)

    def test_tool_fault_recovery_makes_two_real_model_calls(self):
        document, matrix, bindings = inputs()
        # C004 = retry_once + tool_error in the V8.3 factorial order.
        plan = prepare_canary_plan(document, matrix, bindings, cell_id="C004", evaluations=1)
        fake = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            receipt = execute_canary(plan, Path(tmp) / "receipt.json", lambda spec: fake)
            self.assertEqual(receipt["status"], "completed")
            self.assertTrue(receipt["observations"][0]["tool_retry_triggered"])
            self.assertEqual(receipt["observations"][0]["task_success"], 1)
            self.assertEqual(receipt["total_calls"], 2)

    def test_unknown_inflight_never_auto_retries(self):
        document, matrix, bindings = inputs()
        plan = prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)
        fake = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_text(json.dumps({"status": "inflight_unknown_if_interrupted",
                                        "plan_sha256": digest_json(plan)}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing automatic replay"):
                execute_canary(plan, path, lambda spec: fake)
            self.assertEqual(len(fake.prompts), 0)

    def test_provider_usage_over_cap_stops_next_calls(self):
        document, matrix, bindings = inputs()
        plan = prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=2)
        class Greedy:
            def __init__(self): self.calls = 0
            def complete(self, prompt, *, max_output_tokens):
                self.calls += 1
                return LLMCallResult(text='{"answer": 1}', input_tokens=2000, output_tokens=1)
        fake = Greedy()
        with tempfile.TemporaryDirectory() as tmp:
            report = execute_canary(plan, Path(tmp) / "receipt.json", lambda spec: fake)
        self.assertEqual(report["status"], "budget_stopped")
        self.assertEqual(fake.calls, 1)
        self.assertEqual(report["total_calls"], 1)


try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from research.execution_graph import build_execution_graph, pending_execution_approval
except ImportError:
    InMemorySaver = None


@unittest.skipIf(InMemorySaver is None, "langgraph not installed in this test environment")
class V84GraphTests(unittest.TestCase):
    def test_interrupt_approval_rebuild_reject_or_execute(self):
        document, matrix, bindings = inputs()
        plan = prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)
        fake = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            saver = InMemorySaver()
            cfg = {"configurable": {"thread_id": "v84-canary:test"}}
            graph = build_execution_graph(saver, Path(tmp), lambda spec: fake)
            response = graph.invoke({"thread": "test", "plan": plan}, config=cfg)
            self.assertIn("__interrupt__", response)
            self.assertEqual(fake.prompts, [])
            restarted = build_execution_graph(saver, Path(tmp), lambda spec: fake)
            pending = pending_execution_approval(restarted.get_state(cfg))
            self.assertIsNotNone(pending)
            restarted.invoke(Command(resume={"decision": "approve", "plan_sha256": pending["plan_sha256"]}), config=cfg)
            self.assertEqual(len(fake.prompts), 1)
            self.assertEqual(restarted.get_state(cfg).values["status"], "completed")
            self.assertTrue((Path(tmp) / "test" / "v84_execution.json").exists())

    def test_reject_and_wrong_digest_do_not_dispatch(self):
        document, matrix, bindings = inputs()
        plan = prepare_canary_plan(document, matrix, bindings, cell_id="C001", evaluations=1)
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeModel()
            graph = build_execution_graph(InMemorySaver(), Path(tmp), lambda spec: fake)
            cfg = {"configurable": {"thread_id": "v84-canary:reject"}}
            graph.invoke({"thread": "reject", "plan": plan}, config=cfg)
            graph.invoke(Command(resume={"decision": "approve", "plan_sha256": "bad"}), config=cfg)
            self.assertEqual(graph.get_state(cfg).values["status"], "rejected")
            self.assertFalse(fake.prompts)


if __name__ == "__main__":
    unittest.main()
