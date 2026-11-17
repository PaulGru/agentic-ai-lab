"""V8.3 — Offline validation, LLM stub, HITL and artifact lineage tests."""

import json
import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from agent.models import LLMResponse, ToolCall
from research.design_graph import build_design_graph, pending_design_approval
from research.design_planner import ExperimentalDesignPlanner
from research.experimental_design import (
    ExperimentalDesign,
    render_design_markdown,
    validate_approved_literature,
    validate_design_against_inputs,
)
from research.literature import LiteratureReview, digest_json
from research.mission import ResearchMission
from scripts.demo_v83 import config_for, export_approved_design
from tests.fakes import FakeProvider
from tests.test_v82_literature import approved_document, fake_catalog, valid_review


def approved_inputs():
    mission = approved_document()
    catalog = fake_catalog().model_dump()
    review = valid_review()
    literature = {
        "schema_version": "8.2",
        "status": "approved",
        "mission_sha256": mission["mission_sha256"],
        "catalog_sha256": digest_json(catalog),
        "review_sha256": digest_json(review),
        "catalog": catalog,
        "review": review,
        "verification_level": "catalog_metadata_and_exact_abstract_excerpts_only",
    }
    return mission, literature


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


class V83DesignTests(unittest.TestCase):
    def test_happy_path_budget_matrix_and_lineage(self):
        mission_document, literature_document = approved_inputs()
        mission, _, review = validate_approved_literature(mission_document, literature_document)
        design = ExperimentalDesign.model_validate(valid_design())
        validate_design_against_inputs(design, mission, review)
        self.assertEqual(design.design_cells, 4)
        self.assertEqual(design.total_evaluations, 24)
        self.assertEqual(design.worst_case_llm_calls, 48)
        self.assertEqual([x["cell_id"] for x in design.factor_matrix()], ["C001", "C002", "C003", "C004"])
        self.assertEqual(design.factor_matrix()[0]["factor_levels"], {"strategy": "no_retry", "noise": "clean"})
        markdown = render_design_markdown(design, mission, literature_document)
        self.assertIn("Aucune expérience", markdown)
        self.assertIn("C001", markdown)

    def test_rejects_changed_input_and_wrong_mission(self):
        mission, literature = approved_inputs()
        literature["review"]["overview"] = "Résumé modifié après approbation."
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_approved_literature(mission, literature)
        mission, literature = approved_inputs()
        literature["mission_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different research mission"):
            validate_approved_literature(mission, literature)

    def test_only_approved_review_is_accepted(self):
        mission, literature = approved_inputs()
        literature["status"] = "awaiting_approval"
        with self.assertRaisesRegex(ValueError, "approved"):
            validate_approved_literature(mission, literature)

    def test_nonexistent_citation_cannot_justify_protocol(self):
        mission, literature = approved_inputs()
        design = valid_design()
        design["literature_links"][0]["work_id"] = "W999999"
        with self.assertRaisesRegex(ValueError, "not supported"):
            validate_design_against_inputs(
                ExperimentalDesign.model_validate(design),
                ResearchMission.model_validate(mission["mission"]),
                LiteratureReview.model_validate(literature["review"]),
            )

    def test_fails_if_hypothesis_not_covered(self):
        mission, literature = approved_inputs()
        design = ExperimentalDesign.model_validate({**valid_design(), "hypothesis_tests": [{**valid_design()["hypothesis_tests"][0], "hypothesis_id": "H2"}]})
        _, _, review = validate_approved_literature(mission, literature)
        with self.assertRaisesRegex(ValueError, "every"):
            validate_design_against_inputs(design, ResearchMission.model_validate(mission["mission"]), review)

    def test_invalid_or_uncontrolled_contrasts_are_rejected(self):
        for changes in (
            {"level_b": "unknown"},
            {"level_b": "no_retry"},
            {"fixed_levels": {"noise": "unknown"}},
            {"fixed_levels": {}},
            {"comparison_scope": "marginal_paired", "fixed_levels": {"noise": "clean"}},
        ):
            with self.subTest(changes=changes):
                data = valid_design()
                data["hypothesis_tests"][0].update(changes)
                with self.assertRaises(ValidationError):
                    ExperimentalDesign.model_validate(data)

    def test_budget_and_baseline_restrictions(self):
        for mutation in ("budget", "baseline", "seeds", "results"):
            with self.subTest(mutation=mutation):
                data = valid_design()
                if mutation == "budget":
                    data["resource_limits"]["max_total_llm_calls"] = 47
                elif mutation == "baseline":
                    data["baselines"][0]["factor_levels"].pop("noise")
                elif mutation == "seeds":
                    data["resource_limits"]["seeds"] = [42, 42]
                elif mutation == "results":
                    data["results_status"] = "completed"
                with self.assertRaises(ValidationError):
                    ExperimentalDesign.model_validate(data)

    def test_planner_no_local_tools_and_uses_approved_evidence(self):
        mission, literature = approved_inputs()
        m, _, review = validate_approved_literature(mission, literature)
        provider = FakeProvider([LLMResponse(text=json.dumps(valid_design()))])
        result = ExperimentalDesignPlanner(provider).propose(m, review)
        self.assertEqual(result.results_status, "not_executed")
        self.assertEqual(provider.start_calls[0]["tools"], [])
        self.assertEqual(provider.continue_calls, [])
        payload = json.loads(provider.start_calls[0]["prompt"])
        self.assertEqual(len(payload["literature"]["findings"]), 1)

    def test_planner_rejects_tool_calls(self):
        m, _, review = validate_approved_literature(*approved_inputs())
        fake = FakeProvider([LLMResponse(tool_calls=[ToolCall(id="x", name="calculate", arguments={})])])
        with self.assertRaisesRegex(ValueError, "without tools"):
            ExperimentalDesignPlanner(fake).propose(m, review)

    def test_interrupt_reconstruction_reject_and_approved_exports(self):
        mission, literature = approved_inputs()
        memory = InMemorySaver()
        factory = lambda: FakeProvider([LLMResponse(text=json.dumps(valid_design()))])
        graph = build_design_graph(memory, provider_factory=factory)
        config = config_for("design-test")
        initial = graph.invoke({"mission_document": mission, "literature_document": literature}, config=config, durability="sync")
        self.assertIn("__interrupt__", initial)
        fresh = build_design_graph(memory, provider_factory=lambda: FakeProvider([]))
        pending = pending_design_approval(fresh.get_state(config))
        self.assertIsNotNone(pending)
        approved = fresh.invoke(Command(resume={
            "decision": "approve",
            "mission_sha256": pending["mission_sha256"],
            "literature_input_sha256": pending["literature_input_sha256"],
            "design_sha256": pending["design_sha256"],
        }), config=config, durability="sync")
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(fresh.get_state(config).next, ())
        with tempfile.TemporaryDirectory() as d:
            paths = export_approved_design(fresh.get_state(config).values, "design-test", Path(d))
            self.assertEqual(len(paths), 3)
            self.assertEqual(paths, export_approved_design(fresh.get_state(config).values, "design-test", Path(d)))
            self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8"))["execution_status"], "not_executed")
            self.assertEqual(len(json.loads(paths[2].read_text(encoding="utf-8"))["cells"]), 4)

        cfg_bad = config_for("design-wrong-digest")
        wrong = build_design_graph(memory, provider_factory=factory)
        wrong.invoke({"mission_document": mission, "literature_document": literature}, config=cfg_bad)
        bad_pending = pending_design_approval(wrong.get_state(cfg_bad))
        wrong.invoke(Command(resume={
            "decision": "approve",
            "mission_sha256": bad_pending["mission_sha256"],
            "literature_input_sha256": bad_pending["literature_input_sha256"],
            "design_sha256": "f" * 64,
        }), config=cfg_bad)
        self.assertEqual(wrong.get_state(cfg_bad).values["status"], "rejected")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                export_approved_design(wrong.get_state(cfg_bad).values, "design-wrong-digest", Path(d))


if __name__ == "__main__":
    unittest.main()
