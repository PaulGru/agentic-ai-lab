"""V8.1 — Tests sans appel LLM réel ni PostgreSQL."""

import json
import unittest
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from agent.models import LLMResponse, ToolCall
from research.mission import (
    ResearchMission, mission_digest, parse_brief, sha256_text,
)
from research.mission_graph import build_mission_graph, pending_approval
from research.mission_planner import MissionPlanner
from tests.fakes import FakeProvider


BRIEF_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "research_brief.md"
)


def valid_mission():
    return {
        "title": "Robustesse des agents LLM",
        "research_question": "Quels facteurs influencent la robustesse des agents LLM ?",
        "objective": "Estimer les effets des facteurs à protocole comparable",
        "hypotheses": [{
            "id": "H1",
            "claim": "Une orchestration explicite améliore le taux de réussite sous perturbations",
            "falsification_test": "L'effet mesuré est nul ou négatif à conditions identiques",
        }],
        "independent_variables": ["Stratégie d'orchestration"],
        "dependent_metrics": ["Taux de réussite des tâches"],
        "controls": ["Même jeu de tâches et mêmes outils"],
        "baselines": ["Agent mono-modèle sans récupération"],
        "literature_queries": ["LLM agent robustness evaluation", "tool use agent failures"],
        "experiment_outline": ["Définir le benchmark", "Comparer les systèmes"],
        "validity_risks": ["Budget de tokens inégal entre systèmes"],
        "success_criteria": ["Comparer les taux de réussite avec incertitude"],
        "deliverables": ["Rapport d'expériences reproductibles"],
    }


def cfg(name):
    return {"configurable": {"thread_id": f"v81-test:{name}"}}


class MissionV81Tests(unittest.TestCase):
    def setUp(self):
        self.brief = BRIEF_PATH.read_text(encoding="utf-8")

    def test_brief_required_sections(self):
        sections = parse_brief(self.brief)
        self.assertIn("Question", sections)
        with self.assertRaises(ValueError):
            parse_brief(self.brief.replace("## Contraintes", "## Limites"))

    def test_schema_rejects_invalid_hypotheses(self):
        invalid = valid_mission()
        invalid["hypotheses"] = invalid["hypotheses"] * 2
        with self.assertRaises(ValidationError):
            ResearchMission.model_validate(invalid)

    def test_no_tool_execution(self):
        provider = FakeProvider([LLMResponse(text=json.dumps(valid_mission()))])
        result = MissionPlanner(provider).propose(parse_brief(self.brief))
        self.assertEqual(result.hypotheses[0].id, "H1")
        self.assertEqual(provider.start_calls[0]["tools"], [])
        self.assertEqual(provider.continue_calls, [])

    def test_rejects_tool_calls(self):
        provider = FakeProvider([LLMResponse(tool_calls=[
            ToolCall(id="call1", name="calculate", arguments={"expression": "1+1"})
        ])])
        with self.assertRaises(ValueError):
            MissionPlanner(provider).propose(parse_brief(self.brief))

    def test_interrupt_and_resume_after_reconstruction(self):
        saver = InMemorySaver()
        provider = FakeProvider([LLMResponse(text=json.dumps(valid_mission()))])
        graph = build_mission_graph(saver, provider_factory=lambda: provider)
        config = cfg("approval")
        response = graph.invoke({"brief_text": self.brief}, config=config)
        self.assertIn("__interrupt__", response)
        pending = pending_approval(graph.get_state(config))
        self.assertIsNotNone(pending)
        self.assertEqual(provider.continue_calls, [])

        # La reprise n'exige aucun nouvel appel au LLM.
        restarted = build_mission_graph(
            saver,
            provider_factory=lambda: FakeProvider([]),
        )
        restarted.invoke(Command(resume={
            "decision": "approve",
            "brief_sha256": pending["brief_sha256"],
            "mission_sha256": pending["mission_sha256"],
        }), config=config)
        final = restarted.get_state(config)
        self.assertEqual(final.values["status"], "approved")
        self.assertEqual(final.next, ())
        self.assertEqual(final.values["brief_sha256"], sha256_text(self.brief))
        self.assertEqual(final.values["mission_sha256"],
                         mission_digest(final.values["mission"]))

    def test_invalid_digest_blocks_approval(self):
        saver = InMemorySaver()
        graph = build_mission_graph(
            saver,
            provider_factory=lambda: FakeProvider([
                LLMResponse(text=json.dumps(valid_mission()))
            ]),
        )
        config = cfg("digest")
        graph.invoke({"brief_text": self.brief}, config=config)
        pending = pending_approval(graph.get_state(config))
        graph.invoke(Command(resume={
            "decision": "approve",
            "brief_sha256": pending["brief_sha256"],
            "mission_sha256": "incorrect-digest",
        }), config=config)
        self.assertEqual(graph.get_state(config).values["status"], "rejected")

    def test_rejection(self):
        saver = InMemorySaver()
        graph = build_mission_graph(
            saver,
            provider_factory=lambda: FakeProvider([
                LLMResponse(text=json.dumps(valid_mission()))
            ]),
        )
        config = cfg("rejection")
        graph.invoke({"brief_text": self.brief}, config=config)
        pending = pending_approval(graph.get_state(config))
        graph.invoke(Command(resume={
            "decision": "reject",
            "brief_sha256": pending["brief_sha256"],
            "mission_sha256": pending["mission_sha256"],
        }), config=config)
        self.assertEqual(graph.get_state(config).values["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
