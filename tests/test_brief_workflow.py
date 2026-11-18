import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent.models import LLMResponse
from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import run_phase
from providers.replay_provider import FakeProvider
from research.brief_workflow import BriefWorkflow, Decision, Revision
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from research.literature_search import OpenAlexClient
from tests.test_v82_literature import FakeOpenAlex, WORKS, mission_data, valid_review
from tests.test_v83_design import valid_design
from tests.test_v9_retry_pilot_runner import ReplayClient


ROOT = Path(__file__).resolve().parents[1]
BRIEF = (ROOT / "tests/fixtures/research_brief.md").read_text(encoding="utf-8")


class BriefWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name) / "workflow"
        corpus = PilotCorpus.model_validate(read_document(ROOT / "examples/v9_retry_pilot.json"))
        self.remote = ReplayClient(corpus)
        self.catalog = FakeOpenAlex()
        self.agent = FakeProvider([])
        self.provider = Mock(return_value=self.agent)
        self.reopen()

    def reopen(self):
        self.flow = BriefWorkflow(self.directory, provider_factory=self.provider,
                                  search_factory=lambda: OpenAlexClient(api_key="", transport=self.catalog),
                                  experiment_client_factory=lambda: self.remote)

    def start(self, workload="retry_catalog"):
        return self.flow.start("test-study", BRIEF, LLMSettings(provider="openai", model="planner-stub"), workload)

    def approve(self, expected, response=None):
        state = self.flow.status()
        self.assertEqual(state["stage"], expected, state)
        if response is not None:
            self.agent.responses.append(LLMResponse(text=json.dumps(response)))
        digest = state["pending"]["target_sha256"]
        return self.flow.approve(Decision(target_sha256=digest, actor="test"))

    def rewind(self, stage, **options):
        state = self.flow.status()
        return self.flow.revise(Revision(target_sha256=state["checkpoint_sha256"], actor="test", stage=stage,
                                         reason="Périmètre à corriger", **options))

    def to_catalog(self):
        self.approve("mission", mission_data())
        self.approve("review_mission")

    def to_protocol(self):
        self.to_catalog()
        self.approve("catalog")
        self.approve("literature", valid_review())
        self.approve("review_literature")

    def test_brief_to_verified_manuscript_with_two_upstream_revisions(self):
        self.start()
        self.provider.assert_not_called()
        self.to_catalog()
        self.catalog.works = WORKS[:1]
        blocked = self.approve("catalog")
        self.assertEqual(blocked["block"]["reason"], "InsufficientLiteratureError")
        self.assertIn("catalog", blocked["block"]["return_to"])
        self.catalog.works = WORKS
        self.rewind("catalog", queries=["fault recovery in tool using agents"])
        self.reopen()
        self.approve("catalog")
        self.approve("literature", valid_review())
        old_gate = self.flow.status()["pending"]["target_sha256"]
        revised = self.rewind("literature")
        self.assertEqual(set(revised["artifacts"]), {"mission", "catalog"})
        with self.assertRaises(PermissionError):
            self.flow.approve(Decision(target_sha256=old_gate, actor="test"))
        self.approve("literature", valid_review())
        payload = json.loads(self.agent.start_calls[-1]["prompt"])
        self.assertIn("Périmètre à corriger", payload["human_revision_request"])
        self.approve("review_literature")
        self.assertEqual(len(self.remote.calls), 0)
        self.approve("review_protocol")
        self.approve("calibration")
        self.reopen()
        state = self.approve("measurement")
        self.assertEqual(state["stage"], "review_manuscript")
        self.assertEqual(len(self.remote.calls), 72)
        draft = Path(state["draft"]["draft"])
        self.assertIn("0/8 réponses exactes", draft.read_text(encoding="utf-8"))
        self.assertIn("10.5555/example.101", draft.read_text(encoding="utf-8"))
        complete = self.approve("review_manuscript")
        self.assertEqual(complete["status"], "complete")
        self.assertTrue(Path(complete["export"]).is_file())
        self.assertEqual(len([event for event in self.flow.store.history() if event["action"] == "revise"]), 2)
        published = Path(complete["export"])
        redraft = self.rewind("draft")
        self.assertEqual(redraft["stage"], "review_manuscript")
        self.assertNotEqual(redraft["draft"]["draft"], str(draft))
        self.assertEqual(len(self.remote.calls), 72)
        restarted = self.rewind("protocol")
        self.assertEqual(restarted["stage"], "review_protocol")
        self.assertEqual(set(restarted["artifacts"]), {"mission", "catalog", "literature", "protocol"})
        self.assertTrue(published.is_file())
        self.assertEqual(len(self.remote.calls), 72)

    def test_unmapped_design_stops_without_executing_and_can_return_to_protocol(self):
        self.start("unmapped")
        self.to_protocol()
        self.approve("protocol", valid_design())
        blocked = self.approve("review_protocol")
        self.assertEqual(blocked["block"]["reason"], "no_compatible_executor")
        self.assertEqual(len(self.remote.calls), 0)
        self.assertEqual(self.rewind("protocol", workload="retry_catalog")["stage"], "review_protocol")

    def test_incomplete_api_call_survives_restart_without_replay(self):
        self.start()
        self.to_protocol()
        self.approve("review_protocol")
        self.remote.responses.create.side_effect = TimeoutError
        gate = self.flow.status()["pending"]["target_sha256"]
        result = self.approve("calibration")
        self.assertTrue(result["block"]["uncertain"])
        self.reopen()
        with self.assertRaises(PermissionError):
            self.flow.approve(Decision(target_sha256=gate, actor="test"))
        with self.assertRaises(PermissionError):
            self.rewind("protocol")
        self.assertEqual(self.remote.responses.create.call_count, 1)

    def test_failed_calibration_returns_to_a_new_protocol_without_reusing_approval(self):
        self.start()
        self.to_protocol()
        self.approve("review_protocol")
        self.remote.always_abstain = True
        result = self.approve("calibration")
        self.assertEqual(result["block"]["reason"], "calibration_did_not_pass")
        self.assertFalse(result["block"]["uncertain"])
        old_journal = self.directory / "pilot-01" / "pilot.sqlite3"
        self.assertTrue(old_journal.is_file())
        self.assertEqual(self.rewind("protocol")["stage"], "review_protocol")
        next_gate = self.approve("review_protocol")
        self.assertEqual(next_gate["stage"], "calibration")
        self.assertFalse((self.directory / "pilot-02" / "pilot.sqlite3").exists())
        self.assertEqual(len(self.remote.calls), 8)

    def test_recover_completed_measurement_after_checkpoint_interruption(self):
        self.start()
        self.to_protocol()
        self.approve("review_protocol")
        self.approve("calibration")

        def stop_after_receipt(*args, **kwargs):
            run_phase(*args, **kwargs)
            raise KeyboardInterrupt

        with patch("research.brief_workflow.run_phase", side_effect=stop_after_receipt):
            with self.assertRaises(KeyboardInterrupt):
                self.approve("measurement")
        self.reopen()
        state = self.flow.status()
        self.assertEqual(state["status"], "running")
        rebuilt = self.flow.recover(Decision(target_sha256=state["checkpoint_sha256"], actor="test"))
        self.assertEqual(rebuilt["stage"], "review_manuscript")
        self.assertEqual(len(self.remote.calls), 72)


if __name__ == "__main__":
    unittest.main()
