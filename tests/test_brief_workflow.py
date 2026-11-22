from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent.models import LLMResponse
from analysis.contracts import fingerprint
from execution.campaign import run_campaign
from execution.contracts import ExperimentJob
from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import MODEL as PILOT_MODEL, run_phase
from providers.replay_provider import FakeProvider
from research.brief_workflow import BriefWorkflow, Decision, Revision
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from research.critique.llm import bounded_provider
from research.literature_search import OpenAlexClient
from scripts.research_workflow import guide
from tests.test_v82_literature import FakeOpenAlex, WORKS, mission_data, valid_review
from tests.test_v83_design import valid_design
from tests.test_v846_campaign import DeterministicTestTransport
from tests.test_v9_retry_pilot_runner import ReplayClient


ROOT = Path(__file__).resolve().parents[1]
BRIEF = (ROOT / "tests/fixtures/research_brief.md").read_text(encoding="utf-8")


class IncorrectMeanTransport(DeterministicTestTransport):
    def run(self, job, code, cancel):
        outcome = super().run(job, code, cancel)
        output = json.loads(outcome.stdout)
        output["files"]["metrics.json"]["mean"] = 999
        return replace(outcome, stdout=json.dumps(output))


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
        self.synthetic_transport = DeterministicTestTransport()
        self.reopen()

    def reopen(self):
        self.flow = BriefWorkflow(self.directory, provider_factory=self.provider,
                                  search_factory=lambda: OpenAlexClient(api_key="", transport=self.catalog),
                                  experiment_client_factory=lambda: self.remote,
                                  synthetic_transport=self.synthetic_transport)

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
        gate = blocked["pending"]
        self.assertEqual(gate["kind"], "revision")
        self.assertEqual(gate["details"]["proposal"]["stage"], "catalog")
        self.assertIn("agent robustness", gate["details"]["proposal"]["queries"])
        requests = len(self.catalog.urls)
        self.catalog.works = WORKS
        self.reopen()
        self.assertEqual(self.flow.status()["pending"], gate)
        next_step = self.approve("catalog")
        self.assertEqual(len(self.catalog.urls), requests)
        self.assertEqual(next_step["pending"]["kind"], "authorization")
        self.assertNotEqual(next_step["pending"]["target_sha256"], gate["target_sha256"])
        with self.assertRaises(PermissionError):
            self.flow.approve(Decision(target_sha256=gate["target_sha256"], actor="test"))
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
        self.assertEqual(len([event for event in self.flow.store.history()
                              if event["action"] in {"revise", "approve_revision"}]), 2)
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
        self.assertIsNone(blocked["pending"])
        self.assertIn("exécuteur", blocked["attention"])
        self.assertEqual(len(self.remote.calls), 0)
        self.assertEqual(self.rewind("protocol", workload="retry_catalog")["stage"], "review_protocol")

    def test_astra_plans_and_synthesizes_with_separate_experiment_configuration(self):
        self.flow.provider_factory = bounded_provider
        settings = LLMSettings(provider="openai", model="gpt-6.2-astra")
        with patch("providers.openai_provider.OpenAI") as openai:
            client = openai.return_value
            client.responses.create.side_effect = [
                SimpleNamespace(id="mission-test", output=[], output_text=json.dumps(mission_data())),
                SimpleNamespace(id="literature-test", output=[], output_text=json.dumps(valid_review())),
                SimpleNamespace(id="protocol-test", output=[], output_text=json.dumps(valid_design())),
            ]
            starting = self.flow.start("test-study", BRIEF, settings, "unmapped")
            self.assertEqual(starting["pending"]["details"]["reasoning"], {"effort": "low"})
            openai.assert_not_called()
            self.approve("mission")
            self.approve("review_mission")
            self.approve("catalog")
            self.approve("literature")
            self.approve("review_literature")
            self.approve("protocol")
            for invocation in client.responses.create.call_args_list:
                self.assertEqual(invocation.kwargs["model"], settings.model)
                self.assertEqual(invocation.kwargs["reasoning"], {"effort": "low"})
                self.assertEqual(invocation.kwargs["max_output_tokens"], settings.max_output_tokens)
                self.assertNotIn("tools", invocation.kwargs)
            self.assertEqual(client.responses.create.call_count, 3)
            self.assertEqual(self.approve("review_protocol")["block"]["reason"], "no_compatible_executor")
            self.rewind("protocol", workload="retry_catalog")
            self.approve("review_protocol")
            self.assertEqual(self.remote.calls, [])
            self.approve("calibration")
            self.assertEqual(client.responses.create.call_count, 3)
            self.assertEqual(len(self.remote.calls), 8)
            self.assertEqual({request["model"] for request in self.remote.calls}, {PILOT_MODEL})

    def test_brief_declares_synthetic_workload_and_exports_only_verified_receipts(self):
        self.flow.start("test-study", BRIEF + "\n## Exécuteur\nsynthetic.mean\n",
                        LLMSettings(provider="anthropic", model="planner-stub"))
        self.to_protocol()
        protocol_review = self.flow.status()["pending"]
        proposal = protocol_review["details"]["proposal"]["document"]
        self.assertNotEqual(proposal["requested_question"], proposal["measured_question"])
        gate = self.approve("review_protocol")["pending"]
        details = gate["details"]
        self.assertEqual(details["backend"], "docker-local")
        self.assertFalse(details["network"])
        self.assertFalse(details["scientific_evidence"])
        self.assertFalse((self.directory / "synthetic-01").exists())
        self.assertEqual(self.synthetic_transport.calls, [])
        jobs = [ExperimentJob.model_validate(item["spec"]) for item in details["jobs"]]
        self.assertEqual([job.workload.parameters["values"] for job in jobs],
                         [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
        self.assertEqual([job.fingerprint() for job in jobs], [item["sha256"] for item in details["jobs"]])
        self.reopen()
        self.assertEqual(gate, self.flow.status()["pending"])
        with self.assertRaises(PermissionError):
            self.flow.approve(Decision(target_sha256=protocol_review["target_sha256"], actor="test"))
        self.assertEqual(self.synthetic_transport.calls, [])
        complete = self.approve("synthetic_execution")
        self.assertEqual(complete["status"], "complete")
        self.assertIsNone(complete["draft"])
        self.assertEqual(self.synthetic_transport.calls, [job.job_id for job in jobs])
        self.assertEqual(len(self.remote.calls), 0)
        report_path = Path(complete["report"]["path"])
        report = read_document(report_path)
        self.assertEqual(fingerprint(report), complete["report"]["sha256"])
        self.assertEqual(report["brief_protocol_sha256"], details["protocol_sha256"])
        self.assertEqual(report["execution_approval"], {"actor": "test", "target_sha256": gate["target_sha256"]})
        self.assertEqual(report["analysis"]["run_means"], {"baseline": 3.0, "variant": 6.0})
        self.assertTrue(report["simulated"])
        self.assertFalse(report["scientific_evidence"])
        report_path.write_text("{}", encoding="utf-8")
        status = self.flow.status()
        self.assertIsNone(status["report"])
        self.assertEqual(status["block"]["reason"], "invalid_synthetic_report")

    def test_incorrect_mean_blocks_even_when_container_reports_success(self):
        self.synthetic_transport = IncorrectMeanTransport()
        self.reopen()
        self.start("synthetic.mean")
        self.to_protocol()
        self.approve("review_protocol")
        blocked = self.approve("synthetic_execution")
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["block"]["reason"], "synthetic_execution_failed")
        self.assertIn("Inconsistent mean metric", blocked["block"]["message"])
        self.assertFalse(blocked["block"]["uncertain"])
        self.assertIsNone(blocked["report"])
        self.assertEqual(len(self.synthetic_transport.calls), 1)
        self.assertEqual(self.rewind("protocol")["stage"], "review_protocol")
        self.assertFalse((self.directory / "synthetic-02").exists())

    def test_synthetic_export_tampering_prevents_submission(self):
        self.start("unmapped")
        self.to_protocol()
        self.approve("protocol", valid_design())
        self.approve("review_protocol")
        self.rewind("protocol", workload="synthetic.mean")
        gate = self.approve("review_protocol")["pending"]
        path = self.directory / "versions" / "v02" / "protocol.json"
        document = read_document(path)
        document["document"]["campaign"]["runs"][0]["values"][0] = 99
        path.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(self.flow.status()["status"], "blocked")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.flow.approve(Decision(target_sha256=gate["target_sha256"], actor="test"))
        self.assertEqual(self.synthetic_transport.calls, [])
        self.assertFalse((self.directory / "synthetic-02").exists())

    def test_synthetic_recovery_uses_complete_receipts_without_resubmitting(self):
        self.start("synthetic.mean")
        self.to_protocol()
        gate = self.approve("review_protocol")["pending"]

        def stop_after_receipts(*args):
            run_campaign(*args)
            raise KeyboardInterrupt

        with patch("research.brief_workflow.run_campaign", side_effect=stop_after_receipts):
            with self.assertRaises(KeyboardInterrupt):
                self.approve("synthetic_execution")
        self.reopen()
        status = self.flow.status()
        self.assertEqual(status["status"], "running")
        decision = Decision(target_sha256=status["checkpoint_sha256"], actor="recovery-operator")
        complete = self.flow.recover(decision)
        report = read_document(complete["report"]["path"])
        self.assertEqual(report["execution_approval"], {"actor": "test", "target_sha256": gate["target_sha256"]})
        self.assertEqual(len(self.synthetic_transport.calls), 2)

    def test_synthetic_recovery_with_incomplete_receipts_never_submits(self):
        self.start("synthetic.mean")
        self.to_protocol()
        self.approve("review_protocol")
        with patch("research.brief_workflow.run_campaign", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.approve("synthetic_execution")
        self.reopen()
        status = self.flow.status()
        with self.assertRaisesRegex(PermissionError, "No complete synthetic job receipts"):
            self.flow.recover(Decision(target_sha256=status["checkpoint_sha256"], actor="test"))
        with self.assertRaises(PermissionError):
            self.rewind("protocol")
        self.assertEqual(self.synthetic_transport.calls, [])

    def test_incomplete_api_call_survives_restart_without_replay(self):
        self.start()
        self.to_protocol()
        self.approve("review_protocol")
        self.remote.responses.create.side_effect = TimeoutError
        gate = self.flow.status()["pending"]["target_sha256"]
        result = self.approve("calibration")
        self.assertTrue(result["block"]["uncertain"])
        self.assertIsNone(result["pending"])
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

    def test_invalid_citation_requests_a_repair_without_refetching_the_catalog(self):
        self.start()
        self.to_catalog()
        self.approve("catalog")
        invalid = valid_review()
        invalid["findings"][0]["abstract_excerpt"] = "an invented excerpt absent from the catalog"
        result = self.approve("literature", invalid)
        self.assertEqual(result["block"]["reason"], "invalid_model_output")
        self.assertEqual(result["pending"]["details"]["proposal"]["stage"], "literature")
        requests = len(self.catalog.urls)
        planner_calls = len(self.agent.start_calls)
        self.reopen()
        self.approve("literature")
        self.assertEqual(len(self.agent.start_calls), planner_calls)
        self.assertEqual(set(self.flow.status()["artifacts"]), {"mission", "catalog"})
        self.approve("literature", valid_review())
        hints = json.loads(self.agent.start_calls[-1]["prompt"])["human_revision_request"]
        self.assertIn("passage exact du résumé", hints)
        self.assertEqual(len(self.catalog.urls), requests)

    def test_guide_stops_when_the_next_approval_is_declined(self):
        self.start()
        self.agent.responses.append(LLMResponse(text=json.dumps(mission_data())))
        with (patch("scripts.research_workflow.confirm", side_effect=[True, False]) as confirm,
              patch("dotenv.load_dotenv")):
            status = guide(self.flow, "test")
        self.assertEqual([call.args[0]["kind"] for call in confirm.call_args_list], ["authorization", "review"])
        self.assertEqual(status["stage"], "review_mission")
        self.assertEqual(len(self.agent.start_calls), 1)
        self.assertEqual(self.catalog.urls, [])

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
