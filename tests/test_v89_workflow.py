import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.models import LLMResponse
from analysis.contracts import fingerprint
from execution.campaign import PilotProtocol
from execution.backends.local import LocalDockerBackend
from execution.job_manager import DurableJobManager
from research.critique.contracts import HumanDecision, LLMSettings
from research.critique.evidence import read_document
from research.manuscript import (
    apply_order, build_manuscript, EditorialOrder, make_editorial_request,
    record_manuscript, verify_manuscript_export,
)
from research.memory.contracts import MemoryBinding
from research.memory.integration import open_session
from research.memory.sources import pointer
from research.research_workflow import (
    ArithmeticSimulation, ResearchWorkflow, WorkflowInput, config_for, make_plan, pending_decision,
)
from scripts.demo_v89 import load_request
from tests.fakes import FakeProvider


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.context = self.root / "context.json"
        self.context.write_bytes((EXAMPLES / "v86_design_context.json").read_bytes())
        document = load_request(EXAMPLES / "v89_workflow.json").model_dump(mode="json")
        document["context_path"] = str(self.context)
        self.request = WorkflowInput.model_validate(document)
        self.binding = MemoryBinding(project_id="research", path=str(self.root / "memory.sqlite3"))
        self.memory = open_session(self.binding, initialize=True)
        self.saver = InMemorySaver()
        self.cfg = config_for(self.request.run_id)
        self.provider = Mock(side_effect=AssertionError("No LLM expected"))
        self.build()

    def build(self, transport=None):
        self.workflow = ResearchWorkflow(self.root, self.memory, provider_factory=self.provider, transport=transport)
        self.graph = self.workflow.build(self.saver)
        self.store = self.workflow.store

    def start(self):
        self.graph.invoke({"workflow_input": self.request.model_dump(mode="json")}, self.cfg, durability="sync")
        return self.pending()

    def pending(self):
        return pending_decision(self.graph.get_state(self.cfg))

    def resume(self, choice, **extras):
        self.graph.invoke(Command(resume={"decision": choice, "actor": "test-operator",
                                         "target_sha256": self.pending()["target_sha256"], **extras}),
                          self.cfg, durability="sync")
        return self.pending()

    def test_complete_after_graph_restart_with_prospective_revision(self):
        self.assertEqual(self.start()["kind"], "v89_execution_approval")
        self.assertFalse((self.root / "jobs.sqlite3").exists())
        self.build()
        self.assertEqual(self.resume("approve")["kind"], "v86_revision_input")
        draft = read_document(EXAMPLES / "v86_revision_draft.json")
        draft["justifications"]["/analysis_plan/power_limitation"]["issue_ids"] = ["no_replication"]
        self.assertEqual(self.resume("propose", draft=draft)["kind"], "v86_revision_approval")
        self.memory = open_session(self.binding)
        self.build()
        final = self.resume("approve")
        self.assertEqual(final["kind"], "v89_manuscript_approval")
        self.assertFalse((self.root / "output").exists())
        self.assertIsNone(self.resume("approve"))
        result = self.graph.get_state(self.cfg).values
        self.assertEqual(result["workflow_status"], "completed")
        paper = verify_manuscript_export(self.store, self.request.run_id, result["manuscript_sha256"],
                                         self.root / "output" / "paper")
        self.assertFalse(paper.scientific_evidence)
        self.assertEqual([row.mean for row in paper.figure], [2.0, 1.0, 3.0, 3.0])
        self.assertEqual([citation.work_id for citation in paper.bibliography], ["W101"])
        for statement in paper.statements:
            ref = statement.reference
            key = self.request.run_id + "-1" if ref.artifact == "revision" else self.request.run_id
            artifact = "approved_revision" if ref.artifact == "revision" else ref.artifact
            doc = self.store.require(artifact, key)
            self.assertEqual((fingerprint(doc), fingerprint(pointer(doc, ref.pointer))),
                             (ref.artifact_sha256, ref.value_sha256))
        evaluation = read_document(self.root / "output" / "evaluation.json")
        self.assertEqual(evaluation["llm_calls_count"], 0)
        self.assertEqual(len(evaluation["jobs"]), 4)
        self.assertGreater(evaluation["memory_hits"], 0)
        recall = self.memory.memory.search("research", "réplication protocole", kinds=["decision"], audit=False)
        control = self.memory.memory.search("empty-control", "réplication protocole", kinds=["decision"], audit=False)
        self.assertEqual(len(recall.hits), 1)
        self.assertFalse(recall.hits[0].action_authorization)
        self.assertEqual(control.hits, [])
        self.assertTrue(all(job["simulated"] for job in evaluation["jobs"]))
        history = list(self.graph.get_state_history(self.cfg))
        self.assertGreater(len(history), 4)
        self.provider.assert_not_called()
        (self.root / "output" / "paper" / "figure.svg").write_text("altered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verified version"):
            verify_manuscript_export(self.store, self.request.run_id, result["manuscript_sha256"],
                                     self.root / "output" / "paper")

    def test_optional_writer_call_is_exact_bounded_and_separately_approved(self):
        self.request = self.request.model_copy(update={"writer_settings": LLMSettings(provider="anthropic", model="fake")})
        self.start()
        self.resume("approve")
        pending = self.resume("skip")
        self.assertEqual(pending["kind"], "v89_writer_approval")
        self.provider.assert_not_called()
        manuscript = build_manuscript(self.store, self.request.run_id)
        request = make_editorial_request(self.store, manuscript, self.request.writer_settings)
        order = [claim.statement_id for claim in reversed(manuscript.statements)]
        with self.assertRaises(ValueError):
            apply_order(manuscript, request, EditorialOrder(statement_ids=order[:-1] + ["invented"]))
        self.provider.side_effect = None
        self.provider.return_value = FakeProvider([LLMResponse(text=json.dumps({"statement_ids": order}))])
        final = self.resume("approve")
        self.assertEqual(final["kind"], "v89_manuscript_approval")
        self.assertEqual([item["statement_id"] for item in final["manuscript"]["statements"]], order)
        self.assertEqual(final["manuscript"]["editorial_request_sha256"], pending["target_sha256"])
        self.assertEqual(self.provider.call_count, 1)
        self.assertFalse((self.root / "output").exists())
        self.resume("reject")
        self.assertFalse((self.root / "output").exists())
        with self.assertRaises(PermissionError):
            record_manuscript(self.store, manuscript, HumanDecision(
                decision="approve", target_sha256=final["target_sha256"], actor="test"))

    def test_wrong_digest_rejected_and_changed_context_blocks_execution(self):
        self.start()
        with self.assertRaises(PermissionError):
            self.resume("approve", target_sha256="0" * 64)
        self.assertFalse((self.root / "jobs.sqlite3").exists())
        self.context.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.graph.invoke(None, self.cfg, durability="sync")
        self.assertFalse((self.root / "jobs.sqlite3").exists())

    def test_rejected_execution_and_incomplete_binding_have_no_jobs(self):
        invalid = self.request.model_copy(update={"bindings": self.request.bindings.model_copy(
            update={"cells": self.request.bindings.cells[:-1]})})
        with self.assertRaisesRegex(ValueError, "every approved design cell"):
            make_plan(invalid)
        self.start()
        self.assertIsNone(self.resume("reject"))
        self.assertFalse((self.root / "jobs.sqlite3").exists())
        self.assertFalse((self.root / "output").exists())

    def test_transport_failure_blocks_later_jobs_and_manuscript(self):
        transport = ArithmeticSimulation()
        transport.run = Mock(side_effect=RuntimeError("Transport failed"))
        self.build(transport)
        self.start()
        self.resume("approve")
        self.assertEqual(transport.run.call_count, 1)
        protocol = PilotProtocol.model_validate(read_document(self.root / "pilot_protocol.json"))
        manager = DurableJobManager(path=self.root / "jobs.sqlite3", read_only=True)
        self.assertEqual(manager.get(protocol.jobs()[0].job_id).phase, "failed")
        self.assertIsNone(manager.get(protocol.jobs()[1].job_id))
        self.assertFalse((self.root / "output").exists())
        self.build(transport)
        self.assertEqual(self.graph.get_state(self.cfg).values["workflow_status"], "execution_blocked")
        self.assertEqual(transport.run.call_count, 1)

    def test_unconfirmed_submission_is_not_replayed_after_restart(self):
        self.start()
        with patch.object(LocalDockerBackend, "submit", side_effect=KeyboardInterrupt) as submit:
            with self.assertRaises(KeyboardInterrupt):
                self.resume("approve")
            submit.assert_called_once()
        self.build()
        with patch.object(LocalDockerBackend, "submit") as submit:
            self.graph.invoke(None, self.cfg, durability="sync")
            submit.assert_not_called()
        self.assertEqual(self.graph.get_state(self.cfg).values["workflow_status"], "execution_blocked")
        protocol = PilotProtocol.model_validate(read_document(self.root / "pilot_protocol.json"))
        ledger = DurableJobManager(path=self.root / "jobs.sqlite3", read_only=True)
        self.assertEqual(ledger.get(protocol.jobs()[0].job_id).phase, "submission_reserved")
        self.assertIsNone(ledger.get(protocol.jobs()[1].job_id))


if __name__ == "__main__":
    unittest.main()
