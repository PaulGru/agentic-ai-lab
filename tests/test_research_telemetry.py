import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from agent.models import LLMResponse
from experiments.agent import EXPERIMENT_INSTRUCTIONS
from experiments.graph import ExperimentWorkflow, code_digest
from observability.research_telemetry import (export_code_telemetry, export_telemetry, publish_weave,
                                             record_wandb_offline, weave_plan)
from providers.replay_provider import FakeProvider
from research.brief_autonomy import AutonomyBudget, authorize_autonomy, autonomy_plan, run_autonomously
from research.brief_workflow import Decision
from research.critique.contracts import LLMSettings
from tests.test_ood_replay_workflow import BRIEF, offline_workflow


class ResearchTelemetryTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def run_workflow(self):
        flow = offline_workflow(self.root / "workflow")
        provider = flow.provider_factory(None)
        provider.responses.insert(1, LLMResponse(text='{"private_model_output": "forbidden"}'))
        flow.start("private-mission-name", BRIEF, LLMSettings(provider="openai", model="test"))
        budget = AutonomyBudget()
        plan = autonomy_plan(flow, budget)
        authorize_autonomy(flow, budget, Decision(actor="private-operator", target_sha256=plan["target_sha256"]))
        status = run_autonomously(flow)
        self.assertEqual(status["status"], "complete", status)
        return flow, provider

    def test_export_reconciles_prompt_versions_and_rejected_calls_without_exporting_text(self):
        flow, provider = self.run_workflow()
        payload, path = export_telemetry(flow)
        calls = payload["agent_calls"]
        self.assertEqual([call["stage_outcome"] for call in calls], ["accepted", "rejected", "accepted"])
        self.assertEqual([call["prompt_version"] for call in calls], [1, 1, 1])
        for call, effective in zip(calls, provider.start_calls, strict=True):
            self.assertEqual(call["instructions_sha256"], hashlib.sha256(effective["instructions"].encode()).hexdigest())
        self.assertTrue(payload["experiment"]["simulated"])
        self.assertEqual(payload["experiment"]["metrics"]["experiment/simule-a/gap_points"], 50.0)
        report = path.read_text(encoding="utf-8")
        for private in ("private-operator", "private-mission-name", "private_model_output", "Fiche K7", "Fictional notes"):
            self.assertNotIn(private, report)
        self.assertEqual(export_telemetry(flow)[1], path)

    def test_code_agent_tracks_prompt_and_only_exports_numeric_execution_data_and_digests(self):
        code = 'print("private-code")'
        proposed = {"code": code, "hypothesis": "Hypothèse privée", "protocol": ["Étape 1"],
                    "seed": 42, "dependencies": [], "expected_output": "private-output"}
        provider = FakeProvider([LLMResponse(text=json.dumps(proposed))])
        graph = ExperimentWorkflow(lambda: provider, sandbox=MagicMock())
        state = graph.propose_node({"question": "private-question"})
        state["attempts"] = [{"number": 1, "code": code, "code_sha256": code_digest(code),
                              "execution": {"status": "success", "exit_code": 0, "duration_seconds": 1.5,
                                            "stdout": "private-output", "stderr": "private-stderr"}}]
        payload, path = export_code_telemetry(state, self.root)
        call = payload["agent_calls"][0]
        self.assertEqual(call["instructions_sha256"], hashlib.sha256(EXPERIMENT_INSTRUCTIONS.encode()).hexdigest())
        self.assertEqual(payload["code_execution"][0]["duration_seconds"], 1.5)
        for private in ("private-code", "private-output", "private-stderr", "private-question"):
            self.assertNotIn(private, path.read_text(encoding="utf-8"))
        sdk = MagicMock()
        session = sdk.init.return_value.__enter__.return_value
        with patch.dict("sys.modules", {"wandb": sdk}):
            record_wandb_offline(payload, path.parent)
        self.assertEqual(sdk.init.call_args.kwargs["mode"], "offline")
        self.assertEqual(sdk.Settings.call_args.kwargs["console"], "off")
        self.assertTrue(sdk.Settings.call_args.kwargs["x_disable_meta"])
        self.assertEqual(session.log.call_args_list[0].args[0]["code/succeeded"], 1)

    def test_weave_requires_exact_approval_and_sends_only_technical_fields(self):
        flow, _ = self.run_workflow()
        payload, _ = export_telemetry(flow)
        project = "test-team/private-project"
        plan = weave_plan(payload, project)
        sdk = MagicMock()
        client = sdk.init.return_value
        with patch.dict("sys.modules", {"weave": sdk}):
            with self.assertRaises(PermissionError):
                publish_weave(payload, project, approved_sha256="0" * 64)
            sdk.init.assert_not_called()
            result = publish_weave(payload, project, approved_sha256=plan["target_sha256"])
        self.assertEqual(result["calls_submitted"], 3)
        self.assertFalse(sdk.init.call_args.kwargs["settings"]["implicitly_patch_integrations"])
        self.assertFalse(sdk.init.call_args.kwargs["settings"]["capture_code"])
        self.assertEqual(set(client.create_call.call_args.kwargs["inputs"]),
                         {"stage", "prompt_version", "instructions_sha256", "call_id"})
        self.assertEqual(set(client.finish_call.call_args.kwargs["output"]), {"latency_ms", "stage_outcome", "status"})
