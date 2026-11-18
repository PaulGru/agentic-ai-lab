import json
from pathlib import Path
import sqlite3
from contextlib import closing
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import (
    CALIBRATION, MODEL, PhaseApproval, PilotJournal, approval_target, final_plan, run_phase, summarize,
)
from research.critique.evidence import read_document


class ReplayClient:
    def __init__(self, corpus, *, oversized=False, fail=False, always_abstain=False):
        self.tasks = {task.prompt: task for task in corpus.tasks}
        self.tasks.update({task.prompt: task for task, _, _ in CALIBRATION})
        self.calls = []
        self.always_abstain = always_abstain
        self.responses = SimpleNamespace(create=Mock(side_effect=TimeoutError if fail else self.reply),
                                         input_tokens=SimpleNamespace(count=Mock(
                                             return_value=SimpleNamespace(input_tokens=1025 if oversized else 200))))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def reply(self, **request):
        self.calls.append(request)
        items = request["input"]
        task = self.tasks[items[0]["content"]]
        if request["tools"]:
            part = {"type": "function_call", "call_id": "call_lookup", "name": "read_catalog",
                    "arguments": json.dumps({"product": task.product, "field": task.field})}
            answer = ""
        else:
            part = {"type": "message"}
            answer = json.dumps({"answer": None if self.always_abstain else json.loads(items[-1]["output"])["value"]})
        return SimpleNamespace(id=f"response-{len(self.calls)}", model=MODEL, service_tier="default",
                               status="completed", usage=SimpleNamespace(input_tokens=200, output_tokens=30),
                               output_text=answer, output=[SimpleNamespace(model_dump=lambda **_: part)])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.journal = PilotJournal(Path(temp.name) / "pilot.sqlite3")
        example = Path(__file__).resolve().parents[1] / "examples" / "v9_retry_pilot.json"
        self.corpus = PilotCorpus.model_validate(read_document(example))
        self.plan = final_plan(self.corpus)

    def approval(self, phase):
        target = approval_target(self.plan, self.journal, phase)
        return PhaseApproval(phase=phase, target_sha256=target["target_sha256"], actor="test-operator")

    def test_separate_approvals_to_run_all_trials_under_exact_budgets(self):
        replay = ReplayClient(self.corpus)
        with self.assertRaises(PermissionError):
            self.approval("measurement")
        run_phase(self.corpus, self.journal, self.approval("calibration"), client_factory=lambda: replay)
        self.assertEqual(len(replay.calls), 8)
        self.assertTrue(summarize(self.plan, self.journal)["calibration_passed"])
        run_phase(self.corpus, self.journal, self.approval("measurement"), client_factory=lambda: replay)
        self.assertEqual(len(replay.calls), 72)
        self.assertEqual(replay.responses.input_tokens.count.call_count, 72)
        summary = summarize(self.plan, self.journal)
        measurement = summary["measurement"]
        self.assertEqual(measurement["state"], "complete")
        self.assertEqual(measurement["generations_received"], 64)
        self.assertEqual(len(measurement["trial_receipts"]), 32)
        for row in measurement["trial_receipts"]:
            trial = row["trial"]
            expected = "abstention" if trial["policy"] == "no_retry" and trial["profile"] == "transient" else "correct"
            self.assertEqual(row["answer_score"], expected)
        for request in replay.calls:
            self.assertEqual(request["model"], MODEL)
            self.assertFalse(request["store"])
            self.assertFalse(request["parallel_tool_calls"])
            self.assertEqual(request["max_output_tokens"], 256)
            self.assertNotIn("previous_response_id", request)
            self.assertNotIn("retry_once", json.dumps(request))
            self.assertNotIn("no_retry", json.dumps(request))
            if request["tools"]:
                self.assertEqual(len(request["input"]), 1)
        self.assertEqual(self.plan["budget"]["total"]["standard_generation_budget_usd"], "0.13824")

    def test_wrong_approval_does_not_construct_client_or_journal(self):
        factory = Mock()
        approval = self.approval("calibration").model_copy(update={"target_sha256": "0" * 64})
        with self.assertRaises(PermissionError):
            run_phase(self.corpus, self.journal, approval, client_factory=factory)
        factory.assert_not_called()
        self.assertFalse(self.journal.path.exists())

    def test_large_input_is_stopped_before_generation(self):
        replay = ReplayClient(self.corpus, oversized=True)
        with self.assertRaises(RuntimeError):
            run_phase(self.corpus, self.journal, self.approval("calibration"), client_factory=lambda: replay)
        replay.responses.create.assert_not_called()
        self.assertEqual(summarize(self.plan, self.journal)["calibration"]["state"], "incomplete")

    def test_timeout_leaves_reservation_and_blocks_replay(self):
        replay = ReplayClient(self.corpus, fail=True)
        approval = self.approval("calibration")
        with self.assertRaises(RuntimeError):
            run_phase(self.corpus, self.journal, approval, client_factory=lambda: replay)
        self.assertEqual(replay.responses.create.call_count, 1)
        saved = summarize(self.plan, self.journal)["calibration"]
        self.assertEqual((saved["generations_reserved"], saved["generations_received"]), (1, 0))
        self.assertFalse(saved["all_generation_usage_accounted"])
        with self.assertRaises(PermissionError):
            run_phase(self.corpus, PilotJournal(self.journal.path), approval, client_factory=lambda: replay)
        self.assertEqual(replay.responses.create.call_count, 1)

    def test_failed_calibration_cannot_authorize_measurement(self):
        replay = ReplayClient(self.corpus, always_abstain=True)
        run_phase(self.corpus, self.journal, self.approval("calibration"), client_factory=lambda: replay)
        self.assertFalse(summarize(self.plan, self.journal)["calibration_passed"])
        with self.assertRaises(PermissionError):
            self.approval("measurement")

    def test_altered_receipt_blocks_measurement(self):
        replay = ReplayClient(self.corpus)
        run_phase(self.corpus, self.journal, self.approval("calibration"), client_factory=lambda: replay)
        with closing(sqlite3.connect(self.journal.path)) as connection, connection:
            connection.execute("UPDATE pilot_events SET document='{}' WHERE step='receipt'")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.approval("measurement")


if __name__ == "__main__":
    unittest.main()
