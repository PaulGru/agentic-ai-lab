from collections import Counter
from pathlib import Path
import unittest

from analysis.contracts import fingerprint
from experiments.retry_pilot import PilotCorpus, Trial, evaluate_lookup, prepare_plan
from research.critique.evidence import read_document


class RetryPilotTests(unittest.TestCase):
    def setUp(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "v9_retry_pilot.json"
        self.corpus = PilotCorpus.model_validate(read_document(example))

    def test_plan_is_reproducible_complete_and_counterbalanced(self):
        plan = prepare_plan(self.corpus)
        self.assertEqual(plan, prepare_plan(self.corpus))
        self.assertEqual(plan["plan_sha256"], fingerprint({k: v for k, v in plan.items() if k != "plan_sha256"}))
        self.assertEqual(plan["max_model_calls"], 64)
        self.assertFalse(plan["execution_authorized"])
        trials = plan["trials"]
        self.assertEqual(len({trial["trial_id"] for trial in trials}), 32)
        self.assertEqual(set(Counter((trial["profile"], trial["policy"]) for trial in trials).values()), {8})
        self.assertEqual(Counter((trial["profile"], trial["policy"]) for trial in trials[::2]),
                         {("clean", "no_retry"): 4, ("clean", "retry_once"): 4,
                          ("transient", "no_retry"): 4, ("transient", "retry_once"): 4})
        for task in self.corpus.tasks:
            self.assertEqual(sum(trial["task_id"] == task.task_id for trial in trials), 4)

    def test_retry_changes_only_the_handling_of_the_transient_fault(self):
        task = self.corpus.tasks[0]
        expected = self.corpus.catalog[task.product].stock
        args = {"product": task.product, "field": task.field}
        cases = {
            ("clean", "no_retry"): (["ok"], expected),
            ("clean", "retry_once"): (["ok"], expected),
            ("transient", "no_retry"): (["transient_error"], None),
            ("transient", "retry_once"): (["transient_error", "ok"], expected),
        }
        for (profile, policy), outcome in cases.items():
            with self.subTest(profile=profile, policy=policy):
                trial = Trial(trial_id="test", task_id=task.task_id, profile=profile, policy=policy)
                receipt = evaluate_lookup(self.corpus, trial, args)
                self.assertEqual((receipt.attempts, receipt.output), outcome)
                invalid = evaluate_lookup(self.corpus, trial, {**args, "product": "BR-205"})
                self.assertEqual((invalid.attempts, invalid.output), (["invalid_lookup"], None))


if __name__ == "__main__":
    unittest.main()
