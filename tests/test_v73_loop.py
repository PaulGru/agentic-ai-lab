
import json
import unittest

from agent.models import LLMResponse
from experiments.loop import ExperimentLoop
from experiments.proposal import ExperimentProposal
from experiments.reviser import ExperimentReviser
from experiments.sandbox import SandboxResult
from tests.fakes import FakeProvider


def proposal(code="print('ok')"):
    return ExperimentProposal(
        hypothesis="Calculer une moyenne",
        protocol=[
            "Exécuter le programme",
            "Lire la moyenne",
        ],
        seed=42,
        dependencies=[],
        code=code,
        expected_output="Une moyenne numérique",
    )


def result(status, stderr=""):
    return SandboxResult(
        status=status,
        exit_code=0 if status == "success" else 1,
        stdout="3.0" if status == "success" else "",
        stderr=stderr,
        duration_seconds=0.05,
    )


class FakeSandbox:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run(
        self,
        code,
        *,
        approved=False,
        dependencies=None,
    ):
        self.calls.append(
            (code, approved, dependencies)
        )
        return self.results.pop(0)


class FakeReviser:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = []

    def revise(self, current, execution):
        self.calls.append(execution)

        return ExperimentProposal.model_validate(
            {
                **current.model_dump(),
                "code": self.codes.pop(0),
            }
        )


class V73LoopTests(unittest.TestCase):

    def test_success_does_not_trigger_revision(self):
        sandbox = FakeSandbox([result("success")])
        reviser = FakeReviser([])

        report = ExperimentLoop(
            sandbox, reviser
        ).run(
            proposal(),
            approve=lambda p, n: True,
        )

        self.assertEqual(report.status, "completed")
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(len(reviser.calls), 0)

    def test_failure_fix_success_requires_two_approvals(self):
        sandbox = FakeSandbox([
            result("error", "NameError"),
            result("success"),
        ])
        reviser = FakeReviser(["print(3.0)"])
        approved_codes = []

        def approve(current, number):
            approved_codes.append(
                (number, current.code)
            )
            return True

        report = ExperimentLoop(
            sandbox, reviser
        ).run(
            proposal("print(missing)"),
            approve=approve,
        )

        self.assertEqual(report.status, "completed")
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(len(approved_codes), 2)
        self.assertNotEqual(
            approved_codes[0][1],
            approved_codes[1][1],
        )

        self.assertTrue(
            all(call[1] is True for call in sandbox.calls)
        )

    def test_second_approval_can_be_refused(self):
        sandbox = FakeSandbox([result("error")])
        reviser = FakeReviser(["print(3.0)"])

        report = ExperimentLoop(
            sandbox, reviser
        ).run(
            proposal("print(missing)"),
            approve=lambda p, n: n == 1,
        )

        self.assertEqual(report.status, "rejected")
        self.assertEqual(len(sandbox.calls), 1)

    def test_maximum_three_attempts(self):
        sandbox = FakeSandbox([
            result("error"),
            result("error"),
            result("error"),
        ])
        reviser = FakeReviser([
            "print(1)",
            "print(2)",
        ])

        report = ExperimentLoop(
            sandbox, reviser
        ).run(
            proposal("print(missing)"),
            approve=lambda p, n: True,
        )

        self.assertEqual(
            report.status,
            "max_attempts",
        )
        self.assertEqual(len(sandbox.calls), 3)
        self.assertEqual(len(reviser.calls), 2)

    def test_invalid_revision_is_not_executed(self):
        sandbox = FakeSandbox([result("error")])
        reviser = FakeReviser(["def broken("])

        report = ExperimentLoop(
            sandbox, reviser
        ).run(
            proposal("print(missing)"),
            approve=lambda p, n: True,
        )

        self.assertEqual(
            report.status,
            "revision_failed",
        )
        self.assertEqual(len(sandbox.calls), 1)

    def test_unchanged_revision_stops(self):
        sandbox = FakeSandbox([result("error")])
        reviser = FakeReviser(["print(missing)"])

        report = ExperimentLoop(
            sandbox, reviser
        ).run(
            proposal("print(missing)"),
            approve=lambda p, n: True,
        )

        self.assertEqual(report.status, "unchanged")
        self.assertEqual(len(sandbox.calls), 1)

    def test_dependencies_rejected_before_approval(self):
        p = proposal()

        p = ExperimentProposal.model_validate({
            **p.model_dump(),
            "dependencies": ["numpy==2.3.0"],
        })

        sandbox = FakeSandbox([])

        with self.assertRaises(ValueError):
            ExperimentLoop(
                sandbox, FakeReviser([])
            ).run(
                p,
                approve=lambda p, n: True,
            )

        self.assertEqual(sandbox.calls, [])

    def test_attempt_cap_is_hard_limited(self):
        with self.assertRaises(ValueError):
            ExperimentLoop(
                FakeSandbox([]),
                FakeReviser([]),
                max_attempts=4,
            )

    def test_reviser_keeps_experiment_contract(self):
        provider = FakeProvider([
            LLMResponse(
                text=json.dumps({
                    "code": "print(3.0)",
                })
            )
        ])

        original = proposal("print(missing)")
        revised = ExperimentReviser(provider).revise(
            original,
            result("error", "NameError"),
        )

        self.assertEqual(revised.code, "print(3.0)")
        self.assertEqual(
            revised.model_dump(exclude={"code"}),
            original.model_dump(exclude={"code"}),
        )
        self.assertEqual(
            provider.start_calls[0]["tools"],
            [],
        )
        self.assertEqual(provider.continue_calls, [])

    def test_reviser_rejects_extra_fields(self):
        provider = FakeProvider([
            LLMResponse(
                text=json.dumps({
                    "code": "print(3.0)",
                    "seed": 123,
                })
            )
        ])

        with self.assertRaises(ValueError):
            ExperimentReviser(provider).revise(
                proposal("print(missing)"),
                result("error"),
            )


if __name__ == "__main__":
    unittest.main()
