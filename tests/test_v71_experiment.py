
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from agent.models import LLMResponse, ToolCall
from experiments.agent import ExperimentAgent
from experiments.proposal import parse_proposal
from tests.fakes import FakeProvider


def valid_proposal(code="print('hello')"):
    return {
        "hypothesis": "La simulation confirme la théorie.",
        "protocol": ["Générer les données", "Comparer"],
        "seed": 42,
        "dependencies": [],
        "code": code,
        "expected_output": "Résultat de simulation",
    }


class V71ExperimentTests(unittest.TestCase):

    def test_generates_structured_proposal(self):
        provider = FakeProvider([
            LLMResponse(text=json.dumps(valid_proposal()))
        ])

        proposal = ExperimentAgent(provider).propose(
            "Propose une simulation"
        )

        self.assertEqual(proposal.seed, 42)
        self.assertEqual(proposal.dependencies, [])
        self.assertEqual(
            provider.start_calls[0]["tools"], []
        )
        self.assertEqual(provider.continue_calls, [])

    def test_rejects_invalid_json(self):
        provider = FakeProvider([
            LLMResponse(text="not valid JSON")
        ])

        with self.assertRaises(ValueError):
            ExperimentAgent(provider).propose("Test")

    def test_rejects_invalid_python(self):
        data = valid_proposal("def broken(")

        with self.assertRaises(ValidationError):
            parse_proposal(json.dumps(data))

    def test_rejects_unpinned_dependency(self):
        data = valid_proposal()
        data["dependencies"] = ["numpy"]

        with self.assertRaises(ValidationError):
            parse_proposal(json.dumps(data))

    def test_generated_code_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "should_not_exist.txt"

            code = (
                f"open({str(target)!r}, 'w').write('executed')"
            )

            provider = FakeProvider([
                LLMResponse(
                    text=json.dumps(valid_proposal(code))
                )
            ])

            ExperimentAgent(provider).propose("Test")

            self.assertFalse(target.exists())

    def test_rejects_tool_calls(self):
        provider = FakeProvider([
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="calculate",
                        arguments={"expression": "2 + 2"},
                    )
                ]
            )
        ])

        with self.assertRaises(ValueError):
            ExperimentAgent(provider).propose("Test")


if __name__ == "__main__":
    unittest.main()
