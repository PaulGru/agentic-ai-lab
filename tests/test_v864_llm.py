from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent.models import LLMResponse, ToolCall
from research.critique.contracts import HumanDecision
from research.critique.llm import call_llm, create_bounded_provider, validate_suggestions
from research.critique.rules import deterministic_review
from research.critique.store import NeedsReconciliation
from tests.fakes import FakeProvider
from tests.test_v863_store import prepared_store
from tests.v86_helpers import sample_bundle


class LLMTests(unittest.TestCase):
    def test_exact_approval_then_completed_call_is_reused_without_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, request = prepared_store(Path(tmp) / "calls.db")
            bundle = sample_bundle(False)
            validate = lambda text: validate_suggestions(text, bundle, deterministic_review(bundle))
            factory = Mock(return_value=FakeProvider([LLMResponse(text='{"issues":[]}')]))
            wrong = HumanDecision(decision="approve", target_sha256="0" * 64, actor="operator")
            with self.assertRaises(PermissionError):
                call_llm(request, wrong, store, validate=validate, provider_factory=factory)
            factory.assert_not_called()
            decision = wrong.model_copy(update={"target_sha256": request.fingerprint()})
            self.assertEqual(call_llm(request, decision, store, validate=validate, provider_factory=factory), {"issues": []})
            self.assertEqual(call_llm(request, decision, store, validate=validate, provider_factory=factory), {"issues": []})
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.return_value.start_calls[0]["tools"], [])

    def test_invalid_tool_search_and_fabricated_reference_have_no_retry(self):
        responses = [
            LLMResponse(text="not JSON"),
            LLMResponse(text='{"issues":[]}', tool_calls=[ToolCall(id="x", name="shell", arguments={})]),
            LLMResponse(text='{"issues":[]}', web_search_used=True),
            LLMResponse(text='{"issues":[],"scientific_evidence":true}'),
            LLMResponse(text='{"issues":[{"category":"methodology","severity":"minor",'
                             '"finding":"A fabricated concern","recommendation":"Check this concern manually",'
                             '"evidence_refs":["/invented"]}]}'),
        ]
        for response in responses:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as tmp:
                store, request = prepared_store(Path(tmp) / "calls.db")
                decision = HumanDecision(decision="approve", target_sha256=request.fingerprint(), actor="operator")
                bundle = sample_bundle(False)
                validate = lambda text: validate_suggestions(text, bundle, deterministic_review(bundle))
                factory = Mock(return_value=FakeProvider([response]))
                with self.assertRaises(ValueError):
                    call_llm(request, decision, store, validate=validate, provider_factory=factory)
                self.assertEqual(store.call(request.call_id)["phase"], "invalid_response")
                with self.assertRaises(NeedsReconciliation):
                    call_llm(request, decision, store, validate=validate, provider_factory=factory)
                self.assertEqual(factory.call_count, 1)

    def test_timeout_records_ambiguity_without_leaking_error_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, request = prepared_store(Path(tmp) / "calls.db")
            factory = Mock(side_effect=TimeoutError("secret-must-not-be-stored"))
            decision = HumanDecision(decision="approve", target_sha256=request.fingerprint(), actor="operator")
            with self.assertRaises(RuntimeError):
                call_llm(request, decision, store, validate=lambda s: {}, provider_factory=factory)
            self.assertEqual(store.call(request.call_id)["phase"], "needs_reconciliation")
            self.assertNotIn("secret-must-not-be-stored", str(store.call(request.call_id)))

    def test_both_sdk_adapters_receive_no_retries_timeout_and_token_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, request = prepared_store(Path(tmp) / "calls.db")
            openai_client = Mock()
            openai_client.responses.create.return_value = SimpleNamespace(id="response", output_text='{"issues":[]}', output=[])
            with patch("providers.openai_provider.OpenAI", return_value=openai_client) as constructor:
                provider = create_bounded_provider(request)
                provider.start("bounded prompt", tools=[])
                constructor.assert_called_once_with(max_retries=0, timeout=60.0)
            kwargs = openai_client.responses.create.call_args.kwargs
            self.assertEqual(kwargs["max_output_tokens"], 4000)
            self.assertNotIn("tools", kwargs)
            anthropic_client = Mock()
            block = Mock(type="text", text='{"issues":[]}', citations=[])
            block.model_dump.return_value = {"type": "text", "text": block.text}
            anthropic_client.messages.create.return_value = SimpleNamespace(content=[block])
            settings = request.settings.model_copy(update={"provider": "anthropic"})
            with patch("providers.anthropic_provider.Anthropic", return_value=anthropic_client) as constructor:
                provider = create_bounded_provider(request.model_copy(update={"settings": settings}))
                provider.start("bounded prompt", tools=[])
                constructor.assert_called_once_with(max_retries=0, timeout=60.0)
            kwargs = anthropic_client.messages.create.call_args.kwargs
            self.assertEqual(kwargs["max_tokens"], 4000)
            self.assertNotIn("tools", kwargs)


if __name__ == "__main__":
    unittest.main()
