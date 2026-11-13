"""V6.7 — HTTP smoke tests, intentionally without any paid LLM calls."""

import unittest

from scripts.smoke_v54 import call_api, new_thread


class ResearchHTTPTests(unittest.TestCase):
    def test_research_endpoints_require_api_key(self):
        payload = {"thread_id": new_thread("v67_auth"), "question": "Test"}
        endpoints = [
            ("POST", "/research", payload),
            ("POST", "/research/pipeline", payload),
            ("GET", "/research/example", None),
            ("GET", "/research/pipeline/example", None),
        ]
        for method, path, body in endpoints:
            for key in ("missing", "invalid"):
                with self.subTest(method=method, path=path, key=key):
                    status, _ = call_api(method, path, body, key=key)
                    self.assertEqual(status, 401)

    def test_research_rejects_invalid_requests_before_llm(self):
        invalid_payloads = [
            {"thread_id": "invalid id", "question": "Test"},
            {"thread_id": new_thread("v67_empty"), "question": "   "},
            {"thread_id": new_thread("v67_long"), "question": "x" * 4001},
        ]
        for path in ("/research", "/research/pipeline"):
            for payload in invalid_payloads:
                with self.subTest(path=path, payload=payload["thread_id"]):
                    status, _ = call_api("POST", path, payload)
                    self.assertEqual(status, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
