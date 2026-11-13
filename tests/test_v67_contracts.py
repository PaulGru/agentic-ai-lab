"""V6.7 — API contracts without PostgreSQL or real LLM calls."""

import unittest

from pydantic import ValidationError

from api.app import app
from api.models import ResearchRequest
from api.routes.research import pipeline_config, research_config


class V67ContractsTests(unittest.TestCase):
    def test_required_routes_are_protected(self):
        specification = app.openapi()
        security_schemes = specification["components"]["securitySchemes"]

        # Find the API-key scheme without assuming FastAPI's generated name.
        api_key_schemes = {
            name
            for name, details in security_schemes.items()
            if details.get("type") == "apiKey"
            and details.get("in") == "header"
            and details.get("name") == "X-API-Key"
        }
        self.assertTrue(api_key_schemes)

        protected_operations = {
            ("/workflows", "post"),
            ("/workflows/{thread_id}", "get"),
            ("/workflows/{thread_id}/decision", "post"),
            ("/research", "post"),
            ("/research/{thread_id}", "get"),
            ("/research/pipeline", "post"),
            ("/research/pipeline/{thread_id}", "get"),
        }

        for path, method in protected_operations:
            with self.subTest(path=path, method=method):
                operation = specification["paths"][path][method]
                self.assertTrue(
                    any(
                        api_key_schemes.intersection(requirement)
                        for requirement in operation.get("security", [])
                    ),
                    f"Missing X-API-Key security for {method.upper()} {path}",
                )

        self.assertNotIn("security", specification["paths"]["/health"]["get"])

    def test_research_request_validation(self):
        valid = ResearchRequest(thread_id="valid-123", question="  Recherche utile  ")
        self.assertEqual(valid.question, "Recherche utile")

        invalid_payloads = [
            {"thread_id": "bad id", "question": "ok"},
            {"thread_id": "a" * 65, "question": "ok"},
            {"thread_id": "", "question": "ok"},
            {"thread_id": "valid", "question": "   "},
            {"thread_id": "valid", "question": "x" * 4001},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=repr(payload)[:60]):
                with self.assertRaises(ValidationError):
                    ResearchRequest.model_validate(payload)

    def test_checkpoint_namespaces_stay_stable(self):
        self.assertEqual(
            research_config("abc")["configurable"]["thread_id"],
            "v6-research:abc",
        )
        self.assertEqual(
            pipeline_config("abc")["configurable"]["thread_id"],
            "v64-pipeline:abc",
        )


if __name__ == "__main__":
    unittest.main()
