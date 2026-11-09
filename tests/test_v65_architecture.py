
import unittest

from api_v61 import app


class V65ArchitectureTests(unittest.TestCase):

    def test_existing_api_routes(self):
        registered = {
            (method, route.path)
            for route in app.routes
            for method in getattr(route, "methods", [])
        }

        expected = {
            ("GET", "/health"),
            ("POST", "/workflows"),
            ("GET", "/workflows/{thread_id}"),
            ("POST", "/workflows/{thread_id}/decision"),
            ("POST", "/research"),
            ("GET", "/research/{thread_id}"),
            ("POST", "/research/pipeline"),
            ("GET", "/research/pipeline/{thread_id}"),
        }

        self.assertTrue(
            expected.issubset(registered),
            f"Missing API routes: {expected - registered}",
        )

    def test_current_version(self):
        self.assertEqual(app.version, "6.4")


if __name__ == "__main__":
    unittest.main()
