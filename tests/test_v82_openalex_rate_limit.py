"""Tests hors réseau pour le traitement borné des HTTP 429 OpenAlex."""

import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from research.literature_search import OpenAlexClient, OpenAlexRateLimitError


class DummyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return json.dumps({"results": [], "meta": {"count": 0}}).encode("utf-8")


def http429(*, remaining="100", retry_after=None, reset=None):
    headers = {"X-RateLimit-Remaining": remaining}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    if reset is not None:
        headers["X-RateLimit-Reset"] = str(reset)
    return HTTPError(
        "https://api.openalex.org/works", 429, "Too Many Requests", headers, None
    )


class OpenAlexRateLimitTests(unittest.TestCase):
    def test_transient_429_retries_only_once_when_next_attempt_succeeds(self):
        with patch("research.literature_search.urlopen", side_effect=[http429(retry_after=1), DummyResponse()]) as get:
            with patch("research.literature_search.sleep") as pause:
                result = OpenAlexClient()._get_json("https://api.openalex.org/works", {})
        self.assertEqual(result["results"], [])
        self.assertEqual(get.call_count, 2)
        pause.assert_called_once_with(1.0)

    def test_daily_exhaustion_is_not_retried(self):
        with patch("research.literature_search.urlopen", side_effect=http429(remaining="0", reset=40000)) as get:
            with patch("research.literature_search.sleep") as pause:
                with self.assertRaisesRegex(OpenAlexRateLimitError, "Daily budget appears exhausted"):
                    OpenAlexClient()._get_json("https://api.openalex.org/works", {})
        self.assertEqual(get.call_count, 1)
        pause.assert_not_called()

    def test_long_retry_after_is_not_waited_automatically(self):
        with patch("research.literature_search.urlopen", side_effect=http429(retry_after=300)) as get:
            with patch("research.literature_search.sleep") as pause:
                with self.assertRaisesRegex(OpenAlexRateLimitError, "Retry-After: 300"):
                    OpenAlexClient()._get_json("https://api.openalex.org/works", {})
        self.assertEqual(get.call_count, 1)
        pause.assert_not_called()

    def test_retries_are_bounded(self):
        with patch("research.literature_search.urlopen", side_effect=http429()) as get:
            with patch("research.literature_search.sleep") as pause:
                with self.assertRaises(OpenAlexRateLimitError):
                    OpenAlexClient()._get_json("https://api.openalex.org/works", {})
        self.assertEqual(get.call_count, 3)
        self.assertEqual(pause.call_count, 2)

    def test_other_http_error_does_not_retry(self):
        problem = HTTPError("https://api.openalex.org/works", 403, "Forbidden", {}, None)
        with patch("research.literature_search.urlopen", side_effect=problem) as get:
            with patch("research.literature_search.sleep") as pause:
                with self.assertRaisesRegex(RuntimeError, "HTTP error 403"):
                    OpenAlexClient()._get_json("https://api.openalex.org/works", {})
        self.assertEqual(get.call_count, 1)
        pause.assert_not_called()


if __name__ == "__main__":
    unittest.main()
