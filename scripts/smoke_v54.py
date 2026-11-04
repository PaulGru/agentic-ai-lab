import json
import os
import unittest
import uuid

from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.environ["AGENTIC_API_KEY"]
NOTES = Path("/app/data/v45_notes")


def call_api(method, path, body=None, key="valid"):
    headers = {}

    if key == "valid":
        headers["X-API-Key"] = API_KEY
    elif key == "invalid":
        headers["X-API-Key"] = "wrong-key"

    data = None

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        BASE_URL + path,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.loads(
            error.read().decode("utf-8")
        )


def new_thread(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class APISmokeTests(unittest.TestCase):

    def test_authentication(self):
        status, _ = call_api("GET", "/health")
        self.assertEqual(status, 200)

        status, _ = call_api(
            "GET", "/workflows/unknown", key="missing"
        )
        self.assertEqual(status, 401)

        status, _ = call_api(
            "GET", "/workflows/unknown", key="invalid"
        )
        self.assertEqual(status, 401)

        status, _ = call_api(
            "GET", "/workflows/unknown"
        )
        self.assertEqual(status, 404)

    def test_approval(self):
        thread = new_thread("v54_approve")

        status, result = call_api(
            "POST",
            "/workflows",
            {"thread_id": thread},
        )

        self.assertEqual(status, 201)
        self.assertEqual(result["status"], "pending_approval")
        self.assertEqual(
            result["approval"]["action"]["name"],
            "create_note_once",
        )

        note = NOTES / f"{thread}.txt"
        self.assertFalse(note.exists())

        status, _ = call_api(
            "POST",
            "/workflows",
            {"thread_id": thread},
        )
        self.assertEqual(status, 409)

        status, result = call_api(
            "GET", f"/workflows/{thread}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "pending_approval")

        status, result = call_api(
            "POST",
            f"/workflows/{thread}/decision",
            {"decision": "approve"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(note.exists())
        self.assertEqual(
            note.read_text(encoding="utf-8"),
            "125 * 48 = 6000",
        )

        status, _ = call_api(
            "POST",
            f"/workflows/{thread}/decision",
            {"decision": "approve"},
        )
        self.assertEqual(status, 409)

    def test_rejection(self):
        thread = new_thread("v54_reject")

        status, result = call_api(
            "POST", "/workflows", {"thread_id": thread}
        )
        self.assertEqual(status, 201)

        status, result = call_api(
            "POST",
            f"/workflows/{thread}/decision",
            {"decision": "reject"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(
            (NOTES / f"{thread}.txt").exists()
        )

    def test_edit(self):
        thread = new_thread("v54_edit")

        status, _ = call_api(
            "POST", "/workflows", {"thread_id": thread}
        )
        self.assertEqual(status, 201)

        status, _ = call_api(
            "POST",
            f"/workflows/{thread}/decision",
            {"decision": "edit", "content": ""},
        )
        self.assertEqual(status, 422)

        corrected = "125 * 48 = 6000 (validated by human)"

        status, result = call_api(
            "POST",
            f"/workflows/{thread}/decision",
            {
                "decision": "edit",
                "content": corrected,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            (NOTES / f"{thread}.txt").read_text(
                encoding="utf-8"
            ),
            corrected,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)