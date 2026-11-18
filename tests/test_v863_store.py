from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.critique.contracts import LLMSettings, ReviewInput
from research.critique.llm import make_request
from research.critique.reporting import export_review, inspect_export
from research.critique.rules import deterministic_review
from research.critique.store import ReviewStore
from tests.v86_helpers import FIXTURE, sample_bundle


def prepared_store(path):
    store = ReviewStore(path)
    bundle = sample_bundle(False)
    settings = LLMSettings(provider="openai", model="fictional-test-model")
    request = ReviewInput(review_id="audit", source_kind=bundle.source_kind, source_path=str(FIXTURE),
                          llm=settings, assist_critique=True)
    report = deterministic_review(bundle)
    store.put("session", "audit", "audit", request.model_dump(mode="json"))
    store.put("bundle", "audit", "audit", bundle.model_dump(mode="json"))
    store.put("review", "audit", "audit", report.model_dump(mode="json"))
    return store, make_request(bundle, report, settings, review_id="audit", role="critique")


class StoreTests(unittest.TestCase):
    def test_append_only_and_integrity_across_real_process_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.db"
            store, request = prepared_store(path)
            self.assertIsNone(store.reserve_call(request))
            # A NEW interpreter must see the durable reservation and refuse a replay.
            code = ("from research.critique.store import ReviewStore, NeedsReconciliation; "
                    "from research.critique.contracts import LLMRequest; import sys; "
                    "s=ReviewStore(sys.argv[1]); r=LLMRequest.model_validate(s.call(sys.argv[2])['request']); "
                    "s.reserve_call(r)")
            result = subprocess.run([sys.executable, "-c", code, str(path), request.call_id],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("NeedsReconciliation", result.stderr)
            with self.assertRaises(ValueError):
                store.put("session", "audit", "audit", {"different": True})
            with store._connect() as db:
                db.execute("UPDATE artifacts SET sha256=? WHERE kind='review'", ("0" * 64,))
            with self.assertRaisesRegex(ValueError, "checksum"):
                store.require("review", "audit")

    def test_call_completion_cached_policy_bound_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, request = prepared_store(Path(tmp) / "review.db")
            changed = request.model_copy(update={"cycle": 2, "role": "revision"})
            with self.assertRaises(PermissionError):
                store.reserve_call(changed)
            store.reserve_call(request)
            store.finish_call(request, {"issues": []})
            self.assertEqual(ReviewStore(store.path).reserve_call(request), {"issues": []})
            with store._connect() as db:
                db.execute("UPDATE calls SET result_json=?", ('{"issues":[1]}',))
            with self.assertRaisesRegex(ValueError, "checksum"):
                store.call(request.call_id)

    def test_manifest_last_idempotent_and_partial_export_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = prepared_store(Path(tmp) / "review.db")
            destination = Path(tmp) / "révision export"
            export_review(store, "audit", destination)
            export_review(store, "audit", destination)
            self.assertFalse(inspect_export(destination)["source_provenance_rechecked"])
            (destination / "review.md").write_text("truncated", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                inspect_export(destination)
            with self.assertRaises(FileExistsError):
                export_review(store, "audit", destination)
            interrupted = Path(tmp) / "interrupted"
            original = Path.open

            def fail_manifest(path, *args, **kwargs):
                if path.name == "manifest.json" and args and args[0] == "xb":
                    raise OSError("simulated interruption")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", fail_manifest), self.assertRaises(OSError):
                export_review(store, "audit", interrupted)
            self.assertFalse((interrupted / "manifest.json").exists())
            with self.assertRaises(FileNotFoundError):
                inspect_export(interrupted)
            # Complete payloads without a marker can be safely finished without overwriting them.
            export_review(store, "audit", interrupted)
            inspect_export(interrupted)

    def test_readonly_store_never_creates_missing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing" / "review.db"
            with self.assertRaises(FileNotFoundError):
                ReviewStore(path, read_only=True)
            self.assertFalse(path.parent.exists())
            store, _ = prepared_store(Path(tmp) / "exists.db")
            reader = ReviewStore(store.path, read_only=True)
            self.assertIsNotNone(reader.get("review", "audit"))
            with self.assertRaises(PermissionError):
                reader.put("review", "new", "audit", {})


if __name__ == "__main__":
    unittest.main()
