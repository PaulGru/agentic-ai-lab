import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MemoryCLITests(unittest.TestCase):
    def run_cli(self, db, *args, input=""):
        return subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.demo_v87_memory",
                               "--db", str(db), "--project", "demo-project", *args],
                              cwd=ROOT, capture_output=True, input=input, encoding="utf-8", timeout=60)

    def test_offline_import_query_and_human_note_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mémoire.db"
            result = self.run_cli(path, "init")
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_cli(path, "import-context", "--path", "examples/v86_design_context.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_cli(path, "search", "--query", "robustesse agents", "--kind", "hypothesis")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout.split("Context SHA-256:")[0])
            self.assertTrue(payload["hits"])
            self.assertTrue(all(hit["kind"] == "hypothesis" for hit in payload["hits"]))
            result = self.run_cli(path, "note", "--title", "Réplication à vérifier", "--text", "Une note de réplication.",
                                  "--key", "replication", "--actor", "human")
            self.assertEqual(result.returncode, 0, result.stderr)
            from research.memory.stores import SQLiteMemoryStore
            item = SQLiteMemoryStore(path).candidates("demo-project", ["replication"])[0]
            result = self.run_cli(path, "confirm", "--memory-id", item.memory_id, "--target-sha256", item.fingerprint(),
                                  "--actor", "human", "--reason", "Relecture explicite de cette note.", input="NO\n")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(SQLiteMemoryStore(path).events("demo-project", item.memory_id), [])
            result = self.run_cli(path, "confirm", "--memory-id", item.memory_id, "--target-sha256", item.fingerprint(),
                                  "--actor", "human", "--reason", "Relecture explicite de cette note.", input="APPROVE\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_cli(path, "search", "--query", "réplication", "--scope", "human_note")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("human_reviewed", result.stdout)

    def test_memory_aware_critic_offline_cli_exports_and_publishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "memory.db"
            self.assertEqual(self.run_cli(path, "init").returncode, 0)
            self.assertEqual(self.run_cli(path, "import-context", "--path", "examples/v86_design_context.json").returncode, 0)
            result = subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.demo_v86", "--memory-db", str(path),
                                     "--project", "demo-project", "--store", str(root / "reviews.db"), "offline",
                                     "--input", "examples/v86_review_input.json", "--out-dir", str(root / "export")],
                                    cwd=ROOT, capture_output=True, encoding="utf-8", timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Mémoire de recherche consultée", (root / "export" / "review.md").read_text(encoding="utf-8"))
            result = self.run_cli(path, "search", "--query", "indépendance", "--kind", "lesson")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("assumed_independence", result.stdout)


if __name__ == "__main__":
    unittest.main()
