"""Memory survives sessions, preserves source status, and enforces project scope."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis.contracts import fingerprint
from research.memory.contracts import MemoryContext, MemoryItem, SourceOrigin
from research.memory.service import ResearchMemory
from research.memory.stores import SQLiteMemoryStore
from tests.v86_helpers import design_context


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "mémoire avec espaces.db"
        self.store = SQLiteMemoryStore(self.path)
        self.store.setup()
        self.memory = ResearchMemory(self.store)
        self.source = self.root / "context.json"
        self.source.write_text(design_context().model_dump_json(), encoding="utf-8-sig")
        self.origin = SourceOrigin(kind="context", path=str(self.source))

    def test_checked_hypotheses_and_literature_keep_their_scopes(self):
        items = self.memory.ingest("project-A", self.origin)
        self.assertTrue({"hypothesis", "literature", "project_plan"} <= {i.kind for i in items})
        for item in items:
            self.assertFalse(item.scientific_evidence)
            self.assertFalse(item.action_authorization)
        result = self.memory.search("project-A", "récupération erreurs outils", role="design")
        self.assertTrue(result.hits)
        self.memory.verify_context(result)
        self.assertEqual(self.store.retrieval("project-A", result.fingerprint()), result)
        self.assertIsNone(self.store.retrieval("project-B", result.fingerprint()))
        self.assertEqual(self.memory.search("project-B", "récupération erreurs outils").hits, [])

    def test_deduplication_and_real_interpreter_restart(self):
        items = self.memory.ingest("project-A", self.origin)
        self.assertEqual(items, self.memory.ingest("project-A", self.origin))
        first = self.memory.search("project-A", "robustesse agents", role="planner")
        code = ("import sys; from research.memory.stores import SQLiteMemoryStore; "
                "from research.memory.service import ResearchMemory; "
                "s=SQLiteMemoryStore(sys.argv[1]); m=ResearchMemory(s); "
                "print(m.search('project-A','robustesse agents',role='planner').fingerprint())")
        process = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(self.path)],
                                 capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), first.fingerprint())

    def test_notes_are_tentative_then_human_reviewed_never_scientific(self):
        item = self.memory.note("project-A", title="Un enseignement à vérifier", text="La réplication doit être justifiée.",
                                logical_key="replication", actor="test-human", tags=["replication"])
        result = self.memory.search("project-A", "réplication")
        self.assertFalse(result.hits)
        self.assertEqual(result.omitted[0]["reason"], "tentative")
        with self.assertRaises(PermissionError):
            self.memory.change("project-A", item.memory_id, action="confirm", target_sha256="0" * 64,
                               actor="human", reason="Examen explicite de la note.")
        event = self.memory.change("project-A", item.memory_id, action="confirm", target_sha256=item.fingerprint(),
                                   actor="human", reason="Examen explicite de la note.")
        self.assertEqual(event, self.memory.change("project-A", item.memory_id, action="confirm", target_sha256=item.fingerprint(),
                                                   actor="human", reason="Examen explicite de la note."))
        result = self.memory.search("project-A", "réplication")
        self.assertEqual(result.hits[0].status, "human_reviewed")
        self.assertEqual(result.hits[0].verification, "unverified_note")
        self.assertFalse(result.hits[0].scientific_evidence)
        self.memory.verify_context(result)
        self.memory.change("project-A", item.memory_id, action="invalidate", target_sha256=item.fingerprint(),
                           actor="human", reason="La note doit être écartée de la recherche.")
        self.assertFalse(self.memory.search("project-A", "réplication").hits)
        with self.assertRaises(ValueError):
            self.memory.verify_context(result)
        with self.assertRaises(ValueError):
            self.memory.change("project-A", item.memory_id, action="confirm", target_sha256=item.fingerprint(),
                               actor="human", reason="Tentative de réactivation interdite.")

    def test_supersession_is_explicit_and_preserves_old_version(self):
        first = self.memory.note("project-A", title="Méthode précédente", text="Ancienne règle de réplication.",
                                 logical_key="replication", actor="human")
        second = self.memory.note("project-A", title="Méthode suivante", text="Nouvelle règle de réplication.",
                                  logical_key="replication", actor="human")
        for item in (first, second):
            self.memory.change("project-A", item.memory_id, action="confirm", target_sha256=item.fingerprint(),
                               actor="human", reason="La note a été examinée pour ce projet.")
        self.memory.change("project-A", first.memory_id, action="supersede", target_sha256=first.fingerprint(),
                           actor="human", reason="La nouvelle note remplace cette version.", replacement_id=second.memory_id)
        hits = self.memory.search("project-A", "réplication").hits
        self.assertEqual([hit.memory_id for hit in hits], [second.memory_id])
        self.assertEqual(self.memory.get("project-A", first.memory_id)["status"], "superseded")
        with self.assertRaises(ValueError):
            self.memory.change("project-A", second.memory_id, action="supersede", target_sha256=second.fingerprint(),
                               actor="human", reason="Création d'un cycle de remplacement interdite.", replacement_id=first.memory_id)
        with self.assertRaises(PermissionError):
            self.memory.change("project-B", second.memory_id, action="invalidate", target_sha256=second.fingerprint(),
                               actor="human", reason="Un autre projet ne peut pas modifier cette note.")

    def test_tamper_source_and_memory_records_fail_closed(self):
        items = self.memory.ingest("project-A", self.origin)
        result = self.memory.search("project-A", "robustesse agents")
        doc = json.loads(self.source.read_text(encoding="utf-8-sig"))
        doc["design_document"]["design"]["title"] = "Un titre altéré après approbation"
        self.source.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.memory.verify_context(result)
        # History can be inspected explicitly without pretending to revalidate absent/changed sources.
        self.assertFalse(self.memory.get("project-A", items[0].memory_id, verify_source=False)["source_provenance_rechecked"])
        with self.store.connect() as db:
            db.execute("UPDATE memories SET sha256=?", ("0" * 64,))
        with self.assertRaises(ValueError):
            self.memory.search("project-A", "robustesse agents")

    def test_missing_sources_are_excluded_and_reported(self):
        self.memory.ingest("project-A", self.origin)
        # Only this test-owned source is renamed; no user artifact is touched.
        self.source.rename(self.root / "source-moved.json")
        result = self.memory.search("project-A", "robustesse agents")
        self.assertFalse(result.hits)
        self.assertTrue(result.omitted)
        self.assertEqual(result.omitted[0]["reason"], "source_unavailable")

    def test_a_new_valid_source_version_does_not_silently_replace_old_knowledge(self):
        original_items = self.memory.ingest("project-A", self.origin)
        old = next(item for item in original_items if item.logical_key.startswith("design:"))
        doc = json.loads(self.source.read_text(encoding="utf-8-sig"))
        doc["design_document"]["design"]["analysis_plan"]["power_limitation"] = (
            "Une étude de puissance et la définition des unités indépendantes restent nécessaires.")
        from research.literature import digest_json
        digest = digest_json(doc["design_document"]["design"])
        doc["design_document"]["design_sha256"] = digest
        doc["matrix_document"]["design_sha256"] = digest
        self.source.write_text(json.dumps(doc), encoding="utf-8")
        result = self.memory.search("project-A", "robustesse agents")
        self.assertFalse(result.hits)
        self.assertTrue(all(item["reason"] == "source_changed" for item in result.omitted))
        new_items = self.memory.ingest("project-A", self.origin)
        new = next(item for item in new_items if item.logical_key == old.logical_key)
        self.assertNotEqual(old.memory_id, new.memory_id)
        self.memory.change("project-A", old.memory_id, action="supersede", target_sha256=old.fingerprint(),
                           replacement_id=new.memory_id, actor="human", reason="Ce protocole révisé remplace la version précédente.")
        history = self.memory.get("project-A", old.memory_id, verify_source=False)
        self.assertEqual(history["status"], "superseded")
        self.assertEqual(history["item"]["memory_id"], old.memory_id)

    def test_queries_are_bounded_and_not_sql_or_fts_instructions(self):
        self.memory.ingest("project-A", self.origin)
        for query in ("robustesse'; DROP TABLE memories; --", '"robustesse" OR outils:*', "???"):
            result = self.memory.search("project-A", query)
            self.assertIsInstance(result, MemoryContext)
        self.assertTrue(self.memory.search("project-A", "robustesse").hits)
        for options in ({"limit": 11}, {"limit": True}, {"max_bytes": 100}, {"max_bytes": True}):
            with self.assertRaises(ValueError):
                self.memory.search("project-A", "robustesse", **options)
        bounded = self.memory.search("project-A", "robustesse agents", max_bytes=2000)
        self.assertLessEqual(len(bounded.model_dump_json().encode("utf-8")), 2000)

    def test_cannot_promote_to_scientific_evidence_or_write_readonly_store(self):
        item = self.memory.ingest("project-A", self.origin)[0]
        doc = item.model_dump(mode="json")
        doc["scientific_evidence"] = True
        doc["memory_id"] = fingerprint({key: value for key, value in doc.items() if key != "memory_id"})
        with self.assertRaises(ValueError):
            MemoryItem.model_validate(doc)
        readonly = ResearchMemory(SQLiteMemoryStore(self.path, read_only=True))
        self.assertTrue(readonly.search("project-A", "robustesse", audit=False).hits)
        with self.assertRaises(PermissionError):
            readonly.search("project-A", "robustesse")
        with self.assertRaises(FileNotFoundError):
            SQLiteMemoryStore(self.root / "absent.db", read_only=True)


if __name__ == "__main__":
    unittest.main()
