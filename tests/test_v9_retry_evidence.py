from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from analysis.contracts import canonical_json, fingerprint
from analysis.retry_pilot import analyze_retry_pilot
from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import PhaseApproval, PilotJournal, approval_target, final_plan, run_phase
from research.critique.contracts import HumanDecision
from research.critique.evidence import read_document
from research.critique.store import ReviewStore
from research.manuscript import build_manuscript, export_manuscript, record_manuscript, verify_manuscript_export
from research.memory.contracts import MemoryBinding
from research.memory.integration import open_session
from research.memory.sources import pointer
from research.pilot_manuscript import draft_pilot
from tests.test_v9_retry_pilot_runner import ReplayClient


class RetryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.corpus_path = Path(__file__).resolve().parents[1] / "examples" / "v9_retry_pilot.json"
        cls.corpus = PilotCorpus.model_validate(read_document(cls.corpus_path))
        cls.journal = PilotJournal(cls.root / "simulated.sqlite3")
        plan = final_plan(cls.corpus)
        for phase in ("calibration", "measurement"):
            target = approval_target(plan, cls.journal, phase)
            decision = PhaseApproval(phase=phase, target_sha256=target["target_sha256"], actor="test")
            run_phase(cls.corpus, cls.journal, decision, client_factory=lambda: ReplayClient(cls.corpus))

    def test_verified_journal_critique_memory_and_draft(self):
        root = self.root / "draft-run"
        existing = self.journal.path.read_bytes()
        result = draft_pilot(self.corpus_path, self.journal.path, root, "test-v9", "test-project")
        self.assertEqual(draft_pilot(self.corpus_path, self.journal.path, root, "test-v9", "test-project"), result)
        self.assertEqual(existing, self.journal.path.read_bytes())
        self.assertEqual(result["status"], "awaiting_human_approval")
        self.assertEqual(result["memory_indexed_items"], 7)
        store = ReviewStore(root / "review.sqlite3")
        self.assertEqual(store.call_summaries("test-v9"), [])
        manuscript = build_manuscript(store, "test-v9")
        source = store.require("bundle", "test-v9")
        original = store.require("review", "test-v9")
        for statement in manuscript.statements:
            ref = statement.reference
            doc = source if ref.artifact == "bundle" else original
            self.assertEqual(ref.artifact_sha256, fingerprint(doc))
            self.assertEqual(ref.value_sha256, fingerprint(pointer(doc, ref.pointer)))
        self.assertIn("0/8 réponses exactes", Path(result["draft"]).read_text(encoding="utf-8"))
        self.assertIn("8/8 réponses exactes", Path(result["draft"]).read_text(encoding="utf-8"))
        self.assertEqual([row.mean for row in manuscript.figure], [100.0, 100.0, 0.0, 100.0])
        self.assertEqual(original["issues"][1]["issue_id"], "fault_profile")
        self.assertFalse(manuscript.scientific_evidence)
        session = open_session(MemoryBinding(project_id="test-project", path=str(root / "memory.sqlite3")), read_only=True)
        memory = session.memory.search("test-project", "relance exactitude outil", kinds=["observation"],
                                       scopes=["controlled_pilot"], audit=False)
        self.assertEqual(len(memory.hits), 4)
        self.assertTrue(any("0/8" in hit.excerpt for hit in memory.hits))
        self.assertTrue(all(not hit.scientific_evidence and not hit.action_authorization for hit in memory.hits))
        digest = manuscript.fingerprint()
        with self.assertRaises(KeyError):
            export_manuscript(store, "test-v9", digest, root / "approved")
        record_manuscript(store, manuscript, HumanDecision(decision="approve", target_sha256=digest, actor="test"))
        export_manuscript(store, "test-v9", digest, root / "approved")
        export_manuscript(store, "test-v9", digest, root / "approved")
        self.assertEqual(verify_manuscript_export(store, "test-v9", digest, root / "approved"), manuscript)

    def test_rehashed_score_tampering_is_rejected_by_reconstruction(self):
        journal = self.root / "altered.sqlite3"
        shutil.copyfile(self.journal.path, journal)
        with closing(sqlite3.connect(journal)) as db, db:
            index, text = db.execute("""SELECT seq, document FROM pilot_events
                WHERE phase='measurement' AND step='receipt' ORDER BY seq LIMIT 1""").fetchone()
            changed = json.loads(text)
            changed["answer_score"] = "incorrect"
            db.execute("UPDATE pilot_events SET document=?, sha256=? WHERE seq=?",
                       (canonical_json(changed), fingerprint(changed), index))
        with self.assertRaisesRegex(ValueError, "differs from raw replies"):
            analyze_retry_pilot(self.corpus, PilotJournal(journal))

    def test_incomplete_measurement_refuses_to_create_a_draft(self):
        journal = self.root / "partial.sqlite3"
        shutil.copyfile(self.journal.path, journal)
        with closing(sqlite3.connect(journal)) as db, db:
            db.execute("DELETE FROM pilot_events WHERE phase='measurement' AND step='completed'")
        output = self.root / "incomplete-run"
        with self.assertRaisesRegex(ValueError, "Phase is incomplete"):
            draft_pilot(self.corpus_path, journal, output, "partial", "test-project")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
