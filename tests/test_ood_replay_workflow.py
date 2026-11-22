from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from agent.models import LLMResponse
from analysis.contracts import canonical_json, fingerprint
from execution.ood_replay import run_ood_replay
from providers.replay_provider import FakeProvider
from research.brief_workflow import BriefWorkflow, Decision, Revision
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from research.critique.store import ReviewStore
from research.literature_search import OpenAlexClient
from research.manuscript import verify_manuscript_export
from tests.test_v82_literature import FakeOpenAlex, make_work, mission_data, valid_review

BRIEF = (Path(__file__).parent / "fixtures" / "ood_replay_brief.md").read_text(encoding="utf-8")
QUESTION = "Comment la performance varie-t-elle lorsque les mêmes tâches sont formulées d'une façon différente ?"


def offline_workflow(directory):
    mission = mission_data()
    mission.update(
        title="Répétition fictive sur les reformulations",
        research_question=QUESTION,
        objective="Vérifier la reconstruction des écarts entre formulations appariées.",
        hypotheses=[{"id": "H1", "claim": "Changer la formulation peut réduire le taux de réponses exactes.",
                     "falsification_test": "Comparer les taux avec les deux formulations sur les mêmes tâches."}],
        independent_variables=["Formulation de référence ou décalée", "Identifiant du modèle simulé"],
        controls=["Mêmes tâches et mêmes réponses attendues"], baselines=["Formulation de référence"],
        literature_queries=["paraphrase robustness paired evaluation", "language model wording sensitivity"],
        experiment_outline=["Préparer deux formulations par tâche", "Comparer les taux appariés"],
        validity_risks=["Cette répétition utilise des réponses fictives"],
        success_criteria=["Reconstruire les écarts depuis les huit reçus"],
    )
    review = valid_review()
    review.update(findings=[{"work_id": "W301", "claim": "La notice fictive décrit des tâches appariées.",
                             "abstract_excerpt": "Equivalent prompts preserve expected answers"}],
                  possible_gaps=["Vérifier les autres formes de décalage lors d'une vraie étude."],
                  methodological_risks=["Une reformulation peut modifier la difficulté de la tâche."],
                  next_queries=["paired evaluation under wording shifts"],
                  questions_for_v83=["Comment conserver la réponse attendue en changeant la formulation ?"])
    search = FakeOpenAlex([
        make_work("W301", "Fictional notes on paired prompts", "Equivalent prompts preserve expected answers"),
        make_work("W302", "Fictional wording evaluation", "Paired tasks support comparisons under reformulation"),
        make_work("W303", "Fictional controls for shifted tasks", "Keep answers fixed when changing prompt wording"),
    ])
    provider = FakeProvider([LLMResponse(text=json.dumps(item)) for item in (mission, review)])
    return BriefWorkflow(directory, provider_factory=lambda _: provider,
                         search_factory=lambda: OpenAlexClient(api_key="", transport=search))


class OODReplayWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "ood"
        self.flow = offline_workflow(self.root)
        self.flow.start("ood-rehearsal", BRIEF, LLMSettings(provider="openai", model="gpt-6.2-astra"))

    def approve(self, stage):
        gate = self.flow.status()["pending"]
        self.assertEqual(gate["stage"], stage)
        return self.flow.approve(Decision(actor="test", target_sha256=gate["target_sha256"]))

    def ready(self):
        for stage in ("mission", "review_mission", "catalog", "literature", "review_literature", "review_protocol"):
            self.approve(stage)
        gate = self.flow.status()["pending"]
        self.assertEqual(gate["stage"], "ood_execution")
        return gate

    def test_ood_brief_to_approved_manuscript_with_verified_scores(self):
        gate = self.ready()
        details = gate["details"]
        self.assertEqual(len(details["jobs"]), 8)
        self.assertTrue(details["simulated"])
        self.assertFalse(details["gpu"])
        self.assertFalse(details["network"])
        self.assertFalse((self.root / "pilot-01" / "jobs.sqlite3").exists())
        self.assertEqual(read_document(self.root / "pilot-01" / "study.json")["protocol_sha256"],
                         details["protocol_sha256"])
        external = Mock(side_effect=AssertionError("No external work in the OOD replay"))
        self.flow = BriefWorkflow(self.root, provider_factory=external, search_factory=external,
                                  experiment_client_factory=external)
        awaiting_review = self.approve("ood_execution")
        self.assertEqual(awaiting_review["stage"], "review_manuscript")
        draft = Path(awaiting_review["draft"]["draft"])
        paper = draft.parent.parent
        scores = read_document(paper / "analysis.json")
        self.assertEqual(scores["degradation_points"], {"simule-a": 50.0, "simule-b": 0.0})
        self.assertEqual({key: arm["correct"] for key, arm in scores["arms"].items()}, {
            "simule-a-reference": 2, "simule-a-ood": 1, "simule-b-reference": 1, "simule-b-ood": 1,
        })
        self.assertEqual({row["job_sha256"] for row in scores["trials"]},
                         {item["sha256"] for item in details["jobs"]})
        self.assertTrue(all(len(row["artifacts_sha256"]) == 64 for row in scores["trials"]))
        self.assertTrue(scores["simulated"])
        self.assertFalse(scores["scientific_evidence"])
        self.assertFalse(scores["ready_for_scientific_inference"])
        text = draft.read_text(encoding="utf-8")
        self.assertIn(QUESTION, text)
        self.assertIn("50 points de pourcentage", text)
        store = ReviewStore(paper / "review.sqlite3", read_only=True)
        review_id = awaiting_review["draft"]["review_id"]
        report = store.require("review", review_id)
        self.assertIn("/analysis/arms/simule-a-ood", report["observation_refs"])
        self.assertEqual(report["issues"][0]["issue_id"], "ood_replay_scope")
        self.assertGreater(awaiting_review["draft"]["memory_indexed_items"], 0)
        self.assertEqual(store.call_summaries(review_id), [])
        complete = self.approve("review_manuscript")
        self.assertEqual(complete["status"], "complete")
        manifest = Path(complete["export"])
        manuscript = verify_manuscript_export(store, review_id, awaiting_review["draft"]["manuscript_sha256"],
                                              manifest.parent)
        self.assertEqual(manuscript.schema_version, "9.0-ood-replay")
        self.assertFalse(manuscript.scientific_evidence)
        external.assert_not_called()

    def test_protocol_revision_invalidates_the_old_execution_approval(self):
        old_gate = self.ready()
        status = self.flow.status()
        self.flow.revise(Revision(actor="test", target_sha256=status["checkpoint_sha256"],
                                  stage="protocol", reason="Vérifier la nouvelle version du protocole."))
        new_gate = self.approve("review_protocol")["pending"]
        self.assertNotEqual(new_gate["details"]["protocol_sha256"], old_gate["details"]["protocol_sha256"])
        with self.assertRaises(PermissionError):
            self.flow.approve(Decision(actor="test", target_sha256=old_gate["target_sha256"]))
        self.assertFalse((self.root / "pilot-01" / "jobs.sqlite3").exists())
        self.assertFalse((self.root / "pilot-02" / "jobs.sqlite3").exists())

    def test_missing_receipt_blocks_the_draft(self):
        self.ready()

        def omit_receipt(protocol, ledger):
            run_ood_replay(protocol, ledger)
            with closing(sqlite3.connect(ledger)) as db, db:
                db.execute("DELETE FROM jobs WHERE job_id=?", (protocol.jobs()[0].job_id,))

        with patch("research.brief_workflow.run_ood_replay", side_effect=omit_receipt):
            state = self.approve("ood_execution")
        self.assertEqual((state["status"], state["stage"]), ("blocked", "draft"))
        self.assertEqual(state["block"]["reason"], "IncompleteOODReceipt")
        self.assertIsNone(state["draft"])
        self.assertFalse((self.root / "pilot-01" / "paper" / "draft").exists())

    def test_manuscript_approval_refuses_a_rehashed_response(self):
        self.ready()
        state = self.approve("ood_execution")
        self.assertEqual(state["stage"], "review_manuscript")
        ledger = self.root / "pilot-01" / "jobs.sqlite3"
        with closing(sqlite3.connect(ledger)) as db, db:
            job, document = db.execute("SELECT job_id, artifacts_json FROM jobs ORDER BY job_id LIMIT 1").fetchone()
            artifacts = json.loads(document)
            artifacts["files"]["response.json"]["answer"] = "modifiée"
            db.execute("UPDATE jobs SET artifacts_json=?, artifacts_sha256=? WHERE job_id=?",
                       (canonical_json(artifacts), fingerprint(artifacts), job))
        with self.assertRaisesRegex(ValueError, "fixed fixture"):
            self.flow.approve(Decision(actor="test", target_sha256=state["pending"]["target_sha256"]))
        self.assertIsNone(self.flow.status()["export"])

    def test_recovery_after_interruption_reads_receipts_without_replaying(self):
        self.ready()

        def interrupt_after_receipts(protocol, ledger):
            run_ood_replay(protocol, ledger)
            raise KeyboardInterrupt

        with patch("research.brief_workflow.run_ood_replay", side_effect=interrupt_after_receipts):
            with self.assertRaises(KeyboardInterrupt):
                self.approve("ood_execution")
        self.flow = BriefWorkflow(self.root)
        status = self.flow.status()
        self.assertEqual(status["status"], "running")
        path = self.root / "pilot-01" / "jobs.sqlite3"
        before = path.read_bytes()
        with patch("research.brief_workflow.run_ood_replay") as replay:
            rebuilt = self.flow.recover(Decision(actor="test", target_sha256=status["checkpoint_sha256"]))
            replay.assert_not_called()
        self.assertEqual(rebuilt["stage"], "review_manuscript")
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
