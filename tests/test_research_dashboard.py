from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.brief_autonomy import AutonomyBudget, authorize_autonomy, autonomy_plan, run_autonomously
from research.brief_workflow import Decision, Revision
from research.critique.contracts import LLMSettings
from scripts.research_dashboard import Dashboard, timeline
from tests.test_ood_replay_workflow import BRIEF, offline_workflow


class DashboardTests(unittest.TestCase):
    def test_draft_revision_and_approval_keep_distinct_verified_archives(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name).resolve()
        workflow = offline_workflow(root / "mission")
        workflow.start("replay", BRIEF, LLMSettings(provider="openai", model="test"))
        budget = AutonomyBudget()
        plan = autonomy_plan(workflow, budget)
        authorize_autonomy(workflow, budget, Decision(actor="operator", target_sha256=plan["target_sha256"]))
        completed = run_autonomously(workflow)
        self.assertEqual(completed["status"], "complete")
        first_history = workflow.store.history()
        style = root / "acl.sty"
        style.write_text("fixture", encoding="utf-8")
        dashboard = Dashboard(workflow, compiler=root / "pdflatex", acl_style=style)

        def compile_fake(command, *, cwd, **options):
            self.assertIn("-no-shell-escape", command)
            self.assertTrue(options["timeout"] > 0)
            tex = (cwd / "manuscript.tex").read_text(encoding="utf-8")
            self.assertIn(r"\usepackage[preprint]{acl}", tex)
            self.assertIn(r"\begin{tabular}", tex)
            (cwd / "export" / "manuscript.pdf").write_bytes(b"%PDF-1.7\nfixture")
            return SimpleNamespace(returncode=0)

        with patch("research.acl_pdf.subprocess.run", side_effect=compile_fake) as compile_pdf:
            dashboard.archive_pending()
            first = dashboard.snapshot()["editions"]
            self.assertEqual([entry["approved"] for entry in first], [False, True])
            self.assertEqual(first_history, workflow.store.history())
            first_path = dashboard.pdf(first[0]["step"])
            first_bytes = first_path.read_bytes()
            status = workflow.revise(Revision(actor="operator", target_sha256=completed["checkpoint_sha256"],
                                              stage="draft", reason="Relire un nouveau brouillon."))
            self.assertEqual(status["stage"], "review_manuscript")
            dashboard.archive_pending()
            self.assertEqual(compile_pdf.call_count, 3)
            reopened = Dashboard(workflow, compiler=root / "pdflatex", acl_style=style)
            reopened.archive_pending()
            self.assertEqual(compile_pdf.call_count, 3)

        steps, _ = timeline(workflow.store.history())
        self.assertEqual(steps[-1]["version"], 2)
        self.assertEqual(steps[0]["highlights"], [])
        self.assertEqual(steps[-1]["highlights"][0]["label"], "Question de recherche")
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual([entry["status"] for entry in reopened.snapshot()["editions"]], ["ready"] * 3)
        self.assertIsNone(reopened.pdf(999))
        first_path.write_bytes(b"%PDF-1.7\naltered")
        with self.assertRaisesRegex(ValueError, "intégrité"):
            reopened.pdf(first[0]["step"])

    def test_tex_escapes_untrusted_statement_instead_of_running_it(self):
        from research.acl_pdf import render_acl

        statement = SimpleNamespace(section="methods", text=r"\input{private.txt} \write18{echo 1} 50%", statement_id="S001")
        manuscript = SimpleNamespace(schema_version="8.8", statements=[statement], figure=[], bibliography=[])
        tex = render_acl(manuscript, approved=False)
        self.assertIn(r"\textbackslash{}input\{private.txt\}", tex)
        self.assertIn(r"50\%", tex)
        self.assertNotIn(r"\write18", tex)
