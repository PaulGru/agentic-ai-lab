from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.models import LLMResponse
from execution.ood_gpu import run_ood_gpu
from research.brief_autonomy import AutonomyBudget, authorize_autonomy, autonomy_plan, run_autonomously
from research.brief_workflow import BriefWorkflow, Decision, Revision
from research.critique.contracts import LLMSettings
from scripts.research_workflow import main
from tests.test_ood_gpu_workflow import CONFIG, fake_receipt
from tests.test_ood_replay_workflow import BRIEF, offline_workflow


class BriefAutonomyTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "workflow"
        self.flow = offline_workflow(self.root)
        self.provider = self.flow.provider_factory(None)
        self.search = self.flow.search_factory()
        self.settings = LLMSettings(provider="openai", model="gpt-6.2-astra")

    def start(self, *, gpu=True):
        if gpu:
            self.flow.start("calibration", BRIEF.replace("ood_replay", "ood_gpu"), self.settings, gpu_config=CONFIG)
        else:
            self.flow.start("calibration", BRIEF, self.settings)

    def authorize(self, budget=None):
        budget = budget or AutonomyBudget()
        plan = autonomy_plan(self.flow, budget)
        authorize_autonomy(self.flow, budget, Decision(actor="operator", target_sha256=plan["target_sha256"]))
        return plan

    def test_cli_one_confirmation_runs_repairs_and_exports_the_gpu_manuscript(self):
        self.start()
        self.provider.responses.insert(1, LLMResponse(text="{}"))
        search = self.search.search
        query_log = []

        def initially_too_broad(query, per_page):
            query_log.append(query)
            return ([], 0) if len(query_log) <= 2 else search(query, per_page)

        self.search.search = initially_too_broad
        self.flow.search_factory = lambda: self.search
        plan = autonomy_plan(self.flow, AutonomyBudget())
        with (patch("scripts.research_workflow.BriefWorkflow", return_value=self.flow),
              patch("dotenv.load_dotenv"), patch("builtins.input", return_value="APPROVE") as prompt,
              patch("execution.ood_gpu.ssh_receipt", side_effect=lambda _, request: fake_receipt(request)) as ssh,
              redirect_stdout(StringIO())):
            main(["--dir", str(self.root), "autonomy", "--actor", "operator", "--target-sha256", plan["target_sha256"]])
        prompt.assert_called_once()
        ssh.assert_called_once()
        history = self.flow.store.history()
        self.assertEqual(history[1]["detail"]["decision"]["actor"], "operator")
        self.assertEqual([e["detail"]["revision"]["stage"] for e in history if e["action"] == "approve_revision"],
                         ["catalog", "literature"])
        self.assertEqual(len(self.provider.start_calls), 3)
        self.assertEqual(len(query_log), 6)
        self.assertTrue(all(e["detail"]["decision"]["actor"] == f"autonomy:{plan['target_sha256']}"
                            for e in history if e["action"] == "reserve"))
        status = self.flow.status()
        self.assertEqual(status["status"], "complete", status)
        self.assertTrue(Path(status["export"]).is_file())
        log = StringIO()
        with redirect_stdout(log):
            main(["--dir", str(self.root), "watch", "--once"])
        self.assertIn("catalogue bibliographique insuffisant", log.getvalue())
        self.assertIn("Révision · retour à l’étape", log.getvalue())
        self.assertIn("Lot GPU terminé · reçu enregistré", log.getvalue())
        self.assertIn(status["export"], log.getvalue())
        self.assertEqual(self.flow.store.history(), history)

    def test_no_delegated_effect_without_matching_approval(self):
        self.start(gpu=False)
        with self.assertRaises(PermissionError):
            run_autonomously(self.flow)
        plan = autonomy_plan(self.flow, AutonomyBudget())
        with self.assertRaises(PermissionError):
            authorize_autonomy(self.flow, AutonomyBudget(llm_calls=5),
                               Decision(actor="operator", target_sha256=plan["target_sha256"]))
        self.assertEqual(len(self.flow.store.history()), 1)
        self.assertEqual(self.provider.start_calls, [])

    def test_replay_finishes_under_the_same_delegation_rules(self):
        self.start(gpu=False)
        plan = self.authorize()
        self.assertTrue(plan["details"]["scope"]["execution"]["simulated"])
        with patch("execution.ood_gpu.ssh_receipt") as ssh:
            completed = run_autonomously(self.flow)
        ssh.assert_not_called()
        self.assertEqual(completed["status"], "complete", completed)

    def test_watch_displays_committed_changes_without_repeating_old_steps(self):
        self.start(gpu=False)
        self.authorize()
        log = StringIO()

        def advance(_):
            self.assertIn("initialisée", log.getvalue())
            self.assertNotIn("Mission terminée", log.getvalue())
            run_autonomously(self.flow)

        with patch("scripts.research_progress.sleep", side_effect=advance) as wait, redirect_stdout(log):
            main(["--dir", str(self.root), "watch"])
        wait.assert_called_once()
        self.assertEqual(log.getvalue().count("initialisée"), 1)
        self.assertIn("Étape réservée · recherche bibliographique OpenAlex", log.getvalue())
        self.assertIn("Mission terminée", log.getvalue())

    def test_budget_is_still_exhausted_after_reopening_workflow(self):
        self.start(gpu=False)
        self.provider.responses[:0] = [LLMResponse(text="{}"), LLMResponse(text="{}")]
        self.authorize(AutonomyBudget(llm_calls=2))
        stopped = run_autonomously(self.flow)
        self.assertEqual(stopped["autonomy_stop"], "Budget d'appels au fournisseur épuisé.")
        resumed = BriefWorkflow(self.root, provider_factory=lambda _: self.provider)
        self.assertEqual(run_autonomously(resumed), stopped)
        self.assertEqual(resumed.status()["autonomy_stop"], stopped["autonomy_stop"])
        self.assertEqual(self.flow.store.current()["action"], "autonomy_pause")
        log = StringIO()
        with redirect_stdout(log):
            main(["--dir", str(self.root), "watch", "--once"])
        self.assertIn("Autonomie suspendue · Budget d'appels au fournisseur épuisé.", log.getvalue())
        self.assertEqual(len(self.provider.start_calls), 2)
        self.assertFalse((self.root / "pilot-02" / "jobs.sqlite3").exists())

    def test_changed_gpu_configuration_stops_before_execution(self):
        self.start()
        self.authorize()
        for stage in ("mission", "review_mission", "catalog", "literature", "review_literature"):
            gate = self.flow.status()["pending"]
            self.assertEqual(gate["stage"], stage)
            self.flow.approve(Decision(actor="operator", target_sha256=gate["target_sha256"]))
        status = self.flow.status()
        self.flow.revise(Revision(stage="protocol", actor="operator", target_sha256=status["checkpoint_sha256"],
                                  reason="Changer de carte", gpu_config={**CONFIG, "gpu_index": 0}))
        with patch("execution.ood_gpu.ssh_receipt") as ssh:
            self.assertEqual(run_autonomously(self.flow)["autonomy_stop"], "Le périmètre de la mission a changé.")
        ssh.assert_not_called()
        self.assertEqual(len(self.provider.start_calls), 2)

    def test_ambiguous_execution_stops_without_resubmission_on_restart(self):
        self.start()
        self.authorize()
        with patch("execution.ood_gpu.ssh_receipt", side_effect=TimeoutError) as ssh:
            stopped = run_autonomously(self.flow)
            self.assertEqual(stopped["stage"], "ood_gpu_execution")
            self.assertTrue(stopped["block"]["uncertain"])
            self.assertEqual(run_autonomously(BriefWorkflow(self.root)), stopped)
        ssh.assert_called_once()

    def test_restart_uses_saved_receipt_and_finishes_without_another_job(self):
        self.start()
        self.authorize()

        def interrupt_after_receipt(protocol, ledger):
            run_ood_gpu(protocol, ledger)
            raise KeyboardInterrupt

        with patch("execution.ood_gpu.ssh_receipt", side_effect=lambda _, request: fake_receipt(request)) as ssh:
            with patch("research.brief_workflow.run_ood_gpu", side_effect=interrupt_after_receipt):
                with self.assertRaises(KeyboardInterrupt):
                    run_autonomously(self.flow)
            self.assertTrue(self.flow.status()["block"]["uncertain"])
            resumed = run_autonomously(BriefWorkflow(self.root))
        ssh.assert_called_once()
        self.assertEqual(resumed["status"], "complete", resumed)
        self.assertIn("recover_local", [event["action"] for event in self.flow.store.history()])

    def test_failed_gpu_calibration_leaves_manuscript_for_review(self):
        self.start()
        self.authorize()

        def incorrect_reference(_, request):
            receipt = fake_receipt(request)
            receipt["responses"][0]["text"] = "0"
            return receipt

        with patch("execution.ood_gpu.ssh_receipt", side_effect=incorrect_reference):
            stopped = run_autonomously(self.flow)
        self.assertIn("calibration de référence a échoué", stopped["autonomy_stop"])
        self.assertEqual(stopped["stage"], "review_manuscript")
        self.assertIsNone(stopped["export"])
