"""Regression: demo V8.4.3 and V8.4.4 may share a persisted simulator job."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from execution.contracts import ExperimentJob
from execution.backends.slurm_sim import SlurmSimBackend
from execution.job_manager import DurableJobManager
from scripts.tutorials.durable_jobs import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "v841_job.json"


class DemoV844ResumeTests(unittest.TestCase):
    def test_already_completed_simulated_job_is_not_advanced_again(self):
        with tempfile.TemporaryDirectory() as d:
            sim_db = str(Path(d) / "sim.db")
            jobs_db = str(Path(d) / "jobs.db")
            job = ExperimentJob.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))
            backend = SlurmSimBackend(sim_db)
            handle = backend.submit(job)  # V8.4.3 simulation
            backend.advance(handle)
            backend.advance(handle)
            DurableJobManager(backend, jobs_db).submit(job)  # V8.4.4 ledger sees submitted
            argv = ["durable_jobs", "advance", "--job", str(EXAMPLE),
                    "--sim-db", sim_db, "--db", jobs_db]
            out = io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                main()
            self.assertIn("already succeeded", out.getvalue())
            self.assertIn('"phase": "succeeded"', out.getvalue())

    def test_fresh_simulated_job_advances_twice(self):
        with tempfile.TemporaryDirectory() as d:
            sim_db = str(Path(d) / "sim.db")
            jobs_db = str(Path(d) / "jobs.db")
            job = ExperimentJob.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))
            backend = SlurmSimBackend(sim_db)
            manager = DurableJobManager(backend, jobs_db)
            manager.submit(job)
            argv = ["durable_jobs", "advance", "--job", str(EXAMPLE),
                    "--sim-db", sim_db, "--db", jobs_db]
            for expected in ("running", "succeeded"):
                out = io.StringIO()
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                    main()
                self.assertIn(f'"phase": "{expected}"', out.getvalue())

if __name__ == "__main__":
    unittest.main()
