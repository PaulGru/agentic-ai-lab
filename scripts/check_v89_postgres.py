"""Exercise V8.9 CLI start and three decisions in separate processes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4

from research.critique.evidence import read_document
from research.critique.store import ReviewStore
from research.manuscript import verify_manuscript_export
from scripts.demo_v89 import load_request


ROOT = Path(__file__).resolve().parents[1]


def main():
    dsn = os.getenv("V89_CHECKPOINT_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("Configure an explicitly approved PostgreSQL test service")
    env = {**os.environ, "V89_CHECKPOINT_DATABASE_URL": dsn}
    run_id = "v89-ci-" + uuid4().hex[:22]
    with tempfile.TemporaryDirectory(prefix="v89-postgres-") as directory:
        root = Path(directory)
        request = load_request(ROOT / "examples" / "v89_workflow.json")
        document = root / "input.json"
        document.write_text(request.model_copy(update={"run_id": run_id}).model_dump_json(), encoding="utf-8")
        command = [sys.executable, "-X", "utf8", "-m", "scripts.demo_v89", "--root", str(root),
                   "--run-id", run_id]

        def run(*arguments, confirmation=""):
            result = subprocess.run([*command, *arguments], cwd=ROOT, env=env, input=confirmation,
                                    capture_output=True, encoding="utf-8", timeout=90)
            if result.returncode:
                raise RuntimeError(f"V8.9 CLI {arguments[0]} failed:\n{result.stderr}")
            # 'status' emits a single JSON document in its own interpreter.
            if arguments[0] == "status":
                return json.loads(result.stdout)

        run("start", "--input", str(document))
        for kind, choice in (("v89_execution_approval", "approve"), ("v86_revision_input", "skip"),
                             ("v89_manuscript_approval", "approve")):
            pending = run("status")["pending"]
            if pending["kind"] != kind:
                raise AssertionError(f"Expected {kind}, got {pending['kind']}")
            run("resume", "--decision", choice, "--actor", "ci-operator", "--target-sha256",
                pending["target_sha256"], confirmation="APPROVE\n" if choice == "approve" else "")
        if run("status")["workflow_status"] != "completed":
            raise AssertionError("Workflow did not complete")
        evaluation = read_document(root / "output" / "evaluation.json")
        if evaluation["llm_calls_count"] or len(evaluation["jobs"]) != 4:
            raise AssertionError("Unexpected job/API count")
        verify_manuscript_export(ReviewStore(root / "review.sqlite3", read_only=True), run_id,
                                 evaluation["manuscript_sha256"], root / "output" / "paper")
    print("V8.9 PostgreSQL: separate CLI processes resumed execution, review and manuscript approval.")


if __name__ == "__main__":
    main()
