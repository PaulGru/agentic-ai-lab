"""Explicit integration check: separate processes resume an isolated PG schema.

Requires V86_TEST_DATABASE_URL or DATABASE_URL. Creates a fresh v86_test_<uuid>
schema, never deletes schemas/tables/volumes, never calls an LLM or executor.
Run separately from unittest; missing configuration is an error, not a skip.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from uuid import uuid4

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from research.critique.contracts import RevisionDraft, ReviewInput
from research.critique.evidence import read_document
from research.critique.graph import ReviewWorkflow, config_for, pending_review
from research.critique.store import ReviewStore
from scripts.tutorials.scientific_critic import verify_recorded_review

ROOT = Path(__file__).resolve().parents[2]


def no_provider(_):
    raise AssertionError("PostgreSQL integration must never construct an LLM provider")


def worker(action: str, root: Path, schema: str, review_id: str, dsn: str):
    if not re.fullmatch(r"v86_test_[a-f0-9]{32}", schema):
        raise ValueError("Only a fresh integration-test schema may be used")
    scoped_dsn = make_conninfo(dsn, options=f"-csearch_path={schema}")
    with PostgresSaver.from_conn_string(scoped_dsn) as saver:
        saver.setup()
        store = ReviewStore(root / "reviews.db")
        graph = ReviewWorkflow(store, provider_factory=no_provider).build(saver)
        cfg = config_for(review_id)
        if action == "start":
            request = ReviewInput(review_id=review_id, source_kind="pedagogical_paired_fixture",
                                  source_path=str(ROOT / "examples" / "v85_paired_fixture.json"),
                                  context_path=str(ROOT / "examples" / "v86_design_context.json"))
            graph.invoke({"input_document": request.model_dump(mode="json")}, cfg, durability="sync")
            assert pending_review(graph.get_state(cfg))["kind"] == "v86_revision_input"
        elif action == "propose":
            pending = pending_review(graph.get_state(cfg))
            assert pending["kind"] == "v86_revision_input"
            draft = RevisionDraft.model_validate(read_document(ROOT / "examples" / "v86_revision_draft.json"))
            graph.invoke(Command(resume={"decision": "propose", "target_sha256": pending["target_sha256"],
                                         "actor": "integration-test", "draft": draft.model_dump(mode="json")}),
                         cfg, durability="sync")
            assert pending_review(graph.get_state(cfg))["kind"] == "v86_revision_approval"
        else:
            pending = pending_review(graph.get_state(cfg))
            assert pending["kind"] == "v86_revision_approval"
            graph.invoke(Command(resume={"decision": action, "target_sha256": pending["target_sha256"],
                                         "actor": "integration-test", "comment": ""}), cfg, durability="sync")
            state = graph.get_state(cfg)
            assert state.next == ()
            assert state.values["status"] == ("revision_approved" if action == "approve" else "revision_rejected")
            assert (store.get("approved_revision", review_id + "-1") is not None) == (action == "approve")
            verify_recorded_review(store, review_id)
        assert store.call_summaries(review_id) == []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["start", "propose", "approve", "reject"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--schema")
    parser.add_argument("--review")
    args = parser.parse_args()
    dsn = os.getenv("V86_TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("V86_TEST_DATABASE_URL or DATABASE_URL must point to an approved PostgreSQL test service")
    if args.worker:
        worker(args.worker, args.root, args.schema, args.review, dsn)
        return
    schema = "v86_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    with tempfile.TemporaryDirectory(prefix="v86-pg-") as tmp:
        for review_id, final_action in (("postgres-approve", "approve"), ("postgres-reject", "reject")):
            for action in ("start", "propose", final_action):
                subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.checks.critic_postgres",
                                "--worker", action, "--root", tmp, "--schema", schema, "--review", review_id],
                               check=True, cwd=ROOT, timeout=90)
    print("V8.6 PostgreSQL: approval/rejection resumed across six separate worker processes; no LLM or job submission.")
    print("Preserved isolated test schema:", schema)


if __name__ == "__main__":
    main()
