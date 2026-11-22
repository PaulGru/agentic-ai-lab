"""Explicit PostgreSQL research-memory integration in an isolated schema.

Uses only fictional examples. Preserves the created schema and all existing
data. Separate workers test cross-session retrieval and lifecycle decisions.
Requires RESEARCH_MEMORY_TEST_DATABASE_URL or DATABASE_URL; never skips.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research.critique.contracts import ReviewInput
from research.critique.graph import ReviewWorkflow, config_for, pending_review
from research.critique.store import ReviewStore
from research.memory.contracts import MemoryBinding, MemoryContext, SourceOrigin
from research.memory.integration import MemorySession
from research.memory.service import ResearchMemory
from research.memory.stores import PostgresMemoryStore

ROOT = Path(__file__).resolve().parents[2]


def worker(action, root, schema, dsn):
    if not re.fullmatch(r"v87_test_[a-f0-9]{32}", schema):
        raise ValueError("Only an isolated test schema is permitted")
    store = PostgresMemoryStore(dsn, schema_name=schema)
    memory = ResearchMemory(store)
    state_path = root / "state.json"
    if action == "seed":
        store.setup()
        memory.ingest("project-A", SourceOrigin(kind="context", path=str(ROOT / "examples" / "v86_design_context.json")))
        note = memory.note("project-A", title="Réplication à contrôler", text="La réplication demande des unités indépendantes.",
                           logical_key="replication", actor="test-human")
        memory.change("project-A", note.memory_id, action="confirm", target_sha256=note.fingerprint(), actor="test-human",
                       reason="Relecture explicite dans un test fictif.")
        context = memory.search("project-A", "réplication", kinds=["lesson"], scopes=["human_note"])
        assert [hit.memory_id for hit in context.hits] == [note.memory_id]
        state_path.write_text(json.dumps({"note": note.memory_id, "context": context.model_dump(mode="json")}), encoding="utf-8")
    elif action == "retrieve":
        state = json.loads(state_path.read_text(encoding="utf-8"))
        context = memory.search("project-A", "réplication", kinds=["lesson"], scopes=["human_note"])
        assert context.model_dump(mode="json") == state["context"]
        memory.verify_context(context)
        assert memory.search("project-B", "réplication").hits == []
        # A fresh critic receives PostgreSQL memory, then publishes its own checked review.
        binding = MemoryBinding(project_id="project-A", backend="postgres", schema_name=schema)
        session = MemorySession(memory, binding)
        reviews = ReviewStore(root / "reviews.db")
        graph = ReviewWorkflow(reviews, memory=session).build(InMemorySaver())
        request = ReviewInput(review_id="new-session", source_kind="pedagogical_paired_fixture",
                              source_path=str(ROOT / "examples" / "v85_paired_fixture.json"),
                              context_path=str(ROOT / "examples" / "v86_design_context.json"))
        cfg = config_for(request.review_id)
        graph.invoke({"input_document": request.model_dump(mode="json")}, cfg, durability="sync")
        assert reviews.require("memory_context", request.review_id)["hits"]
        assert graph.get_state(cfg).values["memory_indexed_items"] > 0
        pending = pending_review(graph.get_state(cfg))
        graph.invoke(Command(resume={"decision": "skip", "target_sha256": pending["target_sha256"], "actor": "test-human"}), cfg)
        assert reviews.call_summaries(request.review_id) == []
    elif action == "supersede":
        state = json.loads(state_path.read_text(encoding="utf-8"))
        original = store.item("project-A", state["note"])
        replacement = memory.note("project-A", title="Réplication précisée", text="La réplication sera définie par tâche indépendante.",
                                  logical_key="replication", actor="test-human")
        memory.change("project-A", replacement.memory_id, action="confirm", target_sha256=replacement.fingerprint(),
                       actor="test-human", reason="Nouvelle note examinée pour ce test fictif.")
        memory.change("project-A", original.memory_id, action="supersede", target_sha256=original.fingerprint(),
                       replacement_id=replacement.memory_id, actor="test-human", reason="La nouvelle version remplace cette note.")
        try:
            memory.verify_context(MemoryContext.model_validate(state["context"]))
        except ValueError:
            pass
        else:
            raise AssertionError("A superseded memory must invalidate the pending context")
        state["replacement"] = replacement.memory_id
        state_path.write_text(json.dumps(state), encoding="utf-8")
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        context = memory.search("project-A", "réplication", kinds=["lesson"], scopes=["human_note"])
        assert [hit.memory_id for hit in context.hits] == [state["replacement"]]
        assert context.hits[0].scientific_evidence is False
        assert context.hits[0].action_authorization is False
        assert memory.get("project-A", state["note"])["status"] == "superseded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["seed", "retrieve", "supersede", "verify"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--schema")
    args = parser.parse_args()
    dsn = os.getenv("RESEARCH_MEMORY_TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("Configure an explicitly approved PostgreSQL test service; this check is not skipped")
    if args.worker:
        worker(args.worker, args.root, args.schema, dsn)
        return
    schema = "v87_test_" + uuid4().hex
    with tempfile.TemporaryDirectory(prefix="v87-memory-") as root:
        for action in ("seed", "retrieve", "supersede", "verify"):
            subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.checks.memory_postgres", "--worker", action,
                            "--root", root, "--schema", schema], check=True, timeout=90, cwd=ROOT)
    print("V8.7 PostgreSQL memory: FTS retrieval, project isolation, critic integration and supersession passed across four processes.")
    print("Preserved isolated test schema:", schema)


if __name__ == "__main__":
    main()
