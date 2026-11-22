"""Rebuild an offline pilot critique and manuscript from verified evidence."""
from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver

from analysis.retry_pilot import SOURCE_KIND
from execution.campaign import save_report
from research.critique.contracts import ReviewInput
from research.critique.evidence import load_input, local_path
from research.critique.graph import ReviewWorkflow, config_for
from research.critique.reporting import export_review
from research.critique.store import ReviewStore
from research.manuscript import export_manuscript_draft
from research.memory.contracts import MemoryBinding
from research.memory.integration import open_session


def draft_pilot(corpus, journal, out_dir, review_id, project, *, memory_path=None, source_kind=SOURCE_KIND):
    corpus, journal, root = (local_path(path).resolve() for path in (corpus, journal, out_dir))
    memory_path = local_path(memory_path).resolve() if memory_path is not None else root / "memory.sqlite3"
    output_stores = {(root / "review.sqlite3").resolve(), memory_path.resolve()}
    if {corpus, journal} & output_stores or len(output_stores) != 2:
        raise ValueError("The review and memory databases must be separate from the original sources")
    request = ReviewInput(review_id=review_id, source_kind=source_kind, source_path=str(corpus), jobs_db=str(journal))
    bundle = load_input(request)  # Validate before creating output stores.
    store = ReviewStore(root / "review.sqlite3")
    memory_binding = MemoryBinding(project_id=project, path=str(memory_path))
    session = open_session(memory_binding, initialize=True)
    graph = ReviewWorkflow(store, memory=session).build(InMemorySaver())
    cfg = config_for(review_id)
    result = graph.invoke({"input_document": request.model_dump(mode="json")}, cfg, durability="sync")
    if result["status"] != "review_complete" or store.call_summaries(review_id):
        raise RuntimeError("Expected a complete offline critique")
    if bundle.model_dump(mode="json") != store.require("bundle", review_id):
        raise ValueError("Evidence changed during review")
    save_report(bundle.analysis.model_dump(mode="json"), root / "analysis.json")
    export_review(store, review_id, root / "review")
    manifest_path = export_manuscript_draft(store, review_id, root / "draft")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"review_id": review_id, "manuscript_sha256": manifest["manuscript_sha256"],
            "memory_indexed_items": result["memory_indexed_items"], "status": manifest["status"],
            "draft": str(manifest_path.parent / "manuscript.md")}
