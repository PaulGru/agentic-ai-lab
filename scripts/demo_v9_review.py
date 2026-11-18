"""Draft a V9 retry pilot manuscript offline from the original journal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from analysis.retry_pilot import SOURCE_KIND
from execution.campaign import save_report
from research.critique.contracts import ReviewInput
from research.critique.evidence import load_input
from research.critique.graph import ReviewWorkflow, config_for
from research.critique.reporting import export_review
from research.critique.store import ReviewStore
from research.manuscript import export_manuscript_draft
from research.memory.contracts import MemoryBinding
from research.memory.integration import open_session


def local_path(path):
    if str(path).startswith(("\\\\", "//")) or "://" in str(path):
        raise ValueError("Use local files for the corpus, journal and exports")
    return Path(path).resolve()


def draft_pilot(corpus, journal, out_dir, review_id, project, *, memory_path=None):
    corpus, journal, root = (local_path(path) for path in (corpus, journal, out_dir))
    memory_path = local_path(memory_path) if memory_path is not None else root / "memory.sqlite3"
    output_stores = {(root / "review.sqlite3").resolve(), memory_path.resolve()}
    if {corpus, journal} & output_stores or len(output_stores) != 2:
        raise ValueError("The review and memory databases must be separate from the original sources")
    request = ReviewInput(review_id=review_id, source_kind=SOURCE_KIND, source_path=str(corpus), jobs_db=str(journal))
    bundle = load_input(request)  # validate before creating output stores
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    result = draft_pilot(args.corpus, args.journal, args.out_dir, args.review_id, args.project)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
