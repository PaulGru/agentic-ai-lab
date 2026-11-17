"""Complete V8.5 CLI. No API requests or job submissions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.pipeline import (
    build_fixture_analysis, build_verified_pilot_analysis, export_report, inspect_report,
)
from analysis.stat_contracts import AnalysisReport, PedagogicalPairedFixture
from execution.campaign import PilotProtocol


def _read_doc(path: str) -> dict:
    source = Path(path)
    if source.stat().st_size > 65_536:
        raise ValueError("Input JSON exceeds 64 KiB")
    return json.loads(source.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="V8.5 verified descriptive + pedagogical inference")
    sub = parser.add_subparsers(dest="action", required=True)
    pilot = sub.add_parser("pilot", help="Reverify existing V8.4.6 SQLite ledger and describe aggregates")
    pilot.add_argument("--protocol", default="examples/v846_protocol.json")
    pilot.add_argument("--jobs-db", default="runs_control/v846_jobs.sqlite3")
    fixture = sub.add_parser("fixture", help="Pedagogical paired dataset, no real scientific evidence")
    fixture.add_argument("--fixture", default="examples/v85_paired_fixture.json")
    for cmd in (pilot, fixture):
        cmd.add_argument("--out-dir", default=None)
        cmd.add_argument("--graph", action="store_true", help="Use read-only LangGraph DAG")
    inspect = sub.add_parser("inspect", help="Check exported report integrity (not ledger provenance)")
    inspect.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.action == "inspect":
        report = inspect_report(args.out_dir)
        print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
        print("Integrity: exported SHA-256 verified; source provenance NOT rechecked")
        return

    manager = None
    if args.action == "pilot":
        from execution.backends.local import LocalDockerBackend
        from execution.job_manager import DurableJobManager
        protocol = PilotProtocol.model_validate(_read_doc(args.protocol))
        if not Path(args.jobs_db).is_file():
            parser.error("V8.4.6 jobs database absent; run the approved campaign first")
        manager = DurableJobManager(LocalDockerBackend(execution_authorized=False), args.jobs_db)
        report = build_verified_pilot_analysis(protocol, manager)
        state = {"source_kind": "verified_pilot_aggregates",
                 "protocol_document": protocol.model_dump(mode="json")}
    else:
        example = PedagogicalPairedFixture.model_validate(_read_doc(args.fixture))
        report = build_fixture_analysis(example)
        state = {"source_kind": "pedagogical_paired_fixture",
                 "fixture_document": example.model_dump(mode="json")}

    if args.graph:
        from langgraph.checkpoint.memory import MemorySaver
        from analysis.graph import AnalysisWorkflow
        graph = AnalysisWorkflow(manager=manager).build(MemorySaver())
        result = graph.invoke(state, config={"configurable": {"thread_id": "v85-" + report.source_sha256}})
        if result["status"] != "completed":
            raise RuntimeError("LangGraph analysis did not complete")
        graphed = AnalysisReport.model_validate(result["report_document"])
        if graphed != report:
            raise RuntimeError("LangGraph and standalone analysis disagree")
        report = graphed

    destination = args.out_dir or f"runs_control/v85_analysis/{report.source_id}"
    outputs = export_report(report, destination)
    print("Report:", outputs[0])
    print("SHA-256:", outputs[1])
    print(json.dumps({
        "source_kind": report.source_kind,
        "source_id": report.source_id,
        "source_sha256": report.source_sha256,
        "descriptive": report.descriptive.model_dump(mode="json"),
        "validity": report.validity.model_dump(mode="json"),
        "inference": report.inference.model_dump(mode="json") if report.inference else None,
        "scientific_evidence": report.scientific_evidence,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
