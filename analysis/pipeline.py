"""V8.5 complete: side-effect-free reporting on *verified* V8.4.6 ledger data.

Optional synthetic replicated fixture is exclusively for validating mathematical
operators; no LLM calls, job submissions, or scientific efficacy claims.
"""
from __future__ import annotations

import json
from pathlib import Path

from analysis.contracts import fingerprint
from analysis.descriptive import describe_paired_fixture, describe_verified_pilot
from analysis.inference import infer_paired_fixture
from analysis.ingest import ingest_verified_pilot
from analysis.stat_contracts import AnalysisReport, PedagogicalPairedFixture
from analysis.validity import assess_paired_fixture, assess_verified_pilot
from execution.campaign import PilotProtocol, save_report
from execution.job_manager import DurableJobManager


def build_verified_pilot_analysis(protocol: PilotProtocol, manager: DurableJobManager) -> AnalysisReport:
    """The trust boundary is the V8.4.6 SQLite ledger (not a standalone JSON)."""
    dataset, quality = ingest_verified_pilot(protocol, manager)
    return AnalysisReport(
        source_kind="verified_pilot_aggregates", source_id=protocol.campaign_id,
        source_sha256=dataset.fingerprint(),
        descriptive=describe_verified_pilot(dataset, quality),
        validity=assess_verified_pilot(dataset, quality),
        inference=None,
    )


def build_fixture_analysis(fixture: PedagogicalPairedFixture) -> AnalysisReport:
    """Illustrate statistics on *explicitly synthetic* paired observations."""
    gate = assess_paired_fixture(fixture)
    return AnalysisReport(
        source_kind="pedagogical_paired_fixture", source_id=fixture.dataset_id,
        source_sha256=fixture.fingerprint(),
        descriptive=describe_paired_fixture(fixture), validity=gate,
        inference=infer_paired_fixture(fixture, validity=gate) if gate.ready_for_pedagogical_inference else None,
    )


def export_report(report: AnalysisReport, out_dir: str | Path) -> tuple[Path, Path]:
    """Idempotent export; never overwrite divergent reports/checksums."""
    output = Path(out_dir)
    data = output / "analysis_report.json"
    checksum = output / "analysis_report.sha256"
    text = json.dumps(report.model_dump(mode="json"), ensure_ascii=False,
                      sort_keys=True, indent=2, allow_nan=False) + "\n"
    digest = report.fingerprint() + "\n"
    for path, expected in ((data, text), (checksum, digest)):
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Existing analysis artifact differs: {path}")
    save_report(report.model_dump(mode="json"), data)
    output.mkdir(parents=True, exist_ok=True)
    if not checksum.exists():
        with checksum.open("x", encoding="utf-8") as dst:
            dst.write(digest)
    return data, checksum


def inspect_report(out_dir: str | Path) -> AnalysisReport:
    root = Path(out_dir)
    report = AnalysisReport.model_validate_json(
        (root / "analysis_report.json").read_text(encoding="utf-8"))
    recorded = (root / "analysis_report.sha256").read_text(encoding="utf-8").strip()
    if recorded != report.fingerprint():
        raise ValueError("Analysis report SHA-256 mismatch")
    return report
