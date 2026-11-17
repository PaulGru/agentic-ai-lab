"""V8.5.1: normalize V8.4.6 outputs with a verified ledger as the trust boundary.

The only public ingestion path regenerates a V8.4.6 report from the durable
SQLite ledger using V8.4.6's integrity, job identity and metadata checks.
An unverified report JSON alone is intentionally NOT accepted as source data.
"""
from __future__ import annotations

from pathlib import Path

from analysis.contracts import (
    AnalysisDataset, MetricObservation, QualityAssessment,
    fingerprint, verify_export,
)
from execution.campaign import PilotProtocol, build_report, save_report
from execution.job_manager import DurableJobManager


def ingest_verified_pilot(
    protocol: PilotProtocol, manager: DurableJobManager,
) -> tuple[AnalysisDataset, QualityAssessment]:
    """Create a normalized dataset from completed, checked V8.4.6 jobs.

    This function never calls submit/refresh, never starts Docker or an LLM,
    and refuses missing, corrupt, or incomplete ledger artifacts.
    """
    report = build_report(protocol, manager)  # revalidates SQLite artifact checksums
    if (report["kind"] != "engineering_pilot_report"
            or report["campaign_id"] != protocol.campaign_id
            or report["protocol_sha256"] != protocol.fingerprint()
            or report["backend"] != "docker-local"
            or report["scientific_evidence"] is not False):
        raise ValueError("Pilot report provenance or scope is invalid")

    expected_jobs = protocol.jobs()
    if len(report["results"]) != len(expected_jobs):
        raise ValueError("Incomplete set of campaign jobs")

    observations = []
    for run, job, result in zip(protocol.runs, expected_jobs, report["results"]):
        if (result["run_id"] != run.run_id or result["job_id"] != job.job_id
                or result["job_sha256"] != job.fingerprint()):
            raise ValueError("Report run identity differs from approved protocol")
        metrics = result["metrics"]
        # V8.4.6 already checked count/mean/metadata against the approved job.
        observations.append(MetricObservation(
            observation_id=f"{job.job_id}-mean",
            run_id=run.run_id,
            job_id=job.job_id,
            job_sha256=result["job_sha256"],
            artifacts_sha256=result["artifacts_sha256"],
            metric_name="synthetic.mean",
            value=float(metrics["mean"]),
            input_value_count=metrics["count"],
            simulated=result["simulated"],
        ))

    dataset = AnalysisDataset(
        campaign_id=protocol.campaign_id,
        protocol_sha256=protocol.fingerprint(),
        source_report_sha256=fingerprint(report),
        observations=observations,
    )
    simulated_count = sum(obs.simulated for obs in observations)
    blockers = [
        "Engineering-only synthetic pilot; not a research study",
        "One aggregate per job; input_value_count is not a scientific replication count",
        "No independently replicated conditions or inferential design",
    ]
    warnings = []
    if simulated_count:
        warnings.append(f"{simulated_count} observation(s) use a simulated test transport")
    assessment = QualityAssessment(
        campaign_id=protocol.campaign_id,
        dataset_sha256=dataset.fingerprint(),
        observation_count=len(observations),
        verified_job_count=len({obs.job_id for obs in observations}),
        simulated_observation_count=simulated_count,
        blockers=blockers,
        warnings=warnings,
    )
    verify_export(dataset, assessment)
    return dataset, assessment


def export_analysis(
    dataset: AnalysisDataset, assessment: QualityAssessment, output_directory: str | Path,
) -> tuple[Path, Path]:
    """Save deterministic JSON; a divergent file can never be overwritten."""
    verify_export(dataset, assessment)
    out = Path(output_directory)
    data_path = out / "observations.json"
    quality_path = out / "quality.json"
    # Check BOTH outputs before writing either, avoiding known partial updates.
    from json import dumps
    for path, document in (
        (data_path, dataset.model_dump(mode="json")),
        (quality_path, assessment.model_dump(mode="json")),
    ):
        expected = dumps(document, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Existing analysis output differs: {path}")
    save_report(dataset.model_dump(mode="json"), data_path)
    save_report(assessment.model_dump(mode="json"), quality_path)
    return data_path, quality_path


def inspect_export(output_directory: str | Path) -> tuple[AnalysisDataset, QualityAssessment]:
    """Verify the consistency of exported JSON files, not source provenance."""
    out = Path(output_directory)
    dataset = AnalysisDataset.model_validate_json((out / "observations.json").read_text(encoding="utf-8"))
    quality = QualityAssessment.model_validate_json((out / "quality.json").read_text(encoding="utf-8"))
    verify_export(dataset, quality)
    return dataset, quality
