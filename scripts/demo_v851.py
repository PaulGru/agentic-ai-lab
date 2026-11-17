"""V8.5.1 CLI — verified pilot ingestion and integrity-only export inspection.

PowerShell:
  python -m scripts.demo_v851 ingest --protocol examples/v846_protocol.json
  python -m scripts.demo_v851 inspect --out-dir runs_control/v851_analysis/v846-pilot-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.ingest import export_analysis, ingest_verified_pilot, inspect_export
from execution.backends.local import LocalDockerBackend
from execution.campaign import PilotProtocol
from execution.job_manager import DurableJobManager


def main() -> None:
    parser = argparse.ArgumentParser(description="V8.5.1: verified, synthetic results ingestion")
    commands = parser.add_subparsers(dest="action", required=True)
    ingest = commands.add_parser("ingest", help="Validate source SQLite ledger and normalize results")
    ingest.add_argument("--protocol", required=True)
    ingest.add_argument("--jobs-db", default="runs_control/v846_jobs.sqlite3")
    ingest.add_argument("--out-dir", help="Output directory (default runs_control/v851_analysis/<campaign_id>)")
    inspect = commands.add_parser("inspect", help="Check exported JSON consistency (does not re-check ledger)")
    inspect.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.action == "ingest":
        source_db = Path(args.jobs_db)
        if not source_db.is_file():
            parser.error(f"Source SQLite ledger not found: {source_db}; execute V8.4.6 first")
        protocol = PilotProtocol.model_validate_json(Path(args.protocol).read_text(encoding="utf-8"))
        # We only read the completed ledger. No Docker daemon, API keys, Postgres,
        # or permission to submit jobs is required for V8.5.1.
        manager = DurableJobManager(LocalDockerBackend(execution_authorized=False), source_db)
        dataset, quality = ingest_verified_pilot(protocol, manager)
        output_dir = args.out_dir or f"runs_control/v851_analysis/{protocol.campaign_id}"
        path1, path2 = export_analysis(dataset, quality, output_dir)
        print("Source: V8.4.6 verified SQLite ledger")
        print("Dataset:", path1)
        print("Quality:", path2)
    else:
        dataset, quality = inspect_export(args.out_dir)
        print("Export consistency: OK (source SQLite NOT re-verified)")
    print(json.dumps({
        "campaign_id": dataset.campaign_id,
        "observation_count": len(dataset.observations),
        "observations": {x.run_id: x.value for x in dataset.observations},
        "dataset_sha256": dataset.fingerprint(),
        "simulated_count": quality.simulated_observation_count,
        "ready_for_scientific_inference": quality.ready_for_scientific_inference,
        "blockers": quality.blockers,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
