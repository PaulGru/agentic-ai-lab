"""Standalone, domain-neutral V8.4.1 canary specification."""
from execution.contracts import ExperimentJob, ResourceRequest, WorkloadSpec


def sample_job() -> ExperimentJob:
    return ExperimentJob(
        experiment_id="demo-experiment-001",
        job_id="synthetic-mean-001",
        workload=WorkloadSpec(
            name="synthetic.mean",
            parameters={"values": [1, 2, 3, 4, 5]},
        ),
        seed=42,
        resources=ResourceRequest(cpus=1, memory_mb=256, gpus=0, walltime_minutes=2),
        expected_artifacts=["metrics.json", "run_metadata.json"],
    )
