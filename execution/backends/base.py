"""Abstract interface shared by future Docker and Slurm backends."""
from abc import ABC, abstractmethod

from execution.contracts import (
    ExperimentJob, JobArtifacts, JobHandle, JobSnapshot,
)


class ExecutionBackend(ABC):
    @abstractmethod
    def submit(self, job: ExperimentJob) -> JobHandle:
        """Submit an APPROVED executable job in real backends.

        This generic interface does not itself provide approval or isolation.
        """

    @abstractmethod
    def status(self, handle: JobHandle) -> JobSnapshot:
        """Fetch current state by backend handle."""

    @abstractmethod
    def cancel(self, handle: JobHandle) -> JobSnapshot:
        """Request cancellation (may race with completion on a real backend)."""

    @abstractmethod
    def collect(self, handle: JobHandle) -> JobArtifacts:
        """Collect outputs only after the job succeeds."""
