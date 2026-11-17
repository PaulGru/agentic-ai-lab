"""Minimal local Docker transport, applying V7 sandbox hardening principles.

No host mounts, network, credentials, privileged mode, shell or arbitrary code.
The only program provided through stdin comes from execution.workloads.
"""
from __future__ import annotations

from dataclasses import dataclass
import subprocess
from threading import Event, Lock, Thread
import time
from uuid import uuid4

from execution.contracts import ExperimentJob

IMAGE = "agentic-lab-sandbox:v7.2"
MAX_OUTPUT_BYTES = 8_192


@dataclass(frozen=True)
class DockerOutcome:
    status: str  # success | error | timeout | cancelled | output_limit
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float


def docker_command(job: ExperimentJob, name: str) -> list[str]:
    """Pure command builder; limits checked by validate_workload first."""
    from execution.workloads import validate_workload
    validate_workload(job)
    memory = job.resources.memory_mb
    return [
        "docker", "run", "--rm", "-i", "--pull=never",
        "--name", name,
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true", "--security-opt=seccomp=builtin",
        "--pids-limit=32", f"--memory={memory}m", f"--memory-swap={memory}m",
        f"--cpus={job.resources.cpus}", "--shm-size=8m",
        "--ulimit=nofile=64:64", "--tmpfs=/tmp:rw,noexec,nosuid,size=16m,mode=1777",
        "--workdir=/tmp", "--user=10001:10001", "--log-driver=none",
        "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONUNBUFFERED=1",
        "--env=HOME=/tmp", "--env=TMPDIR=/tmp",
        IMAGE,
    ]


class DockerTransport:
    simulated = False

    def run(self, job: ExperimentJob, code: str, cancel: Event) -> DockerOutcome:
        name = f"agentic-v842-{uuid4().hex}"
        command = docker_command(job, name)
        start = time.monotonic()
        # Docker's own memory/pid/cpu isolation is enforced by the daemon.
        # The wall clock timeout is enforced here by killing the container.
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outputs = [bytearray(), bytearray()]
        total = [0]
        lock = Lock()
        overflow = Event()

        def drain(stream, destination):
            try:
                while chunk := stream.read(4096):
                    with lock:
                        remain = max(0, MAX_OUTPUT_BYTES - total[0])
                        destination.extend(chunk[:remain])
                        total[0] += len(chunk)
                        if total[0] > MAX_OUTPUT_BYTES:
                            overflow.set()
            finally:
                stream.close()

        def feed():
            try:
                process.stdin.write(code.encode("utf-8"))
            except (OSError, BrokenPipeError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        threads = [
            Thread(target=drain, args=(process.stdout, outputs[0]), daemon=True),
            Thread(target=drain, args=(process.stderr, outputs[1]), daemon=True),
            Thread(target=feed, daemon=True),
        ]
        reason = None
        try:
            for thread in threads:
                thread.start()
            while process.poll() is None:
                if cancel.is_set():
                    reason = "cancelled"
                    break
                if overflow.is_set():
                    reason = "output_limit"
                    break
                if time.monotonic() - start >= job.resources.walltime_minutes * 60:
                    reason = "timeout"
                    break
                time.sleep(0.05)
        finally:
            # Always remove the specific random container name, never user input.
            try:
                subprocess.run(["docker", "rm", "-f", name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                process.kill()
            process.wait()
            for thread in threads:
                thread.join(timeout=2)
        if reason is None and overflow.is_set():
            reason = "output_limit"
        status = reason or ("success" if process.returncode == 0 else "error")
        return DockerOutcome(
            status=status, exit_code=process.returncode,
            stdout=outputs[0].decode("utf-8", errors="replace"),
            stderr=outputs[1].decode("utf-8", errors="replace"),
            duration_seconds=round(time.monotonic() - start, 3),
        )
