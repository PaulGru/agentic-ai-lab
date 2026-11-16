
"""Sandbox Docker appelé depuis le système hôte."""

from dataclasses import dataclass
import subprocess
from threading import Event, Lock, Thread
import time
from uuid import uuid4


IMAGE = "agentic-lab-sandbox:v7.2"
TIMEOUT_SECONDS = 10
MAX_OUTPUT_BYTES = 8192


@dataclass(frozen=True)
class SandboxResult:
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float


def build_command(name: str) -> list[str]:
    """Construit la commande Docker avec des restrictions fixes."""
    return [
        "docker", "run", "--rm", "-i", "--pull=never",
        "--name", name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--security-opt=seccomp=builtin",
        "--pids-limit=32",
        "--memory=128m",
        "--memory-swap=128m",
        "--cpus=0.5",
        "--shm-size=8m",
        "--ulimit=nofile=64:64",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m,mode=1777",
        "--workdir=/tmp",
        "--user=10001:10001",
        "--log-driver=none",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONUNBUFFERED=1",
        "--env=HOME=/tmp",
        "--env=TMPDIR=/tmp",
        IMAGE,
    ]


class SandboxRunner:
    def run(
        self,
        code: str,
        *,
        approved: bool = False,
        dependencies: list[str] | None = None,
    ) -> SandboxResult:
        if not approved:
            raise PermissionError(
                "Human approval required"
            )

        if dependencies:
            raise ValueError(
                "External dependencies are disabled in V7.2"
            )

        if not isinstance(code, str) or not code.strip():
            raise ValueError("Python code cannot be empty")

        if len(code.encode("utf-8")) > 12000:
            raise ValueError(
                "Python code exceeds 12000 bytes"
            )

        name = f"agentic-v72-{uuid4().hex}"
        started = time.monotonic()

        process = subprocess.Popen(
            build_command(name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        buffers = [bytearray(), bytearray()]
        total = [0]
        lock = Lock()
        exceeded = Event()

        def drain(stream, destination):
            """Lit une sortie sans dépasser la limite mémoire."""
            try:
                while chunk := stream.read(4096):
                    with lock:
                        remaining = max(
                            0, MAX_OUTPUT_BYTES - total[0]
                        )
                        destination.extend(chunk[:remaining])
                        total[0] += len(chunk)

                        if total[0] > MAX_OUTPUT_BYTES:
                            exceeded.set()
            finally:
                stream.close()

        def feed_input():
            """Transmet le code sans bloquer le timeout principal."""
            try:
                process.stdin.write(code.encode("utf-8"))
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        threads = [
            Thread(
                target=drain,
                args=(process.stdout, buffers[0]),
                daemon=True,
            ),
            Thread(
                target=drain,
                args=(process.stderr, buffers[1]),
                daemon=True,
            ),
            Thread(target=feed_input, daemon=True),
        ]

        reason = None

        try:
            for thread in threads:
                thread.start()

            while process.poll() is None:
                if exceeded.is_set():
                    reason = "output_limit"
                    break

                if time.monotonic() - started >= TIMEOUT_SECONDS:
                    reason = "timeout"
                    break

                time.sleep(0.05)

        finally:
            # Important : arrêter aussi le conteneur Docker,
            # pas seulement le processus CLI local.
            try:
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            except (
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                pass

            if process.poll() is None:
                process.kill()

            process.wait()

            for thread in threads:
                thread.join(timeout=2)

        if reason is None and exceeded.is_set():
            reason = "output_limit"

        status = reason or (
            "success" if process.returncode == 0 else "error"
        )

        return SandboxResult(
            status=status,
            exit_code=process.returncode,
            stdout=buffers[0].decode(
                "utf-8", errors="replace"
            ),
            stderr=buffers[1].decode(
                "utf-8", errors="replace"
            ),
            duration_seconds=round(
                time.monotonic() - started, 3
            ),
        )
