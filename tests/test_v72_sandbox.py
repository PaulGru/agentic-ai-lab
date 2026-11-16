
import unittest
from unittest.mock import patch

from experiments.sandbox import (
    IMAGE,
    SandboxRunner,
    build_command,
)


class SandboxTests(unittest.TestCase):

    def test_security_configuration(self):
        cmd = build_command("test-container")

        required_options = [
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--security-opt=seccomp=builtin",
            "--pids-limit=32",
            "--memory=128m",
            "--memory-swap=128m",
            "--cpus=0.5",
            "--user=10001:10001",
            "--log-driver=none",
        ]

        for flag in required_options:
            self.assertIn(flag, cmd)

        self.assertEqual(cmd[-1], IMAGE)

        self.assertFalse(
            any(
                x.startswith("--volume")
                or x.startswith("--mount")
                or x == "-v"
                for x in cmd
            )
        )

    @patch("experiments.sandbox.subprocess.Popen")
    def test_requires_approval(self, popen):
        with self.assertRaises(PermissionError):
            SandboxRunner().run("print(1)")

        popen.assert_not_called()

    @patch("experiments.sandbox.subprocess.Popen")
    def test_rejects_external_deps(self, popen):
        with self.assertRaises(ValueError):
            SandboxRunner().run(
                "print(1)",
                approved=True,
                dependencies=["numpy==2.3.0"],
            )

        popen.assert_not_called()

    @patch("experiments.sandbox.subprocess.Popen")
    def test_rejects_large_code(self, popen):
        with self.assertRaises(ValueError):
            SandboxRunner().run(
                "#" * 12001,
                approved=True,
            )

        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
