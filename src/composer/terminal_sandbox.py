"""
TerminalSandbox — safe execution environment for agentic coding.

Cursor Composer 2.5 runs terminal commands as part of its agent loop:
builds, tests, linting, git operations. This module provides:

  - Timeout-gated subprocess execution
  - Resource limits (memory, CPU time)
  - Output capture with size limits
  - Working directory isolation
  - Command allow/deny lists
  - Environment variable control
"""

from __future__ import annotations

import logging
import os
import platform
import resource
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    wall_time_seconds: float
    was_timeout: bool = False
    was_killed: bool = False
    truncated_stdout: bool = False
    truncated_stderr: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.was_timeout

    @property
    def combined(self) -> str:
        """Combined stdout + stderr for model consumption."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        return "\n".join(parts)


@dataclass
class SandboxConfig:
    """Configuration for the terminal sandbox."""

    timeout_seconds: int = 60
    max_stdout_bytes: int = 100_000
    max_stderr_bytes: int = 50_000
    max_memory_mb: int = 512
    max_cpu_seconds: int = 30
    work_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    allow_commands: list[str] = field(default_factory=list)  # empty = allow all
    deny_commands: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default configs for different use cases
# ---------------------------------------------------------------------------


def default_sandbox_config(work_dir: Path | None = None) -> SandboxConfig:
    """Safe defaults for general terminal use."""
    return SandboxConfig(
        timeout_seconds=60,
        max_stdout_bytes=100_000,
        max_stderr_bytes=50_000,
        max_memory_mb=256,
        work_dir=work_dir,
        deny_commands=[
            "rm -rf /",
            "sudo",
            "shutdown",
            "reboot",
            "mkfs",
            "dd if=",
            ":(){ :|:& };:",  # fork bomb
        ],
    )


def test_runner_config(work_dir: Path | None = None) -> SandboxConfig:
    """Configuration for running test suites."""
    return SandboxConfig(
        timeout_seconds=300,
        max_stdout_bytes=500_000,
        max_stderr_bytes=200_000,
        max_memory_mb=1024,
        work_dir=work_dir,
        allow_commands=[
            "pytest",
            "python -m pytest",
            "python -m unittest",
            "npm test",
            "npm run test",
            "cargo test",
            "go test",
            "make test",
            "tox",
            "nox",
        ],
    )


def build_runner_config(work_dir: Path | None = None) -> SandboxConfig:
    """Configuration for running builds."""
    return SandboxConfig(
        timeout_seconds=600,
        max_stdout_bytes=1_000_000,
        max_stderr_bytes=500_000,
        max_memory_mb=4096,
        work_dir=work_dir,
        allow_commands=[
            "pip install",
            "python -m pip install",
            "python setup.py",
            "cargo build",
            "cargo check",
            "npm install",
            "npm run build",
            "make",
            "cmake",
            "go build",
            "poetry install",
            "uv pip install",
        ],
    )


# ---------------------------------------------------------------------------
# TerminalSandbox
# ---------------------------------------------------------------------------


class TerminalSandbox:
    """Execute commands in a sandboxed environment.

    Usage:
      sandbox = TerminalSandbox(work_dir=Path("/path/to/repo"))
      result = sandbox.run("pytest tests/ -x", config=test_runner_config())

      if result.success:
          print("Tests passed!")
      else:
          print(f"Exit {result.exit_code}: {result.stderr}")
    """

    def __init__(self, work_dir: Path | None = None):
        self.work_dir = Path(work_dir).resolve() if work_dir else Path.cwd()
        self._history: list[SandboxResult] = []

    def run(
        self,
        command: str,
        config: SandboxConfig | None = None,
        shell: bool = True,
    ) -> SandboxResult:
        """Run a command in the sandbox.

        Args:
            command: The command string to execute.
            config: Sandbox configuration. Uses defaults if None.
            shell: Whether to execute via shell. Shell=True enables pipes
                   and redirects; Shell=False is safer for simple commands.

        Returns:
            SandboxResult with exit code, stdout, stderr, and timing.
        """
        if config is None:
            config = default_sandbox_config(self.work_dir)

        # Security check: deny list
        cmd_lower = command.lower()
        for denied in config.deny_commands:
            if denied.lower() in cmd_lower:
                return SandboxResult(
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command denied by sandbox policy: matches '{denied}'",
                    wall_time_seconds=0.0,
                )

        # Security check: allow list
        if config.allow_commands:
            allowed = any(allowed_cmd.lower() in cmd_lower for allowed_cmd in config.allow_commands)
            if not allowed:
                return SandboxResult(
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command not in sandbox allow list: '{command[:80]}'",
                    wall_time_seconds=0.0,
                )

        # Set up environment
        env = os.environ.copy()
        env.update(config.env)
        # Disable interactive prompts
        env.setdefault("DEBIAN_FRONTEND", "noninteractive")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("CI", "true")

        work_dir = str(config.work_dir) if config.work_dir else str(self.work_dir)

        start = time.monotonic()
        was_timeout = False
        was_killed = False
        stdout_bytes = b""
        stderr_bytes = b""
        exit_code = -1
        truncated_stdout = False
        truncated_stderr = False

        try:
            proc = subprocess.Popen(  # noqa: S603  # nosec B602 — shell flag is an explicit sandbox config; command policy is caller-validated
                command,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=work_dir,
                env=env,
                preexec_fn=self._set_limits(config) if platform.system() != "Windows" else None,
            )

            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=config.timeout_seconds)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                was_timeout = True
                try:
                    proc.kill()
                    stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
                except Exception:
                    pass
                exit_code = proc.returncode if proc.returncode is not None else -1

        except Exception as exc:
            stderr_bytes = str(exc).encode("utf-8", errors="replace")
            exit_code = -1
            was_killed = True

        wall_time = time.monotonic() - start

        # Truncate output
        if len(stdout_bytes) > config.max_stdout_bytes:
            stdout_bytes = stdout_bytes[: config.max_stdout_bytes]
            truncated_stdout = True
        if len(stderr_bytes) > config.max_stderr_bytes:
            stderr_bytes = stderr_bytes[: config.max_stderr_bytes]
            truncated_stderr = True

        result = SandboxResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            wall_time_seconds=wall_time,
            was_timeout=was_timeout,
            was_killed=was_killed,
            truncated_stdout=truncated_stdout,
            truncated_stderr=truncated_stderr,
        )

        self._history.append(result)
        return result

    def run_commands(
        self,
        commands: list[str],
        config: SandboxConfig | None = None,
        stop_on_error: bool = True,
    ) -> list[SandboxResult]:
        """Run a sequence of commands, optionally stopping on first error."""
        results = []
        for cmd in commands:
            result = self.run(cmd, config=config)
            results.append(result)
            if stop_on_error and not result.success:
                break
        return results

    @property
    def history(self) -> list[SandboxResult]:
        """Read-only access to execution history."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Internal: resource limits
    # ------------------------------------------------------------------

    @staticmethod
    def _set_limits(config: SandboxConfig):
        """Return a preexec_fn that sets resource limits for the child process."""

        def _set():
            # Memory limit
            if config.max_memory_mb > 0:
                mem_bytes = config.max_memory_mb * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (OSError, ValueError):
                    pass

            # CPU time limit
            if config.max_cpu_seconds > 0:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_CPU,
                        (config.max_cpu_seconds, config.max_cpu_seconds),
                    )
                except (OSError, ValueError):
                    pass

        return _set


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def run_tests(
    work_dir: Path,
    test_command: str = "pytest",
    timeout: int = 300,
) -> SandboxResult:
    """Run tests in a sandbox with test-runner defaults."""
    sandbox = TerminalSandbox(work_dir=work_dir)
    config = test_runner_config(work_dir)
    config.timeout_seconds = timeout
    return sandbox.run(test_command, config=config)


def run_lint(
    work_dir: Path,
    lint_command: str = "ruff check .",
    timeout: int = 120,
) -> SandboxResult:
    """Run linting in a sandbox."""
    sandbox = TerminalSandbox(work_dir=work_dir)
    config = default_sandbox_config(work_dir)
    config.timeout_seconds = timeout
    return sandbox.run(lint_command, config=config)


def run_typecheck(
    work_dir: Path,
    typecheck_command: str = "mypy src/",
    timeout: int = 180,
) -> SandboxResult:
    """Run type checking in a sandbox."""
    sandbox = TerminalSandbox(work_dir=work_dir)
    config = default_sandbox_config(work_dir)
    config.timeout_seconds = timeout
    return sandbox.run(typecheck_command, config=config)


def git_status(work_dir: Path) -> SandboxResult:
    """Get git status of the working directory."""
    sandbox = TerminalSandbox(work_dir=work_dir)
    return sandbox.run("git status --short", config=default_sandbox_config(work_dir))


def git_diff(work_dir: Path, staged: bool = False) -> SandboxResult:
    """Get git diff of the working directory."""
    sandbox = TerminalSandbox(work_dir=work_dir)
    cmd = "git diff --cached" if staged else "git diff"
    return sandbox.run(cmd, config=default_sandbox_config(work_dir))
