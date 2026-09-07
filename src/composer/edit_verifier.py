"""
EditVerifier — test/lint/typecheck gate for Composer edits.

A Composer-style agent must not merely apply a patch; it must prove the patch is
not obviously broken. This module centralizes verification so the agent loop can
run targeted checks first, then broader checks when warranted.
"""

from __future__ import annotations

import json
import py_compile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .terminal_sandbox import SandboxResult, TerminalSandbox, default_sandbox_config


@dataclass
class VerificationCheck:
    """One verification command or in-process check."""

    name: str
    command: str
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    wall_time_seconds: float = 0.0


@dataclass
class VerificationResult:
    """Aggregated verification output."""

    success: bool
    checks: list[VerificationCheck] = field(default_factory=list)
    files_checked: list[str] = field(default_factory=list)
    failure_summary: str = ""
    wall_time_seconds: float = 0.0

    def render(self, max_output_chars: int = 2_000) -> str:
        lines = [f"Verification: {'PASS' if self.success else 'FAIL'}"]
        lines.append(
            f"Files checked: {', '.join(self.files_checked) if self.files_checked else '(none)'}"
        )
        for check in self.checks:
            status = "PASS" if check.success else "FAIL"
            lines.append(
                f"- {status} {check.name}: {check.command} ({check.wall_time_seconds:.2f}s)"
            )
            if not check.success:
                output = (check.stderr or check.stdout)[-max_output_chars:]
                if output:
                    lines.append(output)
        if self.failure_summary:
            lines.append(f"Failure summary: {self.failure_summary}")
        return "\n".join(lines)


class EditVerifier:
    """Run targeted verification after edits."""

    def __init__(self, repo_root: Path, sandbox: TerminalSandbox | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.sandbox = sandbox or TerminalSandbox(self.repo_root)

    def verify(
        self,
        changed_files: list[str],
        commands: list[str] | None = None,
        run_lint: bool = False,
        run_tests: bool = False,
        test_command: str = "python3 -m pytest -q",
        lint_command: str = "ruff check .",
    ) -> VerificationResult:
        """Verify a change set.

        Verification order:
          1. JSON syntax for JSON files
          2. py_compile for changed Python files
          3. targeted commands supplied by the orchestrator
          4. optional lint
          5. optional test command
        """
        start = time.monotonic()
        checks: list[VerificationCheck] = []

        for filepath in changed_files:
            path = self.repo_root / filepath
            if not path.exists():
                continue
            if filepath.endswith(".json"):
                checks.append(self._json_check(filepath))
            if filepath.endswith(".py"):
                checks.append(self._py_compile_check(filepath))

        for cmd in commands or []:
            checks.append(self._command_check(name="targeted", command=cmd))

        if run_lint:
            checks.append(self._command_check(name="lint", command=lint_command))

        if run_tests:
            checks.append(self._command_check(name="tests", command=test_command))

        failures = [c for c in checks if not c.success]
        return VerificationResult(
            success=not failures,
            checks=checks,
            files_checked=changed_files,
            failure_summary=self._summarize_failures(failures),
            wall_time_seconds=time.monotonic() - start,
        )

    def _py_compile_check(self, filepath: str) -> VerificationCheck:
        start = time.monotonic()
        path = self.repo_root / filepath
        try:
            py_compile.compile(str(path), doraise=True)
            return VerificationCheck(
                name="py_compile",
                command=f"python3 -m py_compile {filepath}",
                success=True,
                exit_code=0,
                wall_time_seconds=time.monotonic() - start,
            )
        except py_compile.PyCompileError as exc:
            return VerificationCheck(
                name="py_compile",
                command=f"python3 -m py_compile {filepath}",
                success=False,
                exit_code=1,
                stderr=str(exc),
                wall_time_seconds=time.monotonic() - start,
            )

    def _json_check(self, filepath: str) -> VerificationCheck:
        start = time.monotonic()
        path = self.repo_root / filepath
        try:
            json.loads(path.read_text())
            return VerificationCheck(
                name="json_parse",
                command=f"json.loads({filepath})",
                success=True,
                exit_code=0,
                wall_time_seconds=time.monotonic() - start,
            )
        except Exception as exc:
            return VerificationCheck(
                name="json_parse",
                command=f"json.loads({filepath})",
                success=False,
                exit_code=1,
                stderr=str(exc),
                wall_time_seconds=time.monotonic() - start,
            )

    def _command_check(self, name: str, command: str) -> VerificationCheck:
        result: SandboxResult = self.sandbox.run(
            command, config=default_sandbox_config(self.repo_root)
        )
        return VerificationCheck(
            name=name,
            command=command,
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            wall_time_seconds=result.wall_time_seconds,
        )

    def _summarize_failures(self, failures: list[VerificationCheck]) -> str:
        if not failures:
            return ""
        parts = []
        for failure in failures:
            msg = (failure.stderr or failure.stdout).strip().splitlines()
            first = msg[0] if msg else f"exit {failure.exit_code}"
            parts.append(f"{failure.name}: {first[:200]}")
        return "; ".join(parts)
