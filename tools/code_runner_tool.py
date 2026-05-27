from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field

from agent.tool_sandbox_denylist import (
    DenylistCategory,
    DenylistRule,
    ToolSandboxDenylist,
)


# Python-specific rules for code execution sandbox
PYTHON_CODE_RULES: tuple[DenylistRule, ...] = (
    # Import restrictions
    DenylistRule(
        id="py.import_os",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bimport\s+os\b",
        message="import os is disallowed",
    ),
    DenylistRule(
        id="py.from_os",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bfrom\s+os\b",
        message="from os import is disallowed",
    ),
    DenylistRule(
        id="py.import_sys",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bimport\s+sys\b",
        message="import sys is disallowed",
    ),
    DenylistRule(
        id="py.from_sys",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bfrom\s+sys\b",
        message="from sys import is disallowed",
    ),
    DenylistRule(
        id="py.import_subprocess",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bimport\s+subprocess\b",
        message="import subprocess is disallowed",
    ),
    DenylistRule(
        id="py.from_subprocess",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bfrom\s+subprocess\b",
        message="from subprocess import is disallowed",
    ),
    # Code execution primitives
    DenylistRule(
        id="py.exec_call",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bexec\s*\(",
        message="exec() is disallowed",
    ),
    DenylistRule(
        id="py.eval_call",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\beval\s*\(",
        message="eval() is disallowed",
    ),
    # Builtin access restrictions
    DenylistRule(
        id="py.builtins_getattr",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bgetattr\s*\(\s*__builtins__",
        message="getattr(__builtins__, ...) is disallowed",
    ),
    DenylistRule(
        id="py.builtins_vars",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bvars\s*\(\s*__builtins__",
        message="vars(__builtins__) is disallowed",
    ),
    DenylistRule(
        id="py.builtins_direct",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"__builtins__\.",
        message="direct __builtins__ access is disallowed",
    ),
    # Dynamic import restrictions
    DenylistRule(
        id="py.dynamic_import",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"__import__\s*\(",
        message="__import__() is disallowed",
    ),
    # Open file restrictions (prevent file system access)
    DenylistRule(
        id="py.open_call",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"(?<!\w)open\s*\(",
        message="open() is disallowed",
    ),
)


@dataclass
class CodeRunnerConfig:
    timeout_s: float = 5.0
    max_output_bytes: int = 65536
    allowed_modules: list[str] = field(
        default_factory=lambda: [
            "math",
            "json",
            "re",
            "collections",
            "itertools",
            "functools",
            "string",
            "datetime",
        ]
    )


@dataclass(frozen=True)
class CodeRunnerResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class CodeRunnerTool:
    def __init__(self, config: CodeRunnerConfig | None = None) -> None:
        self.config = config or CodeRunnerConfig()
        # Create denylist with Python-specific rules
        self._denylist = ToolSandboxDenylist(
            rules=list(PYTHON_CODE_RULES),
            strict=True,
        )

    def run(self, code: str) -> CodeRunnerResult:
        if not self.is_safe(code):
            return CodeRunnerResult(
                stdout="", stderr="code rejected: unsafe pattern detected", exit_code=1
            )
        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=self.config.timeout_s,
                text=True,
            )
            limit = self.config.max_output_bytes
            return CodeRunnerResult(
                stdout=proc.stdout[:limit],
                stderr=proc.stderr[:limit],
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return CodeRunnerResult(stdout="", stderr="timeout", exit_code=-1, timed_out=True)

    def is_safe(self, code: str) -> bool:
        """Check code against sandbox denylist using regex patterns."""
        verdict = self._denylist.evaluate(tool_name="", tool_args=code)
        return verdict.allowed


CODE_RUNNER_REGISTRY: dict[str, type[CodeRunnerTool]] = {"default": CodeRunnerTool}
