"""Centralized code-execution mode configuration.

Pre-remediation, the agent's code-execution path ran
untrusted Python in a subprocess with no explicit
distinction between a sandboxed test mode and a real
isolation backend. The audit H6 finding requires:

- AURELIUS_CODE_EXECUTION_MODE in {disabled,
  trusted_subprocess, isolated}.
- Production / untrusted execution refuses unless
  'isolated' backend is configured.
- Default mode is 'disabled' (fail closed).

This module is the single source of truth for the
execution mode. agent.code_execution_tool.CodeExecutionTool
and agent.code_execution_sandbox.CodeExecutionSandbox
both consult it.

Threat model:
  - The pre-remediation 'subprocess' mode was a real
    sandbox (defense in depth) but NOT an isolation
    boundary. Running untrusted user code under that
    mode was a CWE-78-class risk.
  - The new 'isolated' mode requires a backend
    (e.g. gVisor, firecracker, or wasmtime) to be
    configured. If no backend is configured, the
    executor refuses.
  - The 'disabled' mode is the production default;
    any untrusted code is rejected with a 403-style
    refusal.
"""

from __future__ import annotations

import os
from enum import Enum

__all__ = [
    "ExecutionMode",
    "DEFAULT_MODE",
    "IsolatedBackend",
    "get_execution_mode",
    "is_untrusted_execution_allowed",
    "assert_can_execute_untrusted",
    "SandboxRefused",
]


class ExecutionMode(str, Enum):
    """The three allowed execution modes.

    - DISABLED: no code execution. The tool returns
      SandboxRefused on any call. This is the production
      default.
    - TRUSTED_SUBPROCESS: the legacy defense-in-depth
      subprocess wrapper. ONLY for trusted local code
      (e.g. agent self-tests, dev-only). NOT for
      untrusted user input.
    - ISOLATED: requires an isolation backend
      (gVisor/firecracker/wasmtime). Used for untrusted
      user code in production.
    """

    DISABLED = "disabled"
    TRUSTED_SUBPROCESS = "trusted_subprocess"
    ISOLATED = "isolated"


class IsolatedBackend(str, Enum):
    """The allowed isolation backends for ISOLATED mode."""

    GVISOR = "gvisor"
    FIRECRACKER = "firecracker"
    WASMTIME = "wasmtime"


# Production default: fail closed. The audit's H6
# expectation is that the default mode refuses code
# execution; an explicit env var opt-in is required.
DEFAULT_MODE = ExecutionMode.DISABLED

# The env var that selects the mode. Legacy
# 'AURELIUS_SANDBOX_MODE' is also accepted for
# backward compatibility.
_ENV_VAR = "AURELIUS_CODE_EXECUTION_MODE"
_LEGACY_ENV_VAR = "AURELIUS_SANDBOX_MODE"

# The env var that selects the isolation backend when
# mode == ISOLATED.
_BACKEND_ENV_VAR = "AURELIUS_ISOLATED_BACKEND"


class SandboxRefused(RuntimeError):
    """Raised when the executor refuses a call because the
    configured mode does not allow it.

    This is the production-default behavior: any call
    without an explicit mode opt-in is refused.
    """


def get_execution_mode() -> ExecutionMode:
    """Return the configured execution mode. The default
    is DISABLED (fail closed)."""
    raw = os.environ.get(_ENV_VAR) or os.environ.get(_LEGACY_ENV_VAR) or ""
    raw = raw.strip().lower()
    if not raw:
        return DEFAULT_MODE
    if raw in ("disabled", "off", "false", "0", "no"):
        return ExecutionMode.DISABLED
    if raw in ("trusted_subprocess", "subprocess", "trusted", "dev"):
        return ExecutionMode.TRUSTED_SUBPROCESS
    if raw in ("isolated", "isolate", "production"):
        return ExecutionMode.ISOLATED
    # Unknown value: fail closed.
    return ExecutionMode.DISABLED


def get_isolated_backend() -> IsolatedBackend | None:
    """Return the configured isolation backend, or None if
    not configured / not in ISOLATED mode."""
    raw = os.environ.get(_BACKEND_ENV_VAR, "").strip().lower()
    if not raw:
        return None
    for backend in IsolatedBackend:
        if raw == backend.value:
            return backend
    return None


def is_untrusted_execution_allowed() -> bool:
    """True if the current configuration is allowed to
    execute untrusted (user-supplied) code. This requires
    ISOLATED mode with a configured backend."""
    return (
        get_execution_mode() == ExecutionMode.ISOLATED
        and get_isolated_backend() is not None
    )


def assert_can_execute_untrusted() -> None:
    """Raise SandboxRefused unless the current configuration
    is allowed to execute untrusted code.

    The audit H6 finding requires that the default
    behavior is refusal. This function is the gate that
    the code-execution tool calls before invoking the
    sandbox.
    """
    if is_untrusted_execution_allowed():
        return
    mode = get_execution_mode()
    if mode == ExecutionMode.DISABLED:
        raise SandboxRefused(
            "Code execution is disabled (AURELIUS_CODE_EXECUTION_MODE is "
            "'disabled' or unset). Set the mode to 'isolated' and configure "
            "AURELIUS_ISOLATED_BACKEND to run untrusted code."
        )
    if mode == ExecutionMode.TRUSTED_SUBPROCESS:
        raise SandboxRefused(
            "Trusted_subprocess mode is for trusted local code only. "
            "Untrusted user code requires 'isolated' mode with a "
            "configured backend (gvisor/firecracker/wasmtime)."
        )
    if mode == ExecutionMode.ISOLATED and get_isolated_backend() is None:
        raise SandboxRefused(
            "Isolated mode is configured but no isolation backend is set. "
            "Set AURELIUS_ISOLATED_BACKEND to one of: gvisor, firecracker, wasmtime."
        )
    # Catch-all: any other state fails closed.
    raise SandboxRefused("Code execution refused: unknown mode state")
