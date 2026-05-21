"""Agent Mode Registry — behavior presets for ReActLoop.

Inspired by Roo-Code agent modes (Code / Architect / Ask / Debug).
Each mode defines:
  - a system prompt prefix injected at the top of every generation, and
  - an allowed-tools set that filters the runtime ``tool_registry``.

Modes are composable: ``Custom`` lets operators override the prefix and
tool filter explicitly.  A default set of four production-ready modes is
registered at import time.  The registry is global but namespaced under
this module so callers can also add deploy-time modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentMode:
    """One behavioral profile for a ReActLoop invocation."""

    name: str
    """Stable identifier, e.g. ``"code"``, ``"architect"``."""

    prompt_prefix: str = ""
    """Prepended to the system prompt before every LLM call."""

    allowed_tools: frozenset[str] | None = None
    """If *None*, all tools are available.  Otherwise restrict to this set."""

    description: str = ""
    """Human-readable summary for UIs and help text."""


# ── Built-in modes ─────────────────────────────────────────────────────────

CODE_MODE = AgentMode(
    name="code",
    prompt_prefix=(
        "You are a focused coding agent. "
        "Write correct, minimal, well-tested code. "
        "Prefer the simplest solution that works. "
        "Run tests to verify before declaring success."
    ),
    allowed_tools=frozenset(
        {"read_file", "write_file", "search_files", "terminal", "execute_code"}
    ),
    description="Write, edit, test, and debug code.",
)

ARCHITECT_MODE = AgentMode(
    name="architect",
    prompt_prefix=(
        "You are a systems architect. "
        "Think in layers and boundaries. "
        "Before proposing changes, describe the current architecture, "
        "the proposed change, and the migration path. "
        "Prefer backward-compatible evolutions over rewrites."
    ),
    allowed_tools=frozenset(
        {"read_file", "write_file", "search_files", "execute_code", "terminal"}
    ),
    description="Design systems, review architecture, plan migrations.",
)

ASK_MODE = AgentMode(
    name="ask",
    prompt_prefix=(
        "You are a knowledgeable research assistant. "
        "Answer questions clearly and accurately. "
        "Cite sources when you can. "
        "If something is uncertain, say so. "
        "Do not run destructive commands."
    ),
    allowed_tools=frozenset(
        {"web_search", "web_extract", "read_file", "search_files"}
    ),
    description="Answer questions and retrieve information.",
)

DEBUG_MODE = AgentMode(
    name="debug",
    prompt_prefix=(
        "You are a debugging specialist. "
        "Given a bug or failure, gather evidence first: "
        "read stack traces, inspect relevant files, run diagnostic commands. "
        "Form a hypothesis, verify it, then apply the smallest correct fix. "
        "Never guess — always check."
    ),
    allowed_tools=frozenset(
        {
            "read_file", "write_file", "search_files",
            "execute_code", "terminal", "patch",
        }
    ),
    description="Investigate bugs, read traces, apply surgical fixes.",
)

CUSTOM_MODE = AgentMode(
    name="custom",
    prompt_prefix="",
    allowed_tools=None,
    description="Fully open — no prefix or tool restriction applied.",
)

_BUILTINS: dict[str, AgentMode] = {
    m.name: m for m in (CODE_MODE, ARCHITECT_MODE, ASK_MODE, DEBUG_MODE, CUSTOM_MODE)
}


# ── Registry ────────────────────────────────────────────────────────────────

_registry: dict[str, AgentMode] = dict(_BUILTINS)


def register_mode(mode: AgentMode) -> None:
    """Add or replace a mode in the global registry."""
    _registry[mode.name] = mode


def get_mode(name: str) -> AgentMode:
    """Return a mode by name, falling back to ``CUSTOM_MODE``."""
    return _registry.get(name, CUSTOM_MODE)


def get_all_modes() -> list[AgentMode]:
    """Return all registered modes sorted by name."""
    return sorted(_registry.values(), key=lambda m: m.name)


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_system_prompt(
    base_prompt: str,
    mode: AgentMode | str,
) -> str:
    """Return *base_prompt* prepended with *mode*'s prefix (if non-empty).

    When *mode* is a string, it is looked up via :func:`get_mode`.
    """
    if isinstance(mode, str):
        mode = get_mode(mode)
    parts = [p for p in (mode.prompt_prefix, base_prompt) if p]
    return "\n\n".join(parts)


def filter_tools(
    tool_registry: dict[str, callable],
    mode: AgentMode | str,
) -> dict[str, callable]:
    """Return *tool_registry* filtered to *mode*'s ``allowed_tools``.

    If *mode* is a string, it is looked up via :func:`get_mode`.
    When ``allowed_tools`` is *None*, the unfiltered registry is returned.
    """
    if isinstance(mode, str):
        mode = get_mode(mode)
    if mode.allowed_tools is None:
        return tool_registry
    return {k: v for k, v in tool_registry.items() if k in mode.allowed_tools}
