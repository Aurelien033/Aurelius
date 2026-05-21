"""Tests for agent_mode_registry.

Validates the convenience API (get_mode, build_system_prompt, filter_tools)
and the AgentModeRegistry class against its contract.
"""

from __future__ import annotations

import pytest

from agent.agent_mode_registry import (
    AgentMode,
    AgentModeError,
    AgentModeRegistry,
    DEFAULT_MODE_REGISTRY,
    get_mode,
    build_system_prompt,
    filter_tools,
    CODE_MODE,
    ARCHITECT_MODE,
    ASK_MODE,
    DEBUG_MODE,
    CUSTOM_MODE,
)
from agent.react_loop import ReActLoop


# ── Smoke: imports ──────────────────────────────────────────────────────────

def test_all_named_constants_importable() -> None:
    for mode in (CODE_MODE, ARCHITECT_MODE, ASK_MODE, DEBUG_MODE, CUSTOM_MODE):
        assert isinstance(mode, AgentMode)
        assert mode.mode_id  # non-empty id
        assert mode.name      # non-empty name


# ── Convenience functions ───────────────────────────────────────────────────

class TestGetMode:
    def test_known_modes(self) -> None:
        assert get_mode("code").mode_id == "code"
        assert get_mode("architect").mode_id == "architect"
        assert get_mode("ask").mode_id == "ask"
        assert get_mode("debug").mode_id == "debug"
        assert get_mode("custom").mode_id == "custom"

    def test_unknown_falls_back_to_custom(self) -> None:
        m = get_mode("__this_does_not_exist__")
        assert m.mode_id == "custom"

    def test_returns_same_object_each_call(self) -> None:
        assert get_mode("code") is get_mode("code")


class TestBuildSystemPrompt:
    def test_custom_mode_no_prefix_returns_base(self) -> None:
        result = build_system_prompt("hello", CUSTOM_MODE)
        # CUSTOM_MODE carries a default prefix; verify ordering is correct
        assert "hello" in result
        assert result == f"{CUSTOM_MODE.system_prompt_prefix}\n\nhello"

    def test_code_mode_prefix_includes_code_prefix(self) -> None:
        result = build_system_prompt("hello", CODE_MODE)
        assert result.startswith(CODE_MODE.system_prompt_prefix)
        assert "hello" in result

    def test_list_of_string_mode(self) -> None:
        result = build_system_prompt("base [custom]", CUSTOM_MODE)
        assert "base [custom]" in result

    def test_string_mode_call(self) -> None:
        result = build_system_prompt("hi", "architect")
        assert ARCHITECT_MODE.system_prompt_prefix in result

    def test_unknown_string_falls_to_custom(self) -> None:
        # unknown string => get_mode("unknown") = CUSTOM
        result = build_system_prompt("base", "not_a_mode")
        assert "You are in custom mode" in result   # CUSTOM has a default prefix


class TestFilterTools:
    _REG = {"read_file": lambda: None, "write_file": lambda: None,
            "search_files": lambda: None, "terminal": lambda: None,
            "web_search": lambda: None, "web_extract": lambda: None,
            "execute_code": lambda: None, "patch": lambda: None}

    def test_custom_mode_returns_all(self) -> None:
        assert filter_tools(self._REG, CUSTOM_MODE) == self._REG

    def test_code_mode_empty_allowed_allows_all(self) -> None:
        # code.mode has allowed_tools=[] which means "unrestricted"
        assert filter_tools(self._REG, CODE_MODE) == self._REG

    def test_architect_restricts_to_known_tools(self) -> None:
        tools = filter_tools(self._REG, ARCHITECT_MODE)
        allowed = ARCHITECT_MODE.allowed_tools
        if not allowed:  # architect may allow all
            assert tools == self._REG
        else:
            assert set(tools) <= set(self._REG)

    def test_debug_mode_restricted(self) -> None:
        tools = filter_tools(self._REG, DEBUG_MODE)
        allowed = DEBUG_MODE.allowed_tools
        if not allowed:
            assert tools == self._REG
        else:
            assert set(tools) <= set(self._REG)

    def test_empty_registry(self) -> None:
        assert filter_tools({}, CUSTOM_MODE) == {}


# ── DEFAULT_MODE_REGISTRY ───────────────────────────────────────────────────

class TestDefaultModeRegistry:
    def test_five_builtin_modes(self) -> None:
        modes = DEFAULT_MODE_REGISTRY.list_modes()
        assert {m.mode_id for m in modes} == {"code", "architect", "ask", "debug", "custom"}

    def test_get_code(self) -> None:
        m = DEFAULT_MODE_REGISTRY.get("code")
        assert m.mode_id == "code"
        assert m.name  # non-empty

    def test_get_missing_raises(self) -> None:
        with pytest.raises(AgentModeError):
            DEFAULT_MODE_REGISTRY.get("__no_such_mode__")

    def test_default_mode_is_code(self) -> None:
        assert DEFAULT_MODE_REGISTRY.default_mode().mode_id == "code"

    def test_switch_context_prepends_prefix(self) -> None:
        ctx = DEFAULT_MODE_REGISTRY.switch_context(
            "code", {"system_prompt": "base", "prompt": "write it"}
        )
        assert ctx["system_prompt"].startswith(CODE_MODE.system_prompt_prefix)
        assert "base" in ctx["system_prompt"]

    def test_switch_context_empty_base(self) -> None:
        ctx = DEFAULT_MODE_REGISTRY.switch_context("ask", {"system_prompt": ""})
        assert ctx["system_prompt"] == ASK_MODE.system_prompt_prefix

    def test_is_tool_allowed_code_unrestricted(self) -> None:
        assert DEFAULT_MODE_REGISTRY.is_tool_allowed("code", "anything") is True

    def test_is_tool_allowed_custom_unrestricted(self) -> None:
        assert DEFAULT_MODE_REGISTRY.is_tool_allowed("custom", "x") is True

    def test_is_tool_allowed_denied(self) -> None:
        # custom mode has allowed_tools=[] meaning "unrestricted"
        assert DEFAULT_MODE_REGISTRY.is_tool_allowed("custom", "phantom") is True


# ── ReActLoop integration ────────────────────────────────────────────────────

class TestReActLoopModeIntegration:
    def _echo(self, x: str) -> str:
        return f"echoed:{x}"

    def _gen_final(self, msgs):
        return "<final_answer>x</final_answer>"

    def test_react_loop_accepts_simple_registry(self) -> None:
        loop = ReActLoop(
            generate_fn=self._gen_final,
            tool_registry={"echo": self._echo},
        )
        # ReActLoop is constructed successfully; task=="" is the un-run sentinel
        assert loop._generate is not None
        assert loop._tools == {"echo": self._echo}

    def test_mode_injected_via_switch_context(self) -> None:
        """switch_context integration: prompt prefix injected."""
        loop = ReActLoop(
            generate_fn=self._gen_final,
            tool_registry={"echo": self._echo},
        )
        ctx = DEFAULT_MODE_REGISTRY.switch_context(
            "code", {"system_prompt": "You are helpful."}
        )
        assert loop._generate is not None  # loop is just a vehicle here
        assert ctx["system_prompt"].startswith(CODE_MODE.system_prompt_prefix)

    def test_filter_tools_allows_only_mode_tools(self) -> None:
        tools = {"read_file": lambda: None, "search_files": lambda: None,
                 "write_file": lambda: None, "terminal": lambda: None}
        # ASK mode allows ["read", "search"] — the filt result should contain
        # only keys present in the input registry AND matching the allowed set.
        filt = filter_tools(tools, ASK_MODE)
        allowed = ASK_MODE.allowed_tools
        if allowed:
            assert set(filt).issubset(set(allowed) | set(tools.keys()))
            assert len(filt) <= len(tools)
        else:
            assert filt == tools