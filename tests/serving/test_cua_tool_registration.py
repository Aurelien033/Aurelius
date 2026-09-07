"""Tests for CUA tool registration in the agentic runtime."""

from __future__ import annotations

import os

import pytest

from src.serving.agentic_runtime import _make_default_tool_registry
from src.inference.agentic_loop import ToolRegistry


@pytest.fixture(autouse=True)
def _clear_env():
    old = os.environ.pop("AURELIUS_NATIVE_TOOLS_ENABLED", "")
    yield
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = old


def test_default_registry_excludes_computer_use_when_gate_off():
    registry = _make_default_tool_registry()
    assert "computer_use" not in registry
    expected = {"calculator", "word_count", "current_time", "echo"}
    assert set(registry._tools.keys()) == expected


def test_cua_tool_is_registered_when_gate_on():
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = "true"
    registry = _make_default_tool_registry()
    assert "computer_use" in registry


def test_cua_plan_action_returns_json():
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = "true"
    registry = _make_default_tool_registry()
    result = registry.execute("computer_use", {"action": "plan", "goal": "take a screenshot"})
    assert result.startswith('{"ok": true')
    assert '"action": "plan"' in result


def test_cua_denylisted_goal_is_blocked():
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = "true"
    registry = _make_default_tool_registry()
    result = registry.execute("computer_use", {"action": "plan", "goal": "delete the database"})
    assert result.startswith("Error: action blocked by safety verifier")


def test_cua_verify_action():
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = "true"
    registry = _make_default_tool_registry()
    result = registry.execute(
        "computer_use",
        {
            "action": "verify",
            "action_type": "click",
            "target": "Submit",
        },
    )
    assert '"action": "verify"' in result


def test_cua_browser_navigate_requires_url():
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = "true"
    registry = _make_default_tool_registry()
    result = registry.execute("computer_use", {"action": "browser_navigate"})
    assert result.startswith("Error: browser_navigate requires 'url'")


def test_cua_describe_surface_returns_surface_info():
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = "true"
    registry = _make_default_tool_registry()
    result = registry.execute("computer_use", {"action": "describe_surface"})
    assert '"action": "describe_surface"' in result


def test_cua_unknown_action_returns_error():
    os.environ["AURELIUS_NATIVE_TOOLS_ENABLED"] = "true"
    registry = _make_default_tool_registry()
    result = registry.execute("computer_use", {"action": "teleport"})
    assert result.startswith("Error: unknown computer_use action")
