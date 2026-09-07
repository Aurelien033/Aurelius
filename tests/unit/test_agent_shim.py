"""Verify the agent/ shim re-exports src.agent correctly and emits DeprecationWarning."""

from __future__ import annotations

import importlib
import warnings


def _reimport_agent():
    """Force a fresh import of the shim (bypasses sys.modules cache)."""
    import sys

    for key in list(sys.modules):
        if key == "agent" or key.startswith("agent."):
            del sys.modules[key]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.import_module("agent")
    return mod, caught


class TestAgentShim:
    def test_import_triggers_deprecation_warning(self):
        _, caught = _reimport_agent()
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "Expected a DeprecationWarning when importing 'agent'"
        assert any("src.agent" in str(w.message) for w in deprecations)

    def test_session_manager_accessible_via_shim(self):
        mod, _ = _reimport_agent()
        from src.agent.session_manager import SessionManager as Canonical

        assert hasattr(mod, "SessionManager"), "SessionManager not re-exported through agent shim"
        assert mod.SessionManager is Canonical, (
            "agent.SessionManager is not the same object as src.agent.session_manager.SessionManager"
        )

    def test_no_duplicate_class_identity(self):
        mod, _ = _reimport_agent()
        from src.agent.session_manager import SessionManager as Canonical

        shim_cls = getattr(mod, "SessionManager", None)
        if shim_cls is not None:
            assert shim_cls is Canonical
