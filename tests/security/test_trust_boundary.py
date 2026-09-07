"""Tests for trust boundary validator and safe-path resolution."""

from __future__ import annotations

import pytest

from src.agent.session_manager import _resolve_safe_path
from src.security.trust_boundary import TrustBoundary, TrustBoundaryValidator


class TestResolveSafePath:
    def test_valid_subpath_accepted(self, tmp_path):
        result = _resolve_safe_path(tmp_path, tmp_path / "sessions" / "foo.json")
        assert result == (tmp_path / "sessions" / "foo.json").resolve()

    def test_dotdot_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="escapes allowed root"):
            _resolve_safe_path(tmp_path, tmp_path / ".." / "secret.json")

    def test_absolute_path_outside_root_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="escapes allowed root"):
            _resolve_safe_path(tmp_path, "/etc/passwd")

    def test_symlink_outside_root_rejected(self, tmp_path):
        outside = tmp_path.parent / "outside_target.txt"
        outside.write_text("secret")
        link = tmp_path / "evil_link"
        link.symlink_to(outside)
        try:
            with pytest.raises(ValueError):
                _resolve_safe_path(tmp_path, link)
        finally:
            outside.unlink(missing_ok=True)


class TestTrustBoundaryValidator:
    def test_allows_authorized_caller(self):
        tbv = TrustBoundaryValidator()
        tbv.register(
            TrustBoundary(name="mcp", allowed_caller_prefixes=["tool:"], allowed_methods=["call"])
        )
        ok, _ = tbv.check("mcp", "tool:retriever", "call")
        assert ok is True

    def test_rejects_unknown_boundary(self):
        tbv = TrustBoundaryValidator()
        ok, msg = tbv.check("unknown", "caller", "method")
        assert ok is False
        assert "unknown" in msg

    def test_rejects_unauthorized_caller(self):
        tbv = TrustBoundaryValidator()
        tbv.register(
            TrustBoundary(name="mcp", allowed_caller_prefixes=["tool:"], allowed_methods=["call"])
        )
        ok, _ = tbv.check("mcp", "agent:bad", "call")
        assert ok is False

    def test_rejects_unauthorized_method(self):
        tbv = TrustBoundaryValidator()
        tbv.register(
            TrustBoundary(name="mcp", allowed_caller_prefixes=["tool:"], allowed_methods=["call"])
        )
        ok, _ = tbv.check("mcp", "tool:retriever", "exec")
        assert ok is False

    def test_unregister(self):
        tbv = TrustBoundaryValidator()
        tbv.register(TrustBoundary(name="x"))
        tbv.unregister("x")
        ok, _ = tbv.check("x", "", "")
        assert ok is False
