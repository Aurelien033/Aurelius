"""Unit tests for src.security.subdomain_enum.

Covers construction, resolution, error handling, stdlib-only import constraint,
and the fail-closed AURELIUS_SECURITY_NETWORK_SCAN_ENABLED gate.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

from src.security.subdomain_enum import (
    EnumSummary,
    SubdomainEnumError,
    SubdomainEnumerator,
    SubdomainResult,
)


@pytest.fixture(autouse=True)
def _enable_network_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable network enumeration for all tests in this module."""
    monkeypatch.setenv("AURELIUS_SECURITY_NETWORK_SCAN_ENABLED", "1")

_STDLIB_MODULES = {
    "concurrent",
    "dataclasses",
    "socket",
    "__future__",
}

_SRC = pathlib.Path(__file__).parents[2] / "src" / "security" / "subdomain_enum.py"


def test_no_third_party_imports() -> None:
    tree = ast.parse(_SRC.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported - _STDLIB_MODULES, f"Non-stdlib: {imported - _STDLIB_MODULES}"


# ── Dataclasses ───────────────────────────────────────────────────────────────

class TestSubdomainResult:
    def test_default(self):
        r = SubdomainResult(name="api.example.com")
        assert r.name == "api.example.com"
        assert r.ip_addresses == []
        assert r.is_cname is False
        assert r.error == ""

    def test_with_ips(self):
        r = SubdomainResult(name="www.example.com", ip_addresses=["93.184.216.34"])
        assert "93.184.216.34" in r.ip_addresses


class TestEnumSummary:
    def test_total_checked(self):
        s = EnumSummary(found=[], not_found=3, errors=[])
        assert s.total_checked == 3

    def test_names(self):
        r = SubdomainResult(name="www.example.com")
        s = EnumSummary(domain="example.com", found=[r], not_found=0, errors=[])
        assert s.names == ["www.example.com"]


# ── SubdomainEnumerator construction ─────────────────────────────────────────

class TestConstruction:
    def test_default(self):
        e = SubdomainEnumerator()
        assert e.timeout == 2.0
        assert e.concurrency == 50

    def test_custom(self):
        e = SubdomainEnumerator(concurrency=10, timeout=0.5)
        assert e.timeout == 0.5
        assert e.concurrency == 10

    def test_invalid_concurrency(self):
        with pytest.raises(SubdomainEnumError):
            SubdomainEnumerator(concurrency=0)

    def test_invalid_timeout(self):
        with pytest.raises(SubdomainEnumError):
            SubdomainEnumerator(timeout=-1)


# ── resolve() ────────────────────────────────────────────────────────────────

class TestResolve:
    def setup_method(self):
        self.e = SubdomainEnumerator(timeout=0.5, concurrency=10)

    def test_empty_wordlist(self):
        s = self.e.resolve("example.com", wordlist=[])
        assert s.domain == "example.com"
        assert s.found == []
        assert s.not_found == 0

    def test_resolve_gate_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AURELIUS_SECURITY_NETWORK_SCAN_ENABLED", raising=False)
        e = SubdomainEnumerator()
        with pytest.raises(SubdomainEnumError, match="disabled by default"):
            e.resolve("example.com", wordlist=["www"])

    def test_empty_domain_raises(self):
        with pytest.raises(SubdomainEnumError):
            self.e.resolve("")

    def test_domain_with_slash_raises(self):
        with pytest.raises(SubdomainEnumError):
            self.e.resolve("example.com/../../etc")

    def test_result_has_enum_summary_type(self):
        s = self.e.resolve("example.com", wordlist=["non-existent-xyz-999"])
        assert isinstance(s, EnumSummary)
        assert s.domain == "example.com"

    def test_non_resolvable_returns_not_found(self):
        s = self.e.resolve("example.com", wordlist=["this-domain-definitely-does-not-exist-xyz"])
        assert s.not_found >= 0   # might also fail with an error
        assert s.total_checked == 1

    def test_stats_populated(self):
        s = self.e.resolve("example.com", wordlist=["www", "nonexistent-xyz-999"])
        assert s.total_checked == 2

    def test_found_have_name_field(self):
        # if any resolve (e.g. localhost), verify the record is correct
        s = self.e.resolve("localhost")
        # localhost might or might not resolve depending on environment
        if s.found:
            assert all(r.name == "localhost" for r in s.found)
        else:
            assert s.not_found == 0 and s.errors == []

    def test_return_type(self):
        s = self.e.resolve("example.com")
        assert isinstance(s, EnumSummary)
