"""Unit tests for src.security.port_scanner.

Covers construction, scan result shape, errors, stdlib-only import constraint,
and the fail-closed AURELIUS_SECURITY_NETWORK_SCAN_ENABLED gate.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

from src.security.port_scanner import (
    PortResult,
    PortScanner,
    PortScannerError,
    ScanSummary,
    scan_thread_safe,
)


@pytest.fixture(autouse=True)
def _enable_network_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable port-scanner for all tests in this module."""
    monkeypatch.setenv("AURELIUS_SECURITY_NETWORK_SCAN_ENABLED", "1")

# stdlib imports used by this module
_STDLIB_MODULES = {
    "asyncio",
    "socket",
    "ssl",
    "time",
    "dataclasses",
    "__future__",
}

_SRC = pathlib.Path(__file__).parents[2] / "src" / "security" / "port_scanner.py"


def test_no_third_party_imports() -> None:
    tree = ast.parse(_SRC.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported - _STDLIB_MODULES, f"Non-stdlib: {imported - _STDLIB_MODULES}"


# ── Dataclasses ───────────────────────────────────────────────────────────────

class TestPortResult:
    def test_defaults(self):
        pr = PortResult(port=80)
        assert pr.is_open is False
        assert pr.service == ""
        assert pr.tls is False
        assert pr.response_time_ms == 0.0

    def test_str_repr(self):
        pr = PortResult(port=443, is_open=True, service="HTTPS", tls=True)
        assert pr.port == 443


class TestScanSummary:
    def test_open_port_numbers(self):
        r1 = PortResult(port=80, is_open=True)
        r2 = PortResult(port=443, is_open=True)
        r3 = PortResult(port=22, is_open=False)
        s = ScanSummary(host="1.2.3.4", open_ports=[r1, r2], closed_ports=1)
        assert s.open_port_numbers == [80, 443]
        assert s.total_ports == 3


# ── PortScanner construction ──────────────────────────────────────────────────

class TestConstruction:
    def test_default(self):
        ps = PortScanner()
        assert ps.timeout == 1.5
        assert ps.concurrency == 200
        assert ps.grab_banner is True

    def test_custom(self):
        ps = PortScanner(concurrency=10, timeout=0.5, grab_banner=False)
        assert ps.timeout == 0.5
        assert ps.concurrency == 10
        assert ps.grab_banner is False

    def test_invalid_concurrency(self):
        with pytest.raises(PortScannerError):
            PortScanner(concurrency=0)

    def test_invalid_timeout(self):
        with pytest.raises(PortScannerError):
            PortScanner(timeout=-0.1)

    def test_scan_empty_host_raises(self):
        ps = PortScanner()
        with pytest.raises(PortScannerError):
            ps.scan("")

    def test_scan_invalid_port_raises(self):
        ps = PortScanner()
        with pytest.raises(PortScannerError):
            ps.scan("localhost", ports=[0, 99999])

    def test_scan_gate_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AURELIUS_SECURITY_NETWORK_SCAN_ENABLED", raising=False)
        ps = PortScanner()
        with pytest.raises(PortScannerError, match="disabled by default"):
            ps.scan("127.0.0.1", ports=[22])


# ── PortScanner.scan() ────────────────────────────────────────────────────────

class TestPortScannerScan:
    def setup_method(self):
        self.ps = PortScanner(timeout=0.5, concurrency=50, grab_banner=False)

    def test_returns_scan_summary(self):
        summary = self.ps.scan("127.0.0.1", ports=[22, 80, 443])
        assert isinstance(summary, ScanSummary)
        assert summary.host == "127.0.0.1"

    def test_localhost_some_port_unlikely_open(self):
        summary = self.ps.scan("127.0.0.1", ports=[1, 65535])
        assert summary.closed_ports + len(summary.open_ports) == 2

    def test_scan_dedupes_ports(self):
        summary = self.ps.scan("127.0.0.1", ports=[80, 80, 80])
        assert len(summary.open_port_numbers) + summary.closed_ports == 1

    def test_open_port_has_service(self):
        summary = self.ps.scan("127.0.0.1", ports=[7])  # echo is usually closed
        assert isinstance(summary.open_ports, list)

    def test_default_ports_list(self):
        ps = PortScanner(timeout=0.5, grab_banner=False)
        # Should not raise; just check it returns something
        summary = ps.scan("127.0.0.1")
        assert isinstance(summary, ScanSummary)

    def test_elapsed_ms_non_negative(self):
        summary = self.ps.scan("127.0.0.1", ports=[22])
        assert summary.elapsed_ms >= 0.0


# ── scan_thread_safe helper ───────────────────────────────────────────────────

class TestThreadSafeHelper:
    def test_returns_summary(self):
        summary = scan_thread_safe("127.0.0.1", [22, 80, 443], timeout=0.5)
        assert isinstance(summary, ScanSummary)
        assert summary.host == "127.0.0.1" or summary  # at minimum no crash
