"""Tests for src/tools/web_tool.py — at least 18 tests.

No real network calls are made. All fetch tests use unittest.mock.patch.
"""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from tools.web_tool import (
    _MAX_URL_LEN,
    WebTool,
    _is_safe_url,
    _SafeRedirectHandler,
)


@pytest.fixture
def tool():
    return WebTool()


# ── _is_safe_url — loopback / localhost ──────────────────────────────────────


def test_loopback_127_rejected():
    ok, reason = _is_safe_url("http://127.0.0.1/path")
    assert not ok
    assert "SSRF" in reason or "denied" in reason


def test_loopback_127_0_0_2_rejected():
    ok, _ = _is_safe_url("http://127.0.0.2/")
    assert not ok


def test_localhost_rejected():
    ok, reason = _is_safe_url("http://localhost/secret")
    assert not ok


def test_localhost_https_rejected():
    ok, _ = _is_safe_url("https://localhost:8443/api")
    assert not ok


# ── _is_safe_url — RFC1918 ────────────────────────────────────────────────────


def test_rfc1918_10_rejected():
    ok, _ = _is_safe_url("http://10.0.0.1/data")
    assert not ok


def test_rfc1918_172_16_rejected():
    ok, _ = _is_safe_url("http://172.16.0.1/")
    assert not ok


def test_rfc1918_172_31_rejected():
    ok, _ = _is_safe_url("http://172.31.255.255/")
    assert not ok


def test_rfc1918_192_168_rejected():
    ok, _ = _is_safe_url("http://192.168.1.1/admin")
    assert not ok


# ── _is_safe_url — link-local / IMDS ─────────────────────────────────────────


def test_link_local_169_254_rejected():
    ok, _ = _is_safe_url("http://169.254.0.1/")
    assert not ok


def test_aws_imds_rejected():
    ok, reason = _is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert not ok


# ── _is_safe_url — IPv6 loopback ─────────────────────────────────────────────


def test_ipv6_loopback_rejected():
    ok, _ = _is_safe_url("http://[::1]/")
    assert not ok


# ── _is_safe_url — scheme checks ─────────────────────────────────────────────


def test_ftp_scheme_rejected():
    ok, reason = _is_safe_url("ftp://example.com/file")
    assert not ok
    assert "scheme" in reason.lower() or "not allowed" in reason


def test_file_scheme_rejected():
    ok, reason = _is_safe_url("file:///etc/passwd")
    assert not ok


def test_javascript_scheme_rejected():
    ok, _ = _is_safe_url("javascript:alert(1)")
    assert not ok


# ── _is_safe_url — URL length bomb ───────────────────────────────────────────


def test_url_length_bomb_rejected():
    long_url = "https://example.com/" + "a" * (_MAX_URL_LEN + 1)
    ok, reason = _is_safe_url(long_url)
    assert not ok
    assert "exceeds" in reason


# ── _is_safe_url — valid HTTPS passes ────────────────────────────────────────


def test_valid_https_passes():
    ok, reason = _is_safe_url("https://www.example.com/page")
    assert ok
    assert reason == ""


def test_valid_http_passes():
    ok, reason = _is_safe_url("http://www.example.com/page")
    assert ok
    assert reason == ""


# ── WebTool.fetch — rejection paths (no network) ─────────────────────────────


def test_fetch_rejects_localhost(tool):
    result = tool.fetch("http://localhost/secret")
    assert not result.success
    assert result.error != ""


def test_fetch_rejects_ssrf_imds(tool):
    result = tool.fetch("http://169.254.169.254/latest/meta-data/")
    assert not result.success


def test_fetch_rejects_file_scheme(tool):
    result = tool.fetch("file:///etc/passwd")
    assert not result.success


# ── WebTool.fetch — mocked successful fetch ───────────────────────────────────


def test_fetch_success_mocked(tool):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<html>hello</html>"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_opener = MagicMock()
    mock_opener.open.return_value = mock_resp

    with patch("tools.web_tool.urllib.request.build_opener", return_value=mock_opener):
        result = tool.fetch("https://www.example.com/")

    assert result.success
    assert "hello" in result.output


# ── WebTool.fetch — mocked network error ─────────────────────────────────────


def test_fetch_network_error_mocked(tool):
    mock_opener = MagicMock()
    mock_opener.open.side_effect = urllib.error.URLError("timeout")
    with patch("tools.web_tool.urllib.request.build_opener", return_value=mock_opener):
        result = tool.fetch("https://www.example.com/")
    assert not result.success
    assert "timeout" in result.error


# ── Redirect handler — SSRF via redirect ─────────────────────────────────────


def test_redirect_to_imds_denied():
    handler = _SafeRedirectHandler()
    with pytest.raises(urllib.error.URLError, match="unsafe"):
        handler.redirect_request(MagicMock(), None, 302, "Found", {}, "http://169.254.169.254/")


def test_redirect_to_loopback_denied():
    handler = _SafeRedirectHandler()
    with pytest.raises(urllib.error.URLError, match="unsafe"):
        handler.redirect_request(MagicMock(), None, 302, "Found", {}, "http://127.0.0.1/")


def test_redirect_too_many_rejected():
    handler = _SafeRedirectHandler(max_redirects=2)
    handler._redirect_count = 2
    with pytest.raises(urllib.error.URLError, match="too many redirects"):
        handler.redirect_request(MagicMock(), None, 302, "Found", {}, "https://example.com/next")


def test_redirect_safe_public_allowed():
    handler = _SafeRedirectHandler()
    req = urllib.request.Request("https://example.com/page1")
    # Should not raise; returns a new request object (may be None in stub env).
    try:
        handler.redirect_request(req, None, 302, "Found", {}, "https://example.com/page2")
    except urllib.error.URLError as exc:
        pytest.fail(f"safe redirect unexpectedly rejected: {exc}")
