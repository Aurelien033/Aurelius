"""Tests for src/tools/web_tool.py — at least 18 tests.

No real network calls are made. All fetch tests use unittest.mock.patch.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.web_tool import _MAX_URL_LEN, WebTool, _is_safe_url


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
    mock_resp.status = 200
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("tools.web_tool._opener.open", return_value=mock_resp):
        result = tool.fetch("https://www.example.com/")

    assert result.success
    assert "hello" in result.output


# ── WebTool.fetch — mocked network error ─────────────────────────────────────


def test_fetch_network_error_mocked(tool):
    import urllib.error

    with patch("tools.web_tool._opener.open", side_effect=urllib.error.URLError("timeout")):
        result = tool.fetch("https://www.example.com/")
    assert not result.success
    assert "timeout" in result.error


# ── H-24 (CSV): redirect re-validation — SSRF-via-open-redirect ──────────────


def test_fetch_rejects_redirect_to_private_ip(tool):
    """A public URL that redirects to 169.254.169.254 must be rejected at hop 2."""
    import urllib.error

    # First hop: public URL → returns 302 with Location targeting metadata service
    redirect_err = urllib.error.HTTPError(
        "https://public.example.com/x",
        302,
        "Found",
        {"Location": "http://169.254.169.254/latest/meta-data/"},  # noqa: S106
        None,
    )

    with patch("tools.web_tool._opener.open", side_effect=redirect_err):
        result = tool.fetch("https://public.example.com/x")

    assert not result.success
    assert "denied target" in result.error or "169.254" in result.error


def test_fetch_follows_safe_redirect(tool):
    """A redirect to another public URL should be followed."""
    import urllib.error

    final_resp = MagicMock()
    final_resp.read.return_value = b"final content"
    final_resp.status = 200
    final_resp.__enter__ = lambda s: s
    final_resp.__exit__ = MagicMock(return_value=False)

    redirect_err = urllib.error.HTTPError(
        "https://public.example.com/a",
        301,
        "Moved",
        {"Location": "https://public-other.example.com/b"},
        None,
    )

    def side(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "public-other" in url:
            return final_resp
        raise redirect_err

    with patch("tools.web_tool._opener.open", side_effect=side):
        result = tool.fetch("https://public.example.com/a")

    assert result.success
    assert "final content" in result.output


def test_fetch_rejects_redirect_loop(tool):
    """Infinite redirect loops must terminate after _MAX_REDIRECTS."""
    import urllib.error

    redirect_err = urllib.error.HTTPError(
        "https://example.com/a",
        302,
        "Found",
        {"Location": "https://example.com/b"},
        None,
    )

    with patch("tools.web_tool._opener.open", side_effect=redirect_err):
        result = tool.fetch("https://example.com/a")

    assert not result.success
    assert "exceeded" in result.error or "redirect" in result.error
