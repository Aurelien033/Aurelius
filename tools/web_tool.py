"""Web fetch tool for the Aurelius agent surface.

Security-first design: exact-match host blocklist, ipaddress-based private-IP
rejection, per-redirect-hop validation, response size cap, no cookie
persistence, no credential forwarding.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .tool_registry import TOOL_REGISTRY, ToolResult, ToolSpec

_MAX_RESPONSE_BYTES = 500_000  # 500 KB
_MAX_URL_LEN = 2048
_REQUEST_TIMEOUT = 10  # seconds
_MAX_REDIRECTS = 10
_ALLOWED_SCHEMES = frozenset(["https", "http"])

# Exact-match blocklist — no regex, no anchoring errors.
_DENY_HOSTS: frozenset[str] = frozenset(
    {
        "127.0.0.1",
        "localhost",
        "0.0.0.0",  # nosec B104 — SSRF deny-list, not a bind  # noqa: S104
        "::1",
        "169.254.169.254",  # AWS/GCP/Azure IMDS
        "metadata.google.internal",
    }
)


def _is_private_ip(host: str) -> bool:
    """Return True if *host* is or resolves to a private/loopback/link-local address."""
    # Literal IP — no DNS needed.
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_unspecified
    except ValueError:
        pass
    # Hostname — DNS resolution.
    try:
        for info in socket.getaddrinfo(host, None):
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_unspecified:
                return True
    except (socket.gaierror, ValueError):
        pass
    return False


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Return (is_safe, reason). Rejects SSRF-prone targets."""
    if len(url) > _MAX_URL_LEN:
        return False, f"URL exceeds {_MAX_URL_LEN} chars"
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "malformed URL"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"scheme {parsed.scheme!r} not allowed"
    host = (parsed.hostname or "").lower()
    if host in _DENY_HOSTS:
        return False, f"host {host!r} is a denied target (SSRF prevention)"
    if _is_private_ip(host):
        return False, f"host {host!r} is a private IP (SSRF prevention)"
    return True, ""


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate each redirect hop against the SSRF blocklist."""

    def __init__(self, max_redirects: int = _MAX_REDIRECTS) -> None:
        self._redirect_count = 0
        self._max_redirects = max_redirects
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        self._redirect_count += 1
        if self._redirect_count > self._max_redirects:
            raise urllib.error.URLError(f"too many redirects (>{self._max_redirects})")
        safe, reason = _is_safe_url(newurl)
        if not safe:
            raise urllib.error.URLError(f"redirect to unsafe URL rejected: {reason}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebTool:
    def fetch(self, url: str, timeout: int = _REQUEST_TIMEOUT) -> ToolResult:
        """Fetch a URL, returning up to _MAX_RESPONSE_BYTES of content."""
        safe, reason = _is_safe_url(url)
        if not safe:
            return ToolResult(tool_name="web", success=False, output="", error=reason)
        try:
            handler = _SafeRedirectHandler()
            opener = urllib.request.build_opener(handler)
            req = urllib.request.Request(url, headers={"User-Agent": "Aurelius/1.0"})  # noqa: S310
            with opener.open(req, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read(_MAX_RESPONSE_BYTES)
                content = raw.decode("utf-8", errors="replace")
            return ToolResult(tool_name="web", success=True, output=content, error="")
        except urllib.error.URLError as e:
            return ToolResult(tool_name="web", success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(tool_name="web", success=False, output="", error=str(e))

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web",
            description="Fetch content from a URL",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
            },
            required=["url"],
        )


WEB_TOOL = WebTool()
TOOL_REGISTRY.register(WEB_TOOL.spec(), handler=WEB_TOOL.fetch)
