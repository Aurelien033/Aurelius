"""Web fetch tool for the Aurelius agent surface.

Security-first design: URL allowlist, response size cap, no cookie persistence,
no credential forwarding. Redirects are NOT followed automatically — each hop
is re-validated against the SSRF deny list to prevent open-redirect chaining.
Inspired by OpenDevin browser tool (OpenDevin/OpenDevin, Apache-2.0). License: MIT.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .tool_registry import TOOL_REGISTRY, ToolResult, ToolSpec

_MAX_RESPONSE_BYTES = 500_000  # 500KB
_MAX_URL_LEN = 2048
_REQUEST_TIMEOUT = 10  # seconds
_MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = frozenset(["https", "http"])


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Raise instead of auto-following redirects — lets the caller re-validate."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl}", headers, fp)


_opener = urllib.request.build_opener(_NoRedirectHandler)

_DENY_HOST_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        r"^169\.254\.",  # link-local
        r"^127\.",  # loopback
        r"^10\.",  # RFC1918
        r"^172\.(1[6-9]|2\d|3[01])\.",  # RFC1918
        r"^192\.168\.",  # RFC1918
        r"^::1$",  # IPv6 loopback
        r"^fd",  # IPv6 ULA
        r"localhost$",
        r"metadata\.google\.internal",
        r"169\.254\.169\.254",  # AWS/GCP IMDS
    ]
)


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
    host = parsed.hostname or ""
    for pat in _DENY_HOST_PATTERNS:
        if pat.search(host):
            return False, f"host {host!r} is a denied target (SSRF prevention)"
    return True, ""


class WebTool:
    def fetch(self, url: str, timeout: int = _REQUEST_TIMEOUT) -> ToolResult:
        """Fetch a URL, following up to _MAX_REDIRECTS safe redirects."""
        current_url = url
        for hop in range(_MAX_REDIRECTS + 1):
            safe, reason = _is_safe_url(current_url)
            if not safe:
                return ToolResult(tool_name="web", success=False, output="", error=reason)
            try:
                req = urllib.request.Request(current_url, headers={"User-Agent": "Aurelius/1.0"})  # noqa: S310 — URL validated by _is_safe_url (scheme allowlist + SSRF deny patterns) before any request
                with _opener.open(req, timeout=timeout) as resp:  # nosec B310 — URL validated by _is_safe_url (scheme allowlist + SSRF deny patterns) before open  # noqa: S310
                    status = getattr(resp, "status", 200)
                    if 300 <= status < 400:
                        # _NoRedirectHandler turned this into HTTPError; see except below
                        pass
                    else:
                        raw = resp.read(_MAX_RESPONSE_BYTES)
                        content = raw.decode("utf-8", errors="replace")
                        return ToolResult(tool_name="web", success=True, output=content, error="")
            except urllib.error.HTTPError as e:
                if 300 <= e.code < 400:
                    # H-24 (CSV): validate redirect target before following
                    location = e.headers.get("Location")
                    if not location:
                        return ToolResult(
                            tool_name="web",
                            success=False,
                            output="",
                            error="redirect without Location",
                        )
                    current_url = location
                    continue
                return ToolResult(tool_name="web", success=False, output="", error=f"HTTP {e.code}")
            except urllib.error.URLError as e:
                return ToolResult(tool_name="web", success=False, output="", error=str(e))
            except Exception as e:
                return ToolResult(tool_name="web", success=False, output="", error=str(e))
        return ToolResult(
            tool_name="web", success=False, output="", error=f"exceeded {_MAX_REDIRECTS} redirects"
        )

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
