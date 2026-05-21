"""CORS middleware for the stdlib HTTPServer.

Simple middleware that adds CORS headers to all responses
and handles CORS preflight (OPTIONS) requests.

Hardened defaults:
- Credentials are never allowed when origins are empty or "*".
- The Origin request header is validated against the allowlist.
- Vary: Origin is always emitted to prevent cache poisoning.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler


class CORSMiddleware:
    """Add CORS headers to all responses and handle OPTIONS preflight."""

    def __init__(
        self,
        allowed_origins: str | list[str] | None = None,
        allowed_methods: str = "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        allowed_headers: str = "Content-Type, Authorization, X-API-Key, X-Request-ID",
        allow_credentials: bool | None = None,
        max_age: int = 86400,
    ) -> None:
        if allowed_origins is None:
            allowed_origins = ""
        self.allowed_origins = (
            allowed_origins if isinstance(allowed_origins, str) else ", ".join(allowed_origins)
        )
        self.allowed_methods = allowed_methods
        self.allowed_headers = allowed_headers
        # Per CORS spec, Access-Control-Allow-Origin: * cannot be used with credentials.
        # Default: credentials OK when origins are explicitly set (not empty, not *).
        origins_is_wide_open = self.allowed_origins in ("", "*")
        self.allow_credentials = (
            allow_credentials if allow_credentials is not None else not origins_is_wide_open
        )
        # Hardening: explicit attempt to use credentials with wildcard is a configuration error.
        if self.allow_credentials and origins_is_wide_open:
            raise ValueError(
                "CORS misconfiguration: allow_credentials=True is not permitted when "
                f"allowed_origins is {self.allowed_origins!r}"
            )
        self.max_age = max_age

    def _origin_allowed(self, request_origin: str | None) -> str:
        """Return the origin to echo back, respecting the allowlist."""
        if self.allowed_origins in ("", "*"):
            return request_origin or self.allowed_origins or "*"
        if request_origin and request_origin in self.allowed_origins:
            return request_origin
        # No match -> return empty to signal disallowed; caller should 403 if desired.
        return ""

    def add_headers(self, handler: BaseHTTPRequestHandler) -> None:
        """Add CORS headers to the handler's response."""
        request_origin = handler.headers.get("Origin")
        echo_origin = self._origin_allowed(request_origin)
        handler.send_header("Access-Control-Allow-Origin", echo_origin)
        handler.send_header("Access-Control-Allow-Methods", self.allowed_methods)
        handler.send_header("Access-Control-Allow-Headers", self.allowed_headers)
        if self.allow_credentials:
            handler.send_header("Access-Control-Allow-Credentials", "true")
        handler.send_header("Access-Control-Max-Age", str(self.max_age))
        # Prevent shared caches from caching the CORS decision.
        handler.send_header("Vary", "Origin")

    def handle_preflight(self, handler: BaseHTTPRequestHandler) -> bool:
        """Handle CORS preflight OPTIONS request.

        Returns True if the request was handled (OPTIONS), False otherwise.
        """
        if handler.command != "OPTIONS":
            return False

        handler.send_response(204)
        self.add_headers(handler)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return True


CORS = CORSMiddleware()
