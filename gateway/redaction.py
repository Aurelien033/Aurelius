"""Centralized redaction for the Python gateway.

The pre-remediation gateway may have interpolated user
input into log messages without sanitization. This
module exports redact() that handles:

- Authorization: Bearer xxx
- Cookie: aurelius_sid=xxx; aurelius_csrf=xxx
- X-API-Key / X-Aurelius-Session
- AWS-style keys
- PEM blocks
- JWTs
- CR/LF, control chars, ANSI escapes
- <script>/<iframe> tags
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["redact", "redact_value", "redact_json"]


_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Authorization", re.compile(
        r"(?:authorization|Authorization)\s*[:=]\s*"
        r"(?:Bearer|Basic|Token|ApiKey)\s+[A-Za-z0-9._\-+/=]{4,}"
    )),
    ("Cookie", re.compile(
        r"(aurelius_(?:sid|csrf|session|api[-_]?key))\s*=\s*([^;\s]+)",
        re.IGNORECASE,
    )),
    ("X-Key", re.compile(
        r"(x-(?:api[-_]?key|aurelius[-_]?session))\s*[:=]\s*([^\s,;]+)",
        re.IGNORECASE,
    )),
    ("AWS", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("PEM", re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")),
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CRLF = re.compile(r"[\r\n]+")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_HTML = re.compile(r"<\/?(?:script|iframe|object|embed|svg)[^>]*>", re.IGNORECASE)


def redact(input_: Any) -> dict[str, Any]:
    """Return a dict with the redacted string, the hit
    names, and a flag."""
    if input_ is None:
        return {"value": "", "redacted": False, "hits": []}
    s = input_ if isinstance(input_, str) else str(input_)
    hits: list[str] = []
    value = s
    for name, pat in _SECRET_PATTERNS:
        if pat.search(value):
            hits.append(name)
            value = pat.sub(f"[REDACTED:{name}]", value)
    if _CONTROL_CHARS.search(value):
        value = _CONTROL_CHARS.sub("", value)
        hits.append("CTRL")
    if _CRLF.search(value):
        value = _CRLF.sub(" ", value)
        hits.append("CRLF")
    if _ANSI.search(value):
        value = _ANSI.sub("", value)
        hits.append("ANSI")
    if _HTML.search(value):
        value = _HTML.sub("[REDACTED:HTML]", value)
        hits.append("HTML")
    return {"value": value, "redacted": bool(hits), "hits": hits}


def redact_value(input_: Any) -> str:
    return redact(input_)["value"]


def redact_json(obj: Any) -> str:
    """JSON.stringify with redaction of every leaf value."""
    import json
    if obj is None or isinstance(obj, (bool, int, float)):
        return json.dumps(obj)
    if isinstance(obj, str):
        return json.dumps(redact_value(obj))
    if isinstance(obj, list):
        return "[" + ",".join(redact_json(v) for v in obj) + "]"
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out.append(json.dumps(k) + ":" + redact_json(v))
        return "{" + ",".join(out) + "}"
    return json.dumps(redact_value(str(obj)))
