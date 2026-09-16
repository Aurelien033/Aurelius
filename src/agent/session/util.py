"""Shared id/path safety and JSON coercion helpers for session storage."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re

from src.model.interface_framework import InterfaceFrameworkError

__all__ = [
    "_utc_now",
    "_SAFE_ID_RE",
    "_require_non_empty",
    "_safe_filename",
    "_resolve_safe_path",
    "_coerce_optional_text",
    "_json_safe",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
    return value


def _safe_filename(value: str, field_name: str) -> str:
    """Validate *value* is safe to use as a filename component.

    Rejects path separators and parent-directory references to prevent
    directory-traversal attacks.
    """
    _require_non_empty(value, field_name)
    if not _SAFE_ID_RE.match(value):
        raise InterfaceFrameworkError(
            f"{field_name} contains invalid characters; "
            "only alphanumeric characters, hyphens, and underscores are allowed"
        )
    lower = value.lower()
    if ".." in lower:
        raise InterfaceFrameworkError(f"{field_name} contains directory-traversal patterns")
    return value


def _resolve_safe_path(root: Path, candidate: str | Path) -> Path:
    """Resolve *candidate* within *root*, rejecting traversals and symlinks."""
    resolved_root = Path(root).expanduser().resolve()
    raw = Path(candidate).expanduser()
    resolved_candidate = raw.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(f"path {candidate!r} escapes allowed root {root!r}")
    if raw.is_symlink():
        raise ValueError(f"symlink paths are not permitted: {candidate!r}")
    return resolved_candidate


def _coerce_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError(f"{field_name} must be a non-empty string or None")
    return value


def _json_safe(value: Any) -> Any:
    """Return a JSON-round-trippable copy of *value*.

    Non-serializable objects are coerced to strings rather than crashing.
    """

    def _default(obj: Any) -> Any:
        return str(obj)

    try:
        return json.loads(json.dumps(value, sort_keys=True, default=_default))
    except (TypeError, ValueError) as exc:
        raise InterfaceFrameworkError(f"value is not JSON-serializable: {exc}") from exc
