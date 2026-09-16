"""Small shared helpers for the Aurelius shell surface."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC
from datetime import datetime

__all__ = [
    "_dedupe_strings",
    "_utc_now",
]


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
