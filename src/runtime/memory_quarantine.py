"""Runtime helpers for memory-admission quarantine reporting.

This module is the bridge between the deterministic safety admission controller
and AMC memory persistence. It gives CLI/API surfaces a stable, JSON-friendly
report before any untrusted candidate is written to durable memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.safety.admission_controller import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionSignal,
    SafetyAdmissionController,
)


@dataclass(frozen=True)
class MemoryCandidate:
    """Normalized durable-memory candidate."""

    content: str
    source: str = "memory_candidate"


def build_memory_quarantine_report(
    candidates: Sequence[str | Mapping[str, Any] | MemoryCandidate],
    *,
    existing_memories: Sequence[str] | None = None,
    controller: SafetyAdmissionController | None = None,
) -> dict[str, Any]:
    """Assess memory candidates and return trusted/quarantined JSON records."""
    admission = controller or SafetyAdmissionController()
    existing = list(existing_memories or [])
    trusted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    redacted = 0
    blocked = 0

    for raw_candidate in candidates:
        candidate = _coerce_candidate(raw_candidate)
        decision = admission.assess_memory_candidate(
            candidate.content,
            source=candidate.source,
            existing_memories=existing,
        )
        record = _record_from_decision(candidate, decision)
        if decision.action == AdmissionAction.REDACT:
            redacted += 1
        if decision.action == AdmissionAction.BLOCK:
            blocked += 1
        if decision.action == AdmissionAction.QUARANTINE or not decision.allowed:
            quarantined.append(record)
        else:
            trusted.append(record)

    return {
        "summary": {
            "total": len(trusted) + len(quarantined),
            "trusted": len(trusted),
            "quarantined": len(quarantined),
            "redacted": redacted,
            "blocked": blocked,
        },
        "trusted": trusted,
        "quarantined": quarantined,
    }


def _coerce_candidate(raw: str | Mapping[str, Any] | MemoryCandidate) -> MemoryCandidate:
    if isinstance(raw, MemoryCandidate):
        return raw
    if isinstance(raw, str):
        return MemoryCandidate(content=raw)
    if isinstance(raw, Mapping):
        content = raw.get("content")
        if not isinstance(content, str):
            raise TypeError("memory candidate mapping must contain a string 'content' field")
        source = raw.get("source", "memory_candidate")
        if not isinstance(source, str):
            raise TypeError("memory candidate mapping 'source' field must be a string")
        return MemoryCandidate(content=content, source=source)
    raise TypeError(f"unsupported memory candidate type: {type(raw).__name__}")


def _record_from_decision(
    candidate: MemoryCandidate,
    decision: AdmissionDecision,
) -> dict[str, Any]:
    sanitized = decision.sanitized_input != candidate.content
    return {
        "source": candidate.source,
        "action": decision.action.value,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "risk_score": decision.risk_score,
        "content": decision.sanitized_input,
        "sanitized": sanitized,
        "signals": [_signal_to_dict(signal) for signal in decision.signals],
        "metadata": dict(decision.metadata),
    }


def _signal_to_dict(signal: AdmissionSignal) -> dict[str, Any]:
    return {
        "name": signal.name,
        "score": signal.score,
        "severity": signal.severity,
        "details": dict(signal.details),
    }


__all__ = [
    "MemoryCandidate",
    "build_memory_quarantine_report",
]
