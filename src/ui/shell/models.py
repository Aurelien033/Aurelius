"""Value objects and constants shared by the Aurelius shell surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
import json

__all__ = [
    "AureliusShellError",
    "MessageEnvelope",
    "SkillRecord",
    "Workstream",
    "WorkflowStep",
    "WorkflowRun",
    "_KNOWN_WORKFLOW_STEP_KINDS",
]


_KNOWN_WORKFLOW_STEP_KINDS = (
    "message",
    "approval",
    "tool_call",
    "checkpoint",
    "background_job",
    "subagent",
)


class AureliusShellError(ValueError):
    """Raised when the shell surface encounters malformed state or input."""


@dataclass(frozen=True)
class MessageEnvelope:
    """Channel-aware message envelope recorded by the shell."""

    channel: str
    thread_id: str | None
    sender: str
    kind: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("channel", "sender", "kind", "content", "created_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if self.thread_id is not None and not isinstance(self.thread_id, str):
            raise AureliusShellError("thread_id must be a str or None")
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MessageEnvelope:
        return cls(
            channel=payload["channel"],
            thread_id=payload.get("thread_id"),
            sender=payload["sender"],
            kind=payload["kind"],
            content=payload["content"],
            created_at=payload["created_at"],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SkillRecord:
    """Local-first skill catalog record."""

    skill_id: str
    name: str
    source_path: str
    provenance: str
    summary: str = ""
    scope: str = "workspace"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("skill_id", "name", "source_path", "provenance", "scope"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if not isinstance(self.summary, str):
            raise AureliusShellError("summary must be a string")
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SkillRecord:
        return cls(
            skill_id=payload["skill_id"],
            name=payload["name"],
            source_path=payload["source_path"],
            provenance=payload["provenance"],
            summary=payload.get("summary", ""),
            scope=payload.get("scope", "workspace"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class Workstream:
    """Named shell workstream grouping one or more Aurelius threads."""

    workstream_id: str
    name: str
    workspace: str | None
    thread_ids: tuple[str, ...] = field(default_factory=tuple)
    status: str = "open"
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("workstream_id", "name", "status", "created_at", "updated_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise AureliusShellError("workspace must be a str or None")
        if not isinstance(self.thread_ids, tuple):
            raise AureliusShellError("thread_ids must be a tuple")
        if not all(isinstance(item, str) and item for item in self.thread_ids):
            raise AureliusShellError("thread_ids entries must be non-empty strings")
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Workstream:
        return cls(
            workstream_id=payload["workstream_id"],
            name=payload["name"],
            workspace=payload.get("workspace"),
            thread_ids=tuple(payload.get("thread_ids", ())),
            status=payload.get("status", "open"),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorkflowStep:
    """Normalized workflow step for shell execution."""

    index: int
    kind: str
    payload: dict[str, Any]
    approval_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise AureliusShellError("index must be a non-negative integer")
        if self.kind not in _KNOWN_WORKFLOW_STEP_KINDS:
            raise AureliusShellError(
                f"kind must be one of {_KNOWN_WORKFLOW_STEP_KINDS}, got {self.kind!r}"
            )
        if not isinstance(self.payload, dict):
            raise AureliusShellError("payload must be a dict")
        if not isinstance(self.approval_required, bool):
            raise AureliusShellError("approval_required must be bool")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowStep:
        return cls(
            index=payload["index"],
            kind=payload["kind"],
            payload=dict(payload.get("payload", {})),
            approval_required=bool(payload.get("approval_required", False)),
        )


@dataclass(frozen=True)
class WorkflowRun:
    """Result of executing a workflow inside the shell."""

    run_id: str
    thread_id: str
    workflow_name: str
    status: str
    steps: tuple[WorkflowStep, ...]
    transcript: tuple[dict[str, Any], ...]
    created_at: str
    updated_at: str
    halted_step_index: int | None = None
    halted_reason: str | None = None
    final_artifact: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "thread_id",
            "workflow_name",
            "status",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if not isinstance(self.steps, tuple):
            raise AureliusShellError("steps must be a tuple")
        if not all(isinstance(item, WorkflowStep) for item in self.steps):
            raise AureliusShellError("steps entries must be WorkflowStep instances")
        if not isinstance(self.transcript, tuple):
            raise AureliusShellError("transcript must be a tuple")
        if not all(isinstance(item, dict) for item in self.transcript):
            raise AureliusShellError("transcript entries must be dict objects")
        if self.halted_step_index is not None and (
            not isinstance(self.halted_step_index, int) or self.halted_step_index < 0
        ):
            raise AureliusShellError("halted_step_index must be a non-negative int or None")
        if self.halted_reason is not None and not isinstance(self.halted_reason, str):
            raise AureliusShellError("halted_reason must be a str or None")
        if self.final_artifact is not None and not isinstance(self.final_artifact, dict):
            raise AureliusShellError("final_artifact must be a dict or None")
        if self.final_artifact is not None:
            json.dumps(self.final_artifact, sort_keys=True)
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")
        json.dumps(self.metadata, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowRun:
        return cls(
            run_id=payload["run_id"],
            thread_id=payload["thread_id"],
            workflow_name=payload["workflow_name"],
            status=payload["status"],
            steps=tuple(WorkflowStep.from_dict(item) for item in payload.get("steps", ())),
            transcript=tuple(dict(item) for item in payload.get("transcript", ())),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            halted_step_index=payload.get("halted_step_index"),
            halted_reason=payload.get("halted_reason"),
            final_artifact=payload.get("final_artifact"),
            metadata=dict(payload.get("metadata", {})),
        )
