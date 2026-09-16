"""Persisted value objects and payload hydration for session storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from src.agent.session.util import _require_non_empty
from src.agent.session.util import _utc_now
from src.model.interface_framework import ApprovalRequest
from src.model.interface_framework import BackgroundJob
from src.model.interface_framework import Checkpoint
from src.model.interface_framework import InterfaceFrameworkError
from src.model.interface_framework import MessageEnvelope
from src.model.interface_framework import SkillBundle
from src.model.interface_framework import TaskThread
from src.model.interface_framework import Workstream

__all__ = [
    "WorkItem",
    "SessionRecord",
    "_SESSION_STATUSES",
    "_WORKSTREAM_STATUSES",
    "_WORK_ITEM_STATUSES",
    "_BACKGROUND_JOB_STATUSES",
    "_SESSION_EXPORT_FORMAT",
    "_SESSION_EXPORT_SCHEMA_VERSION",
    "_skill_from_payload",
    "_message_from_payload",
    "_thread_from_payload",
    "_workstream_from_payload",
    "_checkpoint_from_payload",
]


_SESSION_STATUSES = frozenset({"active", "paused", "completed", "canceled", "failed"})


_WORKSTREAM_STATUSES = frozenset({"draft", "active", "blocked", "completed", "failed", "canceled"})


_WORK_ITEM_STATUSES = frozenset({"queued", "running", "completed", "failed", "canceled"})


_BACKGROUND_JOB_STATUSES = frozenset({"pending", "running", "completed", "failed", "canceled"})


_SESSION_EXPORT_FORMAT = "aurelius.session.export"


_SESSION_EXPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class WorkItem:
    """Queued session work item."""

    item_id: str
    session_id: str
    workstream_id: str
    kind: str
    title: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: str = ""
    updated_at: str = ""
    thread_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "item_id",
            "session_id",
            "workstream_id",
            "kind",
            "title",
            "created_at",
            "updated_at",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.status not in _WORK_ITEM_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_WORK_ITEM_STATUSES)}, got {self.status!r}"
            )
        if self.thread_id is not None and not isinstance(self.thread_id, str):
            raise InterfaceFrameworkError("thread_id must be str or None")
        if not isinstance(self.payload, dict):
            raise InterfaceFrameworkError("payload must be a dict")
        if self.result is not None and not isinstance(self.result, dict):
            raise InterfaceFrameworkError("result must be dict or None")
        if self.error is not None and not isinstance(self.error, str):
            raise InterfaceFrameworkError("error must be str or None")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass
class SessionRecord:
    """JSON-safe persistent session snapshot."""

    session_id: str
    workspace: str | None
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    active_thread_id: str | None = None
    active_workstream_id: str | None = None
    memory_summary: str = ""
    workstreams: dict[str, Workstream] = field(default_factory=dict)
    threads: dict[str, TaskThread] = field(default_factory=dict)
    approvals: dict[str, ApprovalRequest] = field(default_factory=dict)
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    jobs: dict[str, BackgroundJob] = field(default_factory=dict)
    messages: dict[str, MessageEnvelope] = field(default_factory=dict)
    tool_calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    queue: list[WorkItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.session_id, "session_id")
        _require_non_empty(self.created_at, "created_at")
        _require_non_empty(self.updated_at, "updated_at")
        if self.status not in _SESSION_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_SESSION_STATUSES)}, got {self.status!r}"
            )
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise InterfaceFrameworkError("workspace must be str or None")
        if self.active_thread_id is not None and not isinstance(self.active_thread_id, str):
            raise InterfaceFrameworkError("active_thread_id must be str or None")
        if self.active_workstream_id is not None and not isinstance(self.active_workstream_id, str):
            raise InterfaceFrameworkError("active_workstream_id must be str or None")
        if not isinstance(self.workstreams, dict):
            raise InterfaceFrameworkError("workstreams must be a dict")
        if not isinstance(self.threads, dict):
            raise InterfaceFrameworkError("threads must be a dict")
        if not isinstance(self.approvals, dict):
            raise InterfaceFrameworkError("approvals must be a dict")
        if not isinstance(self.checkpoints, dict):
            raise InterfaceFrameworkError("checkpoints must be a dict")
        if not isinstance(self.jobs, dict):
            raise InterfaceFrameworkError("jobs must be a dict")
        if not isinstance(self.messages, dict):
            raise InterfaceFrameworkError("messages must be a dict")
        if not isinstance(self.tool_calls, dict):
            raise InterfaceFrameworkError("tool_calls must be a dict")
        if not isinstance(self.queue, list):
            raise InterfaceFrameworkError("queue must be a list")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SessionRecord:
        if not isinstance(payload, Mapping):
            raise InterfaceFrameworkError("session payload must be a mapping")
        workstreams = {
            key: _workstream_from_payload(value)
            for key, value in dict(payload.get("workstreams", {})).items()
        }
        threads = {
            key: _thread_from_payload(value)
            for key, value in dict(payload.get("threads", {})).items()
        }
        approvals = {
            key: ApprovalRequest(
                approval_id=_require_non_empty(str(value["approval_id"]), "approval_id"),
                thread_id=_require_non_empty(str(value["thread_id"]), "thread_id"),
                category=_require_non_empty(str(value["category"]), "category"),
                action_summary=_require_non_empty(str(value["action_summary"]), "action_summary"),
                affected_resources=tuple(value.get("affected_resources", ())),
                reason=_require_non_empty(str(value["reason"]), "reason"),
                reversible=value.get("reversible", False),
                minimum_scope=_require_non_empty(
                    str(value.get("minimum_scope", "allow_once")), "minimum_scope"
                ),
                decision=str(value.get("decision", "pending")),
                created_at=_require_non_empty(
                    str(value.get("created_at", _utc_now())), "created_at"
                ),
                decided_at=value.get("decided_at"),
                metadata=dict(value.get("metadata", {})),
            )
            for key, value in dict(payload.get("approvals", {})).items()
        }
        checkpoints = {
            key: _checkpoint_from_payload(value)
            for key, value in dict(payload.get("checkpoints", {})).items()
        }
        jobs = {key: BackgroundJob(**value) for key, value in dict(payload.get("jobs", {})).items()}
        messages = {
            key: _message_from_payload(value)
            for key, value in dict(payload.get("messages", {})).items()
        }
        queue = [WorkItem(**value) for value in payload.get("queue", [])]
        return cls(
            session_id=_require_non_empty(str(payload["session_id"]), "session_id"),
            workspace=payload.get("workspace"),
            status=str(payload.get("status", "active")),
            created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
            updated_at=_require_non_empty(str(payload.get("updated_at", _utc_now())), "updated_at"),
            active_thread_id=payload.get("active_thread_id"),
            active_workstream_id=payload.get("active_workstream_id"),
            memory_summary=str(payload.get("memory_summary", "")),
            workstreams=workstreams,
            threads=threads,
            approvals=approvals,
            checkpoints=checkpoints,
            jobs=jobs,
            messages=messages,
            tool_calls={
                k: [dict(entry) for entry in v]
                for k, v in dict(payload.get("tool_calls", {})).items()
            },
            queue=queue,
            metadata=dict(payload.get("metadata", {})),
        )


def _skill_from_payload(payload: Mapping[str, Any]) -> SkillBundle:
    return SkillBundle(
        skill_id=_require_non_empty(str(payload["skill_id"]), "skill_id"),
        name=_require_non_empty(str(payload.get("name", payload["skill_id"])), "name"),
        description=str(payload.get("description", "")),
        scope=str(payload.get("scope", "thread")),
        instructions=str(payload.get("instructions", "")),
        scripts=tuple(payload.get("scripts", ())),
        resources=tuple(payload.get("resources", ())),
        entrypoints=tuple(payload.get("entrypoints", ())),
        version=payload.get("version"),
        provenance=payload.get("provenance"),
        source_path=payload.get("source_path"),
        metadata=dict(payload.get("metadata", {})),
    )


def _message_from_payload(payload: Mapping[str, Any]) -> MessageEnvelope:
    return MessageEnvelope(
        envelope_id=_require_non_empty(str(payload["envelope_id"]), "envelope_id"),
        channel_id=_require_non_empty(str(payload["channel_id"]), "channel_id"),
        thread_id=payload.get("thread_id"),
        sender=_require_non_empty(str(payload.get("sender", "gateway")), "sender"),
        recipient=payload.get("recipient"),
        kind=_require_non_empty(str(payload.get("kind", "message")), "kind"),
        payload=dict(payload.get("payload", {})),
        created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
        session_id=payload.get("session_id"),
        workstream_id=payload.get("workstream_id"),
        workspace=payload.get("workspace"),
        metadata=dict(payload.get("metadata", {})),
    )


def _thread_from_payload(payload: Mapping[str, Any]) -> TaskThread:
    skills = tuple(_skill_from_payload(item) for item in payload.get("skills", ()))
    messages = tuple(_message_from_payload(item) for item in payload.get("message_history", ()))
    return TaskThread(
        thread_id=_require_non_empty(str(payload["thread_id"]), "thread_id"),
        title=_require_non_empty(str(payload["title"]), "title"),
        mode=_require_non_empty(str(payload["mode"]), "mode"),
        status=_require_non_empty(str(payload["status"]), "status"),
        host=_require_non_empty(str(payload["host"]), "host"),
        session_id=payload.get("session_id"),
        workstream_id=payload.get("workstream_id"),
        workstream_name=payload.get("workstream_name"),
        workspace=payload.get("workspace"),
        workspace_roots=tuple(payload.get("workspace_roots", ())),
        channel=payload.get("channel"),
        repo_instructions=payload.get("repo_instructions"),
        workspace_instructions=payload.get("workspace_instructions"),
        instruction_stack=tuple(payload.get("instruction_stack", ())),
        skills=skills,
        approvals=tuple(payload.get("approvals", ())),
        checkpoints=tuple(payload.get("checkpoints", ())),
        steps=tuple(payload.get("steps", ())),
        created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
        updated_at=_require_non_empty(str(payload.get("updated_at", _utc_now())), "updated_at"),
        parent_thread_id=payload.get("parent_thread_id"),
        parent_checkpoint_id=payload.get("parent_checkpoint_id"),
        lineage=tuple(payload.get("lineage", ())),
        task_prompt=_require_non_empty(str(payload.get("task_prompt", "")), "task_prompt"),
        memory_summary=str(payload.get("memory_summary", "")),
        last_model_response=payload.get("last_model_response"),
        last_tool_result=payload.get("last_tool_result"),
        active_job_ids=tuple(payload.get("active_job_ids", ())),
        message_history=messages,
        metadata=dict(payload.get("metadata", {})),
    )


def _workstream_from_payload(payload: Mapping[str, Any]) -> Workstream:
    messages = tuple(_message_from_payload(item) for item in payload.get("messages", ()))
    queued_items = tuple(dict(item) for item in payload.get("queued_items", ()))
    return Workstream(
        workstream_id=_require_non_empty(str(payload["workstream_id"]), "workstream_id"),
        session_id=_require_non_empty(str(payload["session_id"]), "session_id"),
        name=_require_non_empty(str(payload["name"]), "name"),
        status=_require_non_empty(str(payload.get("status", "draft")), "status"),
        current_thread_id=payload.get("current_thread_id"),
        thread_ids=tuple(payload.get("thread_ids", ())),
        queued_items=queued_items,
        messages=messages,
        checkpoint_ids=tuple(payload.get("checkpoint_ids", ())),
        workspace=payload.get("workspace"),
        created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
        updated_at=_require_non_empty(str(payload.get("updated_at", _utc_now())), "updated_at"),
        metadata=dict(payload.get("metadata", {})),
    )


def _checkpoint_from_payload(payload: Mapping[str, Any]) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=_require_non_empty(str(payload["checkpoint_id"]), "checkpoint_id"),
        thread_id=_require_non_empty(str(payload["thread_id"]), "thread_id"),
        created_at=_require_non_empty(str(payload["created_at"]), "created_at"),
        lineage=tuple(payload.get("lineage", ())),
        thread_snapshot=dict(payload.get("thread_snapshot", {})),
        contract_metadata=dict(payload.get("contract_metadata", {})),
        model_context=dict(payload.get("model_context", {})),
        memory_summary=_require_non_empty(str(payload.get("memory_summary", "")), "memory_summary"),
        last_model_response=payload.get("last_model_response"),
        last_tool_result=payload.get("last_tool_result"),
        session_id=payload.get("session_id"),
        workstream_id=payload.get("workstream_id"),
        workstream_name=payload.get("workstream_name"),
        pending_approval_ids=tuple(payload.get("pending_approval_ids", ())),
        active_job_ids=tuple(payload.get("active_job_ids", ())),
        tool_observations=tuple(dict(item) for item in payload.get("tool_observations", ())),
    )
