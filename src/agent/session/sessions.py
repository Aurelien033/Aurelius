"""Session lifecycle: create, import/export, status and snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any
import json
import uuid

from src.agent.session.models import SessionRecord
from src.agent.session.models import _SESSION_EXPORT_FORMAT
from src.agent.session.models import _SESSION_EXPORT_SCHEMA_VERSION
from src.agent.session.models import _SESSION_STATUSES
from src.agent.session.util import _coerce_optional_text
from src.agent.session.util import _json_safe
from src.agent.session.util import _utc_now
from src.agent.session_journal import SessionJournal
from src.model.interface_framework import InterfaceFrameworkError

__all__ = [
    "SessionLifecycleMixin",
]


class SessionLifecycleMixin:
    """Session lifecycle surface."""

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------
    def create_session(
        self,
        session_id: str | None = None,
        *,
        workspace: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionRecord:
        if session_id is None:
            session_id = f"session-{uuid.uuid4()}"
        if session_id in self._sessions:
            raise InterfaceFrameworkError(f"session already exists: {session_id!r}")
        created_at = _utc_now()
        record = SessionRecord(
            session_id=session_id,
            workspace=_coerce_optional_text(workspace, "workspace"),
            created_at=created_at,
            updated_at=created_at,
            metadata=dict(metadata or {}),
        )
        self._sessions[session_id] = record
        self._persist(record)
        self.append_journal_entry(
            session_id,
            kind="session.created",
            summary=f"Created session {session_id}",
            payload={
                "session": self.snapshot(record),
            },
        )
        return record

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        journal = self.get_journal(session_id, create=False)
        if journal is None:
            journal = SessionJournal.create(
                session.session_id,
                created_at=session.created_at,
                metadata={"workspace": session.workspace},
            )
        return {
            "format": _SESSION_EXPORT_FORMAT,
            "schema_version": _SESSION_EXPORT_SCHEMA_VERSION,
            "exported_at": _utc_now(),
            "session_id": session.session_id,
            "state_dir": str(self.state_dir),
            "session": self.snapshot(session),
            "journal": journal.snapshot(),
        }

    def write_session_export(self, session_id: str, path: str | Path) -> Path:
        raw_target = Path(path).expanduser()
        if raw_target.is_symlink():
            raise ValueError(f"export path must not be a symlink: {path!r}")
        target = raw_target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.export_session(session_id)
        try:
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise InterfaceFrameworkError(f"cannot write session export: {target}") from exc
        return target

    def import_session_export(
        self,
        payload_or_path: Mapping[str, Any] | str | Path,
        *,
        replace: bool = False,
    ) -> SessionRecord:
        if isinstance(payload_or_path, (str, Path)):
            path = Path(payload_or_path).expanduser().resolve()
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except OSError as exc:  # pragma: no cover - filesystem failure
                raise InterfaceFrameworkError(f"cannot read session export: {path}") from exc
            except json.JSONDecodeError as exc:
                raise InterfaceFrameworkError(f"session export is not valid JSON: {path}") from exc
        elif isinstance(payload_or_path, Mapping):
            payload = dict(payload_or_path)
        else:
            raise InterfaceFrameworkError(
                "payload_or_path must be a mapping or a path to a JSON export"
            )
        if not isinstance(payload, dict):
            raise InterfaceFrameworkError("session export payload must be an object")
        if payload.get("format") != _SESSION_EXPORT_FORMAT:
            raise InterfaceFrameworkError("session export format is not recognized")
        if payload.get("schema_version") != _SESSION_EXPORT_SCHEMA_VERSION:
            raise InterfaceFrameworkError("session export schema_version is not supported")
        session_payload = payload.get("session")
        journal_payload = payload.get("journal")
        if not isinstance(session_payload, dict):
            raise InterfaceFrameworkError("session export missing session payload")
        if not isinstance(journal_payload, dict):
            raise InterfaceFrameworkError("session export missing journal payload")
        session = SessionRecord.from_dict(session_payload)
        journal = SessionJournal.from_dict(journal_payload)
        if session.session_id != journal.session_id:
            raise InterfaceFrameworkError("session export session and journal ids do not match")
        existing_session = self.get_session(session.session_id)
        if existing_session is not None and not replace:
            raise InterfaceFrameworkError(f"session already exists: {session.session_id!r}")
        self._sessions[session.session_id] = session
        self._journals[session.session_id] = journal
        self._persist(session)
        self._persist_journal(journal)
        return session

    def ensure_session(
        self,
        session_id: str | None = None,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionRecord:
        if session_id is None:
            return self.create_session(
                workspace=_coerce_optional_text(workspace, "workspace"),
                metadata=metadata,
            )
        session = self.get_session(session_id)
        if session is None:
            return self.create_session(
                session_id=session_id,
                workspace=_coerce_optional_text(workspace, "workspace"),
                metadata=metadata,
            )
        updated = False
        if workspace is not None:
            normalized_workspace = _coerce_optional_text(workspace, "workspace")
            if session.workspace != normalized_workspace:
                session.workspace = normalized_workspace
                updated = True
        if metadata:
            session.metadata.update(dict(metadata))
            updated = True
        if updated:
            session.status = "active"
            self._touch_session(session_id)
            self.append_journal_entry(
                session_id,
                kind="session.updated",
                summary=f"Updated session {session_id}",
                payload={
                    "workspace": session.workspace,
                    "metadata": _json_safe(session.metadata),
                },
            )
        return session

    def get_session(self, session_id: str) -> SessionRecord | None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise InterfaceFrameworkError("session_id must be a non-empty string")
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        path = self._session_path(session_id)
        if not path.exists():
            return None
        session = self._load_session(path)
        self._sessions[session_id] = session
        return session

    def reload_session(self, session_id: str) -> SessionRecord:
        path = self._session_path(session_id)
        if not path.exists():
            raise InterfaceFrameworkError(f"unknown session: {session_id!r}")
        session = self._load_session(path)
        self._sessions[session_id] = session
        return session

    def resume_session(self, session_id: str) -> SessionRecord:
        return self.set_session_status(session_id, "active")

    def pause_session(self, session_id: str) -> SessionRecord:
        return self.set_session_status(session_id, "paused")

    def set_session_status(self, session_id: str, status: str) -> SessionRecord:
        if status not in _SESSION_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_SESSION_STATUSES)}, got {status!r}"
            )
        session = self._require_session(session_id)
        session.status = status
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="session.status.changed",
            summary=f"Set session {session_id} status to {status}",
            payload={"status": status},
        )
        return replace(session)

    def list_sessions(self) -> list[SessionRecord]:
        sessions = list(self._sessions.values())
        if not sessions:
            for path in sorted(self.state_dir.glob("*.json")):
                session = self._load_session(path)
                self._sessions[session.session_id] = session
            sessions = list(self._sessions.values())
        return sorted(sessions, key=lambda item: (item.updated_at, item.session_id))

    def session_count(self) -> int:
        return len(self.list_sessions())

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def status(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise InterfaceFrameworkError(f"unknown session: {session_id!r}")
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        return {
            "session": self.snapshot(session),
            "counts": {
                "threads": len(session.threads),
                "workstreams": len(session.workstreams),
                "approvals": len(session.approvals),
                "checkpoints": len(session.checkpoints),
                "jobs": len(session.jobs),
                "messages": len(session.messages),
                "tool_calls": sum(len(entries) for entries in session.tool_calls.values()),
                "queue": len(session.queue),
                "journal_entries": len(journal.entries),
                "journal_branches": len(journal.branches),
                "journal_compactions": len(journal.compactions),
            },
            "thread_ids": list(session.threads),
            "workstream_ids": list(session.workstreams),
            "job_ids": list(session.jobs),
            "checkpoint_ids": list(session.checkpoints),
            "approval_ids": list(session.approvals),
            "journal": journal.describe(),
        }

    def snapshot(self, session: SessionRecord) -> dict[str, Any]:
        return _json_safe(asdict(session))
