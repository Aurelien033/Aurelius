"""Named workstreams inside a session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any
import uuid

from src.agent.session.models import _WORKSTREAM_STATUSES
from src.agent.session.util import _coerce_optional_text
from src.agent.session.util import _json_safe
from src.agent.session.util import _utc_now
from src.model.interface_framework import InterfaceFrameworkError
from src.model.interface_framework import Workstream

__all__ = [
    "SessionWorkstreamMixin",
]


class SessionWorkstreamMixin:
    """Workstream create/lookup/status surface."""

    # ------------------------------------------------------------------
    # workstreams
    # ------------------------------------------------------------------
    def create_workstream(
        self,
        session_id: str,
        name: str,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workstream:
        session = self._require_session(session_id)
        workstream_id = f"workstream-{uuid.uuid4()}"
        created_at = _utc_now()
        workstream = Workstream(
            workstream_id=workstream_id,
            session_id=session.session_id,
            name=name,
            status="active",
            workspace=_coerce_optional_text(workspace, "workspace") or session.workspace,
            created_at=created_at,
            updated_at=created_at,
            metadata=dict(metadata or {}),
        )
        session.workstreams[workstream_id] = workstream
        session.active_workstream_id = workstream_id
        session.updated_at = created_at
        self._persist(session)
        self.append_journal_entry(
            session_id,
            kind="workstream.created",
            summary=f"Created workstream {name}",
            workstream_id=workstream_id,
            payload={"workstream": _json_safe(asdict(workstream))},
        )
        return workstream

    def ensure_workstream(
        self,
        session_id: str,
        name_or_id: str,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workstream:
        session = self._require_session(session_id)
        workstream = self._resolve_workstream(session, name_or_id)
        if workstream is None:
            return self.create_workstream(
                session_id,
                name_or_id,
                workspace=workspace,
                metadata=metadata,
            )
        updated = False
        normalized_workspace = _coerce_optional_text(workspace, "workspace")
        if normalized_workspace is not None and workstream.workspace != normalized_workspace:
            workstream = replace(workstream, workspace=normalized_workspace)
            updated = True
        if metadata:
            workstream = replace(workstream, metadata={**workstream.metadata, **dict(metadata)})
            updated = True
        if updated:
            workstream = replace(workstream, updated_at=_utc_now())
            session.workstreams[workstream.workstream_id] = workstream
            self._touch_session(session_id)
            self.append_journal_entry(
                session_id,
                kind="workstream.updated",
                summary=f"Updated workstream {workstream.name}",
                workstream_id=workstream.workstream_id,
                payload={"workstream": _json_safe(asdict(workstream))},
            )
        return workstream

    def get_workstream(
        self,
        session_id: str,
        workstream_id: str,
        *,
        missing_ok: bool = False,
    ) -> Workstream | None:
        session = self.get_session(session_id)
        if session is None:
            if missing_ok:
                return None
            raise InterfaceFrameworkError(f"unknown session: {session_id!r}")
        workstream = session.workstreams.get(workstream_id)
        if workstream is not None:
            return workstream
        for candidate in session.workstreams.values():
            if candidate.name == workstream_id:
                return candidate
        if missing_ok:
            return None
        raise InterfaceFrameworkError(
            f"unknown workstream: {workstream_id!r} in session {session_id!r}"
        )

    def set_workstream_status(
        self, session_id: str, workstream_id_or_name: str, status: str
    ) -> Workstream:
        if status not in _WORKSTREAM_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_WORKSTREAM_STATUSES)}, got {status!r}"
            )
        session = self._require_session(session_id)
        workstream = self._resolve_workstream(session, workstream_id_or_name)
        if workstream is None:
            raise InterfaceFrameworkError(f"unknown workstream: {workstream_id_or_name!r}")
        updated = replace(workstream, status=status, updated_at=_utc_now())
        session.workstreams[updated.workstream_id] = updated
        if status == "active":
            session.active_workstream_id = updated.workstream_id
        self._touch_session(session_id)
        self.append_journal_entry(
            session_id,
            kind="workstream.status.changed",
            summary=f"Set workstream {updated.name} status to {status}",
            workstream_id=updated.workstream_id,
            payload={"status": status, "workstream": _json_safe(asdict(updated))},
        )
        return updated

    def list_workstreams(self, session_id: str) -> list[Workstream]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return sorted(session.workstreams.values(), key=lambda item: (item.updated_at, item.name))
